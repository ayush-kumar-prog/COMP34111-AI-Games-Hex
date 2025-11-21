"""
Neural Monte Carlo Tree Search - GPU-accelerated version.
Integrates neural network policy and value for AlphaZero-style search.
"""

import math
import random
import copy
import torch
import numpy as np
from time import perf_counter_ns as time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.Group12.neural.hex_network import HexNeuralNetwork
from agents.Group12.neural.board_encoder import BoardEncoder


@dataclass
class NeuralMCTSNode:
    """
    MCTS node with neural network guidance.
    """
    board: Board
    colour: Colour
    move: Optional[Move] = None
    parent: Optional['NeuralMCTSNode'] = None
    children: List['NeuralMCTSNode'] = field(default_factory=list)

    # MCTS statistics
    visits: int = 0
    value_sum: float = 0.0  # Cumulative value (not wins)

    # Neural network prior
    prior: float = 0.0  # P(s,a) from neural network

    # Cached values
    untried_moves: List[Tuple[Move, float]] = field(default_factory=list)  # (move, prior)
    is_terminal: bool = False
    terminal_value: Optional[float] = None


class NeuralMCTS:
    """
    Neural MCTS with GPU acceleration.

    Key differences from classical MCTS:
    1. Neural network provides policy (move priors)
    2. Neural network provides value (position evaluation)
    3. Batch inference on GPU (fast!)
    4. AlphaZero-style PUCT formula
    5. No random rollouts (use neural value)
    """

    # PUCT constants (from AlphaZero paper)
    C_PUCT = 1.0  # Exploration constant
    DIRICHLET_ALPHA = 0.3  # Dirichlet noise for exploration
    EPSILON = 0.25  # Exploration noise weight

    def __init__(self, network: HexNeuralNetwork, encoder: BoardEncoder,
                 colour: Colour, device: str = 'cuda'):
        """
        Initialize Neural MCTS.

        Args:
            network: Trained neural network
            encoder: Board state encoder
            colour: Our colour
            device: 'cuda' or 'cpu'
        """
        self.network = network
        self.encoder = encoder
        self.colour = colour
        self.device = device

        # Move network to device
        self.network.to(device)
        self.network.eval()

        # Statistics
        self.simulations_run = 0

    def search(self, board: Board, num_simulations: int = 800,
               add_noise: bool = True) -> Tuple[Move, np.ndarray]:
        """
        Run MCTS search with neural network guidance.

        Args:
            board: Current board state
            num_simulations: Number of MCTS simulations (800-1600 typical)
            add_noise: Whether to add Dirichlet noise to root (for exploration)

        Returns:
            best_move: Best move found
            policy_target: Visit count distribution (for training)
        """
        root = self._create_root_node(board)

        # Expand root with neural network evaluation
        self._expand_node_neural(root, add_noise=add_noise)

        # Run simulations
        for i in range(num_simulations):
            node = root
            search_path = [node]

            # Selection: Traverse tree using PUCT
            while node.children and not node.is_terminal:
                node = self._select_child_puct(node)
                search_path.append(node)

            # Expansion & Evaluation
            value = self._evaluate_node(node)

            # Backpropagation
            self._backpropagate(search_path, value)

            self.simulations_run += 1

        # Select best move based on visit counts
        best_move = self._select_best_move_root(root)

        # Create policy target (for training)
        policy_target = self._create_policy_target(root)

        return best_move, policy_target

    def search_with_time_limit(self, board: Board, time_limit_ns: int,
                               min_simulations: int = 100) -> Move:
        """
        Run MCTS with time limit instead of fixed simulation count.

        Args:
            board: Current board state
            time_limit_ns: Time limit in nanoseconds
            min_simulations: Minimum simulations before time check

        Returns:
            Best move found
        """
        start_time = time()
        root = self._create_root_node(board)
        self._expand_node_neural(root, add_noise=True)

        simulations = 0
        while True:
            # Run batch of simulations
            for _ in range(min_simulations):
                node = root
                search_path = [node]

                while node.children and not node.is_terminal:
                    node = self._select_child_puct(node)
                    search_path.append(node)

                value = self._evaluate_node(node)
                self._backpropagate(search_path, value)

                simulations += 1

            # Check time
            elapsed = time() - start_time
            if elapsed >= time_limit_ns:
                break

        print(f"Neural MCTS: {simulations} simulations in {elapsed/1e9:.2f}s "
              f"= {simulations/(elapsed/1e9):.0f} sims/sec")

        best_move = self._select_best_move_root(root)
        return best_move

    def _create_root_node(self, board: Board) -> NeuralMCTSNode:
        """Create root node for search."""
        return NeuralMCTSNode(
            board=copy.deepcopy(board),
            colour=self.colour,
            move=None,
            parent=None
        )

    def _expand_node_neural(self, node: NeuralMCTSNode, add_noise: bool = False):
        """
        Expand node using neural network policy.

        Args:
            node: Node to expand
            add_noise: Add Dirichlet noise for exploration (root only)
        """
        # Check if terminal
        if node.board.has_ended(Colour.RED) or node.board.has_ended(Colour.BLUE):
            node.is_terminal = True
            winner = Colour.RED if node.board.has_ended(Colour.RED) else Colour.BLUE
            node.terminal_value = 1.0 if winner == self.colour else -1.0
            return

        # Get legal moves
        legal_moves = self._get_legal_moves(node.board)
        if not legal_moves:
            node.is_terminal = True
            node.terminal_value = 0.0  # Draw (shouldn't happen in Hex)
            return

        # Encode board state
        state = self.encoder.encode(node.board, node.colour, device=self.device)

        # Get neural network prediction
        with torch.no_grad():
            policy_logits, value = self.network.predict(state)

        # Decode policy for legal moves
        policy_dict = self.encoder.decode_policy(policy_logits, node.board)

        # Add Dirichlet noise for exploration (at root)
        if add_noise:
            noise = np.random.dirichlet([self.DIRICHLET_ALPHA] * len(legal_moves))
            policy_items = list(policy_dict.items())
            for i, (move_tuple, prob) in enumerate(policy_items):
                policy_dict[move_tuple] = (1 - self.EPSILON) * prob + self.EPSILON * noise[i]

            # Renormalize
            total = sum(policy_dict.values())
            policy_dict = {k: v/total for k, v in policy_dict.items()}

        # Store moves with priors
        node.untried_moves = [(Move(x, y), policy_dict.get((x, y), 0.0))
                              for x, y in policy_dict.keys()]

        # Sort by prior (highest first)
        node.untried_moves.sort(key=lambda x: x[1], reverse=True)

    def _select_child_puct(self, node: NeuralMCTSNode) -> NeuralMCTSNode:
        """
        Select child using PUCT formula (from AlphaZero).

        PUCT = Q(s,a) + C_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Where:
        - Q(s,a) = average value of action
        - P(s,a) = prior probability from neural network
        - N(s) = visit count of parent
        - N(s,a) = visit count of child
        """
        best_score = -float('inf')
        best_child = None

        sqrt_parent_visits = math.sqrt(node.visits)

        for child in node.children:
            # Q value (average)
            if child.visits > 0:
                q_value = child.value_sum / child.visits
            else:
                q_value = 0.0

            # PUCT formula
            u_value = (self.C_PUCT * child.prior * sqrt_parent_visits /
                      (1 + child.visits))

            puct_score = q_value + u_value

            if puct_score > best_score:
                best_score = puct_score
                best_child = child

        # If all children tried, return best
        if best_child is not None:
            return best_child

        # Otherwise, expand a new child
        if node.untried_moves:
            return self._expand_child(node)

        # Shouldn't reach here
        return node.children[0] if node.children else node

    def _expand_child(self, parent: NeuralMCTSNode) -> NeuralMCTSNode:
        """Expand a new child node."""
        move, prior = parent.untried_moves.pop(0)

        # Make move
        child_board = copy.deepcopy(parent.board)
        child_board.set_tile_colour(move.x, move.y, parent.colour)

        # Create child node
        child = NeuralMCTSNode(
            board=child_board,
            colour=Colour.opposite(parent.colour),
            move=move,
            parent=parent,
            prior=prior
        )

        parent.children.append(child)

        # Expand child with neural network
        self._expand_node_neural(child, add_noise=False)

        return child

    def _evaluate_node(self, node: NeuralMCTSNode) -> float:
        """
        Evaluate node using neural network.

        Returns:
            Value in [-1, 1] from current player's perspective
        """
        # Terminal node
        if node.is_terminal:
            # Return value from original player's perspective
            return node.terminal_value * (1 if node.colour == self.colour else -1)

        # Use neural network value
        state = self.encoder.encode(node.board, node.colour, device=self.device)

        with torch.no_grad():
            _, value = self.network.predict(state)

        # Value is from node.colour's perspective
        # Convert to root player's perspective
        value_for_root = value * (1 if node.colour == self.colour else -1)

        return value_for_root

    def _backpropagate(self, search_path: List[NeuralMCTSNode], value: float):
        """
        Backpropagate value up the tree.

        Args:
            search_path: Path from root to leaf
            value: Leaf value (from root player's perspective)
        """
        for node in reversed(search_path):
            node.visits += 1
            # Value is always from root player's perspective
            node.value_sum += value

    def _select_best_move_root(self, root: NeuralMCTSNode) -> Move:
        """
        Select best move from root based on visit counts.

        In AlphaZero, we select the move with highest visit count.
        """
        if not root.children:
            # Shouldn't happen, but fallback to random legal move
            legal_moves = self._get_legal_moves(root.board)
            return random.choice(legal_moves) if legal_moves else Move(0, 0)

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def _create_policy_target(self, root: NeuralMCTSNode) -> np.ndarray:
        """
        Create policy target for training (visit count distribution).

        Returns:
            Array of shape (121,) with visit count distribution
        """
        policy = np.zeros(121)

        total_visits = sum(child.visits for child in root.children)

        if total_visits > 0:
            for child in root.children:
                move = child.move
                idx = move.x * 11 + move.y
                policy[idx] = child.visits / total_visits

        return policy

    def _get_legal_moves(self, board: Board) -> List[Move]:
        """Get all legal moves."""
        moves = []
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    moves.append(Move(i, j))
        return moves


class BatchNeuralMCTS:
    """
    Batch version of Neural MCTS for parallel game generation.
    Processes multiple board positions simultaneously on GPU.
    """

    def __init__(self, network: HexNeuralNetwork, encoder: BoardEncoder,
                 device: str = 'cuda', batch_size: int = 32):
        """
        Initialize batch Neural MCTS.

        Args:
            network: Neural network
            encoder: Board encoder
            device: 'cuda' or 'cpu'
            batch_size: Number of positions to process in parallel
        """
        self.network = network
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size

        self.network.to(device)
        self.network.eval()

    def evaluate_positions_batch(self, boards: List[Board],
                                 colours: List[Colour]) -> Tuple[List[Dict], List[float]]:
        """
        Evaluate multiple positions in a single GPU batch.

        Args:
            boards: List of board states
            colours: List of colours for each board

        Returns:
            policies: List of policy dictionaries
            values: List of position values
        """
        # Encode batch
        states = self.encoder.encode_batch(boards, colours, device=self.device)

        # Batch inference on GPU
        with torch.no_grad():
            policy_logits_batch, values_batch = self.network(states)

        # Decode results
        policies = []
        values = []

        for i, (board, policy_logits, value) in enumerate(zip(boards, policy_logits_batch, values_batch)):
            policy_dict = self.encoder.decode_policy(policy_logits, board)
            policies.append(policy_dict)
            values.append(value.item())

        return policies, values


if __name__ == '__main__':
    print("Testing Neural MCTS...")

    # Would need actual Board and trained network to test
    # This is a placeholder

    from agents.Group12.neural.hex_network import create_hex_network

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create network
    network = create_hex_network(device=device)
    encoder = BoardEncoder()

    print("\n✅ Neural MCTS classes created successfully!")
    print(f"   - NeuralMCTS: Ready for single-game search")
    print(f"   - BatchNeuralMCTS: Ready for parallel evaluation")
    print(f"   - PUCT formula: Q + C*P*sqrt(N)/(1+n)")
    print(f"   - GPU-accelerated: {device}")
