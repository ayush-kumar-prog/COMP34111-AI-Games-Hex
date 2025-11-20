"""
Enhanced Monte Carlo Tree Search with RAVE (Rapid Action Value Estimation).
This is the core search algorithm for our Hex AI.
"""

import math
import random
import copy
from time import perf_counter_ns as time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


@dataclass
class MCTSNode:
    """
    Node in the MCTS tree with RAVE statistics.
    """
    board: Board
    colour: Colour
    move: Optional[Move] = None
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)

    # Standard MCTS statistics
    visits: int = 0
    wins: int = 0

    # RAVE (All-Moves-As-First) statistics
    amaf_visits: Dict[Tuple[int, int], int] = field(default_factory=dict)
    amaf_wins: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Cached values
    untried_moves: List[Move] = field(default_factory=list)
    is_terminal: bool = False
    winner: Optional[Colour] = None


class EnhancedMCTS:
    """
    Enhanced MCTS with RAVE, virtual connections, and pattern-guided simulations.
    """

    # MCTS parameters
    EXPLORATION_CONSTANT = math.sqrt(2)  # UCB1 exploration parameter
    RAVE_CONSTANT = 300  # RAVE equivalence parameter

    def __init__(self, colour: Colour, evaluator=None, vc_detector=None, use_rave: bool = True):
        """
        Initialize Enhanced MCTS.

        Args:
            colour: Our colour
            evaluator: Position evaluator (optional)
            vc_detector: Virtual connection detector (optional)
            use_rave: Whether to use RAVE enhancement
        """
        self.colour = colour
        self.evaluator = evaluator
        self.vc_detector = vc_detector
        self.use_rave = use_rave

        # Statistics
        self.simulations_run = 0
        self.max_depth_reached = 0

    def search(self, board: Board, time_limit: int, endgame_mode: bool = False) -> Move:
        """
        Run MCTS search for the given time limit.

        Args:
            board: Current board state
            time_limit: Time limit in nanoseconds
            endgame_mode: Whether to use endgame optimizations

        Returns:
            Best move found
        """
        start_time = time()
        root = self._create_root_node(board)

        # Special case: only one legal move
        if len(root.untried_moves) == 1:
            return root.untried_moves[0]

        # Run simulations until time limit
        iterations = 0
        while time() - start_time < time_limit:
            iterations += 1

            # Selection
            node = self._select(root)

            # Expansion
            if node.untried_moves and not node.is_terminal:
                node = self._expand(node)

            # Simulation (rollout)
            if endgame_mode and self.evaluator:
                # In endgame, use evaluation instead of random rollout
                result = self._evaluate_position(node)
            else:
                result, moves_played = self._simulate(node)

                # Update RAVE statistics with moves from simulation
                if self.use_rave:
                    self._update_rave(node, moves_played, result)

            # Backpropagation
            self._backpropagate(node, result)

        # Select best move
        best_move = self._select_best_move(root)

        # Debug information
        self.simulations_run = iterations

        return best_move

    def _create_root_node(self, board: Board) -> MCTSNode:
        """Create root node with initial setup."""
        root = MCTSNode(
            board=copy.deepcopy(board),
            colour=self.colour
        )
        root.untried_moves = self._get_legal_moves(board)

        # Check if terminal
        if board.has_ended(Colour.RED):
            root.is_terminal = True
            root.winner = Colour.RED
        elif board.has_ended(Colour.BLUE):
            root.is_terminal = True
            root.winner = Colour.BLUE

        return root

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a leaf node using UCB1 with RAVE.
        """
        while not node.is_terminal:
            if node.untried_moves:
                # Node is not fully expanded
                return node
            else:
                # Select best child using UCB1+RAVE
                node = self._select_best_child(node)

        return node

    def _select_best_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 with optional RAVE."""
        best_score = -float('inf')
        best_child = None

        for child in node.children:
            if self.use_rave:
                score = self._ucb1_rave_score(child, node)
            else:
                score = self._ucb1_score(child, node)

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _ucb1_score(self, child: MCTSNode, parent: MCTSNode) -> float:
        """Standard UCB1 score."""
        if child.visits == 0:
            return float('inf')

        exploitation = child.wins / child.visits
        exploration = self.EXPLORATION_CONSTANT * math.sqrt(math.log(parent.visits) / child.visits)

        # Adjust for player to move
        if child.colour != self.colour:
            exploitation = 1 - exploitation

        return exploitation + exploration

    def _ucb1_rave_score(self, child: MCTSNode, parent: MCTSNode) -> float:
        """UCB1 score with RAVE enhancement."""
        if child.visits == 0:
            return float('inf')

        # Standard UCB1 value
        q = child.wins / child.visits
        if child.colour != self.colour:
            q = 1 - q

        # RAVE value
        move_tuple = (child.move.x, child.move.y) if child.move else None
        if move_tuple and move_tuple in parent.amaf_visits and parent.amaf_visits[move_tuple] > 0:
            rave_q = parent.amaf_wins[move_tuple] / parent.amaf_visits[move_tuple]
            if child.colour != self.colour:
                rave_q = 1 - rave_q

            # Beta parameter for weighting RAVE vs UCB1
            beta = math.sqrt(self.RAVE_CONSTANT / (3 * parent.visits + self.RAVE_CONSTANT))

            # Combined value
            combined_q = (1 - beta) * q + beta * rave_q
        else:
            combined_q = q

        # Exploration term
        exploration = self.EXPLORATION_CONSTANT * math.sqrt(math.log(parent.visits) / child.visits)

        return combined_q + exploration

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a new child."""
        # Choose move to expand
        if self.vc_detector:
            # Prioritize moves that create/defend virtual connections
            move = self._select_expansion_move_vc(node)
        else:
            # Random expansion
            move = random.choice(node.untried_moves)

        node.untried_moves.remove(move)

        # Create child node
        child_board = copy.deepcopy(node.board)
        child_board.set_tile_colour(move.x, move.y, node.colour)

        child = MCTSNode(
            board=child_board,
            colour=Colour.opposite(node.colour),
            move=move,
            parent=node
        )

        # Set up child node
        child.untried_moves = self._get_legal_moves(child_board)

        # Check if terminal
        if child_board.has_ended(Colour.RED):
            child.is_terminal = True
            child.winner = Colour.RED
        elif child_board.has_ended(Colour.BLUE):
            child.is_terminal = True
            child.winner = Colour.BLUE

        node.children.append(child)
        return child

    def _select_expansion_move_vc(self, node: MCTSNode) -> Move:
        """Select expansion move considering virtual connections."""
        # Get must-play moves
        if self.vc_detector:
            must_play = self.vc_detector.find_must_play_moves(node.board, node.colour)
            if must_play:
                for move in node.untried_moves:
                    if (move.x, move.y) in must_play:
                        return move

        # Default to random
        return random.choice(node.untried_moves)

    def _simulate(self, node: MCTSNode) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Run simulation (rollout) from node.
        Returns result and moves played for RAVE update.
        """
        if node.is_terminal:
            result = 1.0 if node.winner == self.colour else 0.0
            return result, []

        # Clone board for simulation
        sim_board = copy.deepcopy(node.board)
        current_colour = node.colour
        moves_played = []

        # Run simulation
        max_moves = 200  # Safety limit
        move_count = 0

        while not sim_board.has_ended(Colour.RED) and not sim_board.has_ended(Colour.BLUE):
            move_count += 1
            if move_count > max_moves:
                # Evaluate position if simulation takes too long
                if self.evaluator:
                    score = self.evaluator.evaluate(sim_board, self.colour)
                    return (score + 1) / 2, moves_played  # Convert to [0, 1]
                else:
                    return 0.5, moves_played  # Draw

            # Select move for simulation
            move = self._simulation_policy(sim_board, current_colour)

            if move:
                sim_board.set_tile_colour(move.x, move.y, current_colour)
                moves_played.append((move.x, move.y))
                current_colour = Colour.opposite(current_colour)
            else:
                # No legal moves
                break

        # Determine winner
        if sim_board.has_ended(self.colour):
            result = 1.0
        elif sim_board.has_ended(Colour.opposite(self.colour)):
            result = 0.0
        else:
            result = 0.5  # Should not happen in Hex

        return result, moves_played

    def _simulation_policy(self, board: Board, colour: Colour) -> Optional[Move]:
        """
        Simulation policy - not purely random, uses some heuristics.
        """
        legal_moves = self._get_legal_moves(board)

        if not legal_moves:
            return None

        # With small probability, play smart moves
        if random.random() < 0.2:
            # Prefer moves closer to winning edges
            if colour == Colour.RED:
                # RED wins top-bottom
                scored_moves = []
                for move in legal_moves:
                    edge_dist = min(move.x, board.size - 1 - move.x)
                    score = board.size - edge_dist
                    scored_moves.append((score, move))
            else:
                # BLUE wins left-right
                scored_moves = []
                for move in legal_moves:
                    edge_dist = min(move.y, board.size - 1 - move.y)
                    score = board.size - edge_dist
                    scored_moves.append((score, move))

            scored_moves.sort(reverse=True)
            # Select from top moves
            top_moves = [m for s, m in scored_moves[:5]]
            return random.choice(top_moves)

        # Default: random move
        return random.choice(legal_moves)

    def _evaluate_position(self, node: MCTSNode) -> float:
        """
        Evaluate terminal or near-terminal position.
        Used in endgame instead of simulation.
        """
        if node.is_terminal:
            return 1.0 if node.winner == self.colour else 0.0

        if self.evaluator:
            score = self.evaluator.evaluate(node.board, self.colour)
            # Convert from [-1, 1] to [0, 1]
            return (score + 1) / 2
        else:
            # No evaluator, use simulation
            result, _ = self._simulate(node)
            return result

    def _backpropagate(self, node: MCTSNode, result: float):
        """Backpropagate simulation result up the tree."""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    def _update_rave(self, node: MCTSNode, moves_played: List[Tuple[int, int]], result: float):
        """Update RAVE (AMAF) statistics."""
        # Update AMAF statistics for all ancestors
        current = node.parent  # Start from parent
        current_colour = Colour.opposite(node.colour)

        while current is not None:
            # Update AMAF for moves that could have been played from this position
            for move_tuple in moves_played:
                # Check if move was available from this position
                x, y = move_tuple
                if current.board.tiles[x][y].colour is None:
                    # Move was available
                    if move_tuple not in current.amaf_visits:
                        current.amaf_visits[move_tuple] = 0
                        current.amaf_wins[move_tuple] = 0

                    current.amaf_visits[move_tuple] += 1

                    # Adjust result based on player to move
                    adjusted_result = result if current_colour == self.colour else 1 - result
                    current.amaf_wins[move_tuple] += adjusted_result

            current = current.parent
            current_colour = Colour.opposite(current_colour)

    def _select_best_move(self, root: MCTSNode) -> Move:
        """Select the best move from root based on visit count."""
        if not root.children:
            # No children, return random legal move
            return random.choice(root.untried_moves) if root.untried_moves else Move(0, 0)

        # Select child with most visits (robust selection)
        best_child = max(root.children, key=lambda c: c.visits)

        return best_child.move

    def _get_legal_moves(self, board: Board) -> List[Move]:
        """Get all legal moves for current position."""
        moves = []
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    moves.append(Move(i, j))
        return moves