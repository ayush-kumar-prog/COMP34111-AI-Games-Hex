"""
Ultimate Hex AI Agent for Group12 - Tournament Version

This is the ultimate tournament agent combining:
1. Neural network evaluation (via NumPy - no PyTorch needed)
2. AlphaZero-style MCTS (policy-guided, value-based evaluation)
3. Virtual connection detection
4. Adaptive time management
5. Opening book with Nash equilibrium decisions

Key Features:
- Uses trained neural network for position evaluation
- MCTS guided by neural policy (focuses search on promising moves)
- No random rollouts - uses neural value head directly
- Falls back to heuristic MCTS if neural weights unavailable
- Fully self-contained, works in tournament Docker

Author: Group12
Date: November 2024
"""

import math
import copy
import random
import numpy as np
from pathlib import Path
from time import perf_counter_ns
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import deque

# Try to import scipy for efficient convolutions
try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


# ==============================================================================
# NUMPY NEURAL NETWORK INFERENCE (Self-contained)
# ==============================================================================

def numpy_conv2d(x: np.ndarray, weight: np.ndarray, padding: int = 1,
                 dilation: int = 1) -> np.ndarray:
    """2D convolution using scipy or pure numpy."""
    batch_size, in_c, h, w = x.shape
    out_c = weight.shape[0]

    # Handle dilation
    if dilation > 1:
        kh, kw = weight.shape[2], weight.shape[3]
        new_kh = kh + (kh - 1) * (dilation - 1)
        new_kw = kw + (kw - 1) * (dilation - 1)
        dilated_weight = np.zeros((out_c, in_c, new_kh, new_kw), dtype=weight.dtype)
        dilated_weight[:, :, ::dilation, ::dilation] = weight
        weight = dilated_weight
        padding = padding * dilation

    # Pad input
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    output = np.zeros((batch_size, out_c, h, w), dtype=np.float32)

    for b in range(batch_size):
        for oc in range(out_c):
            for ic in range(in_c):
                if HAS_SCIPY:
                    output[b, oc] += signal.correlate2d(
                        x[b, ic], weight[oc, ic], mode='valid'
                    )
                else:
                    # Pure numpy fallback (slower)
                    kh, kw = weight.shape[2], weight.shape[3]
                    for i in range(h):
                        for j in range(w):
                            output[b, oc, i, j] += np.sum(
                                x[b, ic, i:i+kh, j:j+kw] * weight[oc, ic]
                            )
    return output


def numpy_batchnorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                    mean: np.ndarray, var: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Batch normalization (inference mode)."""
    mean = mean.reshape(1, -1, 1, 1)
    var = var.reshape(1, -1, 1, 1)
    weight = weight.reshape(1, -1, 1, 1)
    bias = bias.reshape(1, -1, 1, 1)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def numpy_relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def numpy_linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Linear layer."""
    return np.dot(x, weight.T) + bias


def numpy_softmax(x: np.ndarray) -> np.ndarray:
    """Softmax (numerically stable)."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class NumpyNeuralNetwork:
    """Neural network inference using pure NumPy."""

    def __init__(self, weights_path: str):
        """Load weights from .npz file."""
        self.weights = dict(np.load(weights_path))
        self._detect_architecture()

    def _detect_architecture(self):
        """Detect number of residual blocks."""
        self.num_blocks = 0
        while f'res_blocks.{self.num_blocks}.conv1.weight' in self.weights:
            self.num_blocks += 1

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass."""
        w = self.weights

        # Input conv + bn + relu
        x = numpy_conv2d(x, w['conv_input.weight'], padding=1)
        x = numpy_batchnorm(x, w['bn_input.weight'], w['bn_input.bias'],
                           w['bn_input.running_mean'], w['bn_input.running_var'])
        x = numpy_relu(x)

        # Residual blocks
        for i in range(self.num_blocks):
            residual = x
            dilation = 2 if i in [5, 10] else 1

            # Conv1
            x = numpy_conv2d(x, w[f'res_blocks.{i}.conv1.weight'],
                            padding=dilation, dilation=dilation)
            x = numpy_batchnorm(x, w[f'res_blocks.{i}.bn1.weight'],
                               w[f'res_blocks.{i}.bn1.bias'],
                               w[f'res_blocks.{i}.bn1.running_mean'],
                               w[f'res_blocks.{i}.bn1.running_var'])
            x = numpy_relu(x)

            # Conv2
            x = numpy_conv2d(x, w[f'res_blocks.{i}.conv2.weight'], padding=1)
            x = numpy_batchnorm(x, w[f'res_blocks.{i}.bn2.weight'],
                               w[f'res_blocks.{i}.bn2.bias'],
                               w[f'res_blocks.{i}.bn2.running_mean'],
                               w[f'res_blocks.{i}.bn2.running_var'])

            # SE block if present
            if f'res_blocks.{i}.se.fc.0.weight' in w:
                batch, channels, h, width = x.shape
                y = np.mean(x, axis=(2, 3))
                y = numpy_relu(np.dot(y, w[f'res_blocks.{i}.se.fc.0.weight'].T))
                y = 1 / (1 + np.exp(-np.dot(y, w[f'res_blocks.{i}.se.fc.2.weight'].T)))
                x = x * y.reshape(batch, channels, 1, 1)

            x = numpy_relu(x + residual)

        # Policy head
        policy = numpy_conv2d(x, w['policy_conv.weight'], padding=0)
        policy = numpy_batchnorm(policy, w['policy_bn.weight'], w['policy_bn.bias'],
                                w['policy_bn.running_mean'], w['policy_bn.running_var'])
        policy = numpy_relu(policy)
        policy = policy.reshape(policy.shape[0], -1)
        policy = numpy_linear(policy, w['policy_fc.weight'], w['policy_fc.bias'])
        policy = numpy_softmax(policy)

        # Value head
        value = numpy_conv2d(x, w['value_conv.weight'], padding=0)
        value = numpy_batchnorm(value, w['value_bn.weight'], w['value_bn.bias'],
                               w['value_bn.running_mean'], w['value_bn.running_var'])
        value = numpy_relu(value)
        value = value.reshape(value.shape[0], -1)
        value = numpy_linear(value, w['value_fc1.weight'], w['value_fc1.bias'])
        value = numpy_relu(value)
        value = numpy_linear(value, w['value_fc2.weight'], w['value_fc2.bias'])
        value = np.tanh(value)

        return policy, value

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict for single state."""
        if state.ndim == 3:
            state = state[np.newaxis, ...]
        policy, value = self.forward(state)
        return policy[0], float(value[0, 0])


# ==============================================================================
# BOARD ENCODER
# ==============================================================================

class BoardEncoder:
    """Encode board state for neural network (8 channels)."""

    def __init__(self, board_size: int = 11):
        self.board_size = board_size
        self._precompute_distances()

    def _precompute_distances(self):
        """Pre-compute edge distances."""
        size = self.board_size
        self.red_top = np.zeros((size, size), dtype=np.float32)
        self.red_bottom = np.zeros((size, size), dtype=np.float32)
        self.blue_left = np.zeros((size, size), dtype=np.float32)
        self.blue_right = np.zeros((size, size), dtype=np.float32)

        for i in range(size):
            for j in range(size):
                self.red_top[i, j] = 1.0 - i / (size - 1)
                self.red_bottom[i, j] = i / (size - 1)
                self.blue_left[i, j] = 1.0 - j / (size - 1)
                self.blue_right[i, j] = j / (size - 1)

    def encode(self, board: Board, colour: Colour) -> np.ndarray:
        """Encode board to 8-channel tensor."""
        state = np.zeros((8, self.board_size, self.board_size), dtype=np.float32)
        opp = Colour.opposite(colour)

        for i in range(self.board_size):
            for j in range(self.board_size):
                tile = board.tiles[i][j].colour
                if tile == colour:
                    state[0, i, j] = 1.0
                elif tile == opp:
                    state[1, i, j] = 1.0
                else:
                    state[2, i, j] = 1.0

        # Edge distances
        if colour == Colour.RED:
            state[3] = self.red_top
            state[4] = self.red_bottom
            state[5] = self.blue_left
            state[6] = self.blue_right
        else:
            state[3] = self.blue_left
            state[4] = self.blue_right
            state[5] = self.red_top
            state[6] = self.red_bottom

        # Channel 7: Bridge patterns
        state[7] = self._detect_bridges(board, colour)

        return state

    def _detect_bridges(self, board: Board, colour: Colour) -> np.ndarray:
        """Detect bridge patterns."""
        bridges = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        patterns = [((0, 1), (1, 0), (1, 1)), ((0, 1), (-1, 1), (-1, 2)),
                   ((1, 0), (1, -1), (2, -1))]

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board.tiles[i][j].colour != colour:
                    continue
                for c1, c2, s2 in patterns:
                    ni, nj = i + s2[0], j + s2[1]
                    ci1, cj1 = i + c1[0], j + c1[1]
                    ci2, cj2 = i + c2[0], j + c2[1]

                    if not (0 <= ni < self.board_size and 0 <= nj < self.board_size):
                        continue
                    if not (0 <= ci1 < self.board_size and 0 <= cj1 < self.board_size):
                        continue
                    if not (0 <= ci2 < self.board_size and 0 <= cj2 < self.board_size):
                        continue

                    if (board.tiles[ni][nj].colour == colour and
                        board.tiles[ci1][cj1].colour is None and
                        board.tiles[ci2][cj2].colour is None):
                        bridges[ci1, cj1] = 1.0
                        bridges[ci2, cj2] = 1.0

        return bridges


# ==============================================================================
# NEURAL-GUIDED MCTS
# ==============================================================================

@dataclass
class MCTSNode:
    """MCTS tree node."""
    board: Board
    colour: Colour
    move: Optional[Move] = None
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    untried_moves: List[Tuple[Move, float]] = field(default_factory=list)
    is_terminal: bool = False
    terminal_value: float = 0.0


class NeuralMCTS:
    """AlphaZero-style MCTS with neural network guidance."""

    C_PUCT = 1.5  # Exploration constant
    DIRICHLET_ALPHA = 0.3
    NOISE_WEIGHT = 0.25

    def __init__(self, network: Optional[NumpyNeuralNetwork],
                 encoder: BoardEncoder, colour: Colour):
        self.network = network
        self.encoder = encoder
        self.colour = colour
        self.use_neural = network is not None

    def search(self, board: Board, num_simulations: int,
               add_noise: bool = True) -> Move:
        """Run MCTS search."""
        root = MCTSNode(board=copy.deepcopy(board), colour=self.colour)
        self._expand(root, add_noise=add_noise)

        for _ in range(num_simulations):
            node = root
            path = [node]

            # Selection
            while node.children and not node.is_terminal:
                node = self._select_child(node)
                path.append(node)

            # Expansion & Evaluation
            if not node.is_terminal and not node.children:
                self._expand(node)

            value = self._evaluate(node)

            # Backpropagation
            for n in reversed(path):
                n.visits += 1
                n.value_sum += value if n.colour == self.colour else -value

        # Select best move
        if not root.children:
            return self._random_move(board)

        best = max(root.children, key=lambda c: c.visits)
        return best.move

    def _expand(self, node: MCTSNode, add_noise: bool = False):
        """Expand node using neural network policy."""
        # Check terminal
        if node.board.has_ended(Colour.RED) or node.board.has_ended(Colour.BLUE):
            node.is_terminal = True
            winner = Colour.RED if node.board.has_ended(Colour.RED) else Colour.BLUE
            node.terminal_value = 1.0 if winner == self.colour else -1.0
            return

        # Get legal moves
        legal_moves = self._get_legal_moves(node.board)
        if not legal_moves:
            node.is_terminal = True
            return

        # Get policy from neural network
        if self.use_neural:
            state = self.encoder.encode(node.board, node.colour)
            policy, _ = self.network.predict(state)
        else:
            # Uniform policy fallback
            policy = np.ones(121) / 121

        # Filter to legal moves
        priors = {}
        for move in legal_moves:
            idx = move.x * 11 + move.y
            priors[move] = policy[idx]

        # Normalize
        total = sum(priors.values())
        if total > 0:
            priors = {m: p/total for m, p in priors.items()}

        # Add Dirichlet noise at root
        if add_noise and len(priors) > 0:
            noise = np.random.dirichlet([self.DIRICHLET_ALPHA] * len(priors))
            for i, move in enumerate(priors.keys()):
                priors[move] = ((1 - self.NOISE_WEIGHT) * priors[move] +
                               self.NOISE_WEIGHT * noise[i])

        node.untried_moves = [(m, p) for m, p in priors.items()]
        node.untried_moves.sort(key=lambda x: x[1], reverse=True)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child using PUCT formula."""
        # First, expand if untried moves exist
        if node.untried_moves:
            move, prior = node.untried_moves.pop(0)
            child_board = copy.deepcopy(node.board)
            child_board.set_tile_colour(move.x, move.y, node.colour)

            child = MCTSNode(
                board=child_board,
                colour=Colour.opposite(node.colour),
                move=move,
                parent=node,
                prior=prior
            )
            node.children.append(child)
            return child

        # PUCT selection
        sqrt_total = math.sqrt(node.visits)
        best_score = -float('inf')
        best_child = None

        for child in node.children:
            q = child.value_sum / max(1, child.visits)
            if child.colour != self.colour:
                q = -q
            u = self.C_PUCT * child.prior * sqrt_total / (1 + child.visits)
            score = q + u

            if score > best_score:
                best_score = score
                best_child = child

        return best_child or node.children[0]

    def _evaluate(self, node: MCTSNode) -> float:
        """Evaluate node using neural network value."""
        if node.is_terminal:
            return node.terminal_value

        if self.use_neural:
            state = self.encoder.encode(node.board, node.colour)
            _, value = self.network.predict(state)
            # Convert to root player's perspective
            if node.colour != self.colour:
                value = -value
            return value
        else:
            # Heuristic fallback
            return self._heuristic_eval(node.board, self.colour)

    def _heuristic_eval(self, board: Board, colour: Colour) -> float:
        """Simple heuristic evaluation."""
        score = 0.0
        size = board.size
        center = size // 2

        for i in range(size):
            for j in range(size):
                tile = board.tiles[i][j].colour
                if tile is None:
                    continue

                # Center control
                dist = abs(i - center) + abs(j - center)
                cell_score = 0.1 * (1 - dist / size)

                if tile == colour:
                    score += cell_score
                else:
                    score -= cell_score

        return np.tanh(score)

    def _get_legal_moves(self, board: Board) -> List[Move]:
        """Get all legal moves."""
        moves = []
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    moves.append(Move(i, j))
        return moves

    def _random_move(self, board: Board) -> Move:
        """Get random legal move."""
        moves = self._get_legal_moves(board)
        return random.choice(moves) if moves else Move(0, 0)


# ==============================================================================
# TIME MANAGEMENT
# ==============================================================================

class TimeManager:
    """Adaptive time management."""

    TOTAL_TIME_NS = 3 * 60 * 10**9  # 3 minutes
    SAFETY_MARGIN = 0.10  # 10% safety buffer

    def __init__(self):
        self.time_used = 0
        self.moves_made = 0

    def get_time_for_move(self, board: Board, remaining_ns: int) -> int:
        """Calculate time allocation for this move."""
        empty_cells = sum(1 for i in range(board.size)
                         for j in range(board.size)
                         if board.tiles[i][j].colour is None)

        # Estimate remaining moves
        est_remaining = max(10, empty_cells // 2)

        # Base allocation
        safe_remaining = remaining_ns * (1 - self.SAFETY_MARGIN)
        base_time = safe_remaining / est_remaining

        # Adjust based on game phase
        progress = 1 - (empty_cells / 121)

        if progress < 0.2:
            # Opening: spend more time
            time_budget = base_time * 1.5
        elif progress > 0.7:
            # Endgame: play faster
            time_budget = base_time * 0.7
        else:
            time_budget = base_time

        # Emergency mode
        if remaining_ns < self.TOTAL_TIME_NS * 0.05:
            time_budget = min(time_budget, 0.5 * 10**9)  # Max 0.5 seconds

        return int(time_budget)


# ==============================================================================
# OPENING BOOK
# ==============================================================================

class OpeningBook:
    """Nash equilibrium opening decisions."""

    # Strong opening moves for RED
    STRONG_RED_MOVES = [(5, 5), (5, 6), (4, 5), (5, 4), (6, 5)]

    # Swap threshold: swap if opponent move is >= this value
    SWAP_THRESHOLD = 0.52

    @staticmethod
    def get_first_move() -> Move:
        """Get opening move as first player."""
        return Move(5, 6)  # Slightly off-center, discourages swap

    @staticmethod
    def should_swap(opp_move: Move) -> bool:
        """Decide whether to swap based on opponent's first move."""
        # Move strength based on position
        x, y = opp_move.x, opp_move.y
        center = 5

        # Distance from center (closer = stronger)
        dist = math.sqrt((x - center)**2 + (y - center)**2)
        strength = 1 - (dist / 7)  # Normalize

        # Corners are weak
        if (x, y) in [(0, 0), (0, 10), (10, 0), (10, 10)]:
            strength = 0.35

        # Center is very strong
        if (x, y) == (5, 5):
            strength = 0.65

        return strength >= OpeningBook.SWAP_THRESHOLD


# ==============================================================================
# MAIN AGENT
# ==============================================================================

class Group12Agent(AgentBase):
    """
    Ultimate Hex AI Agent combining neural network and MCTS.

    Features:
    - Neural network evaluation (NumPy inference)
    - AlphaZero-style MCTS
    - Virtual connection detection
    - Adaptive time management
    - Opening book
    """

    # Path to neural network weights (relative to agent file)
    WEIGHTS_PATH = Path(__file__).parent / "models" / "hex_model_numpy.npz"
    FALLBACK_WEIGHTS = Path(__file__).parent.parent.parent / "models" / "hex_model_numpy.npz"

    # MCTS configuration
    BASE_SIMULATIONS = 400
    MIN_SIMULATIONS = 50
    MAX_SIMULATIONS = 800

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.time_manager = TimeManager()
        self.encoder = BoardEncoder()
        self.start_time = None
        self.total_time_ns = 3 * 60 * 10**9

        # Try to load neural network
        self.network = self._load_network()

        if self.network:
            print(f"Group12Agent: Neural network loaded successfully")
        else:
            print(f"Group12Agent: Running in heuristic mode (no neural weights)")

    def _load_network(self) -> Optional[NumpyNeuralNetwork]:
        """Attempt to load neural network weights."""
        paths_to_try = [self.WEIGHTS_PATH, self.FALLBACK_WEIGHTS]

        for path in paths_to_try:
            if path.exists():
                try:
                    return NumpyNeuralNetwork(str(path))
                except Exception as e:
                    print(f"Warning: Failed to load weights from {path}: {e}")

        return None

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """Make a move."""
        if self.start_time is None:
            self.start_time = perf_counter_ns()

        move_start = perf_counter_ns()
        elapsed = move_start - self.start_time
        remaining = self.total_time_ns - elapsed

        # Opening book
        if turn == 1:
            return OpeningBook.get_first_move()

        if turn == 2 and opp_move is not None:
            if OpeningBook.should_swap(opp_move):
                return Move(-1, -1)  # Swap

        # Calculate time for this move
        time_budget = self.time_manager.get_time_for_move(board, remaining)

        # Calculate simulations based on time
        sims = self._calculate_simulations(time_budget, board)

        # Check for immediate win
        win_move = self._find_winning_move(board)
        if win_move:
            return win_move

        # Check for must-block
        block_move = self._find_blocking_move(board)
        if block_move:
            return block_move

        # Run MCTS
        mcts = NeuralMCTS(self.network, self.encoder, self.colour)
        move = mcts.search(board, sims, add_noise=(turn < 20))

        return move

    def _calculate_simulations(self, time_ns: int, board: Board) -> int:
        """Calculate number of MCTS simulations based on time."""
        # Estimate time per simulation
        if self.network:
            time_per_sim = 0.01 * 10**9  # ~10ms with neural (conservative)
        else:
            time_per_sim = 0.001 * 10**9  # ~1ms heuristic

        max_sims = int(time_ns * 0.8 / time_per_sim)
        return max(self.MIN_SIMULATIONS, min(self.MAX_SIMULATIONS, max_sims))

    def _find_winning_move(self, board: Board) -> Optional[Move]:
        """Check if we can win immediately."""
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    test_board = copy.deepcopy(board)
                    test_board.set_tile_colour(i, j, self.colour)
                    if test_board.has_ended(self.colour):
                        return Move(i, j)
        return None

    def _find_blocking_move(self, board: Board) -> Optional[Move]:
        """Check if opponent can win and block."""
        opp = Colour.opposite(self.colour)
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    test_board = copy.deepcopy(board)
                    test_board.set_tile_colour(i, j, opp)
                    if test_board.has_ended(opp):
                        return Move(i, j)
        return None


if __name__ == "__main__":
    print("Group12 Ultimate Agent")
    print("=" * 50)

    # Test initialization
    agent = Group12Agent(Colour.RED)
    print(f"Agent colour: {agent.colour}")
    print(f"Neural network: {'Loaded' if agent.network else 'Not available'}")

    # Test opening
    board = Board(11)
    move = agent.make_move(1, board, None)
    print(f"Opening move: ({move.x}, {move.y})")
