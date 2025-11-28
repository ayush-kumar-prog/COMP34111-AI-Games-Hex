"""
Group12 V5 Tournament Agent - Neural Network + MCTS

This agent uses a trained neural network (V5 Expert Iteration) with
neural-guided MCTS for tournament play.

Tournament Environment:
- PyTorch 2.5.1 (CPU)
- 8 CPUs, 8GB RAM, NO GPU
- 5 minutes (300 seconds) total time per match
- Scoring: 75% win rate + 25% move speed

Strategy:
- Load pre-trained .pth model with PyTorch CPU
- Neural-guided MCTS with 150 simulations per move
- Adaptive time management
- Opening book + swap logic from tournament agent
"""

import math
import random
import copy
from time import perf_counter_ns
from typing import Optional, List, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


# ============================================================================
# Neural Network (V2 Architecture - must match training)
# ============================================================================

class SqueezeExcitation(torch.nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channels // reduction, channels, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        return x * y.expand_as(x)


class HexResBlockV2(torch.nn.Module):
    """Enhanced residual block with SE attention."""

    def __init__(self, num_channels: int, use_se: bool = True, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.conv1 = torch.nn.Conv2d(num_channels, num_channels, kernel_size=3,
                                     padding=padding, dilation=dilation, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = torch.nn.Conv2d(num_channels, num_channels, kernel_size=3,
                                     padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        out = out + residual
        out = F.relu(out, inplace=True)
        return out


class HexNeuralNetworkV2(torch.nn.Module):
    """V2 neural network architecture for Hex."""

    def __init__(self, board_size: int = 11, num_res_blocks: int = 15,
                 num_channels: int = 256, input_channels: int = 8, use_se: bool = True):
        super().__init__()
        self.board_size = board_size
        self.num_channels = num_channels

        # Input processing
        self.conv_input = torch.nn.Conv2d(input_channels, num_channels,
                                          kernel_size=3, padding=1, bias=False)
        self.bn_input = torch.nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = torch.nn.ModuleList()
        for i in range(num_res_blocks):
            dilation = 2 if i in [5, 10] else 1
            self.res_blocks.append(HexResBlockV2(num_channels, use_se=use_se, dilation=dilation))

        # Policy head
        self.policy_conv = torch.nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = torch.nn.BatchNorm2d(32)
        self.policy_fc = torch.nn.Linear(32 * board_size * board_size, board_size * board_size)

        # Value head
        self.value_conv = torch.nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.value_bn = torch.nn.BatchNorm2d(32)
        self.value_fc1 = torch.nn.Linear(32 * board_size * board_size, 256)
        self.value_fc2 = torch.nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x, inplace=True)

        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy, inplace=True)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy_logits = F.log_softmax(policy, dim=1)

        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value, inplace=True)
        value = value.view(value.size(0), -1)
        value = self.value_fc1(value)
        value = F.relu(value, inplace=True)
        value = self.value_fc2(value)
        value = torch.tanh(value)

        return policy_logits, value


# ============================================================================
# Board Encoder
# ============================================================================

class V5BoardEncoder:
    """8-channel board encoder for V5 network."""

    def __init__(self, size: int = 11):
        self.size = size
        self._precompute_distances()

    def _precompute_distances(self):
        n = self.size
        self.dist_top = np.zeros((n, n), dtype=np.float32)
        self.dist_bottom = np.zeros((n, n), dtype=np.float32)
        self.dist_left = np.zeros((n, n), dtype=np.float32)
        self.dist_right = np.zeros((n, n), dtype=np.float32)
        self.dist_center = np.zeros((n, n), dtype=np.float32)

        center = n // 2
        for i in range(n):
            for j in range(n):
                self.dist_top[i, j] = i / (n - 1)
                self.dist_bottom[i, j] = (n - 1 - i) / (n - 1)
                self.dist_left[i, j] = j / (n - 1)
                self.dist_right[i, j] = (n - 1 - j) / (n - 1)
                self.dist_center[i, j] = 1.0 - (abs(i - center) + abs(j - center)) / (n - 1)

    def encode(self, board: Board, colour: Colour) -> torch.Tensor:
        """Encode board state for neural network."""
        n = self.size
        state = np.zeros((8, n, n), dtype=np.float32)

        if hasattr(colour, 'value'):
            colour_int = 1 if colour.value == 1 else 2
        else:
            colour_int = colour

        opp_int = 3 - colour_int

        for i in range(n):
            for j in range(n):
                tile = board.tiles[i][j]
                if tile.colour is not None:
                    if tile.colour.value == colour_int:
                        state[0, i, j] = 1.0
                    else:
                        state[1, i, j] = 1.0
                else:
                    state[2, i, j] = 1.0

        if colour_int == 1:  # RED
            state[3] = self.dist_top
            state[4] = self.dist_bottom
            state[5] = self.dist_left
            state[6] = self.dist_right
        else:  # BLUE
            state[3] = self.dist_left
            state[4] = self.dist_right
            state[5] = self.dist_top
            state[6] = self.dist_bottom

        state[7] = self.dist_center

        return torch.from_numpy(state).unsqueeze(0)


# ============================================================================
# Time Manager
# ============================================================================

class TimeManager:
    """Intelligent time allocation for 5-minute matches."""

    def __init__(self, total_time_ns: int = 300_000_000_000):  # 5 minutes
        self.total_time = total_time_ns
        self.buffer = 0.05  # 5% safety buffer
        self.usable_time = int(total_time_ns * (1 - self.buffer))
        self.time_used = 0
        self.moves_made = 0

    def update(self, time_spent: int):
        self.time_used += time_spent
        self.moves_made += 1

    def get_allocation(self, turn: int, empty_cells: int) -> int:
        """Get time allocation for this move in nanoseconds."""
        remaining = max(0, self.usable_time - self.time_used)

        if remaining < 1_000_000_000:  # Less than 1 second
            return 50_000_000  # 50ms emergency mode

        # Opening: quick moves
        if turn <= 5:
            return min(1_000_000_000, int(remaining * 0.03))  # 1s max

        # Estimate remaining moves
        est_moves_left = max(1, (empty_cells * 0.6) / 2)
        base_time = remaining / est_moves_left

        # Midgame: standard allocation
        if empty_cells > 30:
            return int(min(base_time * 1.2, 8_000_000_000))  # 8s max

        # Endgame: more time for critical moves
        return int(min(base_time * 1.5, 15_000_000_000))  # 15s max

    def get_mcts_sims(self, turn: int, empty_cells: int) -> int:
        """Get number of MCTS simulations based on time budget."""
        remaining = max(0, self.usable_time - self.time_used)
        time_ratio = remaining / self.usable_time

        if time_ratio < 0.1:  # Emergency
            return 30
        elif turn <= 5:  # Opening
            return 100
        elif time_ratio > 0.5:  # Plenty of time
            return 200
        else:  # Normal
            return 150

    def is_emergency(self) -> bool:
        remaining = self.usable_time - self.time_used
        return remaining < self.usable_time * 0.05


# ============================================================================
# Neural MCTS for Tournament
# ============================================================================

class NeuralMCTS:
    """Neural-guided MCTS for tournament play."""

    EXPLORATION = 1.5
    NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    def __init__(self, network, encoder: V5BoardEncoder, colour: Colour):
        self.network = network
        self.encoder = encoder
        self.colour = colour

    @torch.inference_mode()
    def search(self, board: Board, num_sims: int = 150) -> Move:
        """Run neural MCTS and return best move."""
        n = board.size

        # Get legal moves
        legal_moves = []
        for i in range(n):
            for j in range(n):
                if board.tiles[i][j].colour is None:
                    legal_moves.append((i, j))

        if not legal_moves:
            return Move(5, 5)

        if len(legal_moves) == 1:
            return Move(legal_moves[0][0], legal_moves[0][1])

        # Get neural network prior
        state = self.encoder.encode(board, self.colour)
        policy_logits, root_value = self.network(state)
        prior = torch.softmax(policy_logits, dim=1).squeeze().numpy()

        # Mask illegal moves
        legal_mask = np.zeros(n * n)
        for x, y in legal_moves:
            legal_mask[x * n + y] = 1.0
        prior = prior * legal_mask
        if prior.sum() > 0:
            prior /= prior.sum()
        else:
            prior = legal_mask / legal_mask.sum()

        # MCTS
        visits = np.zeros(n * n)
        total_values = np.zeros(n * n)
        legal_indices = [x * n + y for x, y in legal_moves]

        for _ in range(num_sims):
            # UCB selection
            best_score = -float('inf')
            best_idx = legal_indices[0]
            total_visits = visits.sum() + 1

            for idx in legal_indices:
                if visits[idx] == 0:
                    score = prior[idx] * 10 + random.random() * 0.01
                else:
                    q = total_values[idx] / visits[idx]
                    u = self.EXPLORATION * prior[idx] * math.sqrt(total_visits) / (1 + visits[idx])
                    score = q + u

                if score > best_score:
                    best_score = score
                    best_idx = idx

            # Make move
            x, y = best_idx // n, best_idx % n
            sim_board = copy.deepcopy(board)
            sim_board.set_tile_colour(x, y, self.colour)

            # Evaluate
            if sim_board.has_ended(self.colour):
                leaf_value = 1.0
            elif sim_board.has_ended(Colour.opposite(self.colour)):
                leaf_value = 0.0
            else:
                # Neural network evaluation
                opp_colour = Colour.opposite(self.colour)
                child_state = self.encoder.encode(sim_board, opp_colour)
                _, opp_value = self.network(child_state)
                leaf_value = 1.0 - (opp_value.item() + 1) / 2

            visits[best_idx] += 1
            total_values[best_idx] += leaf_value

        # Select best move by visit count
        best_idx = max(legal_indices, key=lambda idx: visits[idx])
        return Move(best_idx // n, best_idx % n)


# ============================================================================
# Opening Book
# ============================================================================

class OpeningBook:
    """Nash equilibrium-based opening moves."""

    FIRST_MOVES = [(5, 6), (6, 5), (5, 5), (4, 5), (5, 4)]
    CORNERS = [(0, 0), (0, 10), (10, 0), (10, 10)]
    OPENING_VALUES = {}

    def __init__(self):
        for i in range(11):
            for j in range(11):
                dist = abs(i - 5) + abs(j - 5)
                if (i, j) in self.CORNERS:
                    self.OPENING_VALUES[(i, j)] = 0.35
                elif dist == 0:
                    self.OPENING_VALUES[(i, j)] = 0.55
                elif dist <= 2:
                    self.OPENING_VALUES[(i, j)] = 0.52
                else:
                    self.OPENING_VALUES[(i, j)] = max(0.4, 0.5 - dist * 0.02)

    def get_first_move(self) -> Move:
        return Move(5, 6)

    def should_swap(self, opponent_move: Tuple[int, int]) -> bool:
        value = self.OPENING_VALUES.get(opponent_move, 0.45)
        return value > 0.52


# ============================================================================
# Main Tournament Agent
# ============================================================================

class Group12Agent(AgentBase):
    """
    V5 Tournament Agent using neural network + MCTS.

    Features:
    - PyTorch CPU inference (~50-200 pos/sec)
    - Neural-guided MCTS (100-200 simulations)
    - Opening book + Nash equilibrium swap
    - Intelligent time management
    """

    _board_size: int = 11
    _colour: Colour = None

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._colour = colour

        # Load neural network
        self.device = 'cpu'  # Tournament has no GPU
        self.network = self._load_network()
        self.encoder = V5BoardEncoder()

        # Components
        self.opening_book = OpeningBook()
        self.time_manager = TimeManager()
        self.mcts = None  # Created after colour is known

        self.turn = 0

    def _load_network(self) -> HexNeuralNetworkV2:
        """Load trained network."""
        # Try multiple paths - includes both expected and uploaded model names
        possible_paths = [
            # Uploaded model files (from git)
            Path('models/v5_supervised_final.pth'),
            Path('models/hex_model_best.pth'),
            # Relative to agent location
            Path(__file__).parent / 'models' / 'hex_v5_best.pth',
            Path(__file__).parent / 'models' / 'hex_v5_expert_final.pth',
            Path(__file__).parent / 'models' / 'hex_v5_supervised_best.pth',
            # Other common locations
            Path('agents/Group12/models/hex_v5_best.pth'),
            Path('models/v5_best.pth'),
            Path('models/v5_supervised_best.pth'),
        ]

        network = HexNeuralNetworkV2(
            board_size=11,
            num_res_blocks=15,
            num_channels=256
        )

        for path in possible_paths:
            if path.exists():
                try:
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                    if 'model' in checkpoint:
                        network.load_state_dict(checkpoint['model'])
                    else:
                        network.load_state_dict(checkpoint)
                    print(f"Loaded V5 model from: {path}")
                    break
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    continue
        else:
            print("WARNING: No trained model found! Using random weights.")

        network.eval()
        return network

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Make the best move for the current position."""
        start_time = perf_counter_ns()
        self.turn = turn

        try:
            # Initialize MCTS if needed
            if self.mcts is None:
                self.mcts = NeuralMCTS(self.network, self.encoder, self._colour)

            # Count empty cells
            empty_cells = sum(1 for i in range(board.size)
                            for j in range(board.size)
                            if board.tiles[i][j].colour is None)

            # Turn 1: First move
            if turn == 1:
                move = self.opening_book.get_first_move()
                self._update_time(start_time)
                return move

            # Turn 2: Swap decision
            if turn == 2 and opp_move:
                if self.opening_book.should_swap((opp_move.x, opp_move.y)):
                    self._update_time(start_time)
                    return Move(-1, -1)  # Swap

            # Check for immediate winning move
            win_move = self._find_winning_move(board)
            if win_move:
                self._update_time(start_time)
                return win_move

            # Check for blocking moves
            block_move = self._find_blocking_move(board)
            if block_move:
                self._update_time(start_time)
                return block_move

            # Emergency mode: quick heuristic
            if self.time_manager.is_emergency():
                move = self._quick_move(board)
                self._update_time(start_time)
                return move

            # Get MCTS simulations based on time
            num_sims = self.time_manager.get_mcts_sims(turn, empty_cells)

            # Run neural MCTS
            move = self.mcts.search(board, num_sims=num_sims)

            self._update_time(start_time)
            return move

        except Exception as e:
            print(f"Error in make_move: {e}")
            self._update_time(start_time)
            return self._quick_move(board)

    def _update_time(self, start_time: int):
        elapsed = perf_counter_ns() - start_time
        self.time_manager.update(elapsed)

    def _find_winning_move(self, board: Board) -> Optional[Move]:
        """Check if we can win immediately."""
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    board.tiles[i][j].colour = self._colour
                    if board.has_ended(self._colour):
                        board.tiles[i][j].colour = None
                        return Move(i, j)
                    board.tiles[i][j].colour = None
        return None

    def _find_blocking_move(self, board: Board) -> Optional[Move]:
        """Check if opponent can win and block."""
        opp = Colour.opposite(self._colour)
        threats = []

        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    board.tiles[i][j].colour = opp
                    if board.has_ended(opp):
                        threats.append(Move(i, j))
                    board.tiles[i][j].colour = None

        if len(threats) == 1:
            return threats[0]
        elif len(threats) > 1:
            return threats[0]

        return None

    def _quick_move(self, board: Board) -> Move:
        """Quick heuristic move for emergency or failsafe."""
        n = board.size
        center = n // 2

        best_move = None
        best_score = -float('inf')

        for i in range(n):
            for j in range(n):
                if board.tiles[i][j].colour is None:
                    score = 0

                    # Avoid corners
                    if (i, j) in [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]:
                        score -= 100

                    # Prefer center
                    dist = abs(i - center) + abs(j - center)
                    score += (10 - dist) * 2

                    # Prefer near our stones
                    for di, dj in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n:
                            if board.tiles[ni][nj].colour == self._colour:
                                score += 10

                    if score > best_score:
                        best_score = score
                        best_move = Move(i, j)

        return best_move if best_move else Move(center, center)


# Allow importing the class by old name for compatibility
Agent = Group12Agent
