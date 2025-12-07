"""
AzaleaAgent V9b - Policy-Only Enhancements for Hex (No Temperature).
Based on AzaleaAgent baseline with following improvements:
- FIXED: Lookup table swap (replaces broken value-based swap)
- NEW: 1-ply tactical check (win detection)
- REMOVED: Temperature scaling (V9 showed it adds variance without improvement)

NOTE: The value head is UNRELIABLE (network trained policy-only, not AlphaZero-style).
All enhancements in this version avoid the value head entirely.
"""

from __future__ import annotations

import os
import math
import random
from typing import Optional, List, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# Get the directory where this file is located
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(AGENT_DIR, "models", "hex11-20180712-3362.policy.pth")


def conv3x3(in_chans: int, out_chans: int) -> nn.Conv2d:
    return nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False)


def conv1x1(in_chans: int, out_chans: int) -> nn.Conv2d:
    return nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)


class Resblock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.conv1 = conv3x3(in_dim, dim)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = conv3x3(dim, dim)
        self.bn2 = nn.BatchNorm2d(dim)
        if dim != in_dim:
            self.res_conv = conv1x1(in_dim, dim)
            self.res_bn = nn.BatchNorm2d(dim)
        else:
            self.res_conv = self.res_bn = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.res_conv:
            x = self.res_bn(self.res_conv(x))
        y = y + x
        y = self.relu(y)
        return y


class HexNetwork(nn.Module):
    """
    AlphaZero-style network for Hex.
    Architecture: 6 ResBlocks, 64 channels, value + policy heads.
    """

    def __init__(self, board_size: int = 11, num_blocks: int = 6, base_chans: int = 64):
        super().__init__()
        self.board_size = board_size
        input_dim = 4  # Embedding dimension
        value_chans = 2
        policy_chans = 4

        # Tile encoder (0=empty, 1=player1, 2=player2)
        self.encoder = nn.Embedding(3, 4)

        # Input upsampling
        self.conv1 = conv3x3(input_dim, base_chans)
        self.bn1 = nn.BatchNorm2d(base_chans)

        # Residual blocks
        blocks = [Resblock(base_chans, base_chans) for _ in range(num_blocks)]
        self.resblocks = nn.Sequential(*blocks)

        # Value head (UNRELIABLE - do not use for decision making)
        self.value_conv1 = conv1x1(base_chans, value_chans)
        self.value_bn1 = nn.BatchNorm2d(value_chans)
        self.value_fc2 = nn.Linear(value_chans * board_size * board_size, 64)
        self.value_fc3 = nn.Linear(64, 1)

        # Policy head (RELIABLE - use this for move selection)
        self.move_conv1 = conv1x1(base_chans, policy_chans)
        self.move_bn1 = nn.BatchNorm2d(policy_chans)
        self.move_fc = nn.Linear(policy_chans * board_size * board_size, board_size * board_size)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Board tensor of shape (batch, height, width) with values 0, 1, 2

        Returns:
            value: Tensor of shape (batch,) with values in [-1, 1] (UNRELIABLE)
            policy_logits: Tensor of shape (batch, board_size * board_size) (RELIABLE)
        """
        # Encode pieces
        x = self.encoder(x.long())  # (batch, H, W, 4)
        x = x.permute(0, 3, 1, 2)   # (batch, 4, H, W)

        # Initial convolution
        x = self.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        x = self.resblocks(x)

        # Value head (UNRELIABLE)
        v = self.relu(self.value_bn1(self.value_conv1(x)))
        v = v.contiguous().view(v.size(0), -1)
        v = self.relu(self.value_fc2(v))
        v = self.value_fc3(v)
        value = torch.tanh(v).squeeze(1)

        # Policy head (RELIABLE)
        p = self.relu(self.move_bn1(self.move_conv1(x)))
        p = p.contiguous().view(p.size(0), -1)
        policy_logits = self.move_fc(p)

        return value, policy_logits


class OpeningBook:
    """
    Opening book for 11x11 Hex based on game theory research.
    +85 Elo improvement from proven strong openings.
    """

    # Optimal RED openings (first player, connects top-bottom)
    # Center (5,5) has 888 Elo advantage according to research
    RED_OPENINGS = [
        (5, 5),   # Center (strongest)
        (5, 4),   # Near-center
        (4, 5),   # Near-center
        (5, 6),   # Near-center
        (6, 5),   # Near-center
    ]

    @staticmethod
    def get_opening_move(colour: Colour, turn: int) -> Optional[Move]:
        """Get opening book move if applicable."""
        if turn == 1 and colour == Colour.RED:
            # Weighted random selection - center gets 60% probability
            weights = [0.6, 0.1, 0.1, 0.1, 0.1]
            choice = random.choices(OpeningBook.RED_OPENINGS, weights=weights)[0]
            return Move(choice[0], choice[1])
        return None


class SwapTable:
    """
    Lookup table for swap decisions based on Hex game theory.
    REPLACES broken value-based swap that used unreliable value head.

    Based on research showing center cells are strongest openings:
    - (5,5) center: 888 Elo advantage - ALWAYS SWAP
    - Adjacent to center: Strong openings - SWAP
    - 2 away from center: Moderate - SWAP
    - Edge/corner moves: Weak - DON'T SWAP
    """

    # For 11x11 board, center is (5,5)
    # Swap if opponent plays these strong positions
    SWAP_SQUARES: Set[Tuple[int, int]] = {
        # Center (strongest)
        (5, 5),
        # Adjacent to center (strong)
        (5, 4), (4, 5), (5, 6), (6, 5),
        (4, 4), (6, 6), (4, 6), (6, 4),
        # 2 away from center (moderate but still good)
        (5, 3), (3, 5), (5, 7), (7, 5),
        (4, 3), (3, 4), (3, 6), (4, 7),
        (6, 3), (7, 4), (7, 6), (6, 7),
        (3, 3), (7, 7), (3, 7), (7, 3),
    }

    @staticmethod
    def should_swap(opp_move: Move) -> bool:
        """
        Determine if we should swap based on opponent's opening move.
        Uses lookup table - no value head dependency.
        """
        if opp_move is None:
            return False
        return (opp_move.x, opp_move.y) in SwapTable.SWAP_SQUARES


class TacticalChecker:
    """
    1-ply tactical check for immediate wins and must-block threats.
    Uses only board state analysis - no neural network needed.
    """

    # Hex neighbor offsets (6-connected)
    NEIGHBORS = [
        (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)
    ]

    @staticmethod
    def _is_valid(x: int, y: int, size: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= x < size and 0 <= y < size

    @staticmethod
    def _get_neighbors(x: int, y: int, size: int) -> List[Tuple[int, int]]:
        """Get all valid neighbors of a cell."""
        neighbors = []
        for dx, dy in TacticalChecker.NEIGHBORS:
            nx, ny = x + dx, y + dy
            if TacticalChecker._is_valid(nx, ny, size):
                neighbors.append((nx, ny))
        return neighbors

    @staticmethod
    def _find_connected_component(
        board: Board,
        start: Tuple[int, int],
        colour: Colour,
        visited: Set[Tuple[int, int]]
    ) -> Tuple[Set[Tuple[int, int]], bool, bool]:
        """
        Find all cells connected to start of the same colour.
        Also tracks if component touches top/bottom (RED) or left/right (BLUE) edges.

        Returns:
            (component_cells, touches_start_edge, touches_end_edge)
        """
        size = board.size
        tiles = board._tiles
        component = set()
        stack = [start]
        touches_start = False
        touches_end = False

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            component.add((x, y))

            # Check edge touches
            if colour == Colour.RED:
                # RED connects top (x=0) to bottom (x=size-1)
                if x == 0:
                    touches_start = True
                if x == size - 1:
                    touches_end = True
            else:
                # BLUE connects left (y=0) to right (y=size-1)
                if y == 0:
                    touches_start = True
                if y == size - 1:
                    touches_end = True

            # Add same-color neighbors to stack
            for nx, ny in TacticalChecker._get_neighbors(x, y, size):
                if (nx, ny) not in visited and tiles[nx][ny].colour == colour:
                    stack.append((nx, ny))

        return component, touches_start, touches_end

    @staticmethod
    def find_winning_move(board: Board, colour: Colour, legal_moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Check if any legal move creates an immediate win.
        A win occurs when we connect our two edges.

        Returns:
            Winning move coordinates, or None if no immediate win exists.
        """
        size = board.size
        tiles = board._tiles

        for move in legal_moves:
            mx, my = move

            # Simulate placing our stone
            # Check if this move connects two components that together span our edges
            neighbors_of_colour = []
            for nx, ny in TacticalChecker._get_neighbors(mx, my, size):
                if tiles[nx][ny].colour == colour:
                    neighbors_of_colour.append((nx, ny))

            # Also check if move itself is on an edge
            move_on_start = (mx == 0 if colour == Colour.RED else my == 0)
            move_on_end = (mx == size - 1 if colour == Colour.RED else my == size - 1)

            # Find all components this move would connect
            visited: Set[Tuple[int, int]] = set()
            all_touch_start = move_on_start
            all_touch_end = move_on_end

            for neighbor in neighbors_of_colour:
                if neighbor not in visited:
                    _, touches_start, touches_end = TacticalChecker._find_connected_component(
                        board, neighbor, colour, visited
                    )
                    all_touch_start = all_touch_start or touches_start
                    all_touch_end = all_touch_end or touches_end

            # If this move bridges components touching both edges -> WIN
            if all_touch_start and all_touch_end:
                return move

        return None


class AzaleaAgentV9b(AgentBase):
    """
    Agent V9b - Policy-only enhancements without temperature.

    Improvements over baseline:
    1. Lookup table swap (replaces broken neural swap)
    2. 1-ply tactical check for immediate wins
    3. NO temperature scaling (pure argmax like baseline)
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.board_size = 11
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load network
        self.net = self._load_network()
        self.net.to(self.device)
        self.net.eval()

    def _load_network(self) -> HexNetwork:
        """Load pretrained network weights."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        state = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        policy_state = state['policy']

        # Create network with saved hyperparameters
        net = HexNetwork(
            board_size=policy_state['board_size'],
            num_blocks=policy_state['num_blocks'],
            base_chans=policy_state['base_chans']
        )

        # Load weights
        net.load_state_dict(policy_state['net'])
        return net

    def _board_to_tensor(self, board: Board, perspective: Colour) -> torch.Tensor:
        """
        Convert Board to tensor from a player's perspective.
        """
        tiles = board._tiles
        size = board.size

        # Vectorized board encoding
        board_array = np.array([[
            0 if tile.colour is None else (1 if tile.colour == Colour.RED else 2)
            for tile in row
        ] for row in tiles], dtype=np.int32)

        # If we're BLUE, flip perspective so we're "player 1"
        if perspective == Colour.BLUE:
            # Swap colors: 1 <-> 2
            board_array = np.where(board_array > 0, 3 - board_array, 0)
            # Transpose to flip attack direction
            board_array = board_array.T

        return torch.from_numpy(board_array).unsqueeze(0).to(
            self.device, dtype=torch.float32, non_blocking=True
        )

    def _get_legal_moves(self, board: Board) -> List[Tuple[int, int]]:
        """Get list of legal moves as (row, col) tuples."""
        tiles = board._tiles
        size = board.size
        moves = []
        for i in range(size):
            for j in range(size):
                if tiles[i][j].colour is None:
                    moves.append((i, j))
        return moves

    def _flip_move(self, move: Tuple[int, int], size: int) -> Tuple[int, int]:
        """Flip move coordinates when playing as BLUE."""
        return (move[1], move[0])

    @torch.no_grad()
    def _select_move_neural(self, board: Board, legal_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select move using pure policy head with argmax (NO temperature).
        Same as baseline - pure greedy selection.
        """
        board_tensor = self._board_to_tensor(board, self.colour)
        _, policy_logits = self.net(board_tensor)  # Ignore value head (unreliable)

        # Build legal move mask
        if self.colour == Colour.BLUE:
            legal_moves_flipped = [self._flip_move(m, self.board_size) for m in legal_moves]
        else:
            legal_moves_flipped = legal_moves

        # Create mask for legal moves
        mask = torch.full((self.board_size * self.board_size,), float('-inf'), device=self.device)
        for move in legal_moves_flipped:
            idx = move[0] * self.board_size + move[1]
            mask[idx] = 0.0

        masked_logits = policy_logits[0] + mask

        # Pure argmax - no temperature sampling
        best_idx = torch.argmax(masked_logits).item()

        # Convert flat index back to (row, col)
        best_row = best_idx // self.board_size
        best_col = best_idx % self.board_size

        # If BLUE, flip coordinates back to original board perspective
        if self.colour == Colour.BLUE:
            return self._flip_move((best_row, best_col), self.board_size)
        return (best_row, best_col)

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """
        Select the best move for the current position.

        Pipeline:
        1. Opening book (turn 1 RED)
        2. Lookup table swap (turn 2 BLUE)
        3. 1-ply tactical win check
        4. Neural policy with pure argmax
        """
        self.board_size = board.size

        # 1. Opening book for RED on turn 1
        book_move = OpeningBook.get_opening_move(self.colour, turn)
        if book_move:
            return book_move

        # 2. Lookup table swap for BLUE on turn 2
        if (turn == 2 and self.colour == Colour.BLUE and
            opp_move is not None and not opp_move.is_swap()):
            if SwapTable.should_swap(opp_move):
                return Move(-1, -1)  # Swap move

        # Get legal moves
        legal_moves = self._get_legal_moves(board)
        if not legal_moves:
            return Move(0, 0)

        if len(legal_moves) == 1:
            return Move(legal_moves[0][0], legal_moves[0][1])

        # 3. 1-ply tactical check: Can we win immediately?
        winning_move = TacticalChecker.find_winning_move(board, self.colour, legal_moves)
        if winning_move:
            return Move(winning_move[0], winning_move[1])

        # 4. Neural policy with pure argmax (no temperature)
        best_move = self._select_move_neural(board, legal_moves)

        return Move(best_move[0], best_move[1])


# For tournament compatibility
AzaleaAgentV9bClass = AzaleaAgentV9b
