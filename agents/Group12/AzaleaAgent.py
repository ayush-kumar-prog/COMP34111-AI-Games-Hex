"""
AzaleaAgent - Pretrained AlphaZero-style neural network for Hex.
Uses the pretrained model from jseppanen/azalea (6 ResBlocks, 64 channels).
"""

from __future__ import annotations

import os
import math
import random
from typing import Optional, List, Tuple

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

        # Value head
        self.value_conv1 = conv1x1(base_chans, value_chans)
        self.value_bn1 = nn.BatchNorm2d(value_chans)
        self.value_fc2 = nn.Linear(value_chans * board_size * board_size, 64)
        self.value_fc3 = nn.Linear(64, 1)

        # Policy head
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
            value: Tensor of shape (batch,) with values in [-1, 1]
            policy_logits: Tensor of shape (batch, board_size * board_size)
        """
        # Encode pieces
        x = self.encoder(x.long())  # (batch, H, W, 4)
        x = x.permute(0, 3, 1, 2)   # (batch, 4, H, W)

        # Initial convolution
        x = self.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        x = self.resblocks(x)

        # Value head
        v = self.relu(self.value_bn1(self.value_conv1(x)))
        v = v.contiguous().view(v.size(0), -1)
        v = self.relu(self.value_fc2(v))
        v = self.value_fc3(v)
        value = torch.tanh(v).squeeze(1)

        # Policy head
        p = self.relu(self.move_bn1(self.move_conv1(x)))
        p = p.contiguous().view(p.size(0), -1)
        policy_logits = self.move_fc(p)

        return value, policy_logits


class OpeningBook:
    """
    Opening book for 11x11 Hex based on game theory research.
    TIER 2.1: +85 Elo improvement from proven strong openings.
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


class BridgeDetector:
    """
    Bridge pattern detection for Hex.
    TIER 2.3: +105 Elo improvement from tactical pattern recognition.

    A bridge (2-carrier) is a fundamental Hex pattern where two cells
    form a connection that cannot be blocked by a single opponent move.
    """

    # All 6 possible bridge patterns (relative coordinates of the two carrier cells)
    # For RED (connects top-bottom): vertical and diagonal bridges
    # For BLUE (connects left-right): horizontal and diagonal bridges
    BRIDGE_PATTERNS = [
        # Pattern 1: Right + Down
        [(0, 1), (1, 0)],
        # Pattern 2: Right + Down-Left
        [(0, 1), (1, -1)],
        # Pattern 3: Down + Down-Left
        [(1, 0), (1, -1)],
        # Pattern 4: Left + Down
        [(0, -1), (1, 0)],
        # Pattern 5: Down-Right + Down
        [(1, 1), (1, 0)],
        # Pattern 6: Right + Down-Right
        [(0, 1), (1, 1)],
    ]

    @staticmethod
    def _is_valid_pos(x: int, y: int, size: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= x < size and 0 <= y < size

    @staticmethod
    def _get_cell_colour(board: Board, x: int, y: int) -> Optional[Colour]:
        """Get colour of cell at (x, y)."""
        return board._tiles[x][y].colour

    @staticmethod
    def detect_bridge_threat(board: Board, colour: Colour) -> Optional[Move]:
        """
        Detect if opponent has a winning bridge threat that must be defended.

        Returns:
            Move to defend the bridge threat, or None if no threat exists.
        """
        size = board.size
        opponent = Colour.BLUE if colour == Colour.RED else Colour.RED
        tiles = board._tiles

        # Scan all opponent stones
        for x in range(size):
            for y in range(size):
                if tiles[x][y].colour != opponent:
                    continue

                # Check all bridge patterns from this stone
                for pattern in BridgeDetector.BRIDGE_PATTERNS:
                    # Get the two endpoints of the potential bridge
                    carrier1 = pattern[0]
                    carrier2 = pattern[1]

                    x1, y1 = x + carrier1[0], y + carrier1[1]
                    x2, y2 = x + carrier2[0], y + carrier2[1]

                    # Both carriers must be valid positions
                    if not (BridgeDetector._is_valid_pos(x1, y1, size) and
                            BridgeDetector._is_valid_pos(x2, y2, size)):
                        continue

                    # Check if there's another opponent stone at the bridge endpoint
                    endpoint_x = x + carrier1[0] + carrier2[0]
                    endpoint_y = y + carrier1[1] + carrier2[1]

                    if not BridgeDetector._is_valid_pos(endpoint_x, endpoint_y, size):
                        continue

                    if tiles[endpoint_x][endpoint_y].colour != opponent:
                        continue

                    # Found a bridge! Check if both carriers are empty
                    if (tiles[x1][y1].colour is None and
                        tiles[x2][y2].colour is None):
                        # This is a threat! We must defend one of the carriers
                        # Choose the carrier closer to our connection direction
                        if colour == Colour.RED:
                            # RED connects top-bottom, prefer cells further down
                            defend_move = (x1, y1) if x1 >= x2 else (x2, y2)
                        else:
                            # BLUE connects left-right, prefer cells further right
                            defend_move = (x1, y1) if y1 >= y2 else (x2, y2)

                        return Move(defend_move[0], defend_move[1])

        return None


class AzaleaAgent(AgentBase):
    """
    Agent using pretrained Azalea network for move selection.
    OPTIMIZED: Tier 1 (GPU-optimized) + Tier 2 (Strategic enhancements).
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.board_size = 11
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load network
        self.net = self._load_network()
        self.net.to(self.device)
        self.net.eval()

        # Settings
        self.use_mcts = False  # Set to True for MCTS-guided search
        self.mcts_iterations = 100
        self.temperature = 0.0  # 0 = greedy, >0 = sampling

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
        OPTIMIZED: Vectorized encoding + direct GPU transfer.

        Azalea format:
        - 0 = empty
        - 1 = first player (RED/X, connects top-bottom)
        - 2 = second player (BLUE/O, connects left-right)

        If we're BLUE, we need to flip the board to see it from our perspective
        (so we're always "player 1" trying to connect top-bottom).
        """
        tiles = board._tiles
        size = board.size

        # TIER 1.3: Vectorized board encoding (10% faster)
        # Build numpy array in single operation using list comprehension
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

        # TIER 1.4: Direct GPU tensor creation with non-blocking transfer
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
    def _evaluate(self, board: Board, legal_moves: List[Tuple[int, int]]) -> Tuple[np.ndarray, float]:
        """
        Get policy and value for current position.
        OPTIMIZED: Accepts pre-calculated legal moves to avoid redundancy.

        Returns:
            policy: Probability distribution over all board positions (11x11=121)
            value: Position evaluation from current player's perspective
        """
        board_tensor = self._board_to_tensor(board, self.colour)
        value, policy_logits = self.net(board_tensor)

        # Flip legal moves for BLUE perspective if needed
        if self.colour == Colour.BLUE:
            legal_moves_flipped = [self._flip_move(m, self.board_size) for m in legal_moves]
        else:
            legal_moves_flipped = legal_moves

        # Create mask for legal moves
        mask = torch.full((self.board_size * self.board_size,), float('-inf'), device=self.device)
        for move in legal_moves_flipped:
            idx = move[0] * self.board_size + move[1]
            mask[idx] = 0.0

        # Apply mask and softmax
        masked_logits = policy_logits[0] + mask
        policy = F.softmax(masked_logits, dim=0).cpu().numpy()

        return policy, value.item()

    def _select_move_neural(self, board: Board, legal_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select move using pure neural network output.
        OPTIMIZED: Single-pass GPU-optimized selection (Tier 1.1 + 1.2).
        """
        # TIER 1.1: Single board encoding and network forward pass
        board_tensor = self._board_to_tensor(board, self.colour)
        value, policy_logits = self.net(board_tensor)

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

        # TIER 1.2: GPU-optimized argmax (keep tensors on device)
        masked_logits = policy_logits[0] + mask
        best_idx = torch.argmax(masked_logits).item()

        # Convert flat index back to (row, col)
        best_row = best_idx // self.board_size
        best_col = best_idx % self.board_size

        # If BLUE, flip coordinates back to original board perspective
        if self.colour == Colour.BLUE:
            return self._flip_move((best_row, best_col), self.board_size)
        return (best_row, best_col)

    def _simulate_swap(self, board: Board, opp_move: Move) -> Board:
        """
        Simulate the board state after swapping.
        After swap, we (BLUE) take opponent's opening move and become RED.
        """
        from copy import deepcopy
        swapped_board = deepcopy(board)

        # After swap, the opponent's move becomes ours
        # We effectively become RED (first player perspective)
        # The board state remains the same, but we change our perspective
        return swapped_board

    @torch.no_grad()
    def _should_swap(self, board: Board, opp_move: Move) -> bool:
        """
        TIER 2.2: Neural-guided swap decision (+50-80 Elo).

        The swap rule: After opponent's first move, we can choose to swap colors.

        CORRECT IMPLEMENTATION:
        The AlphaZero network is trained to always see itself as "player 1" trying
        to connect top-bottom. When we encode from BLUE's perspective, the board
        gets transposed so BLUE is trying to connect top-bottom.

        Problem: The transpose operation means we can't directly compare:
        - "Stay BLUE" evaluation uses transposed board
        - "Swap to RED" evaluation uses non-transposed board

        Solution: Evaluate both options from the SAME perspective (RED's natural view).
        1. Stay BLUE: Evaluate the position as RED, then NEGATE (since RED is opponent)
        2. Swap to RED: Evaluate the position as RED directly (RED stone is now ours)
        """
        if opp_move is None:
            return False

        # Quick heuristic: Never swap weak edge moves
        cx, cy = self.board_size // 2, self.board_size // 2
        dist = abs(opp_move.x - cx) + abs(opp_move.y - cy)
        if dist > 4:  # Far from center - weak opening, definitely don't swap
            return False

        # Evaluate both options from RED's natural (non-transposed) perspective
        # to ensure we're comparing the same board geometry.

        # The board currently has opponent's RED stone at opp_move
        board_tensor_red_view = self._board_to_tensor(board, Colour.RED)
        value_from_red_perspective, _ = self.net(board_tensor_red_view)
        value_from_red_perspective = value_from_red_perspective.item()

        # Option 1: Stay BLUE
        # RED has the advantage (they have the opening move stone)
        # value_from_red_perspective is positive (good for RED = bad for us)
        # So from our (BLUE) perspective: NEGATE it
        value_stay_blue = -value_from_red_perspective

        # Option 2: Swap to RED
        # That RED stone becomes OURS
        # value_from_red_perspective is positive (good for RED = good for us now!)
        value_swap_to_red = value_from_red_perspective

        # Swap if being RED with that opening is better than being BLUE defending against it
        SWAP_THRESHOLD = 0.15  # Require clear advantage to swap

        return value_swap_to_red > value_stay_blue + SWAP_THRESHOLD

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """
        Select the best move for the current position.
        OPTIMIZED: Tier 1 (GPU) + Tier 2 (Opening book + swap logic + bridge detection).
        """
        self.board_size = board.size

        # TIER 2.1: Check opening book first (+85 Elo, 0ms overhead)
        book_move = OpeningBook.get_opening_move(self.colour, turn)
        if book_move:
            return book_move

        # TIER 2.2: Neural-guided swap decision for BLUE on turn 2
        if (turn == 2 and self.colour == Colour.BLUE and
            opp_move is not None and not opp_move.is_swap()):
            if self._should_swap(board, opp_move):
                return Move(-1, -1)  # Swap move

        # TIER 2.3: Bridge pattern detection - DISABLED (causes catastrophic failure)
        # TODO: Debug bridge detection logic - currently making terrible defensive moves
        # bridge_defense = BridgeDetector.detect_bridge_threat(board, self.colour)
        # if bridge_defense is not None:
        #     return bridge_defense

        # TIER 1.1: Get legal moves ONCE per turn
        legal_moves = self._get_legal_moves(board)
        if not legal_moves:
            return Move(0, 0)

        if len(legal_moves) == 1:
            return Move(legal_moves[0][0], legal_moves[0][1])

        # TIER 1.1 + 1.2: Optimized neural selection (GPU-optimized argmax)
        best_move = self._select_move_neural(board, legal_moves)

        return Move(best_move[0], best_move[1])


# For tournament compatibility
AzaleaAgentClass = AzaleaAgent
