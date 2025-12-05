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


class AzaleaAgent(AgentBase):
    """
    Agent using pretrained Azalea network for move selection.

    Can run in pure neural mode (fast) or with lightweight MCTS (stronger).
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

        Azalea format:
        - 0 = empty
        - 1 = first player (RED/X, connects top-bottom)
        - 2 = second player (BLUE/O, connects left-right)

        If we're BLUE, we need to flip the board to see it from our perspective
        (so we're always "player 1" trying to connect top-bottom).
        """
        tiles = board._tiles
        size = board.size

        # Extract board state
        board_array = np.zeros((size, size), dtype=np.int32)
        for i in range(size):
            for j in range(size):
                c = tiles[i][j].colour
                if c == Colour.RED:
                    board_array[i, j] = 1
                elif c == Colour.BLUE:
                    board_array[i, j] = 2

        # If we're BLUE, flip perspective so we're "player 1"
        if perspective == Colour.BLUE:
            # Swap colors: 1 <-> 2
            board_array = np.where(board_array > 0, 3 - board_array, 0)
            # Transpose to flip attack direction
            board_array = board_array.T

        # Convert to tensor
        tensor = torch.from_numpy(board_array).unsqueeze(0)  # (1, H, W)
        return tensor.to(self.device)

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
    def _evaluate(self, board: Board) -> Tuple[np.ndarray, float]:
        """
        Get policy and value for current position.

        Returns:
            policy: Probability distribution over all board positions (11x11=121)
            value: Position evaluation from current player's perspective
        """
        board_tensor = self._board_to_tensor(board, self.colour)
        value, policy_logits = self.net(board_tensor)

        # Get legal moves mask
        legal_moves = self._get_legal_moves(board)
        if self.colour == Colour.BLUE:
            # Flip legal moves for BLUE perspective
            legal_moves = [self._flip_move(m, self.board_size) for m in legal_moves]

        # Create mask for legal moves
        mask = torch.full((self.board_size * self.board_size,), float('-inf'), device=self.device)
        for move in legal_moves:
            idx = move[0] * self.board_size + move[1]
            mask[idx] = 0.0

        # Apply mask and softmax
        masked_logits = policy_logits[0] + mask
        policy = F.softmax(masked_logits, dim=0).cpu().numpy()

        return policy, value.item()

    def _select_move_neural(self, board: Board) -> Tuple[int, int]:
        """Select move using pure neural network output."""
        policy, value = self._evaluate(board)

        # Get legal moves
        legal_moves = self._get_legal_moves(board)
        if self.colour == Colour.BLUE:
            legal_moves_flipped = [self._flip_move(m, self.board_size) for m in legal_moves]
        else:
            legal_moves_flipped = legal_moves

        # Find best legal move
        best_prob = -1
        best_move = None
        best_move_original = None

        for orig_move, flipped_move in zip(legal_moves, legal_moves_flipped):
            idx = flipped_move[0] * self.board_size + flipped_move[1]
            prob = policy[idx]
            if prob > best_prob:
                best_prob = prob
                best_move = flipped_move
                best_move_original = orig_move

        # If BLUE, we need to flip back to original coordinates
        if self.colour == Colour.BLUE:
            return best_move_original
        return best_move

    def _should_swap(self, board: Board, opp_move: Move) -> bool:
        """Decide whether to swap based on opponent's opening strength."""
        if opp_move is None:
            return False

        # Strong central openings are worth swapping
        cx, cy = self.board_size // 2, self.board_size // 2
        dist = abs(opp_move.x - cx) + abs(opp_move.y - cy)

        # Swap if opponent played close to center
        return dist <= 2

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """Select the best move for the current position."""
        self.board_size = board.size

        # Handle swap decision for BLUE on turn 2
        if (turn == 2 and self.colour == Colour.BLUE and
            opp_move is not None and not opp_move.is_swap()):
            if self._should_swap(board, opp_move):
                return Move(-1, -1)

        # Get legal moves
        legal_moves = self._get_legal_moves(board)
        if not legal_moves:
            return Move(0, 0)

        if len(legal_moves) == 1:
            return Move(legal_moves[0][0], legal_moves[0][1])

        # Select move using neural network
        best_move = self._select_move_neural(board)

        return Move(best_move[0], best_move[1])


# For tournament compatibility
AzaleaAgentClass = AzaleaAgent
