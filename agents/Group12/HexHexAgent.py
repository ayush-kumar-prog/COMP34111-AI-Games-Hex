"""
HexHexAgent - Pure CNN agent for Hex trained via self-play.
Based on the HexHex project: https://github.com/harbecke/HexHex

Architecture: 18 skip layers with Swish activation, 64 channels.
Uses 2-channel input (player stones) with border padding.
No MCTS - pure neural network inference.
"""

from __future__ import annotations

import os
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
MODEL_PATH = os.path.join(AGENT_DIR, "models", "hexhex_11x11.pt")


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class SkipLayerBias(nn.Module):
    """Skip layer with convolution, batch norm, and swish activation."""

    def __init__(self, channels: int, reach: int = 1, scale: float = 1.0):
        super().__init__()
        kernel_size = reach * 2 + 1
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                              padding=reach, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x + self.scale * self.bn(self.conv(x)))


class HexHexConv(nn.Module):
    """
    HexHex convolutional network for Hex.

    Architecture:
    - Input: 2 channels (player 0 stones, player 1 stones) with border padding
    - Initial conv: 2 -> intermediate_channels
    - Skip layers: intermediate_channels -> intermediate_channels with residual
    - Policy conv: intermediate_channels -> 1
    - Output: logits for each board position
    """

    def __init__(self, board_size: int = 11, layers: int = 18,
                 intermediate_channels: int = 64, reach: int = 1,
                 export_mode: bool = True):
        super().__init__()
        self.board_size = board_size
        self.export_mode = export_mode

        # Input convolution: 2 channels (with border) -> intermediate
        # With reach=1: kernel_size=3, padding=0
        # Input shape: (batch, 2, board_size+2, board_size+2)
        # Output shape: (batch, intermediate, board_size, board_size)
        kernel_size = 2 * reach + 1
        self.conv = nn.Conv2d(2, intermediate_channels, kernel_size=kernel_size,
                              padding=reach - 1)

        # Skip layers
        self.skiplayers = nn.ModuleList([
            SkipLayerBias(intermediate_channels, reach=1)
            for _ in range(layers)
        ])

        # Policy output: intermediate -> 1
        self.policyconv = nn.Conv2d(intermediate_channels, 1, kernel_size=kernel_size,
                                    padding=reach, bias=False)

        # Learnable bias for each position
        self.bias = nn.Parameter(torch.zeros(board_size ** 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Board tensor with borders, shape (batch, 2, board_size+2, board_size+2)

        Returns:
            Policy logits, shape (batch, board_size * board_size)
        """
        # Compute illegal move mask from occupied positions
        # Inner board is x[:, :, 1:-1, 1:-1]
        x_sum = torch.sum(x[:, :, 1:-1, 1:-1], dim=1).view(-1, self.board_size ** 2)

        # Initial convolution
        x = self.conv(x)

        # Skip layers
        for skiplayer in self.skiplayers:
            x = skiplayer(x)

        # Policy output
        policy = self.policyconv(x).view(-1, self.board_size ** 2) + self.bias

        if self.export_mode:
            return policy

        # Apply illegal move masking for training/generation
        illegal = x_sum * torch.exp(
            torch.tanh((x_sum.sum(dim=1) - 1) * 1000) * 10
        ).unsqueeze(1).expand_as(x_sum) - x_sum
        return policy - illegal


class RotationWrapperModel(nn.Module):
    """
    Wrapper that evaluates input and its 180-degree rotation,
    averaging both predictions for rotational symmetry.
    """

    def __init__(self, model: nn.Module, export_mode: bool = True):
        super().__init__()
        self.board_size = model.board_size
        self.internal_model = model
        self.export_mode = export_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.export_mode:
            return self.internal_model(x)

        # Evaluate both original and 180-rotated board
        x_flip = torch.flip(x, [2, 3])
        y_flip = self.internal_model(x_flip)
        # Flip the 1D output to match rotated coordinates
        y = torch.flip(y_flip, [1])
        return (self.internal_model(x) + y) / 2


class HexHexAgent(AgentBase):
    """
    Agent using pretrained HexHex network for move selection.
    Pure neural network inference - no MCTS.
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.board_size = 11
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load network
        self.net = self._load_network()
        self.net.to(self.device)
        self.net.eval()

    def _load_network(self) -> nn.Module:
        """Load pretrained HexHex network weights."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"HexHex model not found at {MODEL_PATH}")

        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)

        # Read config from checkpoint
        config = checkpoint.get('config', None)
        if config is not None:
            # Config is a ConfigParser object - try different access methods
            try:
                # Method 1: ConfigParser with section
                board_size = config.getint('DEFAULT', 'board_size')
                layers = config.getint('DEFAULT', 'layers')
                intermediate_channels = config.getint('DEFAULT', 'intermediate_channels')
                reach = config.getint('DEFAULT', 'reach')
                switch_model = config.getboolean('DEFAULT', 'switch_model')
                rotation_model = config.getboolean('DEFAULT', 'rotation_model')
            except (TypeError, KeyError):
                try:
                    # Method 2: ConfigParser without section (uses DEFAULT implicitly)
                    board_size = config.getint('board_size')
                    layers = config.getint('layers')
                    intermediate_channels = config.getint('intermediate_channels')
                    reach = config.getint('reach')
                    switch_model = config.getboolean('switch_model')
                    rotation_model = config.getboolean('rotation_model')
                except (TypeError, KeyError, AttributeError):
                    # Method 3: Dict-like access
                    board_size = int(config.get('board_size', 11))
                    layers = int(config.get('layers', 18))
                    intermediate_channels = int(config.get('intermediate_channels', 64))
                    reach = int(config.get('reach', 1))
                    switch_model = str(config.get('switch_model', 'true')).lower() == 'true'
                    rotation_model = str(config.get('rotation_model', 'true')).lower() == 'true'
            self.board_size = board_size
        else:
            # Fallback to defaults if config not in checkpoint
            board_size = self.board_size
            layers = 18
            intermediate_channels = 64
            reach = 1
            switch_model = True  # Enable switch by default
            rotation_model = True  # Use rotation wrapper

        # Create base model
        internal_model = HexHexConv(
            board_size=board_size,
            layers=layers,
            intermediate_channels=intermediate_channels,
            reach=reach,
            export_mode=True
        )

        # Apply wrappers based on config
        model = internal_model

        if rotation_model:
            model = RotationWrapperModel(internal_model, export_mode=True)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _board_to_tensor(self, board: Board) -> torch.Tensor:
        """
        Convert Board to HexHex tensor format.

        HexHex format:
        - 2 channels: player 0 stones, player 1 stones
        - Border padding: +2 in each dimension
        - Player 0 (RED): border on top/bottom rows
        - Player 1 (BLUE): border on left/right columns

        The current player is always represented as player 0,
        so we need to swap/transpose when playing as BLUE.
        """
        size = board.size
        tiles = board._tiles

        # Create logical board tensor (2, size, size)
        logical_board = torch.zeros(2, size, size)

        for i in range(size):
            for j in range(size):
                c = tiles[i][j].colour
                if c == Colour.RED:
                    logical_board[0, i, j] = 1.0
                elif c == Colour.BLUE:
                    logical_board[1, i, j] = 1.0

        # If we're BLUE, we need to transform perspective
        # HexHex always evaluates from current player's perspective
        if self._colour == Colour.BLUE:
            # Swap channels (we become player 0)
            logical_board = torch.stack([logical_board[1], logical_board[0]])
            # Transpose board (BLUE's top-bottom becomes left-right)
            logical_board = logical_board.transpose(1, 2)

        # Add border padding
        bordered = torch.zeros(2, size + 2, size + 2)

        # Set borders for win condition encoding
        bordered[0, 0, 1:-1] = 1.0      # Player 0: top edge
        bordered[0, -1, 1:-1] = 1.0     # Player 0: bottom edge
        bordered[1, 1:-1, 0] = 1.0      # Player 1: left edge
        bordered[1, 1:-1, -1] = 1.0     # Player 1: right edge

        # Copy inner board
        bordered[:, 1:-1, 1:-1] = logical_board

        # Add batch dimension
        return bordered.unsqueeze(0).to(self.device)

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

    def _transform_move_index(self, idx: int, size: int) -> Tuple[int, int]:
        """
        Transform move index from network output to board coordinates.

        For BLUE: network outputs in transposed space, need to un-transpose.
        """
        row = idx // size
        col = idx % size

        if self._colour == Colour.BLUE:
            # Un-transpose: swap row and col
            row, col = col, row

        return (row, col)

    @torch.no_grad()
    def _select_move(self, board: Board) -> Tuple[int, int]:
        """Select move using neural network."""
        # Encode board
        board_tensor = self._board_to_tensor(board)

        # Get network output
        logits = self.net(board_tensor)[0]  # (board_size * board_size,)

        # Create legal move mask
        legal_moves = self._get_legal_moves(board)
        mask = torch.full((self.board_size ** 2,), float('-inf'), device=self.device)

        for move in legal_moves:
            # Transform to network space if BLUE
            if self._colour == Colour.BLUE:
                net_row, net_col = move[1], move[0]  # Transpose
            else:
                net_row, net_col = move

            idx = net_row * self.board_size + net_col
            mask[idx] = 0.0

        # Apply mask and find best move
        masked_logits = logits + mask
        best_idx = masked_logits.argmax().item()

        # Transform back to board coordinates
        return self._transform_move_index(best_idx, self.board_size)

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
        if (turn == 2 and self._colour == Colour.BLUE and
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
        best_move = self._select_move(board)

        return Move(best_move[0], best_move[1])


# For tournament compatibility
HexHexAgentClass = HexHexAgent
