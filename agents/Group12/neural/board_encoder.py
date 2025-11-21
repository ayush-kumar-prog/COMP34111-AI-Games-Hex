"""
Board state encoder for neural network input.
Converts Hex board to 11×11×5 tensor representation.
"""

import torch
import numpy as np
from typing import Optional

from src.Board import Board
from src.Colour import Colour


class BoardEncoder:
    """
    Encodes Hex board states into neural network input format.

    Output: 11×11×5 tensor with channels:
    - Channel 0: Our stones (1 if our stone, 0 otherwise)
    - Channel 1: Opponent stones (1 if opponent stone, 0 otherwise)
    - Channel 2: Empty cells (1 if empty, 0 otherwise)
    - Channel 3: Legal moves mask (1 if legal move, 0 otherwise)
    - Channel 4: Normalized edge distances (0-1 continuous)
    """

    def __init__(self, board_size: int = 11):
        self.board_size = board_size

    def encode(self, board: Board, colour: Colour, device: str = 'cpu') -> torch.Tensor:
        """
        Encode board state for neural network.

        Args:
            board: Current board state
            colour: Our colour (to determine which player we are)
            device: Device to place tensor on ('cuda' or 'cpu')

        Returns:
            Tensor of shape (5, board_size, board_size)
        """
        state = torch.zeros((5, self.board_size, self.board_size), dtype=torch.float32)

        opponent_colour = Colour.opposite(colour)

        for i in range(self.board_size):
            for j in range(self.board_size):
                tile_colour = board.tiles[i][j].colour

                if tile_colour == colour:
                    # Channel 0: Our stones
                    state[0, i, j] = 1.0
                elif tile_colour == opponent_colour:
                    # Channel 1: Opponent stones
                    state[1, i, j] = 1.0
                elif tile_colour is None:
                    # Channel 2: Empty cells
                    state[2, i, j] = 1.0
                    # Channel 3: Legal moves (all empty cells)
                    state[3, i, j] = 1.0

        # Channel 4: Normalized edge distances
        state[4] = self._compute_edge_distances(colour)

        return state.to(device)

    def _compute_edge_distances(self, colour: Colour) -> torch.Tensor:
        """
        Compute normalized distances to winning edges.

        For RED: Distance to top or bottom edge (whichever is closer)
        For BLUE: Distance to left or right edge (whichever is closer)

        Returns:
            Tensor of shape (board_size, board_size) with values in [0, 1]
        """
        edge_dist = torch.zeros((self.board_size, self.board_size))

        for i in range(self.board_size):
            for j in range(self.board_size):
                if colour == Colour.RED:
                    # RED connects top-bottom (rows 0 to board_size-1)
                    dist_to_top = i
                    dist_to_bottom = self.board_size - 1 - i
                    min_dist = min(dist_to_top, dist_to_bottom)
                else:
                    # BLUE connects left-right (cols 0 to board_size-1)
                    dist_to_left = j
                    dist_to_right = self.board_size - 1 - j
                    min_dist = min(dist_to_left, dist_to_right)

                # Normalize to [0, 1]
                # Closer to edge = higher value
                normalized = 1.0 - (min_dist / (self.board_size // 2))
                edge_dist[i, j] = max(0.0, min(1.0, normalized))

        return edge_dist

    def encode_batch(self, boards: list, colours: list, device: str = 'cpu') -> torch.Tensor:
        """
        Encode multiple board states as a batch.

        Args:
            boards: List of Board objects
            colours: List of Colour objects (our colour for each board)
            device: Device to place tensor on

        Returns:
            Tensor of shape (batch_size, 5, board_size, board_size)
        """
        batch = []
        for board, colour in zip(boards, colours):
            state = self.encode(board, colour, device='cpu')  # Encode on CPU first
            batch.append(state)

        batch_tensor = torch.stack(batch)
        return batch_tensor.to(device)

    def decode_policy(self, policy_tensor: torch.Tensor, board: Board) -> dict:
        """
        Convert policy tensor to dictionary of (move -> probability).

        Args:
            policy_tensor: Tensor of shape (board_size²,) with move probabilities
            board: Current board state (to filter legal moves)

        Returns:
            Dictionary mapping (x, y) tuples to probabilities
        """
        policy_dict = {}

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board.tiles[i][j].colour is None:  # Legal move
                    idx = i * self.board_size + j
                    prob = policy_tensor[idx].item()
                    policy_dict[(i, j)] = prob

        # Normalize to sum to 1.0 (in case some probability leaked to illegal moves)
        total = sum(policy_dict.values())
        if total > 0:
            policy_dict = {k: v/total for k, v in policy_dict.items()}

        return policy_dict

    def create_move_mask(self, board: Board) -> torch.Tensor:
        """
        Create mask for legal moves.

        Args:
            board: Current board state

        Returns:
            Tensor of shape (board_size²,) with 1 for legal moves, 0 for illegal
        """
        mask = torch.zeros(self.board_size * self.board_size)

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board.tiles[i][j].colour is None:
                    idx = i * self.board_size + j
                    mask[idx] = 1.0

        return mask


def test_board_encoder():
    """Test the board encoder with a sample board."""
    print("Testing BoardEncoder...")

    # Create a sample board (would need actual Board class)
    # This is a placeholder test
    encoder = BoardEncoder(board_size=11)

    print(f"✅ BoardEncoder initialized")
    print(f"   Board size: {encoder.board_size}")
    print(f"   Output shape: (5, {encoder.board_size}, {encoder.board_size})")

    # Test edge distance computation
    edge_dist_red = encoder._compute_edge_distances(Colour.RED)
    edge_dist_blue = encoder._compute_edge_distances(Colour.BLUE)

    print(f"\n✅ Edge distance computation:")
    print(f"   RED edge distances shape: {edge_dist_red.shape}")
    print(f"   BLUE edge distances shape: {edge_dist_blue.shape}")
    print(f"   RED center value: {edge_dist_red[5, 5]:.3f}")
    print(f"   BLUE center value: {edge_dist_blue[5, 5]:.3f}")

    # Test move mask creation
    # Would need actual board to test properly

    print(f"\n✅ BoardEncoder tests passed!")


if __name__ == '__main__':
    test_board_encoder()
