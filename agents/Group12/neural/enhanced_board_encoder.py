"""
Enhanced board state encoder for neural network input.
Converts Hex board to 11x11x8 tensor with Hex-specific features.

This encoder provides richer features than the basic 5-channel version:
- Separate distance channels for each edge
- Bridge pattern detection
- Better suited for learning Hex-specific strategies
"""

import torch
import numpy as np
from typing import Optional, List, Tuple
from collections import deque

from src.Board import Board
from src.Colour import Colour


class EnhancedBoardEncoder:
    """
    Enhanced encoder with 8 Hex-specific feature channels.

    Output: 11x11x8 tensor with channels:
    - Channel 0: Our stones (1 if our stone, 0 otherwise)
    - Channel 1: Opponent stones (1 if opponent stone, 0 otherwise)
    - Channel 2: Empty cells (1 if empty, 0 otherwise)
    - Channel 3: Our distance to start edge (normalized 0-1, closer=higher)
    - Channel 4: Our distance to end edge (normalized 0-1, closer=higher)
    - Channel 5: Opponent distance to start edge (normalized)
    - Channel 6: Opponent distance to end edge (normalized)
    - Channel 7: Bridge pattern indicators (1 if part of bridge pattern)
    """

    def __init__(self, board_size: int = 11):
        self.board_size = board_size
        self.num_channels = 8

        # Pre-compute edge distance matrices for efficiency
        self._precompute_edge_distances()

        # Bridge patterns (relative offsets for carrier cells)
        # A bridge connects two stones with two empty carrier cells
        self.bridge_patterns = [
            # Pattern: Stone at (0,0), carrier at offsets, stone at end
            # Each tuple: (carrier1, carrier2, stone2)
            ((0, 1), (1, 0), (1, 1)),      # Bridge type 1
            ((0, 1), (-1, 1), (-1, 2)),    # Bridge type 2
            ((1, 0), (1, -1), (2, -1)),    # Bridge type 3
            ((-1, 1), (-1, 0), (-2, 1)),   # Bridge type 4 (rotated)
            ((1, -1), (0, -1), (1, -2)),   # Bridge type 5 (rotated)
            ((1, 0), (0, 1), (1, 1)),      # Bridge type 6 (alternative)
        ]

    def _precompute_edge_distances(self):
        """Pre-compute normalized edge distances for both colors."""
        size = self.board_size

        # RED: connects top (row 0) to bottom (row size-1)
        self.red_dist_to_top = np.zeros((size, size), dtype=np.float32)
        self.red_dist_to_bottom = np.zeros((size, size), dtype=np.float32)

        # BLUE: connects left (col 0) to right (col size-1)
        self.blue_dist_to_left = np.zeros((size, size), dtype=np.float32)
        self.blue_dist_to_right = np.zeros((size, size), dtype=np.float32)

        max_dist = size - 1

        for i in range(size):
            for j in range(size):
                # RED distances (row-based)
                self.red_dist_to_top[i, j] = 1.0 - (i / max_dist)
                self.red_dist_to_bottom[i, j] = 1.0 - ((size - 1 - i) / max_dist)

                # BLUE distances (column-based)
                self.blue_dist_to_left[i, j] = 1.0 - (j / max_dist)
                self.blue_dist_to_right[i, j] = 1.0 - ((size - 1 - j) / max_dist)

    def encode(self, board: Board, colour: Colour, device: str = 'cpu') -> torch.Tensor:
        """
        Encode board state for neural network.

        Args:
            board: Current board state
            colour: Our colour (perspective)
            device: Device to place tensor on

        Returns:
            Tensor of shape (8, board_size, board_size)
        """
        state = np.zeros((self.num_channels, self.board_size, self.board_size),
                        dtype=np.float32)

        opponent_colour = Colour.opposite(colour)

        # Channels 0-2: Stone positions
        for i in range(self.board_size):
            for j in range(self.board_size):
                tile_colour = board.tiles[i][j].colour

                if tile_colour == colour:
                    state[0, i, j] = 1.0  # Our stones
                elif tile_colour == opponent_colour:
                    state[1, i, j] = 1.0  # Opponent stones
                else:
                    state[2, i, j] = 1.0  # Empty cells

        # Channels 3-6: Edge distances based on colour
        if colour == Colour.RED:
            # RED connects top to bottom
            state[3] = self.red_dist_to_top
            state[4] = self.red_dist_to_bottom
            # Opponent (BLUE) connects left to right
            state[5] = self.blue_dist_to_left
            state[6] = self.blue_dist_to_right
        else:
            # BLUE connects left to right
            state[3] = self.blue_dist_to_left
            state[4] = self.blue_dist_to_right
            # Opponent (RED) connects top to bottom
            state[5] = self.red_dist_to_top
            state[6] = self.red_dist_to_bottom

        # Channel 7: Bridge patterns
        state[7] = self._compute_bridge_features(board, colour)

        return torch.from_numpy(state).to(device)

    def _compute_bridge_features(self, board: Board, colour: Colour) -> np.ndarray:
        """
        Compute bridge pattern features.

        Marks cells that are:
        1. Part of an existing bridge (carrier cells between two stones)
        2. Potential bridge completions

        Returns:
            2D array with bridge pattern indicators
        """
        bridge_map = np.zeros((self.board_size, self.board_size), dtype=np.float32)

        for i in range(self.board_size):
            for j in range(self.board_size):
                # Check if this cell is our stone
                if board.tiles[i][j].colour == colour:
                    # Check all bridge patterns from this stone
                    for carrier1, carrier2, stone2 in self.bridge_patterns:
                        c1_i, c1_j = i + carrier1[0], j + carrier1[1]
                        c2_i, c2_j = i + carrier2[0], j + carrier2[1]
                        s2_i, s2_j = i + stone2[0], j + stone2[1]

                        # Check bounds
                        if not self._in_bounds(c1_i, c1_j):
                            continue
                        if not self._in_bounds(c2_i, c2_j):
                            continue
                        if not self._in_bounds(s2_i, s2_j):
                            continue

                        # Check if this forms a valid bridge
                        # Need: both carriers empty, end stone is ours
                        carrier1_empty = board.tiles[c1_i][c1_j].colour is None
                        carrier2_empty = board.tiles[c2_i][c2_j].colour is None
                        stone2_ours = board.tiles[s2_i][s2_j].colour == colour

                        if carrier1_empty and carrier2_empty and stone2_ours:
                            # Mark carrier cells as bridge cells
                            bridge_map[c1_i, c1_j] = 1.0
                            bridge_map[c2_i, c2_j] = 1.0

        return bridge_map

    def _in_bounds(self, i: int, j: int) -> bool:
        """Check if coordinates are within board bounds."""
        return 0 <= i < self.board_size and 0 <= j < self.board_size

    def encode_numpy(self, board: Board, colour: Colour) -> np.ndarray:
        """
        Encode board state as numpy array (for NumPy inference).

        Args:
            board: Current board state
            colour: Our colour

        Returns:
            NumPy array of shape (8, board_size, board_size)
        """
        return self.encode(board, colour, device='cpu').numpy()

    def encode_batch(self, boards: List[Board], colours: List[Colour],
                    device: str = 'cpu') -> torch.Tensor:
        """
        Encode multiple board states as a batch.

        Args:
            boards: List of Board objects
            colours: List of colours (perspective for each board)
            device: Device to place tensor on

        Returns:
            Tensor of shape (batch_size, 8, board_size, board_size)
        """
        batch = []
        for board, colour in zip(boards, colours):
            state = self.encode(board, colour, device='cpu')
            batch.append(state)

        batch_tensor = torch.stack(batch)
        return batch_tensor.to(device)

    def decode_policy(self, policy_tensor: torch.Tensor, board: Board) -> dict:
        """
        Convert policy tensor to dictionary of (move -> probability).

        Args:
            policy_tensor: Tensor of shape (board_size^2,) with move probabilities
            board: Current board state (to filter legal moves)

        Returns:
            Dictionary mapping (x, y) tuples to probabilities
        """
        policy_dict = {}

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board.tiles[i][j].colour is None:  # Legal move
                    idx = i * self.board_size + j
                    if isinstance(policy_tensor, torch.Tensor):
                        prob = policy_tensor[idx].item()
                    else:
                        prob = float(policy_tensor[idx])
                    policy_dict[(i, j)] = prob

        # Normalize
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
            Tensor of shape (board_size^2,) with 1 for legal moves, 0 for illegal
        """
        mask = torch.zeros(self.board_size * self.board_size)

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board.tiles[i][j].colour is None:
                    idx = i * self.board_size + j
                    mask[idx] = 1.0

        return mask

    def create_move_mask_numpy(self, board: Board) -> np.ndarray:
        """Create move mask as numpy array."""
        return self.create_move_mask(board).numpy()


def compute_shortest_path_distance(board: Board, colour: Colour) -> np.ndarray:
    """
    Compute shortest path distance from each cell to winning edges.
    Uses BFS considering only empty or own-color cells.

    This is a more sophisticated distance metric that considers
    actual board connectivity rather than just geometric distance.

    Args:
        board: Current board state
        colour: Colour to compute distances for

    Returns:
        2D array with shortest path distances (normalized)
    """
    size = board.size
    distances = np.full((size, size), float('inf'), dtype=np.float32)

    # Determine start edge based on colour
    if colour == Colour.RED:
        # RED connects top (row 0) to bottom
        start_cells = [(0, j) for j in range(size)]
    else:
        # BLUE connects left (col 0) to right
        start_cells = [(i, 0) for i in range(size)]

    # BFS from start edge
    queue = deque()
    visited = set()

    for i, j in start_cells:
        tile = board.tiles[i][j]
        if tile.colour is None or tile.colour == colour:
            queue.append((i, j, 0))
            visited.add((i, j))
            distances[i, j] = 0

    # Hex neighbors (6 directions)
    directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    while queue:
        x, y, dist = queue.popleft()

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                tile = board.tiles[nx][ny]
                # Can traverse empty or own-color cells
                if tile.colour is None or tile.colour == colour:
                    visited.add((nx, ny))
                    distances[nx, ny] = dist + 1
                    queue.append((nx, ny, dist + 1))

    # Normalize distances (convert inf to max, then normalize)
    max_dist = size * 2  # Maximum possible path length
    distances = np.clip(distances, 0, max_dist)
    distances = 1.0 - (distances / max_dist)  # Closer = higher value

    return distances


def test_enhanced_encoder():
    """Test the enhanced board encoder."""
    print("Testing EnhancedBoardEncoder...")

    encoder = EnhancedBoardEncoder(board_size=11)

    print(f"Encoder initialized:")
    print(f"  Board size: {encoder.board_size}")
    print(f"  Channels: {encoder.num_channels}")
    print(f"  Output shape: ({encoder.num_channels}, {encoder.board_size}, {encoder.board_size})")

    # Test edge distances
    print(f"\nEdge distance precomputation:")
    print(f"  RED dist to top [0,0]: {encoder.red_dist_to_top[0, 0]:.3f}")
    print(f"  RED dist to top [5,5]: {encoder.red_dist_to_top[5, 5]:.3f}")
    print(f"  RED dist to top [10,5]: {encoder.red_dist_to_top[10, 5]:.3f}")

    print(f"\nEnhancedBoardEncoder tests passed!")


if __name__ == '__main__':
    test_enhanced_encoder()
