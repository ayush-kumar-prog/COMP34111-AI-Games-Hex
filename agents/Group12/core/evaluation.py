"""
Position evaluation functions including the revolutionary electrical resistance model.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import math

from src.Board import Board
from src.Colour import Colour
from src.Tile import Tile


class PositionEvaluator:
    """
    Basic position evaluation using heuristics.
    """

    def __init__(self):
        self.cache = {}  # Position cache

    def evaluate(self, board: Board, colour: Colour) -> float:
        """
        Evaluate position for given colour.
        Returns value in range [-1, 1] where positive is good for colour.
        """
        # Check for terminal positions first
        if board.has_ended(colour):
            return 1.0
        if board.has_ended(Colour.opposite(colour)):
            return -1.0

        # Compute various features
        connection_score = self._evaluate_connections(board, colour)
        edge_distance = self._evaluate_edge_distance(board, colour)
        center_control = self._evaluate_center_control(board, colour)

        # Weighted combination
        score = (0.4 * connection_score +
                 0.3 * edge_distance +
                 0.3 * center_control)

        return max(-1.0, min(1.0, score))  # Clamp to [-1, 1]

    def _evaluate_connections(self, board: Board, colour: Colour) -> float:
        """Evaluate connection strength."""
        my_connections = self._count_connections(board, colour)
        opp_connections = self._count_connections(board, Colour.opposite(colour))

        if my_connections + opp_connections == 0:
            return 0.0

        return (my_connections - opp_connections) / (my_connections + opp_connections)

    def _count_connections(self, board: Board, colour: Colour) -> int:
        """Count connections for a colour."""
        count = 0
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour == colour:
                    # Count adjacent same-color tiles
                    for di, dj in self._get_neighbors(i, j, board.size):
                        if board.tiles[di][dj].colour == colour:
                            count += 1
        return count // 2  # Each connection counted twice

    def _evaluate_edge_distance(self, board: Board, colour: Colour) -> float:
        """Evaluate minimum distance to winning edges."""
        if colour == Colour.RED:
            # Red connects top-bottom
            my_dist = self._min_distance_to_edges(board, colour, 'vertical')
            opp_dist = self._min_distance_to_edges(board, Colour.BLUE, 'horizontal')
        else:
            # Blue connects left-right
            my_dist = self._min_distance_to_edges(board, colour, 'horizontal')
            opp_dist = self._min_distance_to_edges(board, Colour.RED, 'vertical')

        # Normalize and compute difference
        max_dist = board.size
        my_score = 1.0 - (my_dist / max_dist)
        opp_score = 1.0 - (opp_dist / max_dist)

        return my_score - opp_score

    def _min_distance_to_edges(self, board: Board, colour: Colour, direction: str) -> int:
        """Calculate minimum distance to winning edges."""
        min_dist = board.size

        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour == colour:
                    if direction == 'vertical':
                        # Distance to top and bottom
                        dist = min(i, board.size - 1 - i)
                    else:
                        # Distance to left and right
                        dist = min(j, board.size - 1 - j)
                    min_dist = min(min_dist, dist)

        return min_dist

    def _evaluate_center_control(self, board: Board, colour: Colour) -> float:
        """Evaluate control of center region."""
        center = board.size // 2
        radius = board.size // 4

        my_center_count = 0
        opp_center_count = 0

        for i in range(max(0, center - radius), min(board.size, center + radius + 1)):
            for j in range(max(0, center - radius), min(board.size, center + radius + 1)):
                if board.tiles[i][j].colour == colour:
                    my_center_count += 1
                elif board.tiles[i][j].colour == Colour.opposite(colour):
                    opp_center_count += 1

        if my_center_count + opp_center_count == 0:
            return 0.0

        return (my_center_count - opp_center_count) / (my_center_count + opp_center_count)

    def _get_neighbors(self, x: int, y: int, board_size: int) -> List[Tuple[int, int]]:
        """Get valid neighbor positions."""
        neighbors = []
        # Hex has 6 neighbors
        deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                neighbors.append((nx, ny))

        return neighbors


class ResistanceEvaluator:
    """
    Revolutionary electrical resistance model for position evaluation.
    Models the board as an electrical circuit where:
    - Empty cells = 1Ω resistors
    - Our pieces = 0Ω (perfect conductors)
    - Opponent pieces = ∞Ω (insulators)

    Lower resistance between our edges = stronger position.
    """

    def __init__(self):
        self.cache = {}

    def evaluate(self, board: Board, colour: Colour) -> float:
        """
        Compute electrical resistance for position.
        Returns value in [0, 1] where lower resistance is better.
        """
        try:
            resistance = self._compute_resistance(board, colour)
            # Convert to score (lower resistance = higher score)
            # Use sigmoid to map to [0, 1]
            score = 1.0 / (1.0 + resistance)
            return score
        except:
            # Fallback to basic evaluation if resistance computation fails
            return PositionEvaluator().evaluate(board, colour)

    def _compute_resistance(self, board: Board, colour: Colour) -> float:
        """
        Compute effective resistance between winning edges.
        Uses sparse matrix for efficiency.
        """
        n = board.size
        num_nodes = n * n + 2  # Board cells + 2 virtual nodes for edges

        # Create adjacency matrix (conductance)
        # Using sparse matrix for efficiency
        rows, cols, data = [], [], []

        # Virtual nodes
        source_node = n * n
        target_node = n * n + 1

        # Connect pieces to virtual nodes based on colour
        for i in range(n):
            for j in range(n):
                node_id = i * n + j

                if board.tiles[i][j].colour == colour:
                    # Our piece - perfect conductor
                    if colour == Colour.RED:
                        # RED connects top-bottom
                        if i == 0:
                            # Top edge to source
                            rows.extend([source_node, node_id])
                            cols.extend([node_id, source_node])
                            data.extend([1000.0, 1000.0])  # High conductance
                        elif i == n - 1:
                            # Bottom edge to target
                            rows.extend([target_node, node_id])
                            cols.extend([node_id, target_node])
                            data.extend([1000.0, 1000.0])
                    else:
                        # BLUE connects left-right
                        if j == 0:
                            # Left edge to source
                            rows.extend([source_node, node_id])
                            cols.extend([node_id, source_node])
                            data.extend([1000.0, 1000.0])
                        elif j == n - 1:
                            # Right edge to target
                            rows.extend([target_node, node_id])
                            cols.extend([node_id, target_node])
                            data.extend([1000.0, 1000.0])

                    # Connect to neighbors with high conductance
                    for ni, nj in self._get_hex_neighbors(i, j, n):
                        neighbor_id = ni * n + nj
                        if board.tiles[ni][nj].colour != Colour.opposite(colour):
                            rows.extend([node_id, neighbor_id])
                            cols.extend([neighbor_id, node_id])
                            data.extend([10.0, 10.0])  # High conductance

                elif board.tiles[i][j].colour is None:
                    # Empty cell - 1Ω resistor (conductance = 1)
                    for ni, nj in self._get_hex_neighbors(i, j, n):
                        neighbor_id = ni * n + nj
                        if board.tiles[ni][nj].colour != Colour.opposite(colour):
                            rows.extend([node_id, neighbor_id])
                            cols.extend([neighbor_id, node_id])
                            data.extend([1.0, 1.0])

                # Opponent pieces are insulators (no connections)

        if not data:
            return float('inf')  # No connections

        # Build Laplacian matrix
        G = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        L = self._compute_laplacian(G)

        # Compute resistance between source and target
        resistance = self._compute_effective_resistance(L, source_node, target_node)

        return resistance

    def _get_hex_neighbors(self, x: int, y: int, board_size: int) -> List[Tuple[int, int]]:
        """Get valid hex neighbors."""
        neighbors = []
        # Hex neighbor offsets
        deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                neighbors.append((nx, ny))

        return neighbors

    def _compute_laplacian(self, G: csr_matrix) -> csr_matrix:
        """Compute graph Laplacian from conductance matrix."""
        # Degree matrix
        D = np.array(G.sum(axis=1)).flatten()
        D_matrix = csr_matrix((D, (range(len(D)), range(len(D)))))

        # Laplacian = D - G
        L = D_matrix - G
        return L

    def _compute_effective_resistance(self, L: csr_matrix, source: int, target: int) -> float:
        """
        Compute effective resistance between two nodes.
        Uses pseudoinverse method for numerical stability.
        """
        n = L.shape[0]

        # Create current vector (1A from source to target)
        b = np.zeros(n)
        b[source] = 1.0
        b[target] = -1.0

        # Remove one row/column to make matrix invertible (ground node)
        # Ground the target node
        L_reduced = L.tocsr()[:-1, :-1]
        b_reduced = b[:-1]

        try:
            # Solve for voltages
            v = spsolve(L_reduced, b_reduced)

            # Add back ground voltage (0)
            v_full = np.append(v, 0)

            # Resistance = voltage difference / current
            resistance = abs(v_full[source] - v_full[target])

            return resistance
        except:
            return float('inf')  # Failed to solve

    def compute_gradient(self, board: Board, colour: Colour) -> Dict[Tuple[int, int], float]:
        """
        Compute resistance gradient for each empty cell.
        Shows which moves most reduce resistance.
        """
        gradients = {}
        current_resistance = self._compute_resistance(board, colour)

        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    # Simulate placing our piece
                    board_copy = copy.deepcopy(board)
                    board_copy.tiles[i][j].colour = colour

                    new_resistance = self._compute_resistance(board_copy, colour)
                    gradient = current_resistance - new_resistance  # Positive = improvement

                    gradients[(i, j)] = gradient

        return gradients