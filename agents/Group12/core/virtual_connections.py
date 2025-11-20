"""
Virtual connection detection - the foundation of strong Hex play.
Detects unbreakable connections through bridge patterns and more.
"""

from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
import copy

from src.Board import Board
from src.Colour import Colour


@dataclass
class VirtualConnection:
    """
    Represents a virtual (unbreakable) connection between stones.
    """
    order: int  # 0=adjacent, 1=bridge, 2=ladder, etc.
    keys: Set[Tuple[int, int]]  # Our stones that are connected
    carriers: Set[Tuple[int, int]]  # Empty cells that maintain connection
    colour: Colour


class VirtualConnectionDetector:
    """
    Detects virtual connections (bridges, ladders, etc.) on the board.
    This is crucial for strong Hex play as it identifies unbreakable connections.
    """

    # Bridge templates (relative positions)
    # Format: [(key1, key2), (carrier1, carrier2)]
    BRIDGE_TEMPLATES = [
        # Standard bridge pattern
        [[(0, 0), (1, 1)], [(0, 1), (1, 0)]],
        # Rotations and reflections
        [[(0, 0), (1, -1)], [(0, -1), (1, 0)]],
        [[(0, 0), (-1, 1)], [(-1, 0), (0, 1)]],
        [[(0, 0), (-1, -1)], [(-1, 0), (0, -1)]],
        # Additional patterns
        [[(0, 0), (2, 1)], [(1, 0), (1, 1)]],
        [[(0, 0), (1, 2)], [(0, 1), (1, 1)]],
    ]

    def __init__(self):
        self.cache = {}

    def find_all_virtual_connections(self, board: Board, colour: Colour) -> List[VirtualConnection]:
        """
        Find all virtual connections for the given colour.
        """
        vcs = []

        # Find 0th order (adjacent) connections
        vcs.extend(self._find_adjacent_connections(board, colour))

        # Find 1st order (bridge) connections
        vcs.extend(self._find_bridge_connections(board, colour))

        # Could add 2nd order (ladder) connections here
        # vcs.extend(self._find_ladder_connections(board, colour))

        return vcs

    def _find_adjacent_connections(self, board: Board, colour: Colour) -> List[VirtualConnection]:
        """Find all adjacent (0th order) connections."""
        connections = []
        visited = set()

        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour == colour and (i, j) not in visited:
                    # Find connected component
                    component = self._find_connected_component(board, i, j, colour)
                    visited.update(component)

                    if len(component) > 1:
                        # Create virtual connection for this component
                        vc = VirtualConnection(
                            order=0,
                            keys=component,
                            carriers=set(),  # No carriers for direct connections
                            colour=colour
                        )
                        connections.append(vc)

        return connections

    def _find_connected_component(self, board: Board, x: int, y: int, colour: Colour) -> Set[Tuple[int, int]]:
        """Find all stones connected to (x, y)."""
        component = set()
        stack = [(x, y)]

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in component:
                continue

            component.add((cx, cy))

            # Check all neighbors
            for nx, ny in self._get_hex_neighbors(cx, cy, board.size):
                if board.tiles[nx][ny].colour == colour and (nx, ny) not in component:
                    stack.append((nx, ny))

        return component

    def _find_bridge_connections(self, board: Board, colour: Colour) -> List[VirtualConnection]:
        """
        Find all bridge (1st order virtual) connections.
        A bridge is two stones with two empty mutual neighbors.
        """
        connections = []

        # Check every pair of our stones
        our_stones = []
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour == colour:
                    our_stones.append((i, j))

        # Check each pair for bridge patterns
        for i, stone1 in enumerate(our_stones):
            for stone2 in our_stones[i+1:]:
                bridge = self._check_bridge(board, stone1, stone2, colour)
                if bridge:
                    connections.append(bridge)

        return connections

    def _check_bridge(self, board: Board, stone1: Tuple[int, int],
                      stone2: Tuple[int, int], colour: Colour) -> Optional[VirtualConnection]:
        """
        Check if two stones form a bridge.
        A bridge exists if the stones share exactly two empty neighbors.
        """
        x1, y1 = stone1
        x2, y2 = stone2

        # Check if stones are in bridge position (not adjacent, but close)
        dx = x2 - x1
        dy = y2 - y1

        # Standard bridge patterns have specific distances
        if (dx, dy) not in [(1, 1), (-1, -1), (1, -1), (-1, 1), (2, 1), (1, 2),
                            (-2, -1), (-1, -2), (2, -1), (-2, 1), (1, -2), (-1, 2)]:
            return None

        # Find common empty neighbors (carriers)
        neighbors1 = set(self._get_hex_neighbors(x1, y1, board.size))
        neighbors2 = set(self._get_hex_neighbors(x2, y2, board.size))
        common = neighbors1.intersection(neighbors2)

        # Filter for empty cells
        carriers = set()
        for nx, ny in common:
            if board.tiles[nx][ny].colour is None:
                carriers.add((nx, ny))

        # A bridge needs exactly 2 empty carriers
        if len(carriers) == 2:
            return VirtualConnection(
                order=1,
                keys={stone1, stone2},
                carriers=carriers,
                colour=colour
            )

        return None

    def is_virtual_connection_threatened(self, vc: VirtualConnection, board: Board) -> bool:
        """
        Check if a virtual connection is threatened.
        A VC is threatened if opponent occupies one of its carriers.
        """
        if vc.order == 0:
            # Adjacent connections can't be threatened
            return False

        opp_colour = Colour.opposite(vc.colour)
        for cx, cy in vc.carriers:
            if board.tiles[cx][cy].colour == opp_colour:
                return True

        return False

    def get_defensive_moves(self, vc: VirtualConnection, board: Board) -> List[Tuple[int, int]]:
        """
        Get moves that defend a threatened virtual connection.
        """
        defensive_moves = []

        if self.is_virtual_connection_threatened(vc, board):
            # Return empty carriers as defensive moves
            for cx, cy in vc.carriers:
                if board.tiles[cx][cy].colour is None:
                    defensive_moves.append((cx, cy))

        return defensive_moves

    def find_connection_path(self, board: Board, colour: Colour,
                            start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[VirtualConnection]]:
        """
        Find a path of virtual connections between two positions.
        Used to check if a winning path exists.
        """
        # This would implement a search through virtual connections
        # to find a path from start to end
        # For now, return None (not implemented)
        return None

    def evaluate_connection_strength(self, board: Board, colour: Colour) -> float:
        """
        Evaluate overall connection strength for a colour.
        Higher score = stronger connection network.
        """
        vcs = self.find_all_virtual_connections(board, colour)

        if not vcs:
            return 0.0

        score = 0.0

        # Score based on connection types
        for vc in vcs:
            if vc.order == 0:
                # Direct connections are strongest
                score += len(vc.keys) * 2.0
            elif vc.order == 1:
                # Bridges are valuable
                score += 3.0
                # Extra points if bridge connects to edges
                if self._connects_to_edge(vc, board, colour):
                    score += 2.0
            # Higher order connections would score here

        # Normalize by board size
        max_score = board.size * board.size
        return min(1.0, score / max_score)

    def _connects_to_edge(self, vc: VirtualConnection, board: Board, colour: Colour) -> bool:
        """Check if virtual connection connects to a winning edge."""
        for x, y in vc.keys:
            if colour == Colour.RED:
                # RED wins top-bottom
                if x == 0 or x == board.size - 1:
                    return True
            else:
                # BLUE wins left-right
                if y == 0 or y == board.size - 1:
                    return True
        return False

    def _get_hex_neighbors(self, x: int, y: int, board_size: int) -> List[Tuple[int, int]]:
        """Get valid hex neighbors for a position."""
        neighbors = []
        # Hex has 6 neighbors
        deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                neighbors.append((nx, ny))

        return neighbors

    def find_must_play_moves(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """
        Find moves that must be played to maintain virtual connections.
        These are high-priority moves.
        """
        must_play = set()
        opp_colour = Colour.opposite(colour)

        # Get our virtual connections
        our_vcs = self.find_all_virtual_connections(board, colour)

        # Check which are threatened
        for vc in our_vcs:
            if vc.order > 0:  # Only non-adjacent connections can be threatened
                # Simulate opponent moves
                for cx, cy in vc.carriers:
                    if board.tiles[cx][cy].colour is None:
                        # Simulate opponent playing here
                        board_copy = copy.deepcopy(board)
                        board_copy.tiles[cx][cy].colour = opp_colour

                        # Check if this breaks our virtual connection
                        # If so, we must play in the other carrier
                        remaining_carriers = vc.carriers - {(cx, cy)}
                        for rx, ry in remaining_carriers:
                            if board.tiles[rx][ry].colour is None:
                                must_play.add((rx, ry))

        return list(must_play)