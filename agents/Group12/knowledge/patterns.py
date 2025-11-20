"""
Pattern recognition and template matching for tactical play.
"""

from typing import List, Set, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from src.Board import Board
from src.Colour import Colour


class PatternType(Enum):
    """Types of patterns in Hex."""
    BRIDGE = 1
    EDGE_TEMPLATE = 2
    LADDER = 3
    FORCING_MOVE = 4
    DEAD_CELL = 5


@dataclass
class Pattern:
    """Represents a tactical pattern."""
    type: PatternType
    positions: Set[Tuple[int, int]]  # Positions involved
    key_cells: Set[Tuple[int, int]]  # Critical cells to control
    priority: int  # 1=highest priority
    colour: Colour


class PatternMatcher:
    """
    Matches tactical patterns on the board.
    This provides instant recognition of important tactical features.
    """

    def __init__(self):
        # Pre-compute common patterns
        self.bridge_patterns = self._generate_bridge_patterns()
        self.edge_templates = self._generate_edge_templates()

        # Cache for efficiency
        self.cache = {}

    def find_all_patterns(self, board: Board, colour: Colour) -> List[Pattern]:
        """Find all patterns for the given colour."""
        patterns = []

        # Find bridges
        patterns.extend(self._find_bridges(board, colour))

        # Find edge templates
        patterns.extend(self._find_edge_templates(board, colour))

        # Find forcing moves
        patterns.extend(self._find_forcing_moves(board, colour))

        # Find dead cells
        patterns.extend(self._find_dead_cells(board, colour))

        return patterns

    def _find_bridges(self, board: Board, colour: Colour) -> List[Pattern]:
        """Find all bridge patterns."""
        bridges = []

        # Check all pairs of our stones
        our_stones = []
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour == colour:
                    our_stones.append((i, j))

        # Check each pair for bridge pattern
        for i, stone1 in enumerate(our_stones):
            for stone2 in our_stones[i+1:]:
                if self._is_bridge(board, stone1, stone2):
                    # Find the carrier cells
                    carriers = self._get_bridge_carriers(stone1, stone2, board)
                    if carriers:
                        pattern = Pattern(
                            type=PatternType.BRIDGE,
                            positions={stone1, stone2} | carriers,
                            key_cells=carriers,
                            priority=2,  # High priority
                            colour=colour
                        )
                        bridges.append(pattern)

        return bridges

    def _is_bridge(self, board: Board, pos1: Tuple[int, int],
                   pos2: Tuple[int, int]) -> bool:
        """Check if two positions form a bridge."""
        x1, y1 = pos1
        x2, y2 = pos2

        # Bridge patterns have specific relative positions
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Standard bridge is at (1,1) offset
        return (dx == 1 and dy == 1) or (dx == 2 and dy == 1) or (dx == 1 and dy == 2)

    def _get_bridge_carriers(self, pos1: Tuple[int, int], pos2: Tuple[int, int],
                             board: Board) -> Set[Tuple[int, int]]:
        """Get carrier cells for a bridge."""
        x1, y1 = pos1
        x2, y2 = pos2
        carriers = set()

        # Find common neighbors
        neighbors1 = self._get_neighbors(x1, y1, board.size)
        neighbors2 = self._get_neighbors(x2, y2, board.size)
        common = neighbors1.intersection(neighbors2)

        # Filter for empty cells
        for nx, ny in common:
            if board.tiles[nx][ny].colour is None:
                carriers.add((nx, ny))

        return carriers if len(carriers) == 2 else set()

    def _find_edge_templates(self, board: Board, colour: Colour) -> List[Pattern]:
        """Find edge template patterns."""
        templates = []

        # Check top edge (for RED)
        if colour == Colour.RED:
            for j in range(board.size):
                if board.tiles[0][j].colour == colour:
                    # Look for edge template patterns
                    template = self._check_edge_template(board, 0, j, 'top', colour)
                    if template:
                        templates.append(template)

                if board.tiles[board.size-1][j].colour == colour:
                    template = self._check_edge_template(board, board.size-1, j, 'bottom', colour)
                    if template:
                        templates.append(template)

        # Check side edges (for BLUE)
        elif colour == Colour.BLUE:
            for i in range(board.size):
                if board.tiles[i][0].colour == colour:
                    template = self._check_edge_template(board, i, 0, 'left', colour)
                    if template:
                        templates.append(template)

                if board.tiles[i][board.size-1].colour == colour:
                    template = self._check_edge_template(board, i, board.size-1, 'right', colour)
                    if template:
                        templates.append(template)

        return templates

    def _check_edge_template(self, board: Board, x: int, y: int,
                             edge: str, colour: Colour) -> Optional[Pattern]:
        """Check for edge template at position."""
        # Edge templates are patterns that guarantee connection to edge
        # Simplified version - would need full template library

        key_cells = set()

        if edge == 'top' and x == 0:
            # Check for top edge template
            # Pattern: stone at (0,j) can connect via (1,j-1) and (1,j)
            if y > 0 and board.tiles[1][y-1].colour is None:
                key_cells.add((1, y-1))
            if y < board.size - 1 and board.tiles[1][y].colour is None:
                key_cells.add((1, y))

        elif edge == 'bottom' and x == board.size - 1:
            # Bottom edge template
            if y > 0 and board.tiles[x-1][y-1].colour is None:
                key_cells.add((x-1, y-1))
            if y < board.size - 1 and board.tiles[x-1][y].colour is None:
                key_cells.add((x-1, y))

        # Similar for left/right edges

        if key_cells:
            return Pattern(
                type=PatternType.EDGE_TEMPLATE,
                positions={(x, y)} | key_cells,
                key_cells=key_cells,
                priority=3,  # Medium priority
                colour=colour
            )

        return None

    def _find_forcing_moves(self, board: Board, colour: Colour) -> List[Pattern]:
        """Find forcing move patterns."""
        forcing = []

        # A forcing move is one that threatens to win immediately
        # Simplified version - check for near-win positions

        # Check if we're one move from winning
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    # Simulate our move
                    board.tiles[i][j].colour = colour

                    # Check if this creates a win threat
                    if self._creates_win_threat(board, colour, i, j):
                        pattern = Pattern(
                            type=PatternType.FORCING_MOVE,
                            positions={(i, j)},
                            key_cells={(i, j)},
                            priority=1,  # Highest priority
                            colour=colour
                        )
                        forcing.append(pattern)

                    # Restore board
                    board.tiles[i][j].colour = None

        return forcing

    def _creates_win_threat(self, board: Board, colour: Colour, x: int, y: int) -> bool:
        """Check if move creates immediate win threat."""
        # Simplified check - would need full path analysis
        # Check if this move connects to both edges

        if colour == Colour.RED:
            # Check connection to top and bottom
            has_top = x == 0 or self._has_path_to_edge(board, colour, x, y, 'top')
            has_bottom = x == board.size - 1 or self._has_path_to_edge(board, colour, x, y, 'bottom')
            return has_top and has_bottom
        else:
            # Check connection to left and right
            has_left = y == 0 or self._has_path_to_edge(board, colour, x, y, 'left')
            has_right = y == board.size - 1 or self._has_path_to_edge(board, colour, x, y, 'right')
            return has_left and has_right

    def _has_path_to_edge(self, board: Board, colour: Colour,
                          x: int, y: int, edge: str) -> bool:
        """Check if position has path to specified edge."""
        # Simplified BFS to edge
        visited = set()
        queue = [(x, y)]

        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) in visited:
                continue

            visited.add((cx, cy))

            # Check if reached edge
            if edge == 'top' and cx == 0:
                return True
            elif edge == 'bottom' and cx == board.size - 1:
                return True
            elif edge == 'left' and cy == 0:
                return True
            elif edge == 'right' and cy == board.size - 1:
                return True

            # Add connected neighbors
            for nx, ny in self._get_neighbors(cx, cy, board.size):
                if board.tiles[nx][ny].colour == colour and (nx, ny) not in visited:
                    queue.append((nx, ny))

        return False

    def _find_dead_cells(self, board: Board, colour: Colour) -> List[Pattern]:
        """Find dead cells that can't affect the game outcome."""
        dead = []

        # Dead cells are those that are completely surrounded
        # or in regions that can't connect to winning edges

        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    if self._is_dead_cell(board, i, j, colour):
                        pattern = Pattern(
                            type=PatternType.DEAD_CELL,
                            positions={(i, j)},
                            key_cells=set(),
                            priority=5,  # Lowest priority
                            colour=colour
                        )
                        dead.append(pattern)

        return dead

    def _is_dead_cell(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """Check if cell is dead (can't affect outcome)."""
        # Simplified check - corners are often dead in late game
        if (x, y) in [(0, 0), (0, board.size-1), (board.size-1, 0), (board.size-1, board.size-1)]:
            # Check if corner is isolated
            neighbors_occupied = 0
            for nx, ny in self._get_neighbors(x, y, board.size):
                if board.tiles[nx][ny].colour is not None:
                    neighbors_occupied += 1

            # Dead if all neighbors occupied
            return neighbors_occupied == len(self._get_neighbors(x, y, board.size))

        return False

    def _get_neighbors(self, x: int, y: int, board_size: int) -> Set[Tuple[int, int]]:
        """Get hex neighbors."""
        neighbors = set()
        deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                neighbors.add((nx, ny))

        return neighbors

    def _generate_bridge_patterns(self) -> List[List[Tuple[int, int]]]:
        """Generate all bridge pattern templates."""
        # Bridge patterns relative to (0,0)
        patterns = [
            # Standard bridges
            [[(0, 0), (1, 1)], [(0, 1), (1, 0)]],  # Diagonal bridge
            [[(0, 0), (1, -1)], [(0, -1), (1, 0)]],  # Other diagonal
            # Extended bridges
            [[(0, 0), (2, 1)], [(1, 0), (1, 1)]],
            [[(0, 0), (1, 2)], [(0, 1), (1, 1)]],
        ]
        return patterns

    def _generate_edge_templates(self) -> Dict[str, List[List[Tuple[int, int]]]]:
        """Generate edge template patterns."""
        templates = {
            'top': [],
            'bottom': [],
            'left': [],
            'right': []
        }

        # Add various edge templates
        # These guarantee connection to edge
        # Simplified version - real implementation would have hundreds

        return templates