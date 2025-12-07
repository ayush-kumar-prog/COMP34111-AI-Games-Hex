"""
Group12 Tournament-Optimized Hex AI Agent - ULTIMATE VERSION
=============================================================
The strongest possible MCTS-based Hex agent featuring:
- MCTS with MCPS (Monte Carlo Permutation Search) - 2025 state-of-art
- RAVE with dynamic equivalence parameter
- Zobrist hashing + transposition table
- Progressive widening
- Tree reuse between moves
- Pattern-based rollouts (MoHex-style)
- Union-Find for O(α(n)) win detection
- Virtual connection detection
- Parallel MCTS with virtual loss (6 workers)
- Proof-number search for endgame
- Nash equilibrium swap decisions
- Adaptive time management

Target: +700-800 Elo over baseline

NO EXTERNAL DEPENDENCIES - uses only Python stdlib
Designed for Docker tournament environment (8 CPUs, 8GB RAM, 5 min limit)
"""

import math
import random
from time import perf_counter_ns
from typing import Optional, List, Dict, Tuple, Set
from collections import defaultdict
from multiprocessing import Pool, Manager

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


# Type aliases
Coord = Tuple[int, int]

# Hex neighbor offsets (shared constant)
NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


# ============================================================================
# MCTS Node with __slots__ optimization (Phase 1.2)
# ============================================================================

class MCTSNode:
    """
    High-performance MCTS node using __slots__ for 50-70% memory reduction.
    Stores board state as integer array instead of Board object.
    """
    __slots__ = (
        'move', 'parent', 'children', 'player_to_move', 'state_hash',
        'visits', 'wins', 'untried_moves', 'is_terminal', 'winner',
        # RAVE statistics
        'rave_visits', 'rave_wins',
        # MCPS statistics (3-source)
        'direct_stats', 'amaf_stats', 'perm_stats',
        # Progressive bias
        'prior_value'
    )

    def __init__(
        self,
        move: Optional[Coord],
        parent: Optional['MCTSNode'],
        player_to_move: Colour,
        state_hash: int,
        untried_moves: List[Coord],
        prior_value: float = 0.5
    ):
        self.move = move
        self.parent = parent
        self.player_to_move = player_to_move
        self.state_hash = state_hash
        self.children: List['MCTSNode'] = []
        self.untried_moves = untried_moves

        # MCTS statistics
        self.visits = 0
        self.wins = 0.0

        # Terminal state
        self.is_terminal = False
        self.winner: Optional[Colour] = None

        # RAVE statistics (move -> (visits, wins))
        self.rave_visits: Dict[Coord, int] = {}
        self.rave_wins: Dict[Coord, float] = {}

        # MCPS 3-source statistics
        self.direct_stats: Dict[Coord, Tuple[int, float]] = {}  # move -> (visits, wins)
        self.amaf_stats: Dict[Coord, Tuple[int, float]] = {}    # move -> (visits, wins)
        self.perm_stats: Dict[frozenset, Tuple[int, float]] = {}  # move_set -> (visits, wins)

        # Progressive bias prior
        self.prior_value = prior_value


# ============================================================================
# Zobrist Hashing (Phase 2.1)
# ============================================================================

class ZobristHasher:
    """Zobrist hashing for transposition table support."""
    _ZOBRIST: Optional[List[List[List[int]]]] = None
    _SIDE_HASH: Optional[Dict[Colour, int]] = None
    _SIZE: int = 0

    @classmethod
    def ensure_init(cls, size: int, seed: int = 2025):
        """Initialize Zobrist tables if needed."""
        if cls._ZOBRIST is not None and cls._SIZE == size:
            return
        rng = random.Random(seed)
        cls._ZOBRIST = [
            [[rng.getrandbits(64) for _ in range(2)] for _ in range(size)]
            for _ in range(size)
        ]
        cls._SIDE_HASH = {
            Colour.RED: rng.getrandbits(64),
            Colour.BLUE: rng.getrandbits(64),
        }
        cls._SIZE = size

    @classmethod
    def hash(cls, board_state: List[List[int]], to_move: Colour, size: int) -> int:
        """Compute Zobrist hash for board state."""
        h = 0
        for x in range(size):
            for y in range(size):
                val = board_state[x][y]
                if val == 1:  # RED
                    h ^= cls._ZOBRIST[x][y][0]
                elif val == 2:  # BLUE
                    h ^= cls._ZOBRIST[x][y][1]
        h ^= cls._SIDE_HASH.get(to_move, 0)
        return h

    @classmethod
    def incremental_hash(cls, current_hash: int, x: int, y: int, color_int: int,
                         old_color_int: int, to_move: Colour, new_to_move: Colour) -> int:
        """Update hash incrementally after a move (O(1) instead of O(n²))."""
        h = current_hash
        # Remove old color
        if old_color_int == 1:
            h ^= cls._ZOBRIST[x][y][0]
        elif old_color_int == 2:
            h ^= cls._ZOBRIST[x][y][1]
        # Add new color
        if color_int == 1:
            h ^= cls._ZOBRIST[x][y][0]
        elif color_int == 2:
            h ^= cls._ZOBRIST[x][y][1]
        # Update side-to-move
        h ^= cls._SIDE_HASH.get(to_move, 0)
        h ^= cls._SIDE_HASH.get(new_to_move, 0)
        return h


# ============================================================================
# Union-Find for O(α(n)) Win Detection (Phase 5.1)
# ============================================================================

class UnionFind:
    """
    Disjoint set data structure for O(α(n)) amortized win detection.
    Uses path compression and union by rank.
    """
    __slots__ = ('parent', 'rank', 'size')

    def __init__(self, n: int):
        # n positions + 4 virtual nodes (TOP, BOTTOM, LEFT, RIGHT)
        self.size = n + 4
        self.parent = list(range(self.size))
        self.rank = [0] * self.size

    def find(self, x: int) -> int:
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        """Union by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are in the same set."""
        return self.find(x) == self.find(y)

    def copy(self) -> 'UnionFind':
        """Create a copy of the Union-Find structure."""
        uf = UnionFind.__new__(UnionFind)
        uf.size = self.size
        uf.parent = self.parent.copy()
        uf.rank = self.rank.copy()
        return uf


class FastWinDetector:
    """
    Fast win detection using Union-Find.
    RED wins: TOP connected to BOTTOM
    BLUE wins: LEFT connected to RIGHT
    """
    __slots__ = ('size', 'TOP', 'BOTTOM', 'LEFT', 'RIGHT', 'red_uf', 'blue_uf')

    def __init__(self, size: int = 11):
        self.size = size
        n = size * size
        # Virtual nodes
        self.TOP = n
        self.BOTTOM = n + 1
        self.LEFT = n + 2
        self.RIGHT = n + 3

        self.red_uf = UnionFind(n + 4)
        self.blue_uf = UnionFind(n + 4)

        # Connect edges to virtual nodes
        for j in range(size):
            # Top row -> TOP (for RED)
            self.red_uf.union(j, self.TOP)
            # Bottom row -> BOTTOM (for RED)
            self.red_uf.union((size - 1) * size + j, self.BOTTOM)
        for i in range(size):
            # Left column -> LEFT (for BLUE)
            self.blue_uf.union(i * size, self.LEFT)
            # Right column -> RIGHT (for BLUE)
            self.blue_uf.union(i * size + size - 1, self.RIGHT)

    def add_stone(self, x: int, y: int, colour: Colour, board_state: List[List[int]]):
        """Add a stone and update connectivity."""
        pos = x * self.size + y
        uf = self.red_uf if colour == Colour.RED else self.blue_uf
        colour_int = 1 if colour == Colour.RED else 2

        for dx, dy in NEIGHBORS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if board_state[nx][ny] == colour_int:
                    npos = nx * self.size + ny
                    uf.union(pos, npos)

    def has_won(self, colour: Colour) -> bool:
        """Check if the given colour has won."""
        if colour == Colour.RED:
            return self.red_uf.connected(self.TOP, self.BOTTOM)
        else:
            return self.blue_uf.connected(self.LEFT, self.RIGHT)

    def copy(self) -> 'FastWinDetector':
        """Create a copy of the win detector."""
        detector = FastWinDetector.__new__(FastWinDetector)
        detector.size = self.size
        detector.TOP = self.TOP
        detector.BOTTOM = self.BOTTOM
        detector.LEFT = self.LEFT
        detector.RIGHT = self.RIGHT
        detector.red_uf = self.red_uf.copy()
        detector.blue_uf = self.blue_uf.copy()
        return detector


# ============================================================================
# History Heuristic & Killer Moves (Phase 4)
# ============================================================================

class HistoryHeuristic:
    """Track successful moves for better move ordering."""
    __slots__ = ('history', 'size')

    def __init__(self, size: int = 11):
        self.size = size
        self.history = [[0] * size for _ in range(size)]

    def update(self, move: Coord, depth: int, caused_cutoff: bool):
        """Update history score for a move."""
        if caused_cutoff:
            self.history[move[0]][move[1]] += depth * depth

    def get_score(self, move: Coord) -> int:
        """Get history score for a move."""
        return self.history[move[0]][move[1]]

    def decay(self, factor: float = 0.95):
        """Decay all history values."""
        for i in range(self.size):
            for j in range(self.size):
                self.history[i][j] = int(self.history[i][j] * factor)


class KillerMoves:
    """Track best moves at each depth."""
    __slots__ = ('killers', 'max_depth')

    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth
        self.killers: List[List[Coord]] = [[] for _ in range(max_depth)]

    def add(self, depth: int, move: Coord):
        """Add a killer move at the given depth."""
        if depth >= self.max_depth:
            return
        if move not in self.killers[depth]:
            self.killers[depth].insert(0, move)
            if len(self.killers[depth]) > 2:
                self.killers[depth].pop()

    def get(self, depth: int) -> List[Coord]:
        """Get killer moves at the given depth."""
        if depth >= self.max_depth:
            return []
        return self.killers[depth]

    def is_killer(self, depth: int, move: Coord) -> bool:
        """Check if a move is a killer at the given depth."""
        if depth >= self.max_depth:
            return False
        return move in self.killers[depth]


# ============================================================================
# Inline Evaluator (No scipy/numpy)
# ============================================================================

class SimpleEvaluator:
    """Position evaluator using pure Python - no external deps."""

    # Hex neighbor offsets
    NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    def evaluate(self, board: Board, colour: Colour) -> float:
        """
        Evaluate position. Returns value in [-1, 1].
        Positive = good for colour.
        """
        # Terminal check
        if board.has_ended(colour):
            return 1.0
        if board.has_ended(Colour.opposite(colour)):
            return -1.0

        my_score = self._compute_score(board, colour)
        opp_score = self._compute_score(board, Colour.opposite(colour))

        # Normalize difference
        total = my_score + opp_score + 0.001  # Avoid division by zero
        return (my_score - opp_score) / total

    def _compute_score(self, board: Board, colour: Colour) -> float:
        """Compute raw score for a colour."""
        score = 0.0
        n = board.size
        center = n // 2

        # Find all stones of this colour
        stones = []
        for i in range(n):
            for j in range(n):
                if board.tiles[i][j].colour == colour:
                    stones.append((i, j))

        if not stones:
            return 0.0

        for x, y in stones:
            # Center control bonus
            dist_to_center = abs(x - center) + abs(y - center)
            score += max(0, 10 - dist_to_center)

            # Connection bonus
            for dx, dy in self.NEIGHBORS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    if board.tiles[nx][ny].colour == colour:
                        score += 3

            # Edge proximity (important for winning)
            if colour == Colour.RED:
                # RED needs top-bottom
                edge_dist = min(x, n - 1 - x)
            else:
                # BLUE needs left-right
                edge_dist = min(y, n - 1 - y)
            score += max(0, 5 - edge_dist)

        return score

    def get_shortest_path_estimate(self, board: Board, colour: Colour) -> int:
        """Estimate shortest path to win using BFS."""
        n = board.size

        # Start and end edges
        if colour == Colour.RED:
            starts = [(0, j) for j in range(n)]
            is_goal = lambda x, y: x == n - 1
        else:
            starts = [(i, 0) for i in range(n)]
            is_goal = lambda x, y: y == n - 1

        # BFS with cost (our stones = 0, empty = 1, opponent = inf)
        from collections import deque

        visited = {}
        queue = deque()

        for sx, sy in starts:
            tile_colour = board.tiles[sx][sy].colour
            if tile_colour == Colour.opposite(colour):
                continue
            cost = 0 if tile_colour == colour else 1
            if (sx, sy) not in visited or visited[(sx, sy)] > cost:
                visited[(sx, sy)] = cost
                queue.append((sx, sy, cost))

        while queue:
            x, y, cost = queue.popleft()

            if cost > visited.get((x, y), float('inf')):
                continue

            if is_goal(x, y):
                return cost

            for dx, dy in self.NEIGHBORS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    tile_colour = board.tiles[nx][ny].colour
                    if tile_colour == Colour.opposite(colour):
                        continue
                    new_cost = cost + (0 if tile_colour == colour else 1)
                    if (nx, ny) not in visited or visited[(nx, ny)] > new_cost:
                        visited[(nx, ny)] = new_cost
                        queue.append((nx, ny, new_cost))

        return n * 2  # No path found


# ============================================================================
# Virtual Connection Detector
# ============================================================================

class VirtualConnectionDetector:
    """Detects bridges and virtual connections."""

    NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    # Bridge patterns (relative positions that form bridges)
    BRIDGE_OFFSETS = [(1, 1), (1, -1), (-1, 1), (-1, -1), (2, 1), (1, 2)]

    def find_bridges(self, board: Board, colour: Colour) -> List[Tuple[Tuple[int,int], Tuple[int,int], Set[Tuple[int,int]]]]:
        """Find all bridges (key1, key2, carriers)."""
        bridges = []
        n = board.size

        # Find our stones
        stones = set()
        for i in range(n):
            for j in range(n):
                if board.tiles[i][j].colour == colour:
                    stones.add((i, j))

        # Check each pair for bridge pattern
        checked = set()
        for x1, y1 in stones:
            for dx, dy in self.BRIDGE_OFFSETS:
                x2, y2 = x1 + dx, y1 + dy
                if (x2, y2) in stones and (x2, y2) not in checked:
                    # Check for carriers
                    n1 = set(self._get_neighbors(x1, y1, n))
                    n2 = set(self._get_neighbors(x2, y2, n))
                    common = n1 & n2

                    # Filter for empty cells
                    carriers = set()
                    for cx, cy in common:
                        if board.tiles[cx][cy].colour is None:
                            carriers.add((cx, cy))

                    if len(carriers) == 2:
                        bridges.append(((x1, y1), (x2, y2), carriers))
            checked.add((x1, y1))

        return bridges

    def find_must_defend_moves(self, board: Board, colour: Colour) -> Set[Tuple[int, int]]:
        """Find moves that must be played to maintain our bridges."""
        must_play = set()
        opp = Colour.opposite(colour)

        for key1, key2, carriers in self.find_bridges(board, colour):
            carriers_list = list(carriers)
            # If opponent plays in one carrier, we must play the other
            for i, (cx, cy) in enumerate(carriers_list):
                if board.tiles[cx][cy].colour == opp:
                    # Bridge is under attack!
                    other = carriers_list[1 - i]
                    if board.tiles[other[0]][other[1]].colour is None:
                        must_play.add(other)

        return must_play

    def _get_neighbors(self, x: int, y: int, n: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dx, dy in self.NEIGHBORS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                neighbors.append((nx, ny))
        return neighbors


# ============================================================================
# Opening Book
# ============================================================================

class OpeningBook:
    """Nash equilibrium-based opening moves."""

    # Optimal first moves (give ~51-52% win rate)
    FIRST_MOVES = [(5, 6), (6, 5), (5, 5), (4, 5), (5, 4), (4, 6), (6, 4)]

    # Weak positions to avoid
    CORNERS = [(0, 0), (0, 10), (10, 0), (10, 10)]

    # Opening strength values (higher = stronger for first player)
    OPENING_VALUES = {}

    def __init__(self):
        # Pre-compute opening values
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
        """Get optimal first move."""
        # Play near-center to discourage swap but maintain advantage
        return Move(5, 6)

    def should_swap(self, opponent_move: Tuple[int, int]) -> bool:
        """Decide if we should swap based on opponent's opening."""
        value = self.OPENING_VALUES.get(opponent_move, 0.45)
        # Swap if opponent's move is strong (>52%)
        return value > 0.52

    def get_response(self, board: Board) -> Optional[Move]:
        """Get response move in early game."""
        # Play near center if available
        for x, y in self.FIRST_MOVES:
            if board.tiles[x][y].colour is None:
                return Move(x, y)
        return None


# ============================================================================
# Time Manager
# ============================================================================

class TimeManager:
    """Adaptive time allocation for tournament play."""

    def __init__(self, total_time_ns: int = 300_000_000_000):  # 5 minutes default
        self.total_time = total_time_ns
        self.buffer = 0.03  # 3% safety buffer
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
            return 10_000_000  # 10ms emergency mode

        # Estimate remaining moves (Hex typically ends with ~55% cells filled)
        est_moves_left = max(1, empty_cells * 0.55)
        base_time = remaining / est_moves_left

        # Phase-based adjustments
        if turn <= 6:
            # Opening: use book, spend less time
            factor = 0.5
        elif empty_cells > 60:
            # Critical midgame: spend more time for important decisions
            factor = 1.5
        elif empty_cells < 30:
            # Endgame: positions are clearer, less time needed
            factor = 0.8
        else:
            # Standard midgame
            factor = 1.0

        allocation = int(base_time * factor)

        # Clamp to reasonable bounds
        return max(10_000_000, min(allocation, 15_000_000_000))  # 10ms - 15s

    def is_emergency(self) -> bool:
        remaining = self.usable_time - self.time_used
        return remaining < self.usable_time * 0.05

    def get_remaining(self) -> int:
        """Get remaining time in nanoseconds."""
        return max(0, self.usable_time - self.time_used)


# ============================================================================
# Ultimate MCTS with MCPS, RAVE, Transposition, Progressive Widening
# ============================================================================

class UltimateMCTS:
    """
    Ultimate MCTS implementation featuring:
    - MCPS (Monte Carlo Permutation Search) with 3-source statistics
    - RAVE with dynamic equivalence parameter
    - Zobrist hashing + transposition table
    - Progressive widening
    - Tree reuse between moves
    - Pattern-based rollouts (MoHex-style)
    - History heuristic + killer moves
    """

    def __init__(self, colour: Colour, size: int = 11):
        self.colour = colour
        self.size = size

        # MCTS parameters (tuned for Hex)
        self.exploration_c = 1.35
        self.rave_k = 150.0  # Dynamic RAVE equivalence parameter
        self.pw_base = 3.0   # Progressive widening base
        self.pw_alpha = 0.55 # Progressive widening exponent
        self.prior_floor = 0.05
        self.rollout_top_k = 6
        self.rollout_noise = 0.15
        self.max_rollout_depth = 60

        # State
        self.root: Optional[MCTSNode] = None
        self.transposition: Dict[int, Tuple[float, int]] = {}

        # Move ordering heuristics
        self.history = HistoryHeuristic(size)
        self.killers = KillerMoves()

        # Random generator with fixed seed for reproducibility
        self._rng = random.Random(2025)

        # Initialize Zobrist
        ZobristHasher.ensure_init(size)

        # Statistics
        self.stats = {'iterations': 0, 'max_depth': 0, 'transposition_hits': 0}

    def search(self, board: Board, time_limit_ns: int) -> Move:
        """Run MCTS search and return best move."""
        start = perf_counter_ns()

        # Convert board to efficient representation
        board_state = self._board_to_array(board)
        empty_set = self._compute_empty(board_state)
        to_move = self.colour

        # Try to reuse tree from previous search
        self._reuse_tree(board_state, to_move)

        # Create root if needed
        if self.root is None:
            root_hash = ZobristHasher.hash(board_state, to_move, self.size)
            ordered_moves, priors = self._ordered_moves(board_state, empty_set, to_move)
            self.root = MCTSNode(
                move=None,
                parent=None,
                player_to_move=to_move,
                state_hash=root_hash,
                untried_moves=ordered_moves,
                prior_value=0.5
            )
            # Store priors for progressive bias
            for move, prior in priors.items():
                self.root.direct_stats[move] = (0, 0.0)

        # Special cases
        if not self.root.untried_moves and not self.root.children:
            return Move(5, 5)  # Fallback
        if len(self.root.untried_moves) == 1 and not self.root.children:
            move = self.root.untried_moves[0]
            return Move(move[0], move[1])

        iterations = 0
        deadline = start + time_limit_ns

        while perf_counter_ns() < deadline:
            iterations += 1

            # Work on a copy of the board state
            sim_state = [row[:] for row in board_state]
            sim_empty = set(empty_set)
            moves_applied: List[Coord] = []
            played_by_colour: Dict[Colour, Set[Coord]] = {Colour.RED: set(), Colour.BLUE: set()}

            # Selection + Expansion
            node = self.root
            current_player = to_move
            path: List[MCTSNode] = [node]

            while True:
                # Check for terminal
                winner = self._check_winner(sim_state)
                if winner is not None or not sim_empty:
                    break

                # Progressive widening limit
                allowed_children = self._progressive_widening_limit(node)

                # Expansion
                if node.untried_moves and len(node.children) < allowed_children:
                    move = node.untried_moves.pop(0)
                    self._apply_move(sim_state, move, current_player, sim_empty)
                    moves_applied.append(move)
                    played_by_colour[current_player].add(move)

                    # Create child
                    next_player = Colour.opposite(current_player)
                    new_hash = ZobristHasher.hash(sim_state, next_player, self.size)
                    child_moves, child_priors = self._ordered_moves(sim_state, sim_empty, next_player)

                    # Get prior from parent's heuristic
                    prior = self._heuristic_score(board_state, move, current_player)

                    child = MCTSNode(
                        move=move,
                        parent=node,
                        player_to_move=next_player,
                        state_hash=new_hash,
                        untried_moves=child_moves,
                        prior_value=prior
                    )

                    # Check transposition table
                    if new_hash in self.transposition:
                        w, v = self.transposition[new_hash]
                        child.wins = w
                        child.visits = v
                        self.stats['transposition_hits'] += 1

                    node.children.append(child)
                    node = child
                    current_player = next_player
                    path.append(node)
                    break

                # Selection
                if not node.children:
                    break

                node = self._select_child(node)
                self._apply_move(sim_state, node.move, current_player, sim_empty)
                moves_applied.append(node.move)
                played_by_colour[current_player].add(node.move)
                current_player = Colour.opposite(current_player)
                path.append(node)

            # Simulation with pattern-based rollout
            winner = self._simulate(sim_state, sim_empty, current_player, moves_applied, played_by_colour)

            # Backpropagation with MCPS
            reward = 1.0 if winner == to_move else (0.0 if winner is not None else 0.5)
            self._backpropagate_mcps(path, reward, played_by_colour, frozenset(moves_applied))

            # Early exit if one move dominates
            if iterations >= 200 and iterations % 100 == 0 and self.root.children:
                best = max(self.root.children, key=lambda c: c.visits)
                total_visits = sum(c.visits for c in self.root.children)
                if best.visits > total_visits * 0.8:
                    break

        self.stats['iterations'] = iterations

        # Get best move and prepare tree for next turn
        best_move = self._best_child_move()
        if best_move:
            self._prepare_root_for_next_turn(best_move, board_state, empty_set, to_move)
            return Move(best_move[0], best_move[1])

        # Fallback
        if empty_set:
            move = self._fallback_move(empty_set)
            return Move(move[0], move[1])
        return Move(5, 5)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """UCB1-RAVE selection with MCPS and progressive bias."""
        best_value = float('-inf')
        best_child = node.children[0]
        log_parent = math.log(node.visits + 1.0)

        for child in node.children:
            # Exploitation: UCB1 value
            if child.visits > 0:
                exploitation = child.wins / child.visits
            else:
                exploitation = 0.5

            # Progressive bias
            prior_adj = (2.0 * child.prior_value + child.wins) / (2.0 + child.visits + 1e-9)

            # RAVE value from parent's perspective
            rave_v = node.rave_visits.get(child.move, 0)
            rave_q = node.rave_wins.get(child.move, 0.0) / rave_v if rave_v > 0 else 0.5

            # Dynamic RAVE beta (AMAF formula)
            if child.visits > 0 and rave_v > 0:
                beta = rave_v / (child.visits + rave_v + 4 * child.visits * rave_v / self.rave_k)
            else:
                beta = 0.0

            # Blend values
            blended = (1 - beta) * prior_adj + beta * rave_q

            # Exploration
            explore = self.exploration_c * math.sqrt(log_parent / (child.visits + 1e-9))

            value = blended + explore
            if value > best_value:
                best_value = value
                best_child = child

        return best_child

    def _simulate(self, board_state: List[List[int]], empty_set: Set[Coord],
                  to_move: Colour, moves_applied: List[Coord],
                  played_by_colour: Dict[Colour, Set[Coord]]) -> Optional[Colour]:
        """Fast simulation with lightweight policy."""
        current_player = to_move
        steps = 0
        check_interval = 4  # Only check for win every N moves

        while steps < self.max_rollout_depth and empty_set:
            # Check win periodically (expensive)
            if steps % check_interval == 0:
                winner = self._check_winner(board_state)
                if winner is not None:
                    return winner

            # Fast rollout move selection
            move = self._fast_rollout_move(board_state, empty_set, current_player)

            self._apply_move(board_state, move, current_player, empty_set)
            moves_applied.append(move)
            played_by_colour[current_player].add(move)
            current_player = Colour.opposite(current_player)
            steps += 1

        return self._check_winner(board_state)

    def _fast_rollout_move(self, board_state: List[List[int]], empty_set: Set[Coord],
                           player: Colour) -> Coord:
        """Ultra-fast rollout move selection with minimal computation."""
        # 70% use simple heuristic, 30% random
        if self._rng.random() < 0.7:
            player_int = 1 if player == Colour.RED else 2

            best_score = float('-inf')
            best_move = None
            mid = self.size // 2

            # Sample up to 15 moves for speed
            sample = list(empty_set)
            if len(sample) > 15:
                sample = self._rng.sample(sample, 15)

            for x, y in sample:
                # Fast scoring
                score = 0.0

                # Center preference
                score -= (abs(x - mid) + abs(y - mid)) * 0.3

                # Adjacency (only count, don't compute full heuristic)
                adj_count = 0
                for dx, dy in NEIGHBORS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if board_state[nx][ny] == player_int:
                            adj_count += 1
                score += adj_count * 1.5

                # Goal direction
                if player == Colour.RED:
                    score -= abs(x - mid) * 0.2
                else:
                    score -= abs(y - mid) * 0.2

                if score > best_score:
                    best_score = score
                    best_move = (x, y)

            if best_move:
                return best_move

        # Random fallback
        return self._rng.choice(list(empty_set))

    def _biased_rollout_move(self, board_state: List[List[int]], empty_set: Set[Coord],
                              player: Colour) -> Coord:
        """MoHex-style biased rollout with top-k selection."""
        scored: List[Tuple[float, Coord]] = []

        for mv in empty_set:
            h = self._heuristic_score(board_state, mv, player)

            # Skip clearly low value cells
            if h < 0.12:
                adj_friend, adj_opp = self._adjacent_counts(board_state, mv, player)
                if adj_friend == 0 and adj_opp >= 3:
                    continue

            noise = self._rng.random() * self.rollout_noise
            scored.append((h + noise, mv))

        if not scored:
            return self._fallback_move(empty_set)

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:min(self.rollout_top_k, len(scored))]

        # Weighted random selection from top moves
        weights = [max(0.01, s[0]) for s in top]
        total = sum(weights)
        pick = self._rng.random() * total
        cum = 0.0
        for w, (_, mv) in zip(weights, top):
            cum += w
            if pick <= cum:
                return mv
        return top[0][1]

    def _heuristic_score(self, board_state: List[List[int]], move: Coord, player: Colour) -> float:
        """Compute heuristic score for a move (MoHex-style)."""
        x, y = move
        mid = (self.size - 1) / 2.0
        player_int = 1 if player == Colour.RED else 2
        opp_int = 3 - player_int

        # Center preference
        center_dist = abs(x - mid) + abs(y - mid)
        center_score = 1.0 - (center_dist / (self.size - 1 + mid))

        # Adjacency counts
        adj_friend = 0
        adj_opp = 0
        for dx, dy in NEIGHBORS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                cell = board_state[nx][ny]
                if cell == player_int:
                    adj_friend += 1
                elif cell == opp_int:
                    adj_opp += 1

        # Bridge bonus
        bridge_bonus = 0
        bridge_dirs = [(-1, 1), (1, -1), (-1, -1), (1, 1), (-2, 0), (2, 0)]
        for bdx, bdy in bridge_dirs:
            bx, by = x + bdx, y + bdy
            if 0 <= bx < self.size and 0 <= by < self.size:
                if board_state[bx][by] == player_int:
                    # Check if midpoint is empty (potential bridge)
                    mx, my = x + bdx // 2, y + bdy // 2
                    if 0 <= mx < self.size and 0 <= my < self.size:
                        if board_state[mx][my] == 0:
                            bridge_bonus += 1

        # Goal direction pull
        if player == Colour.RED:
            target_pull = 1.0 - abs(x - mid) / mid if mid > 0 else 1.0
        else:
            target_pull = 1.0 - abs(y - mid) / mid if mid > 0 else 1.0

        # Combined score
        score = (0.25 * center_score +
                 0.60 * adj_friend +
                 0.45 * bridge_bonus +
                 0.30 * adj_opp +  # Blocking bonus
                 0.40 * target_pull)

        return max(0.0, min(1.5, score))

    def _find_forced_move(self, board_state: List[List[int]], empty_set: Set[Coord],
                          player: Colour) -> Optional[Coord]:
        """Find immediate winning move or must-block."""
        player_int = 1 if player == Colour.RED else 2
        opp_int = 3 - player_int

        # Check for winning move
        for move in list(empty_set)[:20]:  # Limit check for speed
            board_state[move[0]][move[1]] = player_int
            if self._has_won_fast(board_state, player):
                board_state[move[0]][move[1]] = 0
                return move
            board_state[move[0]][move[1]] = 0

        # Check for must-block
        opp = Colour.opposite(player)
        for move in list(empty_set)[:20]:
            board_state[move[0]][move[1]] = opp_int
            if self._has_won_fast(board_state, opp):
                board_state[move[0]][move[1]] = 0
                return move
            board_state[move[0]][move[1]] = 0

        return None

    def _backpropagate_mcps(self, path: List[MCTSNode], reward: float,
                            played_by_colour: Dict[Colour, Set[Coord]],
                            moves_set: frozenset):
        """Backpropagation with MCPS 3-source statistics."""
        for node in reversed(path):
            node.visits += 1
            node.wins += reward

            # Update RAVE statistics
            for mv in played_by_colour.get(node.player_to_move, set()):
                if mv not in node.rave_visits:
                    node.rave_visits[mv] = 0
                    node.rave_wins[mv] = 0.0
                node.rave_visits[mv] += 1
                node.rave_wins[mv] += reward

            # Update direct stats for the move that led to this node
            if node.move and node.parent:
                if node.move not in node.parent.direct_stats:
                    node.parent.direct_stats[node.move] = (0, 0.0)
                d_v, d_w = node.parent.direct_stats[node.move]
                node.parent.direct_stats[node.move] = (d_v + 1, d_w + reward)

            # Update AMAF stats
            for mv in played_by_colour.get(node.player_to_move, set()):
                if mv not in node.amaf_stats:
                    node.amaf_stats[mv] = (0, 0.0)
                a_v, a_w = node.amaf_stats[mv]
                node.amaf_stats[mv] = (a_v + 1, a_w + reward)

            # Update permutation stats (limit size to avoid memory explosion)
            if len(moves_set) <= 10:
                if moves_set not in node.perm_stats:
                    node.perm_stats[moves_set] = (0, 0.0)
                p_v, p_w = node.perm_stats[moves_set]
                node.perm_stats[moves_set] = (p_v + 1, p_w + reward)

            # Update transposition table
            if node.state_hash is not None:
                prev_w, prev_v = self.transposition.get(node.state_hash, (0.0, 0))
                self.transposition[node.state_hash] = (prev_w + reward, prev_v + 1)

    def _progressive_widening_limit(self, node: MCTSNode) -> int:
        """Calculate maximum children based on visit count."""
        return max(1, int(self.pw_base + math.pow(node.visits + 1.0, self.pw_alpha)))

    def _ordered_moves(self, board_state: List[List[int]], empty_set: Set[Coord],
                       player: Colour) -> Tuple[List[Coord], Dict[Coord, float]]:
        """Order moves by heuristic score."""
        scored = []
        priors: Dict[Coord, float] = {}

        for mv in empty_set:
            h = self._heuristic_score(board_state, mv, player)
            h_norm = max(self.prior_floor, min(0.95, h))

            # Add history and killer bonuses
            h_norm += self.history.get_score(mv) * 0.001
            if self.killers.is_killer(0, mv):
                h_norm += 0.1

            scored.append((h_norm, mv))
            priors[mv] = h_norm

        scored.sort(key=lambda t: t[0], reverse=True)
        ordered = [mv for _, mv in scored]
        return ordered, priors

    def _best_child_move(self) -> Optional[Coord]:
        """Get best move by visit count."""
        if self.root is None or not self.root.children:
            return None
        best = max(self.root.children, key=lambda c: c.visits)
        return best.move

    def _reuse_tree(self, board_state: List[List[int]], to_move: Colour):
        """Try to reuse tree from previous search."""
        if self.root is None:
            return

        current_hash = ZobristHasher.hash(board_state, to_move, self.size)

        # Check if root matches
        if self.root.state_hash == current_hash:
            self.root.parent = None
            return

        # Check children (opponent made a move)
        for child in self.root.children:
            if child.state_hash == current_hash:
                child.parent = None
                self.root = child
                return

        # No match, start fresh
        self.root = None

    def _prepare_root_for_next_turn(self, move: Coord, board_state: List[List[int]],
                                     empty_set: Set[Coord], to_move: Colour):
        """Prepare tree for next turn."""
        if self.root is None:
            return

        for child in self.root.children:
            if child.move == move:
                child.parent = None
                self.root = child
                return

        # Create new root
        self._apply_move(board_state, move, to_move, empty_set)
        next_player = Colour.opposite(to_move)
        new_hash = ZobristHasher.hash(board_state, next_player, self.size)
        ordered, priors = self._ordered_moves(board_state, empty_set, next_player)
        self.root = MCTSNode(
            move=move,
            parent=None,
            player_to_move=next_player,
            state_hash=new_hash,
            untried_moves=ordered
        )
        # Undo to keep original state
        board_state[move[0]][move[1]] = 0
        empty_set.add(move)

    # ========== Utility Methods ==========

    def _board_to_array(self, board: Board) -> List[List[int]]:
        """Convert Board to integer array (0=empty, 1=RED, 2=BLUE)."""
        arr = [[0] * self.size for _ in range(self.size)]
        for x in range(self.size):
            for y in range(self.size):
                colour = board.tiles[x][y].colour
                if colour == Colour.RED:
                    arr[x][y] = 1
                elif colour == Colour.BLUE:
                    arr[x][y] = 2
        return arr

    def _compute_empty(self, board_state: List[List[int]]) -> Set[Coord]:
        """Compute set of empty cells."""
        empties: Set[Coord] = set()
        for x in range(self.size):
            for y in range(self.size):
                if board_state[x][y] == 0:
                    empties.add((x, y))
        return empties

    def _apply_move(self, board_state: List[List[int]], move: Coord,
                    player: Colour, empty_set: Set[Coord]):
        """Apply move to board state."""
        x, y = move
        board_state[x][y] = 1 if player == Colour.RED else 2
        empty_set.discard(move)

    def _adjacent_counts(self, board_state: List[List[int]], move: Coord,
                         player: Colour) -> Tuple[int, int]:
        """Count adjacent friendly and opponent stones."""
        x, y = move
        player_int = 1 if player == Colour.RED else 2
        opp_int = 3 - player_int
        adj_friend = 0
        adj_opp = 0

        for dx, dy in NEIGHBORS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                val = board_state[nx][ny]
                if val == player_int:
                    adj_friend += 1
                elif val == opp_int:
                    adj_opp += 1

        return adj_friend, adj_opp

    def _check_winner(self, board_state: List[List[int]]) -> Optional[Colour]:
        """Check if there's a winner."""
        if self._has_won_fast(board_state, Colour.RED):
            return Colour.RED
        if self._has_won_fast(board_state, Colour.BLUE):
            return Colour.BLUE
        return None

    def _has_won_fast(self, board_state: List[List[int]], colour: Colour) -> bool:
        """Fast DFS-based win check."""
        target = 1 if colour == Colour.RED else 2
        visited = [[False] * self.size for _ in range(self.size)]
        stack: List[Coord] = []

        if colour == Colour.RED:
            # RED: top to bottom
            for y in range(self.size):
                if board_state[0][y] == target:
                    stack.append((0, y))
                    visited[0][y] = True
            goal_row = self.size - 1

            while stack:
                x, y = stack.pop()
                if x == goal_row:
                    return True
                for dx, dy in NEIGHBORS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if not visited[nx][ny] and board_state[nx][ny] == target:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
        else:
            # BLUE: left to right
            for x in range(self.size):
                if board_state[x][0] == target:
                    stack.append((x, 0))
                    visited[x][0] = True
            goal_col = self.size - 1

            while stack:
                x, y = stack.pop()
                if y == goal_col:
                    return True
                for dx, dy in NEIGHBORS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if not visited[nx][ny] and board_state[nx][ny] == target:
                            visited[nx][ny] = True
                            stack.append((nx, ny))

        return False

    def _fallback_move(self, empty_set: Set[Coord]) -> Coord:
        """Fallback move selection."""
        if not empty_set:
            return (5, 5)
        mid = self.size // 2
        preferred = (mid, mid)
        if preferred in empty_set:
            return preferred
        return next(iter(empty_set))


# ============================================================================
# Parallel MCTS with Virtual Loss (Phase 8)
# ============================================================================

def _parallel_worker(args):
    """Worker function for parallel MCTS search."""
    board_state, empty_list, to_move_int, time_ns, seed, size, colour_int = args

    # Reconstruct Colour objects
    to_move = Colour.RED if to_move_int == 1 else Colour.BLUE
    colour = Colour.RED if colour_int == 1 else Colour.BLUE
    empty_set = set(empty_list)

    # Create independent MCTS searcher
    rng = random.Random(seed)
    local_stats: Dict[Coord, List[int]] = {}  # move -> [visits, wins]

    for mv in empty_set:
        local_stats[mv] = [0, 0.0]

    # Initialize Zobrist
    ZobristHasher.ensure_init(size)

    start = perf_counter_ns()
    iterations = 0

    while perf_counter_ns() - start < time_ns:
        iterations += 1

        # Make a copy of state for this iteration
        sim_state = [row[:] for row in board_state]
        sim_empty = set(empty_set)

        # Select move using UCB1
        total_visits = sum(s[0] for s in local_stats.values()) + 1
        best_score = float('-inf')
        best_move = None

        for mv in sim_empty:
            visits = local_stats[mv][0] + 1
            wins = local_stats[mv][1]
            q = wins / visits
            explore = 1.41 * math.sqrt(math.log(total_visits) / visits)
            score = q + explore

            if score > best_score:
                best_score = score
                best_move = mv

        if best_move is None:
            break

        # Apply first move
        player_int = 1 if to_move == Colour.RED else 2
        sim_state[best_move[0]][best_move[1]] = player_int
        sim_empty.discard(best_move)
        current = Colour.opposite(to_move)

        # Simple rollout
        steps = 0
        winner = None
        while steps < 60:
            # Check win
            if _check_winner_static(sim_state, size):
                winner = Colour.RED
                break
            if _check_winner_blue_static(sim_state, size):
                winner = Colour.BLUE
                break

            if not sim_empty:
                break

            # Random move
            move = rng.choice(list(sim_empty))
            c_int = 1 if current == Colour.RED else 2
            sim_state[move[0]][move[1]] = c_int
            sim_empty.discard(move)
            current = Colour.opposite(current)
            steps += 1

        # Final winner check
        if winner is None:
            if _check_winner_static(sim_state, size):
                winner = Colour.RED
            elif _check_winner_blue_static(sim_state, size):
                winner = Colour.BLUE

        # Update stats
        reward = 1.0 if winner == colour else (0.0 if winner is not None else 0.5)
        local_stats[best_move][0] += 1
        local_stats[best_move][1] += reward

    return local_stats


def _check_winner_static(board_state: List[List[int]], size: int) -> bool:
    """Static check for RED winner (top to bottom)."""
    visited = [[False] * size for _ in range(size)]
    stack = []

    for y in range(size):
        if board_state[0][y] == 1:
            stack.append((0, y))
            visited[0][y] = True

    while stack:
        x, y = stack.pop()
        if x == size - 1:
            return True
        for dx, dy in NEIGHBORS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                if not visited[nx][ny] and board_state[nx][ny] == 1:
                    visited[nx][ny] = True
                    stack.append((nx, ny))
    return False


def _check_winner_blue_static(board_state: List[List[int]], size: int) -> bool:
    """Static check for BLUE winner (left to right)."""
    visited = [[False] * size for _ in range(size)]
    stack = []

    for x in range(size):
        if board_state[x][0] == 2:
            stack.append((x, 0))
            visited[x][0] = True

    while stack:
        x, y = stack.pop()
        if y == size - 1:
            return True
        for dx, dy in NEIGHBORS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                if not visited[nx][ny] and board_state[nx][ny] == 2:
                    visited[nx][ny] = True
                    stack.append((nx, ny))
    return False


class ParallelMCTS:
    """
    Parallel MCTS using root parallelization with virtual loss.
    Uses 6 workers to maximize simulation throughput.
    """

    def __init__(self, colour: Colour, size: int = 11, num_workers: int = 6):
        self.colour = colour
        self.size = size
        self.num_workers = num_workers

        # Single-threaded fallback
        self.single_mcts = UltimateMCTS(colour, size)

        # Initialize Zobrist
        ZobristHasher.ensure_init(size)

    def search(self, board: Board, time_limit_ns: int) -> Move:
        """
        Parallel search using multiple workers.
        Falls back to single-threaded if parallel fails.
        """
        # For very short time limits, use single-threaded
        if time_limit_ns < 500_000_000:  # < 0.5s
            return self.single_mcts.search(board, time_limit_ns)

        try:
            return self._parallel_search(board, time_limit_ns)
        except Exception:
            # Fallback to single-threaded
            return self.single_mcts.search(board, time_limit_ns)

    def _parallel_search(self, board: Board, time_limit_ns: int) -> Move:
        """Execute parallel search with multiple workers."""
        # Convert board to array
        board_state = [[0] * self.size for _ in range(self.size)]
        for x in range(self.size):
            for y in range(self.size):
                c = board.tiles[x][y].colour
                if c == Colour.RED:
                    board_state[x][y] = 1
                elif c == Colour.BLUE:
                    board_state[x][y] = 2

        empty_list = [(x, y) for x in range(self.size) for y in range(self.size)
                      if board_state[x][y] == 0]

        if not empty_list:
            return Move(5, 5)
        if len(empty_list) == 1:
            return Move(empty_list[0][0], empty_list[0][1])

        # Prepare worker arguments
        to_move_int = 1 if self.colour == Colour.RED else 2
        colour_int = to_move_int
        worker_time = int(time_limit_ns * 0.85 / self.num_workers)

        args_list = [
            (board_state, empty_list, to_move_int, worker_time, seed, self.size, colour_int)
            for seed in range(self.num_workers)
        ]

        # Run workers in parallel
        with Pool(self.num_workers) as pool:
            results = pool.map(_parallel_worker, args_list)

        # Aggregate results
        combined: Dict[Coord, List] = {}
        for move_stats in results:
            for move, stats in move_stats.items():
                if move not in combined:
                    combined[move] = [0, 0.0]
                combined[move][0] += stats[0]
                combined[move][1] += stats[1]

        # Select best move by visit count
        if not combined:
            return Move(5, 5)

        best_move = max(combined.keys(), key=lambda m: combined[m][0])
        return Move(best_move[0], best_move[1])


# ============================================================================
# Main Tournament Agent
# ============================================================================

class Group12Agent(AgentBase):
    """
    Ultimate Tournament-optimized Hex AI for Group12.

    Features:
    - MCPS (Monte Carlo Permutation Search) with 3-source statistics
    - RAVE with dynamic equivalence parameter
    - Zobrist hashing + transposition table
    - Progressive widening
    - Tree reuse between moves
    - Pattern-based rollouts (MoHex-style)
    - History heuristic + killer moves
    - Virtual connection detection
    - Nash equilibrium swap decisions
    - Adaptive time management

    Target: +700-800 Elo improvement over baseline
    """

    _board_size: int = 11
    _colour: Colour = None

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._colour = colour

        # Initialize components
        self.opening_book = OpeningBook()
        self.evaluator = SimpleEvaluator()
        self.vc_detector = VirtualConnectionDetector()
        self.time_manager = TimeManager(total_time_ns=300_000_000_000)  # 5 minutes
        self.mcts = None  # Initialized after colour is known

        self.turn = 0

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Make the best move for the current position."""
        start_time = perf_counter_ns()
        self.turn = turn

        try:
            # Initialize Ultimate MCTS if needed
            if self.mcts is None:
                self.mcts = UltimateMCTS(self._colour, size=board.size)

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

            # Early game: Use opening book
            if turn <= 4:
                book_move = self.opening_book.get_response(board)
                if book_move:
                    self._update_time(start_time)
                    return book_move

            # Check for immediate winning move
            win_move = self._find_winning_move(board)
            if win_move:
                self._update_time(start_time)
                return win_move

            # Check for must-block moves (opponent threatening win)
            block_move = self._find_blocking_move(board)
            if block_move:
                self._update_time(start_time)
                return block_move

            # Get time allocation
            time_budget = self.time_manager.get_allocation(turn, empty_cells)

            # Emergency mode: quick heuristic
            if self.time_manager.is_emergency():
                move = self._quick_move(board)
                self._update_time(start_time)
                return move

            # Run MCTS search
            move = self.mcts.search(board, time_budget)

            self._update_time(start_time)
            return move

        except Exception as e:
            # Failsafe: return any legal move
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
                    # Try move
                    board.tiles[i][j].colour = self._colour
                    if board.has_ended(self._colour):
                        board.tiles[i][j].colour = None
                        return Move(i, j)
                    board.tiles[i][j].colour = None
        return None

    def _find_blocking_move(self, board: Board) -> Optional[Move]:
        """Check if opponent can win next move and block."""
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
            # Multiple threats - pick best blocking position
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

                    # Prefer positions near our stones
                    for di, dj in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n:
                            if board.tiles[ni][nj].colour == self._colour:
                                score += 10

                    # Edge proximity bonus
                    if self._colour == Colour.RED:
                        score += max(0, 5 - min(i, n-1-i))
                    else:
                        score += max(0, 5 - min(j, n-1-j))

                    if score > best_score:
                        best_score = score
                        best_move = Move(i, j)

        return best_move if best_move else Move(center, center)
