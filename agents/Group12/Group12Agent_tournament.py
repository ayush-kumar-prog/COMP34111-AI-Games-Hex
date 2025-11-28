"""
Group12 Tournament-Optimized Hex AI Agent
==========================================
This is our best tournament agent combining:
- MCTS with RAVE (Rapid Action Value Estimation)
- Virtual connection detection
- Nash equilibrium swap decisions
- Pattern recognition
- Intelligent time management

NO EXTERNAL DEPENDENCIES - uses only Python stdlib
Designed for Docker tournament environment (8 CPUs, 8GB RAM, 3 min limit)
"""

import math
import random
import copy
from time import perf_counter_ns
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, field

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


# ============================================================================
# MCTS Node
# ============================================================================

@dataclass
class MCTSNode:
    """Node in MCTS tree with RAVE statistics."""
    board: Board
    colour: Colour  # Colour to play next
    move: Optional[Move] = None
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)

    # MCTS statistics
    visits: int = 0
    wins: float = 0.0

    # RAVE statistics
    amaf_visits: Dict[Tuple[int, int], int] = field(default_factory=dict)
    amaf_wins: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Cached data
    untried_moves: List[Move] = field(default_factory=list)
    is_terminal: bool = False
    winner: Optional[Colour] = None


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
    """Intelligent time allocation."""

    def __init__(self, total_time_ns: int = 180_000_000_000):  # 3 minutes
        self.total_time = total_time_ns
        self.buffer = 0.05  # 5% safety buffer
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

        # Opening: quick moves
        if turn <= 5:
            return min(500_000_000, int(remaining * 0.02))  # 0.5s max

        # Estimate remaining moves (game usually 60-80 total moves)
        est_moves_left = max(1, (empty_cells * 0.6) / 2)
        base_time = remaining / est_moves_left

        # Midgame: standard allocation
        if empty_cells > 30:
            return int(min(base_time * 1.2, 5_000_000_000))  # 5s max

        # Endgame: more time for critical moves
        return int(min(base_time * 1.5, 10_000_000_000))  # 10s max

    def is_emergency(self) -> bool:
        remaining = self.usable_time - self.time_used
        return remaining < self.usable_time * 0.05


# ============================================================================
# Enhanced MCTS with RAVE
# ============================================================================

class TournamentMCTS:
    """MCTS with RAVE enhancement, optimized for tournament play."""

    EXPLORATION = math.sqrt(2)
    RAVE_K = 300  # RAVE equivalence parameter
    MAX_ITERATIONS = 500  # Increased from 200
    NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    def __init__(self, colour: Colour, evaluator: SimpleEvaluator, vc_detector: VirtualConnectionDetector):
        self.colour = colour
        self.evaluator = evaluator
        self.vc_detector = vc_detector
        self.stats = {'iterations': 0, 'max_depth': 0}

    def search(self, board: Board, time_limit_ns: int) -> Move:
        """Run MCTS search and return best move."""
        start = perf_counter_ns()
        root = self._create_root(board)

        # Special cases
        if len(root.untried_moves) == 0:
            return Move(5, 5)  # Fallback
        if len(root.untried_moves) == 1:
            return root.untried_moves[0]

        # Check for must-defend moves
        must_defend = self.vc_detector.find_must_defend_moves(board, self.colour)
        if must_defend:
            # High priority - defend our bridges
            for move in root.untried_moves:
                if (move.x, move.y) in must_defend:
                    return move

        iterations = 0
        while (perf_counter_ns() - start < time_limit_ns) and iterations < self.MAX_ITERATIONS:
            iterations += 1

            # Selection
            node = self._select(root)

            # Expansion
            if node.untried_moves and not node.is_terminal:
                node = self._expand(node)

            # Simulation
            result, moves_played = self._simulate(node)

            # Backpropagation
            self._backpropagate(node, result)

            # Update RAVE
            self._update_rave(node, moves_played, result)

            # Early exit if one move dominates
            if iterations >= 100 and iterations % 50 == 0 and root.children:
                best = max(root.children, key=lambda c: c.visits)
                if best.visits > root.visits * 0.75:
                    break

        self.stats['iterations'] = iterations
        return self._best_move(root)

    def _create_root(self, board: Board) -> MCTSNode:
        root = MCTSNode(board=copy.deepcopy(board), colour=self.colour)
        root.untried_moves = self._get_legal_moves(board)
        self._check_terminal(root)
        return root

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal:
            if node.untried_moves:
                return node
            if not node.children:
                return node
            node = self._best_child(node)
        return node

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        best_score = -float('inf')
        best = None

        for child in node.children:
            score = self._ucb1_rave(child, node)
            if score > best_score:
                best_score = score
                best = child

        return best if best else node.children[0]

    def _ucb1_rave(self, child: MCTSNode, parent: MCTSNode) -> float:
        if child.visits == 0:
            return float('inf')

        # UCB1 value
        q = child.wins / child.visits
        if child.colour != self.colour:
            q = 1 - q

        # RAVE value
        move_key = (child.move.x, child.move.y) if child.move else None
        if move_key and move_key in parent.amaf_visits and parent.amaf_visits[move_key] > 0:
            rave_q = parent.amaf_wins[move_key] / parent.amaf_visits[move_key]
            if child.colour != self.colour:
                rave_q = 1 - rave_q

            # Weight RAVE vs UCB1
            beta = math.sqrt(self.RAVE_K / (3 * parent.visits + self.RAVE_K))
            q = (1 - beta) * q + beta * rave_q

        # Exploration bonus
        explore = self.EXPLORATION * math.sqrt(math.log(parent.visits) / child.visits)

        return q + explore

    def _expand(self, node: MCTSNode) -> MCTSNode:
        # Prioritize bridge-related moves
        move = self._select_expansion_move(node)
        node.untried_moves.remove(move)

        # Create child
        child_board = copy.deepcopy(node.board)
        child_board.set_tile_colour(move.x, move.y, node.colour)

        child = MCTSNode(
            board=child_board,
            colour=Colour.opposite(node.colour),
            move=move,
            parent=node
        )
        child.untried_moves = self._get_legal_moves(child_board)
        self._check_terminal(child)

        node.children.append(child)
        return child

    def _select_expansion_move(self, node: MCTSNode) -> Move:
        """Select move for expansion, prioritizing good positions."""
        # Check for must-play moves
        must_play = self.vc_detector.find_must_defend_moves(node.board, node.colour)
        for move in node.untried_moves:
            if (move.x, move.y) in must_play:
                return move

        # Prioritize center and avoid corners
        n = node.board.size
        center = n // 2

        scored = []
        for move in node.untried_moves:
            dist = abs(move.x - center) + abs(move.y - center)
            # Penalize corners heavily
            if (move.x, move.y) in [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]:
                score = -100
            else:
                score = -dist
            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Pick from top 3
        top = [m for _, m in scored[:3]]
        return random.choice(top)

    def _simulate(self, node: MCTSNode) -> Tuple[float, List[Tuple[int, int]]]:
        if node.is_terminal:
            return (1.0 if node.winner == self.colour else 0.0), []

        sim_board = copy.deepcopy(node.board)
        current = node.colour
        moves_played = []
        n = sim_board.size

        for _ in range(60):  # Max simulation depth
            # Check win every 5 moves
            if len(moves_played) % 5 == 0:
                if sim_board.has_ended(Colour.RED) or sim_board.has_ended(Colour.BLUE):
                    break

            # Get legal moves
            legal = []
            for i in range(n):
                for j in range(n):
                    if sim_board.tiles[i][j].colour is None:
                        legal.append((i, j))

            if not legal:
                break

            # Simulation policy: 30% smart, 70% random
            if random.random() < 0.3:
                # Prefer center, avoid corners
                center = n // 2
                scored = []
                for x, y in legal:
                    if (x, y) in [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]:
                        score = -50
                    else:
                        score = -(abs(x - center) + abs(y - center))
                        # Bonus for edge proximity
                        if current == Colour.RED:
                            score += max(0, 5 - min(x, n-1-x))
                        else:
                            score += max(0, 5 - min(y, n-1-y))
                    scored.append((score, x, y))
                scored.sort(key=lambda t: t[0], reverse=True)
                x, y = scored[0][1], scored[0][2]
            else:
                x, y = random.choice(legal)

            sim_board.set_tile_colour(x, y, current)
            moves_played.append((x, y))
            current = Colour.opposite(current)

        # Determine result
        if sim_board.has_ended(self.colour):
            return 1.0, moves_played
        elif sim_board.has_ended(Colour.opposite(self.colour)):
            return 0.0, moves_played
        else:
            # Use evaluator for non-terminal
            score = self.evaluator.evaluate(sim_board, self.colour)
            return (score + 1) / 2, moves_played

    def _backpropagate(self, node: MCTSNode, result: float):
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent

    def _update_rave(self, node: MCTSNode, moves: List[Tuple[int, int]], result: float):
        current = node.parent
        colour = Colour.opposite(node.colour)

        while current:
            for mx, my in moves:
                if current.board.tiles[mx][my].colour is None:
                    key = (mx, my)
                    if key not in current.amaf_visits:
                        current.amaf_visits[key] = 0
                        current.amaf_wins[key] = 0.0

                    current.amaf_visits[key] += 1
                    adj_result = result if colour == self.colour else (1 - result)
                    current.amaf_wins[key] += adj_result

            current = current.parent
            colour = Colour.opposite(colour)

    def _best_move(self, root: MCTSNode) -> Move:
        if not root.children:
            return root.untried_moves[0] if root.untried_moves else Move(5, 5)

        # Select by visit count (most robust)
        best = max(root.children, key=lambda c: c.visits)
        return best.move

    def _get_legal_moves(self, board: Board) -> List[Move]:
        moves = []
        n = board.size
        for i in range(n):
            for j in range(n):
                if board.tiles[i][j].colour is None:
                    moves.append(Move(i, j))
        return moves

    def _check_terminal(self, node: MCTSNode):
        if node.board.has_ended(Colour.RED):
            node.is_terminal = True
            node.winner = Colour.RED
        elif node.board.has_ended(Colour.BLUE):
            node.is_terminal = True
            node.winner = Colour.BLUE


# ============================================================================
# Main Tournament Agent
# ============================================================================

class Group12Agent(AgentBase):
    """
    Tournament-optimized Hex AI for Group12.

    Features:
    - MCTS with RAVE enhancement
    - Virtual connection detection
    - Nash equilibrium swap decisions
    - Intelligent time management
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
        self.time_manager = TimeManager()
        self.mcts = None  # Initialized after colour is known

        self.turn = 0

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Make the best move for the current position."""
        start_time = perf_counter_ns()
        self.turn = turn

        try:
            # Initialize MCTS if needed
            if self.mcts is None:
                self.mcts = TournamentMCTS(self._colour, self.evaluator, self.vc_detector)

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
