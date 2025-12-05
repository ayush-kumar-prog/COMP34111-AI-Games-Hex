from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Tuple

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile


Coord = Tuple[int, int]


class MCTSNode:
    """
    A single node in the MCTS tree for HexMast-Ultra.

    - `move` is the coordinate (x, y) that led to this node from its parent.
      For the root, `move` is None.
    - `player_to_move` is the Colour whose turn it is at this node.
    - RAVE statistics are stored per node, keyed by move index.
    """

    __slots__ = (
        "parent",
        "move",
        "player_to_move",
        "children",
        "untried_moves",
        "visits",
        "wins",
        "rave_wins",
        "rave_visits",
    )

    def __init__(
        self,
        parent: Optional["MCTSNode"],
        move: Optional[Coord],
        player_to_move: Colour,
        legal_moves: List[Coord],
    ) -> None:
        self.parent: Optional[MCTSNode] = parent
        self.move: Optional[Coord] = move
        self.player_to_move: Colour = player_to_move

        # children: (x, y) -> node
        self.children: Dict[Coord, MCTSNode] = {}

        # moves not yet expanded at this node
        self.untried_moves: List[Coord] = list(legal_moves)

        # standard MCTS stats
        self.visits: int = 0
        self.wins: float = 0.0  # wins for *our agent* (fixed perspective)

        # RAVE stats: move_index -> (wins, visits)
        self.rave_wins: Dict[int, float] = {}
        self.rave_visits: Dict[int, int] = {}


class HexMastUltra2(AgentBase):
    """
    HexMast-Ultra: a stronger MCTS agent for 11x11 Hex.

    - UCT selection with tuned exploration constant.
    - RAVE (Rapid Action Value Estimation) to speed up early learning.
    - Biased rollouts favouring centre and simple connection patterns.
    - Simple time control (bounded per-move) to respect the 5 min/game limit.
    """

    def __init__(self, colour: Colour) -> None:
        super().__init__(colour)

        self.board_size: int = 11
        self.root: Optional[MCTSNode] = None

        # MCTS / RAVE parameters
        self.exploration_c: float = 0.9   # slightly below sqrt(2)
        self.rave_b: float = 300.0        # RAVE beta parameter

        # Time management
        self.total_time_used: float = 0.0
        self.max_total_time: float = 295.0  # stay below 5 minutes with margin
        self.default_per_move: float = 1.5  # upper bound per move in seconds

        # Random generator
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Public API required by AgentBase
    # ------------------------------------------------------------------

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """
        Called by the engine to request a move.
        - `turn` is the global turn number (1-based).
        - `board` is the current Board object.
        - `opp_move` is the opponent's last Move, or None if we play first.
        """
        start = time.time()

        # Handle swap (pie rule) as Blue on our first move (turn == 2)
        if (
            turn == 2
            and self.colour == Colour.BLUE
            and opp_move is not None
            and not opp_move.is_swap()
        ):
            if self._should_swap(opp_move, board):
                move = Move(-1, -1)
                self.total_time_used += time.time() - start
                return move

        # Convert Board -> simple integer state + list of empty cells
        state, empty = self._extract_state(board)
        self.board_size = len(state)

        if not empty:
            # Should not happen in a normal game, but be defensive
            self.total_time_used += time.time() - start
            return Move(0, 0)

        # Build a fresh root each move (simpler and still strong enough)
        root_player = self.colour
        self.root = MCTSNode(
            parent=None, move=None, player_to_move=root_player, legal_moves=empty
        )

        # Time budget
        remaining = max(0.01, self.max_total_time - self.total_time_used)
        per_move_budget = min(self.default_per_move, remaining - 0.01)
        if per_move_budget <= 0.02:
            per_move_budget = 0.02  # always do at least a few simulations

        deadline = start + per_move_budget

        # MCTS loop
        while time.time() < deadline:
            self._run_simulation(state, empty)

        # Pick best move (most visits from root)
        if self.root.children:
            best_child = max(self.root.children.values(), key=lambda n: n.visits)
            best_move = best_child.move
        else:
            best_move = self._rng.choice(empty)

        self.total_time_used += time.time() - start

        x, y = best_move
        return Move(x, y)

    # ------------------------------------------------------------------
    # MCTS core
    # ------------------------------------------------------------------

    def _run_simulation(
        self,
        root_state: List[List[int]],
        root_empty: List[Coord],
    ) -> None:
        """
        One complete MCTS simulation:
        - selection & expansion down the tree
        - rollout (playout) to the end of the game
        - backpropagation with RAVE updates
        """
        # Local copies
        state = [row[:] for row in root_state]
        empty = list(root_empty)

        node = self.root
        assert node is not None

        path: List[MCTSNode] = [node]

        # Root's player_to_move is our agent
        current_player = self.colour

        # Moves played by *us* in this simulation (for RAVE)
        played_by_self: List[int] = []

        # -------- Selection + Expansion --------
        while True:
            if not empty or self._is_terminal(state):
                break

            # Expand if we have untried moves
            if node.untried_moves:
                move = self._select_untried_move(node)
                self._apply_move(state, empty, move, current_player)

                move_id = self._move_index(move)
                if current_player == self.colour:
                    played_by_self.append(move_id)

                next_player = self._other_player(current_player)

                child = MCTSNode(
                    parent=node,
                    move=move,
                    player_to_move=next_player,
                    legal_moves=empty,
                )
                node.children[move] = child
                path.append(child)
                node = child
                current_player = next_player
                break  # then rollout from this child

            # Otherwise, select best child using UCT + RAVE
            move, child = self._select_child_uct_rave(node)
            self._apply_move(state, empty, move, current_player)

            move_id = self._move_index(move)
            if current_player == self.colour:
                played_by_self.append(move_id)

            node = child
            path.append(node)
            current_player = self._other_player(current_player)

        # -------- Rollout --------
        winner, played_by_self = self._rollout(
            state, empty, current_player, played_by_self
        )

        # -------- Backpropagation --------
        self._backpropagate(path, winner, played_by_self)

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _select_untried_move(self, node: MCTSNode) -> Coord:
        """
        Pick an untried move, biased toward the centre of the board.
        """
        if len(node.untried_moves) == 1:
            return node.untried_moves.pop()

        cx = cy = self.board_size / 2.0
        scored = []
        for m in node.untried_moves:
            x, y = m
            dist = abs(x - cx) + abs(y - cy)
            scored.append((-(dist), m))  # smaller distance => better

        scored.sort(reverse=True)
        top_k = min(6, len(scored))
        pool = [scored[i][1] for i in range(top_k)]

        move = self._rng.choice(pool)
        node.untried_moves.remove(move)
        return move

    def _select_child_uct_rave(self, node: MCTSNode) -> Tuple[Coord, MCTSNode]:
        """
        Select child using UCT + RAVE combination.
        """
        assert node.children

        ln_parent = math.log(max(node.visits, 1))
        best_value = -1e9
        best_move = None
        best_child = None

        for move, child in node.children.items():
            if child.visits == 0:
                value = float("inf")
            else:
                q = child.wins / child.visits

                move_id = self._move_index(move)
                rave_n = node.rave_visits.get(move_id, 0)
                rave_w = node.rave_wins.get(move_id, 0.0)

                if rave_n > 0:
                    rave_q = rave_w / rave_n
                    beta = rave_n / (
                        child.visits
                        + rave_n
                        + 4 * child.visits * rave_n / self.rave_b
                    )
                    mixed_q = (1 - beta) * q + beta * rave_q
                else:
                    mixed_q = q

                value = mixed_q + self.exploration_c * math.sqrt(
                    ln_parent / child.visits
                )

            if value > best_value:
                best_value = value
                best_move = move
                best_child = child

        assert best_move is not None and best_child is not None
        return best_move, best_child

    # ------------------------------------------------------------------
    # Rollout / default policy
    # ------------------------------------------------------------------

    def _rollout(
        self,
        state: List[List[int]],
        empty: List[Coord],
        current_player: Colour,
        played_by_self: List[int],
    ) -> Tuple[Colour, List[int]]:
        """
        Biased playout until the board is full.
        Returns winner Colour and updated list of our own moves.
        """
        while empty:
            move = self._select_rollout_move(state, empty, current_player)
            self._apply_move(state, empty, move, current_player)

            move_id = self._move_index(move)
            if current_player == self.colour:
                played_by_self.append(move_id)

            current_player = self._other_player(current_player)

        winner = self._determine_winner(state)
        return winner, played_by_self

    def _select_rollout_move(
        self,
        state: List[List[int]],
        empty: List[Coord],
        player: Colour,
    ) -> Coord:
        """
        Rollout policy:
        - Prefer moves that connect to at least two same-colour neighbours (simple "bridge").
        - Otherwise, prefer central cells.
        """
        player_val = 1 if player == Colour.RED else 2
        bridge_moves: List[Coord] = []

        for (x, y) in empty:
            same_neigh = 0
            for dx, dy in zip(Tile.I_DISPLACEMENTS, Tile.J_DISPLACEMENTS):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if state[nx][ny] == player_val:
                        same_neigh += 1
                        if same_neigh >= 2:
                            bridge_moves.append((x, y))
                            break

        if bridge_moves:
            return self._rng.choice(bridge_moves)

        # Otherwise, prefer centre
        cx = cy = self.board_size / 2.0
        best_score = -1e9
        best_moves: List[Coord] = []

        for (x, y) in empty:
            dist = abs(x - cx) + abs(y - cy)
            score = -dist
            if score > best_score:
                best_score = score
                best_moves = [(x, y)]
            elif score == best_score:
                best_moves.append((x, y))

        return self._rng.choice(best_moves)

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(
        self,
        path: List[MCTSNode],
        winner: Colour,
        played_by_self: List[int],
    ) -> None:
        """
        Backpropagate from leaf to root:
        - standard MCTS stats (fixed perspective of our agent)
        - RAVE updates for moves we played
        """
        result = 1.0 if winner == self.colour else 0.0
        rave_moves = set(played_by_self)

        for node in path:
            node.visits += 1
            node.wins += result

            for m_id in rave_moves:
                node.rave_visits[m_id] = node.rave_visits.get(m_id, 0) + 1
                node.rave_wins[m_id] = node.rave_wins.get(m_id, 0.0) + result

    # ------------------------------------------------------------------
    # Board / state utilities
    # ------------------------------------------------------------------

    def _extract_state(self, board: Board) -> Tuple[List[List[int]], List[Coord]]:
        """
        Convert Board -> integer grid + list of empty cells.
        0 = empty, 1 = RED, 2 = BLUE
        """
        tiles = board._tiles  # type: ignore[attr-defined]
        size = len(tiles)
        state: List[List[int]] = [[0] * size for _ in range(size)]
        empty: List[Coord] = []

        for x in range(size):
            for y in range(size):
                tile = tiles[x][y]
                if tile.colour is None:
                    state[x][y] = 0
                    empty.append((x, y))
                elif tile.colour == Colour.RED:
                    state[x][y] = 1
                else:
                    state[x][y] = 2

        return state, empty

    def _apply_move(
        self,
        state: List[List[int]],
        empty: List[Coord],
        move: Coord,
        player: Colour,
    ) -> None:
        """
        Apply a move to the local integer board + empties list.
        """
        x, y = move
        val = 1 if player == Colour.RED else 2
        state[x][y] = val
        try:
            empty.remove(move)
        except ValueError:
            # Should not happen, but be robust
            pass

    def _is_terminal(self, state: List[List[int]]) -> bool:
        """
        Check if either player already has a winning connection.
        """
        return self._has_path(state, Colour.RED) or self._has_path(state, Colour.BLUE)

    def _determine_winner(self, state: List[List[int]]) -> Colour:
        """
        Determine which player has a connecting path.
        Exactly one will in Hex.
        """
        if self._has_path(state, Colour.RED):
            return Colour.RED
        return Colour.BLUE

    def _has_path(self, state: List[List[int]], colour: Colour) -> bool:
        """
        BFS to check if `colour` has connected their two opposite sides.
        RED: top -> bottom
        BLUE: left -> right
        """
        size = len(state)
        val = 1 if colour == Colour.RED else 2

        from collections import deque

        visited = [[False] * size for _ in range(size)]
        q = deque()

        if colour == Colour.RED:
            # start from top row
            for y in range(size):
                if state[0][y] == val:
                    visited[0][y] = True
                    q.append((0, y))

            while q:
                x, y = q.popleft()
                if x == size - 1:
                    return True
                for dx, dy in zip(Tile.I_DISPLACEMENTS, Tile.J_DISPLACEMENTS):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        if not visited[nx][ny] and state[nx][ny] == val:
                            visited[nx][ny] = True
                            q.append((nx, ny))
        else:
            # BLUE: left -> right
            for x in range(size):
                if state[x][0] == val:
                    visited[x][0] = True
                    q.append((x, 0))

            while q:
                x, y = q.popleft()
                if y == size - 1:
                    return True
                for dx, dy in zip(Tile.I_DISPLACEMENTS, Tile.J_DISPLACEMENTS):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        if not visited[nx][ny] and state[nx][ny] == val:
                            visited[nx][ny] = True
                            q.append((nx, ny))

        return False

    def _move_index(self, move: Coord) -> int:
        """
        Map (x, y) to a unique index for RAVE arrays.
        """
        x, y = move
        return x * self.board_size + y

    @staticmethod
    def _other_player(player: Colour) -> Colour:
        return Colour.RED if player == Colour.BLUE else Colour.BLUE

    # ------------------------------------------------------------------
    # Swap rule heuristic (as Blue on turn 2)
    # ------------------------------------------------------------------

    def _should_swap(self, opp_move: Move, board: Board) -> bool:
        """
        Very simple swap heuristic:
        - If Red opens near the centre (within Manhattan dist <= 2),
          we take the swap.
        """
        x, y = opp_move.x, opp_move.y
        size = len(board._tiles)  # type: ignore[attr-defined]
        cx = cy = (size - 1) / 2.0
        dist = abs(x - cx) + abs(y - cy)
        return dist <= 2.0
