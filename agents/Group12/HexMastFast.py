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


class HexMastFast(AgentBase):
    """
    HexMast-Fast: ultra-light, root-only MCTS / flat Monte Carlo.
    Focuses on speed with simple heuristics.
    """

    def __init__(self, colour: Colour) -> None:
        super().__init__(colour)
        self.board_size: int = 11
        self.per_move_time: float = 0.2  # very small time budget
        self.exploration_c: float = 1.4
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        start = time.time()
        self.board_size = board.size

        # Simple swap rule: as BLUE on turn 2, swap if Red played near centre
        if (
            turn == 2
            and self.colour == Colour.BLUE
            and opp_move is not None
            and not opp_move.is_swap()
        ):
            if self._should_swap_simple(opp_move, board.size):
                return Move(-1, -1)

        state, empty = self._extract_state(board)
        if not empty:
            return Move(0, 0)

        if len(empty) == 1:
            x, y = empty[0]
            return Move(x, y)

        root_player = self.colour
        legal_moves = list(empty)

        # Root-only bandit stats
        wins: Dict[Coord, float] = {mv: 0.0 for mv in legal_moves}
        visits: Dict[Coord, int] = {mv: 0 for mv in legal_moves}

        # Pre-order moves by static heuristic (centre + adjacency)
        legal_moves.sort(
            key=lambda mv: self._static_move_score(state, mv, root_player),
            reverse=True,
        )

        deadline = start + self.per_move_time
        idx = 0
        n_moves = len(legal_moves)

        while time.time() < deadline:
            # Ensure each move is tried at least once
            if idx < n_moves:
                mv = legal_moves[idx]
                idx += 1
            else:
                # UCB1 selection at root
                total_visits = sum(visits.values()) + 1
                ln_total = math.log(total_visits)
                best_val = -1e9
                best_mv = legal_moves[0]
                for mv_candidate in legal_moves:
                    v = visits[mv_candidate]
                    if v == 0:
                        uct = float("inf")
                    else:
                        q = wins[mv_candidate] / v
                        uct = q + self.exploration_c * math.sqrt(ln_total / v)
                    if uct > best_val:
                        best_val = uct
                        best_mv = mv_candidate
                mv = best_mv

            # Simulate from chosen root move
            tmp_state = [row[:] for row in state]
            tmp_empty = [p for p in empty if p != mv]

            # Apply root move
            self._apply_move(tmp_state, mv, root_player)
            current_player = self._other(root_player)

            winner = self._rollout_fast(tmp_state, tmp_empty, current_player)

            visits[mv] += 1
            if winner == root_player:
                wins[mv] += 1.0

        # Pick move with best win rate (then visits, then heuristic)
        best_mv = max(
            legal_moves,
            key=lambda mv: (
                wins[mv] / visits[mv] if visits[mv] > 0 else -1.0,
                visits[mv],
                self._static_move_score(state, mv, root_player),
            ),
        )

        x, y = best_mv
        return Move(x, y)

    # ------------------------------------------------------------------
    # Rollout policy
    # ------------------------------------------------------------------

    def _rollout_fast(
        self,
        state: List[List[int]],
        empty: List[Coord],
        current_player: Colour,
    ) -> Colour:
        """
        Very lightweight random-biased rollout until terminal or board full.
        """
        while empty and not self._is_terminal(state):
            mv = self._fast_rollout_policy(state, empty, current_player)
            self._apply_move(state, mv, current_player)
            try:
                empty.remove(mv)
            except ValueError:
                pass
            current_player = self._other(current_player)

        if self._has_path(state, Colour.RED):
            return Colour.RED
        if self._has_path(state, Colour.BLUE):
            return Colour.BLUE

        # Fallback: treat as loss for us (should rarely happen)
        return self._other(self.colour)

    def _fast_rollout_policy(
        self,
        state: List[List[int]],
        empty: List[Coord],
        player: Colour,
    ) -> Coord:
        if len(empty) == 1:
            return empty[0]

        size = self.board_size
        cx = cy = (size - 1) / 2.0
        me_val = 1 if player == Colour.RED else 2

        best_score = -1e9
        best_moves: List[Coord] = []

        for (x, y) in empty:
            # centrality
            dist = abs(x - cx) + abs(y - cy)
            center_score = -dist

            # adjacency to own stones
            adj_me = 0
            for dx, dy in zip(Tile.I_DISPLACEMENTS, Tile.J_DISPLACEMENTS):
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if state[nx][ny] == me_val:
                        adj_me += 1

            score = center_score + 0.4 * adj_me

            if score > best_score:
                best_score = score
                best_moves = [(x, y)]
            elif score == best_score:
                best_moves.append((x, y))

        return self._rng.choice(best_moves)

    # ------------------------------------------------------------------
    # Static move heuristic for ordering
    # ------------------------------------------------------------------

    def _static_move_score(
        self, state: List[List[int]], move: Coord, player: Colour
    ) -> float:
        size = self.board_size
        cx = cy = (size - 1) / 2.0
        x, y = move
        dist = abs(x - cx) + abs(y - cy)
        center_score = -dist

        me_val = 1 if player == Colour.RED else 2
        opp_val = 2 if me_val == 1 else 1

        adj_me = 0
        adj_opp = 0
        for dx, dy in zip(Tile.I_DISPLACEMENTS, Tile.J_DISPLACEMENTS):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                if state[nx][ny] == me_val:
                    adj_me += 1
                elif state[nx][ny] == opp_val:
                    adj_opp += 1

        # encourage connecting own stones and also contesting opponent
        return center_score + 0.6 * adj_me + 0.3 * adj_opp

    # ------------------------------------------------------------------
    # Simple swap heuristic
    # ------------------------------------------------------------------

    def _should_swap_simple(self, opp_move: Move, size: int) -> bool:
        cx = cy = (size - 1) / 2.0
        dist = abs(opp_move.x - cx) + abs(opp_move.y - cy)
        # If Red played near centre, swapping is usually correct
        return dist <= 1.5

    # ------------------------------------------------------------------
    # Board utilities
    # ------------------------------------------------------------------

    def _extract_state(self, board: Board) -> Tuple[List[List[int]], List[Coord]]:
        tiles = board._tiles
        size = len(tiles)
        state: List[List[int]] = [[0] * size for _ in range(size)]
        empty: List[Coord] = []
        for x in range(size):
            for y in range(size):
                c = tiles[x][y].colour
                if c is None:
                    state[x][y] = 0
                    empty.append((x, y))
                elif c == Colour.RED:
                    state[x][y] = 1
                else:
                    state[x][y] = 2
        return state, empty

    def _apply_move(self, state: List[List[int]], move: Coord, player: Colour) -> None:
        x, y = move
        state[x][y] = 1 if player == Colour.RED else 2

    def _is_terminal(self, state: List[List[int]]) -> bool:
        return self._has_path(state, Colour.RED) or self._has_path(state, Colour.BLUE)

    def _has_path(self, state: List[List[int]], colour: Colour) -> bool:
        size = self.board_size
        val = 1 if colour == Colour.RED else 2

        from collections import deque

        visited = [[False] * size for _ in range(size)]
        q = deque()

        if colour == Colour.RED:
            # top -> bottom
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
            # left -> right
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

    @staticmethod
    def _other(colour: Colour) -> Colour:
        return Colour.RED if colour == Colour.BLUE else Colour.BLUE
