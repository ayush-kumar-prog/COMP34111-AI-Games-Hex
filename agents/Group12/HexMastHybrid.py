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
    __slots__ = (
        "parent",
        "move",
        "player_to_move",
        "children",
        "untried_moves",
        "visits",
        "wins",
    )

    def __init__(
        self,
        parent: Optional["MCTSNode"],
        move: Optional[Coord],
        player_to_move: Colour,
        legal_moves: List[Coord],
    ) -> None:
        self.parent = parent
        self.move = move
        self.player_to_move = player_to_move
        self.children: Dict[Coord, MCTSNode] = {}
        self.untried_moves: List[Coord] = list(legal_moves)
        self.visits: int = 0
        self.wins: float = 0.0  # from the perspective of the root player


class HexMastHybrid(AgentBase):
    """
    HexMast-Hybrid: MCTS + shallow alpha-beta search.
    Fresh MCTS tree each move, depth-2 minimax at root as a tactical filter.
    """

    def __init__(self, colour: Colour) -> None:
        super().__init__(colour)
        self.board_size: int = 11
        self.exploration_c: float = 1.2
        self.per_move_time: float = 1.0  # ~1s per move
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        start = time.time()
        self.board_size = board.size

        # Swap rule: as BLUE on turn 2, use evaluation-based swap
        if (
            turn == 2
            and self.colour == Colour.BLUE
            and opp_move is not None
            and not opp_move.is_swap()
        ):
            state, _ = self._extract_state(board)
            score = self._evaluate_state(state, self.colour)
            # If this opening is estimated bad for us, swap
            if score < 0.0:
                return Move(-1, -1)

        state, empty = self._extract_state(board)
        if not empty:
            return Move(0, 0)

        # Root-level shallow minimax (depth 2) for tactical scores
        minimax_scores = self._root_minimax_scores(state, empty, self.colour, depth=2)

        # Order moves by minimax score (good moves first)
        ordered_moves = sorted(
            empty, key=lambda mv: minimax_scores.get(mv, 0.0), reverse=True
        )
        root = MCTSNode(
            parent=None,
            move=None,
            player_to_move=self.colour,
            legal_moves=ordered_moves,
        )

        deadline = start + self.per_move_time
        while time.time() < deadline:
            self._run_simulation(root, state, ordered_moves)

        # Choose most visited child
        if root.children:
            best_child = max(root.children.values(), key=lambda n: n.visits)
            best_move = best_child.move
        else:
            best_move = self._rng.choice(ordered_moves)

        x, y = best_move
        return Move(x, y)

    # ------------------------------------------------------------------
    # MCTS core
    # ------------------------------------------------------------------

    def _run_simulation(
        self,
        root: MCTSNode,
        root_state: List[List[int]],
        root_moves: List[Coord],
    ) -> None:
        # Local copies of state and legal moves
        state = [row[:] for row in root_state]
        empty = [mv for mv in root_moves if state[mv[0]][mv[1]] == 0]

        node = root
        path: List[MCTSNode] = [node]
        current_player = self.colour

        # Selection + expansion
        while True:
            if not empty or self._is_terminal(state):
                break

            # Expand if possible
            if node.untried_moves:
                move = node.untried_moves.pop(0)
                self._apply_move(state, empty, move, current_player)
                next_player = self._other(current_player)
                child = MCTSNode(
                    parent=node,
                    move=move,
                    player_to_move=next_player,
                    legal_moves=empty,
                )
                node.children[move] = child
                node = child
                path.append(node)
                current_player = next_player
                break

            # If fully expanded but no children (shouldn't really happen)
            if not node.children:
                break

            # UCT selection
            move, node = self._select_child(node)
            self._apply_move(state, empty, move, current_player)
            path.append(node)
            current_player = self._other(current_player)

        # Rollout with early cutoff + static evaluation
        winner = self._rollout_with_eval(state, empty, current_player)

        # Backpropagate from self.colour's perspective
        reward = 1.0 if winner == self.colour else 0.0
        for n in path:
            n.visits += 1
            n.wins += reward

    def _select_child(self, node: MCTSNode) -> Tuple[Coord, MCTSNode]:
        assert node.children
        ln_parent = math.log(max(node.visits, 1))
        best_val = -1e9
        best_move = None
        best_child = None

        for move, child in node.children.items():
            if child.visits == 0:
                uct = float("inf")
            else:
                q = child.wins / child.visits
                uct = q + self.exploration_c * math.sqrt(ln_parent / child.visits)
            if uct > best_val:
                best_val = uct
                best_move = move
                best_child = child

        return best_move, best_child

    # ------------------------------------------------------------------
    # Rollout with cutoff + eval
    # ------------------------------------------------------------------

    def _rollout_with_eval(
        self,
        state: List[List[int]],
        empty: List[Coord],
        current_player: Colour,
    ) -> Colour:
        steps = 0
        max_steps = 2 * self.board_size  # limit rollout length

        while empty and not self._is_terminal(state) and steps < max_steps:
            move = self._rollout_policy(state, empty, current_player)
            self._apply_move(state, empty, move, current_player)
            current_player = self._other(current_player)
            steps += 1

        # If terminal, return actual winner
        if self._has_path(state, Colour.RED):
            return Colour.RED
        if self._has_path(state, Colour.BLUE):
            return Colour.BLUE

        # Otherwise use heuristic evaluation from our perspective
        score = self._evaluate_state(state, self.colour)
        return self.colour if score >= 0 else self._other(self.colour)

    def _rollout_policy(
        self,
        state: List[List[int]],
        empty: List[Coord],
        player: Colour,
    ) -> Coord:
        # Biased random: prefer centre + adjacency to own stones
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

            score = center_score + 0.5 * adj_me

            if score > best_score:
                best_score = score
                best_moves = [(x, y)]
            elif score == best_score:
                best_moves.append((x, y))

        return self._rng.choice(best_moves)

    # ------------------------------------------------------------------
    # Root-level shallow minimax (negamax depth 2)
    # ------------------------------------------------------------------

    def _root_minimax_scores(
        self,
        state: List[List[int]],
        empty: List[Coord],
        to_move: Colour,
        depth: int,
    ) -> Dict[Coord, float]:
        scores: Dict[Coord, float] = {}
        if depth <= 0:
            return {mv: self._evaluate_state(state, to_move) for mv in empty}

        for mv in empty:
            x, y = mv
            if state[x][y] != 0:
                continue
            state[x][y] = 1 if to_move == Colour.RED else 2
            new_empty = [e for e in empty if e != mv]

            val = -self._negamax(
                state,
                new_empty,
                self._other(to_move),
                depth - 1,
                -1e9,
                1e9,
                to_move,
            )

            state[x][y] = 0
            scores[mv] = val

        return scores

    def _negamax(
        self,
        state: List[List[int]],
        empty: List[Coord],
        player: Colour,
        depth: int,
        alpha: float,
        beta: float,
        root_player: Colour,
    ) -> float:
        # Depth 0 or terminal: evaluate
        if depth == 0 or not empty or self._is_terminal(state):
            eval_score = self._evaluate_state(state, root_player)
            return eval_score

        best = -1e9

        for mv in empty:
            x, y = mv
            if state[x][y] != 0:
                continue
            state[x][y] = 1 if player == Colour.RED else 2
            new_empty = [e for e in empty if e != mv]

            val = -self._negamax(
                state,
                new_empty,
                self._other(player),
                depth - 1,
                -beta,
                -alpha,
                root_player,
            )

            state[x][y] = 0

            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        return best

    # ------------------------------------------------------------------
    # Evaluation function
    # ------------------------------------------------------------------

    def _evaluate_state(self, state: List[List[int]], me: Colour) -> float:
        """
        Lightweight heuristic evaluation.
        Positive if good for 'me', negative if good for the opponent.
        """
        size = self.board_size
        opp = self._other(me)
        me_val = 1 if me == Colour.RED else 2
        opp_val = 1 if opp == Colour.RED else 2

        cx = cy = (size - 1) / 2.0
        score = 0.0

        for x in range(size):
            for y in range(size):
                v = state[x][y]
                if v == 0:
                    continue

                # centrality
                dist = abs(x - cx) + abs(y - cy)
                center_bonus = (size - dist)

                # adjacency to same-colour stones
                adj_same = 0
                for dx, dy in zip(Tile.I_DISPLACEMENTS, Tile.J_DISPLACEMENTS):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        if state[nx][ny] == v:
                            adj_same += 1

                piece_score = center_bonus + 2.0 * adj_same

                if v == me_val:
                    score += piece_score
                elif v == opp_val:
                    score -= piece_score

        return score

    # ------------------------------------------------------------------
    # Board utilities
    # ------------------------------------------------------------------

    def _extract_state(self, board: Board) -> Tuple[List[List[int]], List[Coord]]:
        tiles = board._tiles  # internal engine representation
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

    def _apply_move(
        self,
        state: List[List[int]],
        empty: List[Coord],
        move: Coord,
        player: Colour,
    ) -> None:
        x, y = move
        val = 1 if player == Colour.RED else 2
        state[x][y] = val
        try:
            empty.remove(move)
        except ValueError:
            pass

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
