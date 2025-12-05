import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


Coord = Tuple[int, int]


class HexMastUltra(AgentBase):
    _ZOBRIST: Optional[List[List[List[int]]]] = None
    _SIDE_HASH: Optional[Dict[Colour, int]] = None

    class Node:
        def __init__(
            self,
            move: Optional[Coord],
            parent: Optional["HexMastUltra.Node"],
            player_to_move: Colour,
            state_hash: int,
            unexpanded: List[Coord],
            priors: Dict[Coord, float],
        ):
            self.move = move
            self.parent = parent
            self.player_to_move = player_to_move
            self.state_hash = state_hash
            self.children: List["HexMastUltra.Node"] = []
            self.unexpanded: List[Coord] = unexpanded
            self.priors = priors
            self.visits = 0
            self.wins = 0.0
            self.rave_wins: Dict[Coord, float] = defaultdict(float)
            self.rave_visits: Dict[Coord, int] = defaultdict(int)
            self.bias_mean = 0.5
            self.bias_strength = 0.0

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.size = 11
        self.exploration_c = 1.35
        self.rave_k = 150.0
        self.progressive_bias_strength = 2.0
        self.pw_base = 3.0
        self.pw_alpha = 0.55
        self.prior_floor = 0.05
        self.rollout_top_k = 6
        self.rollout_noise = 0.15
        self.root: Optional[HexMastUltra.Node] = None
        self.transposition: Dict[int, Tuple[float, int]] = {}
        self._rng = random.Random(1337)
        self._ensure_zobrist(self.size)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        self.size = board.size
        self._ensure_zobrist(self.size)
        if self._should_swap(turn, board, opp_move):
            return Move(-1, -1)

        board_state = self._board_to_array(board)
        empty_set = self._compute_empty(board_state)
        to_move = self.colour
        self._reuse_tree(board_state, to_move, empty_set)

        time_budget = self._time_budget(turn, len(empty_set))
        best_move = self._search(board_state, empty_set, to_move, time_budget)
        if best_move is None:
            best_move = self._fallback_move(empty_set)
        self._prepare_root_for_next_turn(best_move, board_state, empty_set, to_move)
        return Move(best_move[0], best_move[1])

    def _search(
        self,
        board_state: List[List[int]],
        empty_set: Set[Coord],
        to_move: Colour,
        time_budget: float,
    ) -> Optional[Coord]:
        deadline = time.time() + max(0.05, time_budget)
        if self.root is None:
            root_hash = self._board_hash(board_state, to_move)
            unexpanded, priors = self._ordered_moves(board_state, empty_set, to_move)
            self.root = HexMastUltra.Node(
                move=None,
                parent=None,
                player_to_move=to_move,
                state_hash=root_hash,
                unexpanded=unexpanded,
                priors=priors,
            )

        while time.time() < deadline:
            path: List[HexMastUltra.Node] = []
            moves_applied: List[Coord] = []
            played_by_colour: Dict[Colour, Set[Coord]] = {Colour.RED: set(), Colour.BLUE: set()}
            node = self.root
            current_player = to_move

            path.append(node)
            # Selection + Expansion
            while True:
                winner = self._winner(board_state)
                if winner is not None or not empty_set:
                    break

                allowed_children = self._progressive_widening_limit(node)
                if node.unexpanded and len(node.children) < allowed_children:
                    move = node.unexpanded.pop(0)
                    self._apply_move(board_state, move, current_player, empty_set)
                    moves_applied.append(move)
                    played_by_colour[current_player].add(move)
                    next_player = Colour.opposite(current_player)
                    new_hash = self._board_hash(board_state, next_player)
                    child_unexp, child_priors = self._ordered_moves(board_state, empty_set, next_player)
                    child = HexMastUltra.Node(
                        move=move,
                        parent=node,
                        player_to_move=next_player,
                        state_hash=new_hash,
                        unexpanded=child_unexp,
                        priors=child_priors,
                    )
                    h_score = node.priors.get(move, 0.5)
                    child.bias_mean = h_score
                    child.bias_strength = self.progressive_bias_strength
                    if new_hash in self.transposition:
                        w, v = self.transposition[new_hash]
                        child.wins = w
                        child.visits = v
                    node.children.append(child)
                    node = child
                    current_player = next_player
                    path.append(node)
                    break

                if not node.children:
                    break

                node = self._select_child(node)
                self._apply_move(board_state, node.move, current_player, empty_set)
                moves_applied.append(node.move)
                played_by_colour[current_player].add(node.move)
                current_player = Colour.opposite(current_player)
                path.append(node)

            # Simulation
            winner = self._simulate(board_state, empty_set, current_player, moves_applied, played_by_colour, deadline)

            # Backpropagation
            reward = 1.0 if winner == to_move else 0.0
            if winner is None:
                reward = 0.5
            self._backpropagate(path, reward, played_by_colour)

            # Undo
            self._undo_moves(board_state, moves_applied, empty_set)

        best = self._best_child_move(self.root)
        return best

    def _simulate(
        self,
        board_state: List[List[int]],
        empty_set: Set[Coord],
        to_move: Colour,
        moves_applied: List[Coord],
        played_by_colour: Dict[Colour, Set[Coord]],
        deadline: float,
    ) -> Optional[Colour]:
        current_player = to_move
        while True:
            if time.time() > deadline:
                return self._winner(board_state)
            winner = self._winner(board_state)
            if winner is not None or not empty_set:
                return winner

            move = self._biased_rollout_move(board_state, empty_set, current_player)
            self._apply_move(board_state, move, current_player, empty_set)
            moves_applied.append(move)
            played_by_colour[current_player].add(move)
            current_player = Colour.opposite(current_player)

        return self._winner(board_state)

    def _backpropagate(
        self,
        path: List["HexMastUltra.Node"],
        reward: float,
        played_by_colour: Dict[Colour, Set[Coord]],
    ):
        for node in reversed(path):
            node.visits += 1
            node.wins += reward
            for mv in played_by_colour.get(node.player_to_move, ()):
                node.rave_visits[mv] += 1
                node.rave_wins[mv] += reward
            if node.state_hash is not None:
                prev_w, prev_v = self.transposition.get(node.state_hash, (0.0, 0))
                self.transposition[node.state_hash] = (prev_w + reward, prev_v + 1)

    def _select_child(self, node: "HexMastUltra.Node") -> "HexMastUltra.Node":
        best_value = -1e9
        best_child = node.children[0]
        log_parent = math.log(node.visits + 1.0)
        for child in node.children:
            exploitation = 0.5
            if child.visits > 0:
                exploitation = child.wins / child.visits
            prior_adj = (child.bias_strength * child.bias_mean + child.wins) / (
                child.bias_strength + child.visits + 1e-9
            )
            rave_v = node.rave_visits.get(child.move, 0)
            rave_q = node.rave_wins.get(child.move, 0.0) / rave_v if rave_v > 0 else 0.5
            beta = rave_v / (child.visits + rave_v + self.rave_k)
            blended = (1 - beta) * prior_adj + beta * rave_q
            explore = self.exploration_c * math.sqrt(log_parent / (child.visits + 1e-9))
            value = blended + explore
            if value > best_value:
                best_value = value
                best_child = child
        return best_child

    def _best_child_move(self, node: Optional["HexMastUltra.Node"]) -> Optional[Coord]:
        if node is None or not node.children:
            return None
        best = max(node.children, key=lambda c: c.visits)
        return best.move

    def _progressive_widening_limit(self, node: "HexMastUltra.Node") -> int:
        return max(1, int(self.pw_base + math.pow(node.visits + 1.0, self.pw_alpha)))

    def _ordered_moves(
        self, board_state: List[List[int]], empty_set: Set[Coord], player: Colour
    ) -> Tuple[List[Coord], Dict[Coord, float]]:
        scored = []
        priors: Dict[Coord, float] = {}
        for mv in empty_set:
            h = self._heuristic_score(board_state, mv, player)
            h_norm = max(self.prior_floor, min(0.95, h))
            scored.append((h_norm, mv))
            priors[mv] = h_norm
        scored.sort(key=lambda t: t[0], reverse=True)
        ordered = [mv for _, mv in scored]
        return ordered, priors

    def _heuristic_score(self, board_state: List[List[int]], move: Coord, player: Colour) -> float:
        x, y = move
        size = self.size
        mid = (size - 1) / 2.0
        center_dist = abs(x - mid) + abs(y - mid)
        center_score = 1.0 - (center_dist / (size - 1 + mid))
        adj_friend = 0
        adj_opp = 0
        bridge_bonus = 0
        target_pull = 0
        for dx, dy in zip(
            [-1, -1, 0, 1, 1, 0],
            [0, 1, 1, 0, -1, -1],
        ):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                cell = board_state[nx][ny]
                if cell == self._p_int(player):
                    adj_friend += 1
                elif cell == self._p_int(Colour.opposite(player)):
                    adj_opp += 1
        bridge_dirs = [(-1, 1), (1, -1), (-1, -1), (1, 1), (-2, 0), (2, 0)]
        for bdx, bdy in bridge_dirs:
            bx, by = x + bdx, y + bdy
            midx, midy = x + bdx // 2, y + bdy // 2
            if 0 <= bx < size and 0 <= by < size and 0 <= midx < size and 0 <= midy < size:
                if board_state[bx][by] == self._p_int(player) and board_state[midx][midy] == 0:
                    bridge_bonus += 1
        if player == Colour.RED:
            target_pull = 1.0 - abs(x - mid) / mid if mid > 0 else 1.0
        else:
            target_pull = 1.0 - abs(y - mid) / mid if mid > 0 else 1.0
        block_score = 0.3 * adj_opp
        connect_score = 0.6 * adj_friend + 0.45 * bridge_bonus
        score = 0.25 * center_score + connect_score + block_score + 0.4 * target_pull
        return max(0.0, min(1.5, score))

    def _biased_rollout_move(
        self, board_state: List[List[int]], empty_set: Set[Coord], player: Colour
    ) -> Coord:
        scored: List[Tuple[float, Coord]] = []
        for mv in empty_set:
            h = self._heuristic_score(board_state, mv, player)
            if h < 0.12:
                # skip clearly low value cells surrounded by opponents
                adj = self._adjacent_counts(board_state, mv, player)
                if adj[0] == 0 and adj[1] >= 3:
                    continue
            noise = self._rng.random() * self.rollout_noise
            scored.append((h + noise, mv))
        if not scored:
            return self._fallback_move(empty_set)
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[: max(1, min(self.rollout_top_k, len(scored)))]
        weights = [max(0.01, s[0]) for s in top]
        total = sum(weights)
        pick = self._rng.random() * total
        cum = 0.0
        for w, (_, mv) in zip(weights, top):
            cum += w
            if pick <= cum:
                return mv
        return top[0][1]

    def _adjacent_counts(self, board_state: List[List[int]], move: Coord, player: Colour) -> Tuple[int, int]:
        x, y = move
        size = self.size
        adj_friend = 0
        adj_opp = 0
        for dx, dy in zip(
            [-1, -1, 0, 1, 1, 0],
            [0, 1, 1, 0, -1, -1],
        ):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                val = board_state[nx][ny]
                if val == self._p_int(player):
                    adj_friend += 1
                elif val == self._p_int(Colour.opposite(player)):
                    adj_opp += 1
        return adj_friend, adj_opp

    def _apply_move(
        self, board_state: List[List[int]], move: Coord, player: Colour, empty_set: Set[Coord]
    ):
        x, y = move
        if move in empty_set:
            empty_set.remove(move)
        board_state[x][y] = self._p_int(player)

    def _undo_moves(self, board_state: List[List[int]], moves: List[Coord], empty_set: Set[Coord]):
        for x, y in reversed(moves):
            board_state[x][y] = 0
            empty_set.add((x, y))

    def _winner(self, board_state: List[List[int]]) -> Optional[Colour]:
        if self._has_won(board_state, Colour.RED):
            return Colour.RED
        if self._has_won(board_state, Colour.BLUE):
            return Colour.BLUE
        return None

    def _has_won(self, board_state: List[List[int]], colour: Colour) -> bool:
        target = self._p_int(colour)
        size = self.size
        visited = [[False for _ in range(size)] for _ in range(size)]
        stack: List[Coord] = []
        if colour == Colour.RED:
            for y in range(size):
                if board_state[0][y] == target:
                    stack.append((0, y))
                    visited[0][y] = True
            goal_row = size - 1
            while stack:
                x, y = stack.pop()
                if x == goal_row:
                    return True
                for dx, dy in zip(
                    [-1, -1, 0, 1, 1, 0],
                    [0, 1, 1, 0, -1, -1],
                ):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size and not visited[nx][ny] and board_state[nx][ny] == target:
                        visited[nx][ny] = True
                        stack.append((nx, ny))
        else:
            for x in range(size):
                if board_state[x][0] == target:
                    stack.append((x, 0))
                    visited[x][0] = True
            goal_col = size - 1
            while stack:
                x, y = stack.pop()
                if y == goal_col:
                    return True
                for dx, dy in zip(
                    [-1, -1, 0, 1, 1, 0],
                    [0, 1, 1, 0, -1, -1],
                ):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size and not visited[nx][ny] and board_state[nx][ny] == target:
                        visited[nx][ny] = True
                        stack.append((nx, ny))
        return False

    def _board_to_array(self, board: Board) -> List[List[int]]:
        size = board.size
        arr = [[0 for _ in range(size)] for _ in range(size)]
        for x in range(size):
            for y in range(size):
                colour = board.tiles[x][y].colour
                if colour == Colour.RED:
                    arr[x][y] = 1
                elif colour == Colour.BLUE:
                    arr[x][y] = 2
        return arr

    def _compute_empty(self, board_state: List[List[int]]) -> Set[Coord]:
        empties: Set[Coord] = set()
        size = self.size
        for x in range(size):
            for y in range(size):
                if board_state[x][y] == 0:
                    empties.add((x, y))
        return empties

    def _p_int(self, colour: Colour) -> int:
        return 1 if colour == Colour.RED else 2

    def _ensure_zobrist(self, size: int):
        if HexMastUltra._ZOBRIST is not None and len(HexMastUltra._ZOBRIST) == size:
            return
        rng = random.Random(2025)
        HexMastUltra._ZOBRIST = [
            [[rng.getrandbits(64) for _ in range(2)] for _ in range(size)]
            for _ in range(size)
        ]
        HexMastUltra._SIDE_HASH = {
            Colour.RED: rng.getrandbits(64),
            Colour.BLUE: rng.getrandbits(64),
        }

    def _board_hash(self, board_state: List[List[int]], to_move: Colour) -> int:
        h = 0
        size = self.size
        zob = HexMastUltra._ZOBRIST
        for x in range(size):
            for y in range(size):
                val = board_state[x][y]
                if val == 1:
                    h ^= zob[x][y][0]
                elif val == 2:
                    h ^= zob[x][y][1]
        side_hash = HexMastUltra._SIDE_HASH or {}
        h ^= side_hash.get(to_move, 0)
        return h

    def _reuse_tree(self, board_state: List[List[int]], to_move: Colour, empty_set: Set[Coord]):
        current_hash = self._board_hash(board_state, to_move)
        if self.root and self.root.state_hash == current_hash:
            self.root.parent = None
            return
        if self.root:
            for child in self.root.children:
                if child.state_hash == current_hash:
                    child.parent = None
                    self.root = child
                    return
        self.root = None

    def _prepare_root_for_next_turn(
        self, move: Coord, board_state: List[List[int]], empty_set: Set[Coord], to_move: Colour
    ):
        self._apply_move(board_state, move, to_move, empty_set)
        next_player = Colour.opposite(to_move)
        new_hash = self._board_hash(board_state, next_player)
        if self.root:
            for child in self.root.children:
                if child.move == move:
                    child.parent = None
                    child.state_hash = new_hash
                    self.root = child
                    break
            else:
                unexp, priors = self._ordered_moves(board_state, empty_set, next_player)
                self.root = HexMastUltra.Node(
                    move=move,
                    parent=None,
                    player_to_move=next_player,
                    state_hash=new_hash,
                    unexpanded=unexp,
                    priors=priors,
                )
        else:
            unexp, priors = self._ordered_moves(board_state, empty_set, next_player)
            self.root = HexMastUltra.Node(
                move=move,
                parent=None,
                player_to_move=next_player,
                state_hash=new_hash,
                unexpanded=unexp,
                priors=priors,
            )
        self._undo_moves(board_state, [move], empty_set)

    def _fallback_move(self, empty_set: Set[Coord]) -> Coord:
        if not empty_set:
            return (0, 0)
        mid = (self.size - 1) // 2
        preferred = (mid, mid)
        if preferred in empty_set:
            return preferred
        return next(iter(empty_set))

    def _time_budget(self, turn: int, empties_remaining: int) -> float:
        total_cells = self.size * self.size
        progress = 1.0 - (empties_remaining / total_cells)
        if progress < 0.25:
            return 1.5
        if progress < 0.65:
            return 2.5
        return 0.9

    def _should_swap(self, turn: int, board: Board, opp_move: Move | None) -> bool:
        if turn != 2 or self.colour != Colour.BLUE or opp_move is None or opp_move.is_swap():
            return False
        center = (self.size - 1) / 2.0
        strength = abs(opp_move.x - center) + abs(opp_move.y - center)
        threshold = (self.size / 3.0)
        return strength <= threshold
