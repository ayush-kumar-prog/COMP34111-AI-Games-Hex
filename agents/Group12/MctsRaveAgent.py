from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

import random
import math

class InternalHexBoard:
    def __init__(self, size: int, tiles=None):
        self.size = size
        self.n = size * size
        self.cells = [0] * self.n
        if tiles is not None:
            self._from_tiles(tiles)

    def _idx(self, x: int, y: int) -> int:
        return y * self.size + x

    def _from_tiles(self, tiles):
        for r in range(self.size):
            for c in range(self.size):
                t = tiles[r][c]
                if t.colour is None:
                    self.cells[self._idx(c, r)] = 0
                elif t.colour == Colour.RED:
                    self.cells[self._idx(c, r)] = 1
                else:
                    self.cells[self._idx(c, r)] = 2

    def copy(self):
        b = InternalHexBoard(self.size)
        b.cells = self.cells[:]
        return b

    def legal_moves(self):
        s = self.size
        return [(x, y)
                for y in range(s)
                for x in range(s)
                if self.cells[self._idx(x, y)] == 0]

    def play(self, x: int, y: int, player: int):
        self.cells[self._idx(x, y)] = player

    def neighbours(self, x: int, y: int):
        s = self.size
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < s and 0 <= ny < s:
                yield nx, ny

    def winner(self) -> int:
        s = self.size

        # RED (top-bottom)
        visited = [[False] * s for _ in range(s)]
        stack = []
        for x in range(s):
            if self.cells[self._idx(x, 0)] == 1:
                stack.append((x, 0))
                visited[0][x] = True
        while stack:
            x, y = stack.pop()
            if y == s - 1:
                return 1
            for nx, ny in self.neighbours(x, y):
                if not visited[ny][nx] and self.cells[self._idx(nx, ny)] == 1:
                    visited[ny][nx] = True
                    stack.append((nx, ny))

        # BLUE (left-right)
        visited = [[False] * s for _ in range(s)]
        stack = []
        for y in range(s):
            if self.cells[self._idx(0, y)] == 2:
                stack.append((0, y))
                visited[y][0] = True
        while stack:
            x, y = stack.pop()
            if x == s - 1:
                return 2
            for nx, ny in self.neighbours(x, y):
                if not visited[ny][nx] and self.cells[self._idx(nx, ny)] == 2:
                    visited[ny][nx] = True
                    stack.append((nx, ny))

        return 0


class MCTSNodeRAVE:
    __slots__ = ("parent", "children", "move", "player_to_move",
                 "wins", "visits", "untried_moves",
                 "rave_wins", "rave_visits")

    def __init__(self, parent, move, player_to_move, untried_moves):
        self.parent = parent
        self.children = []
        self.move = move
        self.player_to_move = player_to_move
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = list(untried_moves)
        # RAVE: statistics per child move, stored as dict move -> (wins, visits)
        self.rave_wins = {}   # (x,y) -> float
        self.rave_visits = {} # (x,y) -> int

    def _get_rave(self, move):
        return self.rave_wins.get(move, 0.0), self.rave_visits.get(move, 0)

    def uct_select_child(self, c: float = 1.4, c_beta: float = 1.0):
        log_N = math.log(self.visits + 1)
        best, best_score = None, -1e9
        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                # standard value
                q_uct = child.wins / child.visits
                # RAVE value
                rw, rv = self._get_rave(child.move)
                q_rave = rw / rv if rv > 0 else 0.5
                # blend
                beta = rv / (child.visits + rv + 4 * child.visits * rv * c_beta)
                q_mix = (1 - beta) * q_uct + beta * q_rave
                # UCB
                score = q_mix + c * math.sqrt(log_N / child.visits)
            if score > best_score:
                best, best_score = child, score
        return best

    def add_child(self, move, next_player, untried_moves):
        child = MCTSNodeRAVE(self, move, next_player, untried_moves)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child

    def update(self, result_for_root: float):
        self.visits += 1
        self.wins += result_for_root


class MctsRaveAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.colour = colour
        self.my_int = 1 if colour == Colour.RED else 2
        self.opp_int = 2 if self.my_int == 1 else 1
        self.simulations_per_move = 300
        self.rng = random.Random()

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        size = board.size

        # Pie rule: only BLUE can swap on its first move
        if self.colour == Colour.BLUE and turn == 2 and opp_move is not None:
            cx = cy = (size - 1) / 2
            dist = abs(opp_move.x - cx) + abs(opp_move.y - cy)
            if dist <= 2:
                return Move(-1, -1)

        internal = InternalHexBoard(size, board.tiles)
        cand = self._mcts_rave_choose_move(internal)
        x, y = self._best_legal_or_fallback(board, cand)
        return Move(x, y)

    def _is_legal(self, board: Board, x: int, y: int) -> bool:
        s = board.size
        if not (0 <= x < s and 0 <= y < s):
            return False
        return board.tiles[y][x].colour is None

    def _best_legal_or_fallback(self, board: Board, cand) -> tuple[int, int]:
        if cand is not None:
            x, y = cand
            if self._is_legal(board, x, y):
                return (x, y)
        s = board.size
        for r in range(s):
            for c in range(s):
                if board.tiles[r][c].colour is None:
                    return (c, r)
        return (0, 0)

    def _mcts_rave_choose_move(self, board: InternalHexBoard):
        root_player = self.my_int
        legal = board.legal_moves()
        if not legal:
            return None

        root = MCTSNodeRAVE(
            parent=None,
            move=None,
            player_to_move=root_player,
            untried_moves=legal
        )

        for _ in range(self.simulations_per_move):
            node = root
            state = board.copy()
            player = root_player
            playout_moves = []  # record moves in this simulation

            # Selection
            while not node.untried_moves and node.children:
                node = node.uct_select_child()
                if node.move is not None:
                    x, y = node.move
                    state.play(x, y, player)
                    playout_moves.append((player, node.move))
                    player = 1 if player == 2 else 2
                    if state.winner() != 0:
                        break

            # Expansion
            if node.untried_moves and state.winner() == 0:
                move = self.rng.choice(node.untried_moves)
                x, y = move
                state.play(x, y, player)
                playout_moves.append((player, move))
                next_player = 1 if player == 2 else 2
                node = node.add_child(move, next_player, state.legal_moves())
                player = next_player

            # Simulation
            winner = state.winner()
            rollout_player = player
            while winner == 0:
                moves = state.legal_moves()
                if not moves:
                    break
                mx, my = self.rng.choice(moves)
                state.play(mx, my, rollout_player)
                playout_moves.append((rollout_player, (mx, my)))
                rollout_player = 1 if rollout_player == 2 else 2
                winner = state.winner()

            # Result
            if winner == 0:
                result = 0.5
            elif winner == root_player:
                result = 1.0
            else:
                result = 0.0

            # Backprop with RAVE
            node_bp = node
            while node_bp is not None:
                node_bp.update(result)
                # RAVE: update for all moves by root_player in this playout
                for p, mv in playout_moves:
                    if p == root_player:
                        rw = node_bp.rave_wins.get(mv, 0.0)
                        rv = node_bp.rave_visits.get(mv, 0)
                        node_bp.rave_wins[mv] = rw + result
                        node_bp.rave_visits[mv] = rv + 1
                node_bp = node_bp.parent

        if not root.children:
            return self.rng.choice(legal)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
