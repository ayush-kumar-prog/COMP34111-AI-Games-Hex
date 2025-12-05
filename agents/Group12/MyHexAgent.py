# agents/Group12/MyHexAgent.py

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

import random
import time
import math


# ---------- Lightweight internal Hex board ----------

class InternalHexBoard:
    """
    Simple k x k Hex board using ints:
      0 = empty, 1 = RED, 2 = BLUE.
    Knows how to:
      - copy itself
      - list legal moves
      - play a move
      - detect winner (via DFS)
    """

    def __init__(self, size: int, tiles=None):
        self.size = size
        self.n = size * size
        self.cells = [0] * self.n
        if tiles is not None:
            self._from_tiles(tiles)

    def _idx(self, x: int, y: int) -> int:
        return y * self.size + x

    def _from_tiles(self, tiles):
        # tiles is board.tiles: tiles[row][col]
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
        # player: 1 (RED) or 2 (BLUE)
        self.cells[self._idx(x, y)] = player

    def neighbours(self, x: int, y: int):
        s = self.size
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < s and 0 <= ny < s:
                yield nx, ny

    def winner(self) -> int:
        """
        Return 0 if no winner, 1 if RED has a connection top-bottom,
        2 if BLUE has a connection left-right.
        """
        s = self.size

        # Check RED (top -> bottom)
        visited = [[False] * s for _ in range(s)]
        stack = []
        for x in range(s):
            if self.cells[self._idx(x, 0)] == 1:
                stack.append((x, 0))
                visited[0][x] = True

        while stack:
            x, y = stack.pop()
            if y == s - 1:
                return 1  # RED connects top to bottom
            for nx, ny in self.neighbours(x, y):
                if not visited[ny][nx] and self.cells[self._idx(nx, ny)] == 1:
                    visited[ny][nx] = True
                    stack.append((nx, ny))

        # Check BLUE (left -> right)
        visited = [[False] * s for _ in range(s)]
        stack = []
        for y in range(s):
            if self.cells[self._idx(0, y)] == 2:
                stack.append((0, y))
                visited[y][0] = True

        while stack:
            x, y = stack.pop()
            if x == s - 1:
                return 2  # BLUE connects left to right
            for nx, ny in self.neighbours(x, y):
                if not visited[ny][nx] and self.cells[self._idx(nx, ny)] == 2:
                    visited[ny][nx] = True
                    stack.append((nx, ny))

        return 0


# ---------- MCTS Node ----------

class MCTSNode:
    __slots__ = ("parent", "children", "move", "player_to_move",
                 "wins", "visits", "untried_moves")

    def __init__(self, parent, move, player_to_move, untried_moves):
        self.parent = parent              # parent node
        self.children = []                # list[MCTSNode]
        self.move = move                  # (x, y) that led here, or None for root
        self.player_to_move = player_to_move  # 1 or 2
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = list(untried_moves)

    def uct_select_child(self, c: float = 1.4):
        # UCT: exploit + explore
        log_N = math.log(self.visits + 1)
        best, best_score = None, -1e9
        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.wins / child.visits
                explore = c * math.sqrt(log_N / child.visits)
                score = exploit + explore
            if score > best_score:
                best, best_score = child, score
        return best

    def add_child(self, move, next_player, untried_moves):
        child = MCTSNode(self, move, next_player, untried_moves)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child

    def update(self, result_for_root: float):
        self.visits += 1
        self.wins += result_for_root


# ---------- Agent using MCTS ----------

class MyHexAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.colour = colour
        self.my_int = 1 if colour == Colour.RED else 2
        self.opp_int = 2 if self.my_int == 1 else 1

        # For now, just a per-move simulation cap (will later tie to 3-min limit)
        self.simulations_per_move = 400
        self.rng = random.Random()

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        size = board.size

        # Pie rule: we are second player and it's our first move (turn == 2)
        if self.colour == Colour.BLUE and turn == 2 and opp_move is not None:
            cx = cy = (board.size - 1) / 2
            dist = abs(opp_move.x - cx) + abs(opp_move.y - cy)
            if dist <= 2:
                return Move(-1, -1)  # swap

        # Build internal board
        internal = InternalHexBoard(size, board.tiles)

        # Run MCTS from this position
        best_xy = self._mcts_choose_move(internal)

        # Safety: if something goes wrong, choose first legal move
        x, y = self._best_legal_or_fallback(board, best_xy)
        return Move(x, y)


    # ---- MCTS driver ----

    def _mcts_choose_move(self, board: InternalHexBoard):
        root_player = self.my_int
        legal = board.legal_moves()
        if not legal:
            return None

        root = MCTSNode(parent=None,
                        move=None,
                        player_to_move=root_player,
                        untried_moves=legal)

        for _ in range(self.simulations_per_move):
            node = root
            state = board.copy()
            player = root_player

            # Selection
            while not node.untried_moves and node.children:
                node = node.uct_select_child()
                if node.move is not None:
                    x, y = node.move
                    state.play(x, y, player)
                    player = 1 if player == 2 else 2
                    if state.winner() != 0:
                        break

            # Expansion
            if node.untried_moves and state.winner() == 0:
                move = self.rng.choice(node.untried_moves)
                x, y = move
                state.play(x, y, player)
                next_player = 1 if player == 2 else 2
                node = node.add_child(move, next_player, state.legal_moves())

            # Simulation (rollout)
            winner = state.winner()
            rollout_player = node.player_to_move
            while winner == 0:
                moves = state.legal_moves()
                if not moves:
                    break
                mx, my = self.rng.choice(moves)
                state.play(mx, my, rollout_player)
                rollout_player = 1 if rollout_player == 2 else 2
                winner = state.winner()

            # Result for root player
            if winner == 0:
                result = 0.5  # draw – should not occur in Hex, but be safe
            elif winner == root_player:
                result = 1.0
            else:
                result = 0.0

            # Backpropagate
            while node is not None:
                node.update(result)
                node = node.parent

        # Pick child with most visits
        if not root.children:
            return self.rng.choice(legal)

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    # ---- Fallback / helper ----

    def _first_legal(self, board: Board) -> tuple[int, int]:
        s = board.size
        for r in range(s):
            for c in range(s):
                if board.tiles[r][c].colour is None:
                    return (c, r)
        return (0, 0)
    
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
