import math
import random
import time
from typing import List, Optional, Tuple

from src.AgentBase import AgentBase
from src.Move import Move
from src.Colour import Colour
from src.Board import Board


Coord = Tuple[int, int]


class GameState:
    """
    Internal game state for MCTS simulations.
    Board cells hold None / Colour.RED / Colour.BLUE.
    """

    def __init__(self, size: int, board: Optional[List[List[Optional[Colour]]]] = None,
                 next_player: Colour = Colour.RED):
        self.size = size
        if board is None:
            self.board: List[List[Optional[Colour]]] = [[None for _ in range(size)] for _ in range(size)]
        else:
            self.board = [row[:] for row in board]

        self.next_player = next_player

    @staticmethod
    def from_board_object(board: Board, my_colour: Colour) -> "GameState":
        tiles_2d = board.tiles
        size = board.size

        internal_board: List[List[Optional[Colour]]] = [
            [None for _ in range(size)] for _ in range(size)
        ]

        for x in range(size):
            for y in range(size):
                tile = tiles_2d[x][y]
                tile_colour = tile.colour
                internal_board[y][x] = tile_colour

        next_player = my_colour
        return GameState(size=size, board=internal_board, next_player=next_player)

    def clone(self) -> "GameState":
        return GameState(self.size, self.board, self.next_player)

    def get_legal_moves(self) -> List[Coord]:
        moves: List[Coord] = []
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] is None:
                    moves.append((x, y))
        return moves

    def play_move(self, move: Coord):
        x, y = move
        assert self.board[y][x] is None, "Illegal move in GameState"
        self.board[y][x] = self.next_player
        self.next_player = Colour.RED if self.next_player == Colour.BLUE else Colour.BLUE

    def is_terminal(self) -> bool:
        if self.has_won(Colour.RED) or self.has_won(Colour.BLUE):
            return True
        for row in self.board:
            for c in row:
                if c is None:
                    return False
        return True

    def winner(self) -> Optional[Colour]:
        if self.has_won(Colour.RED):
            return Colour.RED
        if self.has_won(Colour.BLUE):
            return Colour.BLUE
        return None

    def has_won(self, colour: Colour) -> bool:
        n = self.size
        visited = set()
        stack = []

        def neighbours(x: int, y: int):
            cand = [
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
                (x - 1, y + 1),
                (x + 1, y - 1),
            ]
            for nx, ny in cand:
                if 0 <= nx < n and 0 <= ny < n:
                    yield nx, ny

        if colour == Colour.RED:
            for x in range(n):
                if self.board[0][x] == colour:
                    stack.append((x, 0))
                    visited.add((x, 0))

            target_row = n - 1
            while stack:
                x, y = stack.pop()
                if y == target_row:
                    return True
                for nx, ny in neighbours(x, y):
                    if (nx, ny) not in visited and self.board[ny][nx] == colour:
                        visited.add((nx, ny))
                        stack.append((nx, ny))

        else:
            for y in range(n):
                if self.board[y][0] == colour:
                    stack.append((0, y))
                    visited.add((0, y))

            target_col = n - 1
            while stack:
                x, y = stack.pop()
                if x == target_col:
                    return True
                for nx, ny in neighbours(x, y):
                    if (nx, ny) not in visited and self.board[ny][nx] == colour:
                        visited.add((nx, ny))
                        stack.append((nx, ny))

        return False


class MCTSNode:
    def __init__(self, parent: Optional["MCTSNode"], move: Optional[Coord]):
        self.parent: Optional["MCTSNode"] = parent
        self.move: Optional[Coord] = move
        self.children: List["MCTSNode"] = []
        self.untried_moves: List[Coord] = []
        self.visits: int = 0
        self.wins: float = 0.0  # from root player's perspective

    def select_child_uct(self, c_param: float = math.sqrt(2.0)) -> "MCTSNode":
        """Select child that maximises UCT value."""
        best = None
        best_value = -1e9
        for child in self.children:
            if child.visits == 0:
                uct_value = float("inf")
            else:
                exploit = child.wins / child.visits
                explore = c_param * math.sqrt(math.log(self.visits + 1) / child.visits)
                uct_value = exploit + explore

            if uct_value > best_value:
                best_value = uct_value
                best = child
        return best

    def add_child(self, move: Coord, state: GameState) -> "MCTSNode":
        child = MCTSNode(parent=self, move=move)
        child.untried_moves = state.get_legal_moves()
        self.children.append(child)
        self.untried_moves.remove(move)
        return child

    def update(self, result: float):
        self.visits += 1
        self.wins += result


class BasicMCTSAgent(AgentBase):
    """
    Slightly improved MCTS agent:
      - Plain UCT
      - Own GameState + winner detection
      - Simple constant time budget per move
      - Simple pie-rule heuristic for swapping.
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.colour = colour
        self.opponent_colour = Colour.RED if colour == Colour.BLUE else Colour.BLUE
        self.rng = random.Random()
        self.per_move_time_limit = 0.2  # seconds per decision (tune this)
        self.is_first_move = True       # to handle pie rule

    # ------------- Pie rule handling -------------

    def _should_swap(self, board: Board, opponent_move: Optional[Move]) -> bool:
        """
        Simple heuristic: if we're the second player and opponent's first
        move is very central, swap.
        """
        # If no opponent move (we're opening as Red), cannot swap.
        if opponent_move is None:
            return False

        # Only meaningful on our first move
        if not self.is_first_move:
            return False

        # We can only swap when we are the second player.
        # In this engine, RED starts; so if we are BLUE, we're second.
        if self.colour != Colour.BLUE:
            return False

        x, y = opponent_move.x, opponent_move.y

        tiles = board.tiles
        size = board.size
        cx = cy = size // 2

        # Manhattan distance from centre
        dist = abs(x - cx) + abs(y - cy)
        # Threshold: if they played very centrally, it's strong → we swap.
        return dist <= 2

    # ------------- Core MCTS -------------

    def make_move(self, turn: int, board: Board, opponent_move: Optional[Move]) -> Move:
        """
        Main entrypoint called by the engine.
        Must return a Move object.
        """
        # Pie rule: maybe swap on our very first move as second player
        if self._should_swap(board, opponent_move):
            # After swapping, we won't be asked again in this game, but
            # we still mark first-move flag as used.
            self.is_first_move = False
            return Move(-1, -1)

        # Build internal game state from the Board
        state = GameState.from_board_object(board, self.colour)

        root = MCTSNode(parent=None, move=None)
        root.untried_moves = state.get_legal_moves()

        if not root.untried_moves:
            # No legal moves → should not really happen, but be safe.
            self.is_first_move = False
            return Move(0, 0)

        time_deadline = time.time() + self.per_move_time_limit

        while time.time() < time_deadline:
            # 1. SELECTION
            node = root
            sim_state = state.clone()

            # Traverse the tree until a node with untried moves or a leaf
            while not node.untried_moves and node.children:
                node = node.select_child_uct()
                if node.move is not None:
                    sim_state.play_move(node.move)

            # 2. EXPANSION
            if node.untried_moves and not sim_state.is_terminal():
                move = self.rng.choice(node.untried_moves)
                sim_state.play_move(move)
                node = node.add_child(move, sim_state)

            # 3. ROLLOUT
            # Random playout to the end
            while not sim_state.is_terminal():
                legal = sim_state.get_legal_moves()
                if not legal:
                    break
                move = self.rng.choice(legal)
                sim_state.play_move(move)

            winner = sim_state.winner()

            # 4. BACKPROPAGATION
            # Reward from our perspective: 1 if we won, 0 if lost, 0.5 otherwise.
            if winner is None:
                result = 0.5
            elif winner == self.colour:
                result = 1.0
            else:
                result = 0.0

            while node is not None:
                node.update(result)
                node = node.parent

        # Choose child with highest visit count
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            best_move = best_child.move
        else:
            # Fallback: random legal move
            best_move = self.rng.choice(root.untried_moves)

        self.is_first_move = False

        x, y = best_move
        return Move(x, y)
