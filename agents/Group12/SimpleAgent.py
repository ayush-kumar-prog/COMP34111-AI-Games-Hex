from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import random

class SimpleAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.rng = random.Random()

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        size = board.size
        legal = []
        for y in range(size):
            for x in range(size):
                if board.tiles[y][x].colour is None:
                    legal.append((x, y))
        if not legal:
            return Move(0, 0)
        x, y = self.rng.choice(legal)
        return Move(x, y)
