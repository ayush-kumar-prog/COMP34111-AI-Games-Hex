"""
Group12 Hex AI Agent - Simplified version without external dependencies
COMP34111 AI & Games

This is a working version that doesn't require scipy/numpy.
"""

import math
import random
import copy
from time import perf_counter_ns as time
from typing import Optional, List

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class Group12Agent(AgentBase):
    """
    Advanced Hex AI implementing game theory concepts.
    Simplified version for testing.
    """

    def __init__(self, colour: Colour):
        """Initialize the agent."""
        super().__init__(colour)
        self.total_time_used = 0
        self.moves_made = 0

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """Make an intelligent move based on game phase."""
        start_time = time()

        # Handle swap decision (turn 2 only)
        if turn == 2 and opp_move is not None:
            if self._should_swap(board, opp_move):
                return Move(-1, -1)

        # Get valid moves
        valid_moves = self._get_valid_moves(board)

        if not valid_moves:
            # Should never happen
            return Move(0, 0)

        # Choose best move based on simple heuristics
        if turn == 1:
            # First move - play near center
            return Move(5, 5) if board.size == 11 else Move(2, 2)

        # Score each move
        best_move = None
        best_score = -float('inf')

        for move in valid_moves:
            score = self._evaluate_move(board, move)
            if score > best_score:
                best_score = score
                best_move = move

        # Update time tracking
        self.total_time_used += time() - start_time
        self.moves_made += 1

        return best_move if best_move else random.choice(valid_moves)

    def _should_swap(self, board: Board, opp_move: Move) -> bool:
        """Nash equilibrium swap decision."""
        x, y = opp_move.x, opp_move.y
        center = board.size // 2

        # Distance from center
        dist_from_center = abs(x - center) + abs(y - center)

        # Swap if opponent played very strong opening (center or adjacent)
        return dist_from_center <= 1

    def _evaluate_move(self, board: Board, move: Move) -> float:
        """Simple evaluation of a move."""
        x, y = move.x, move.y
        score = 0.0

        # Prefer center area
        center = board.size // 2
        center_dist = abs(x - center) + abs(y - center)
        score += (board.size - center_dist) / board.size * 10

        # Avoid corners
        if self._is_corner(move, board.size):
            score -= 20

        # Check for connections
        for dx, dy in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board.size and 0 <= ny < board.size:
                if board.tiles[nx][ny].colour == self.colour:
                    score += 5  # Adjacent to our stone
                elif board.tiles[nx][ny].colour == Colour.opposite(self.colour):
                    score += 2  # Block opponent

        # Prefer moves closer to winning edges
        if self.colour == Colour.RED:
            # RED connects top-bottom
            edge_dist = min(x, board.size - 1 - x)
            score += (board.size - edge_dist) / board.size * 5
        else:
            # BLUE connects left-right
            edge_dist = min(y, board.size - 1 - y)
            score += (board.size - edge_dist) / board.size * 5

        return score + random.random() * 0.1  # Small random factor

    def _get_valid_moves(self, board: Board) -> List[Move]:
        """Get all valid moves."""
        moves = []
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    moves.append(Move(i, j))
        return moves

    def _is_corner(self, move: Move, board_size: int) -> bool:
        """Check if move is a corner."""
        corners = [
            (0, 0), (0, board_size-1),
            (board_size-1, 0), (board_size-1, board_size-1)
        ]
        return (move.x, move.y) in corners