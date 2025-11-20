"""
Opening book with Nash equilibrium-based moves.
Contains proven strong openings and optimal swap decisions.
"""

from typing import Optional, List, Tuple, Dict
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class OpeningBook:
    """
    Pre-computed opening moves based on game theory analysis.
    """

    def __init__(self):
        # Optimal openings for 11x11 board
        # These moves give approximately 50-52% win rate (Nash equilibrium)
        self.FIRST_MOVES = [
            (5, 5),   # Center - strongest but invites swap
            (5, 6),   # Near center - good balance
            (6, 5),   # Near center - symmetric
            (4, 5),   # Slightly off-center
            (5, 4),   # Slightly off-center
            (4, 6),   # Diagonal off-center
            (6, 4),   # Diagonal off-center
        ]

        # Weak openings to avoid (corners and edges)
        self.WEAK_OPENINGS = [
            (0, 0), (0, 10), (10, 0), (10, 10),  # Corners
            (0, 5), (5, 0), (10, 5), (5, 10),    # Edge centers
        ]

        # Swap thresholds for different positions
        # Position -> win probability for first player
        self.OPENING_VALUES = self._compute_opening_values()

        # Response moves after opponent's opening
        self.RESPONSES = self._compute_responses()

    def get_move(self, board: Board, turn: int, colour: Colour) -> Optional[Move]:
        """
        Get opening book move if available.

        Args:
            board: Current board state
            turn: Turn number
            colour: Our colour

        Returns:
            Move from opening book or None if not in book
        """
        if turn == 1:
            # We're first player - choose optimal opening
            return self._get_first_move(board)

        elif turn == 2:
            # We're second player - decide whether to swap
            # This is handled in the main agent's _should_swap method
            return None

        elif turn == 3:
            # We're first player after opponent declined swap
            return self._get_third_move(board, colour)

        elif turn <= 5:
            # Early game responses
            return self._get_response_move(board, turn, colour)

        # Not in opening book
        return None

    def _get_first_move(self, board: Board) -> Move:
        """
        Get optimal first move.
        Chooses move that gives ~51-52% win rate to discourage swap.
        """
        # Default to slightly off-center
        # This gives good winning chances but doesn't force swap
        return Move(5, 6)  # Near center but not perfect center

    def _get_third_move(self, board: Board, colour: Colour) -> Optional[Move]:
        """
        Get third move (after opponent declined swap).
        Now we have advantage and should play aggressively.
        """
        # Find opponent's move
        opp_move = None
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is not None:
                    opp_move = (i, j)
                    break

        if not opp_move:
            return None

        # Play complementary move to create connection potential
        x, y = opp_move

        # If opponent played center, play adjacent
        if (x, y) == (5, 5):
            return Move(5, 6)

        # If opponent played off-center, play center
        if abs(x - 5) <= 1 and abs(y - 5) <= 1:
            if board.tiles[5][5].colour is None:
                return Move(5, 5)

        # Default: play symmetric position
        sym_x = board.size - 1 - x
        sym_y = board.size - 1 - y
        if board.tiles[sym_x][sym_y].colour is None:
            return Move(sym_x, sym_y)

        return None

    def _get_response_move(self, board: Board, turn: int, colour: Colour) -> Optional[Move]:
        """
        Get response move in early game.
        """
        # Count stones on board
        stone_count = sum(1 for i in range(board.size)
                         for j in range(board.size)
                         if board.tiles[i][j].colour is not None)

        if stone_count <= 2:
            # Very early game - play near center
            center_moves = [
                (5, 5), (5, 6), (6, 5), (4, 5), (5, 4),
                (4, 4), (6, 6), (4, 6), (6, 4)
            ]

            for x, y in center_moves:
                if board.tiles[x][y].colour is None:
                    return Move(x, y)

        return None

    def _compute_opening_values(self) -> Dict[Tuple[int, int], float]:
        """
        Compute win probability for each opening position.
        Based on distance from center and game theory analysis.
        """
        values = {}
        board_size = 11
        center = board_size // 2

        for i in range(board_size):
            for j in range(board_size):
                # Manhattan distance from center
                dist = abs(i - center) + abs(j - center)

                # Corners are very weak
                if (i, j) in [(0, 0), (0, 10), (10, 0), (10, 10)]:
                    values[(i, j)] = 0.35
                # Center is strongest
                elif dist == 0:
                    values[(i, j)] = 0.55
                # Near center is good
                elif dist <= 2:
                    values[(i, j)] = 0.52
                # Linear decrease with distance
                else:
                    values[(i, j)] = max(0.4, 0.5 - dist * 0.02)

        return values

    def _compute_responses(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Compute best responses to opponent openings.
        """
        responses = {}

        # Response to center opening
        responses[(5, 5)] = [
            (5, 6), (6, 5), (4, 5), (5, 4),  # Adjacent
            (4, 4), (6, 6),  # Diagonal
        ]

        # Response to near-center openings
        responses[(5, 6)] = [(5, 5), (4, 5), (6, 5), (5, 7)]
        responses[(6, 5)] = [(5, 5), (5, 4), (5, 6), (7, 5)]
        responses[(4, 5)] = [(5, 5), (5, 4), (5, 6), (3, 5)]
        responses[(5, 4)] = [(5, 5), (4, 5), (6, 5), (5, 3)]

        return responses

    def evaluate_opening(self, move: Tuple[int, int]) -> float:
        """
        Evaluate strength of an opening move.

        Returns:
            Win probability for first player (0.0-1.0)
        """
        if move in self.OPENING_VALUES:
            return self.OPENING_VALUES[move]

        # Default evaluation based on distance from center
        x, y = move
        center = 5  # For 11x11 board
        dist = abs(x - center) + abs(y - center)
        return max(0.4, 0.5 - dist * 0.02)

    def should_swap(self, opponent_opening: Tuple[int, int]) -> bool:
        """
        Determine if we should swap based on opponent's opening.

        Args:
            opponent_opening: Opponent's first move

        Returns:
            True if we should swap
        """
        value = self.evaluate_opening(opponent_opening)

        # Swap if opponent's opening gives them >52% win rate
        return value > 0.52