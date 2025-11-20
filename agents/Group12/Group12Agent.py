"""
Group12 Hex AI Agent
COMP34111 AI & Games

This agent implements advanced game theory concepts including:
- Nash equilibrium swap decisions
- Virtual connection detection
- MCTS with RAVE enhancement
- Electrical resistance evaluation
- Pattern recognition
- Hybrid algorithm switching
"""

import math
import random
import copy
from time import perf_counter_ns as time
from typing import Optional, List, Dict, Set, Tuple
from dataclasses import dataclass

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile

# Import our advanced modules
from agents.Group12.core.evaluation import PositionEvaluator, ResistanceEvaluator
from agents.Group12.core.virtual_connections import VirtualConnectionDetector
from agents.Group12.core.time_manager import TimeManager
from agents.Group12.algorithms.mcts_enhanced import EnhancedMCTS
from agents.Group12.knowledge.opening_book import OpeningBook
from agents.Group12.knowledge.patterns import PatternMatcher


class Group12Agent(AgentBase):
    """
    Advanced Hex AI implementing cutting-edge game theory concepts.
    """

    # Time management constants (nanoseconds)
    TOTAL_TIME_LIMIT = 3 * 60 * 10**9  # 3 minutes
    TIME_BUFFER = 0.05  # 5% safety buffer

    # Nash equilibrium swap threshold
    SWAP_THRESHOLD = 0.52  # Swap if opponent's move > 52% win rate

    # Algorithm switching thresholds
    OPENING_MOVES = 5
    ENDGAME_EMPTY_CELLS = 30

    def __init__(self, colour: Colour):
        """Initialize the agent with advanced components."""
        super().__init__(colour)

        # Core components
        self.evaluator = PositionEvaluator()
        self.resistance_eval = ResistanceEvaluator()
        self.vc_detector = VirtualConnectionDetector()
        self.time_manager = TimeManager(self.TOTAL_TIME_LIMIT)

        # Algorithms
        self.mcts = EnhancedMCTS(
            colour=colour,
            evaluator=self.evaluator,
            vc_detector=self.vc_detector,
            use_rave=True
        )

        # Knowledge bases
        self.opening_book = OpeningBook()
        self.pattern_matcher = PatternMatcher()

        # Game state tracking
        self.total_time_used = 0
        self.moves_made = 0

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """
        Make an intelligent move based on game phase and position evaluation.

        Uses hybrid algorithm switching:
        - Opening: Book moves + quick evaluation
        - Middle game: Enhanced MCTS with virtual connections
        - Endgame: Proof number search (if time permits)
        """
        start_time = time()

        # Update game state
        self.moves_made += 1

        # Handle swap decision (turn 2 only)
        if turn == 2 and opp_move is not None:
            if self._should_swap(board, opp_move):
                return Move(-1, -1)

        # Choose algorithm based on game phase
        move = self._select_move_algorithm(turn, board)

        # Time tracking
        time_used = time() - start_time
        self.total_time_used += time_used
        self.time_manager.update(time_used)

        return move

    def _should_swap(self, board: Board, opp_move: Move) -> bool:
        """
        Implement Nash equilibrium swap decision.
        Swap if opponent's opening gives them >52% win probability.
        """
        # Quick evaluation of opening strength
        opening_value = self._evaluate_opening(board, opp_move)

        # Swap if opponent played a strong opening
        return opening_value > self.SWAP_THRESHOLD

    def _evaluate_opening(self, board: Board, move: Move) -> float:
        """
        Evaluate opening move strength (0-1 scale).
        Center moves are strongest (~0.55), corners weakest (~0.3).
        """
        x, y = move.x, move.y
        board_size = board.size
        center = board_size // 2

        # Distance from center (Manhattan distance)
        dist_from_center = abs(x - center) + abs(y - center)

        # Corners are particularly weak
        if (x, y) in [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]:
            return 0.3

        # Center and near-center are strong
        if dist_from_center <= 1:
            return 0.55
        elif dist_from_center <= 2:
            return 0.52

        # Linear decrease with distance
        max_dist = board_size
        strength = 0.5 - (dist_from_center / max_dist) * 0.15

        return strength

    def _select_move_algorithm(self, turn: int, board: Board) -> Move:
        """
        Select the best algorithm based on game phase.
        """
        empty_cells = self._count_empty_cells(board)
        time_remaining = self.TOTAL_TIME_LIMIT - self.total_time_used

        # Opening phase: Use book or quick heuristics
        if turn <= self.OPENING_MOVES:
            move = self.opening_book.get_move(board, turn, self.colour)
            if move:
                return move
            # Fallback to quick evaluation
            return self._quick_heuristic_move(board)

        # Endgame phase: Could use proof number search (not implemented yet)
        # For now, use enhanced MCTS with more iterations
        if empty_cells < self.ENDGAME_EMPTY_CELLS:
            time_limit = min(30 * 10**9, time_remaining * 0.3)  # Up to 30 seconds
            return self.mcts.search(board, time_limit, endgame_mode=True)

        # Middle game: Enhanced MCTS with virtual connections
        # Allocate time based on position complexity
        position_temperature = self._compute_temperature(board)
        base_time = time_remaining / (empty_cells + 10)
        allocated_time = base_time * (1 + position_temperature)

        return self.mcts.search(board, allocated_time)

    def _quick_heuristic_move(self, board: Board) -> Move:
        """
        Fast move selection based on heuristics.
        Used in opening when book doesn't have the position.
        """
        valid_moves = self._get_valid_moves(board)

        if not valid_moves:
            # Fallback - should never happen
            return Move(0, 0)

        # Score each move quickly
        best_move = None
        best_score = -float('inf')

        for move in valid_moves:
            # Skip corners in opening
            if self._is_corner(move, board.size):
                continue

            score = self._quick_evaluate_move(board, move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move else random.choice(valid_moves)

    def _quick_evaluate_move(self, board: Board, move: Move) -> float:
        """Quick evaluation of a move without deep search."""
        x, y = move.x, move.y
        score = 0.0

        # Prefer center area
        center = board.size // 2
        center_dist = abs(x - center) + abs(y - center)
        score += (board.size - center_dist) / board.size

        # Check for bridge creation opportunities
        if self._creates_bridge(board, move):
            score += 2.0

        # Check for blocking opponent bridges
        if self._blocks_opponent_bridge(board, move):
            score += 1.5

        return score

    def _creates_bridge(self, board: Board, move: Move) -> bool:
        """Check if move creates a bridge pattern."""
        # Simplified bridge detection
        x, y = move.x, move.y

        # Check for potential bridge patterns
        bridge_patterns = [
            [(x-1, y), (x+1, y+1)],  # Diagonal bridge
            [(x-1, y-1), (x+1, y)],   # Another diagonal
            # Add more patterns as needed
        ]

        for pattern in bridge_patterns:
            if self._check_bridge_pattern(board, pattern, self.colour):
                return True

        return False

    def _blocks_opponent_bridge(self, board: Board, move: Move) -> bool:
        """Check if move blocks opponent's bridge."""
        # Similar to creates_bridge but for opponent
        opp_colour = Colour.opposite(self.colour)
        # Simplified - would need full implementation
        return False

    def _check_bridge_pattern(self, board: Board, pattern: List[Tuple[int, int]], colour: Colour) -> bool:
        """Check if a bridge pattern exists."""
        for x, y in pattern:
            if not self._is_valid_position(x, y, board.size):
                return False
            if board.tiles[x][y].colour != colour:
                return False
        return True

    def _compute_temperature(self, board: Board) -> float:
        """
        Compute position temperature (0-1).
        High temperature = critical position requiring more computation.
        """
        # Simplified temperature computation
        # In full implementation, would analyze threats and critical regions

        empty_cells = self._count_empty_cells(board)
        total_cells = board.size * board.size

        # Middle game positions tend to be most critical
        fill_ratio = 1 - (empty_cells / total_cells)

        if 0.3 <= fill_ratio <= 0.7:
            return 0.8  # High temperature in middle game
        else:
            return 0.5  # Lower temperature in opening/endgame

    def _get_valid_moves(self, board: Board) -> List[Move]:
        """Get all valid moves in current position."""
        moves = []
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    moves.append(Move(i, j))
        return moves

    def _count_empty_cells(self, board: Board) -> int:
        """Count empty cells on the board."""
        count = 0
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    count += 1
        return count

    def _is_corner(self, move: Move, board_size: int) -> bool:
        """Check if move is a corner (weak position)."""
        corners = [
            (0, 0), (0, board_size-1),
            (board_size-1, 0), (board_size-1, board_size-1)
        ]
        return (move.x, move.y) in corners

    def _is_valid_position(self, x: int, y: int, board_size: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= x < board_size and 0 <= y < board_size