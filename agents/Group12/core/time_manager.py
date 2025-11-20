"""
Intelligent time management for optimal resource allocation.
"""

from typing import Optional


class TimeManager:
    """
    Manages time allocation throughout the game.
    Ensures we never timeout while using time optimally.
    """

    def __init__(self, total_time: int, buffer_ratio: float = 0.05):
        """
        Initialize time manager.

        Args:
            total_time: Total time budget in nanoseconds
            buffer_ratio: Safety buffer ratio (default 5%)
        """
        self.total_time = total_time
        self.buffer_ratio = buffer_ratio
        self.usable_time = total_time * (1 - buffer_ratio)
        self.time_used = 0
        self.moves_made = 0

        # Phase thresholds
        self.OPENING_MOVES = 5
        self.ENDGAME_EMPTY_CELLS = 30

    def update(self, time_spent: int):
        """Update time tracking after a move."""
        self.time_used += time_spent
        self.moves_made += 1

    def get_remaining_time(self) -> int:
        """Get remaining usable time (excluding buffer)."""
        return max(0, self.usable_time - self.time_used)

    def allocate_time(self, turn: int, empty_cells: int, temperature: float = 0.5) -> int:
        """
        Allocate time for current move based on game phase and position complexity.

        Args:
            turn: Current turn number
            empty_cells: Number of empty cells on board
            temperature: Position temperature (0-1, higher = more critical)

        Returns:
            Time to allocate in nanoseconds
        """
        remaining_time = self.get_remaining_time()

        if remaining_time <= 0:
            # Emergency mode - make moves instantly
            return 10**6  # 1ms

        # Opening phase - use minimal time
        if turn <= self.OPENING_MOVES:
            return min(0.5 * 10**9, remaining_time * 0.01)  # 0.5s max

        # Estimate remaining moves
        estimated_moves_left = self._estimate_remaining_moves(empty_cells)

        if estimated_moves_left <= 0:
            return 10**6  # 1ms for safety

        # Base allocation
        base_time = remaining_time / estimated_moves_left

        # Adjust for game phase
        if empty_cells < self.ENDGAME_EMPTY_CELLS:
            # Endgame - can use more time for perfect play
            phase_multiplier = 1.5
        elif empty_cells > 100:
            # Early game - quick moves
            phase_multiplier = 0.5
        else:
            # Middle game - standard
            phase_multiplier = 1.0

        # Adjust for position complexity (temperature)
        complexity_multiplier = 0.5 + temperature  # Range: 0.5-1.5

        # Final allocation
        allocated_time = base_time * phase_multiplier * complexity_multiplier

        # Apply caps
        min_time = 0.1 * 10**9  # 0.1 second minimum
        max_time = min(30 * 10**9, remaining_time * 0.3)  # 30s or 30% of remaining

        return int(max(min_time, min(max_time, allocated_time)))

    def _estimate_remaining_moves(self, empty_cells: int) -> int:
        """
        Estimate number of moves remaining in the game.
        Hex games typically last about 50-70% of total cells.
        """
        # Assume game will last about 60% of board capacity
        # And moves alternate between players
        estimated_total_moves = (121 * 0.6) / 2  # For 11x11 board
        moves_left = max(1, estimated_total_moves - self.moves_made)
        return int(moves_left)

    def should_use_emergency_mode(self) -> bool:
        """Check if we should switch to emergency (fast) mode."""
        remaining_ratio = self.get_remaining_time() / self.usable_time

        # Emergency if less than 5% time remaining
        return remaining_ratio < 0.05

    def get_time_pressure(self) -> float:
        """
        Get current time pressure (0-1).
        Higher value = more time pressure.
        """
        remaining_ratio = self.get_remaining_time() / self.usable_time
        return 1.0 - remaining_ratio

    def can_afford_deep_search(self, estimated_time: int) -> bool:
        """Check if we can afford a deep search."""
        remaining_time = self.get_remaining_time()
        # Can afford if search uses less than 20% of remaining time
        return estimated_time < remaining_time * 0.2