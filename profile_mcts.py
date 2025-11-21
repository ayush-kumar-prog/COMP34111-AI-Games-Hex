#!/usr/bin/env python3
"""
Profile MCTS performance to identify bottlenecks.
"""

import time
import cProfile
import pstats
from io import StringIO

from src.Board import Board
from src.Colour import Colour
from agents.Group12.Group12Agent import Group12Agent

def profile_mcts_performance():
    """Profile MCTS iterations and timing."""
    print("=" * 80)
    print("MCTS PERFORMANCE PROFILING")
    print("=" * 80)

    # Create a clean board
    board = Board(11)
    agent = Group12Agent(Colour.RED)

    # Test different game states
    test_cases = [
        ("Empty board (turn 1)", board, 1),
        ("Early game (turn 5)", None, 5),
        ("Mid game (turn 15)", None, 15),
    ]

    for name, test_board, turn in test_cases:
        print(f"\n{'=' * 80}")
        print(f"Test: {name}")
        print(f"{'=' * 80}")

        if test_board is None:
            # Create board with some moves
            test_board = Board(11)
            # Place some pieces
            for i in range((turn - 1) // 2):
                test_board.set_tile_colour(4 + i, 5, Colour.RED)
                if i < turn // 2:
                    test_board.set_tile_colour(5, 4 + i, Colour.BLUE)

        # Profile with cProfile
        profiler = cProfile.Profile()
        profiler.enable()

        start = time.perf_counter()
        move = agent.make_move(turn, test_board, None)
        elapsed = time.perf_counter() - start

        profiler.disable()

        print(f"Move selected: {move}")
        print(f"Time taken: {elapsed:.3f}s ({elapsed*1000:.1f}ms)")

        # Get detailed stats
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions

        print("\nTop 20 functions by cumulative time:")
        print(s.getvalue())

        # Estimate iterations if using MCTS
        if turn > 5:
            # Rough estimate based on typical MCTS performance
            empty_cells = sum(1 for i in range(11) for j in range(11)
                            if test_board.tiles[i][j].colour is None)
            print(f"\nEmpty cells: {empty_cells}")
            print(f"Branch factor: ~{empty_cells}")

def test_mcts_iterations():
    """Test how many MCTS iterations we can do in different time budgets."""
    print("\n" + "=" * 80)
    print("MCTS ITERATION RATE TEST")
    print("=" * 80)

    from agents.Group12.algorithms.mcts_enhanced import EnhancedMCTS
    from agents.Group12.core.evaluation import PositionEvaluator

    board = Board(11)
    # Place a few pieces
    board.set_tile_colour(5, 5, Colour.RED)
    board.set_tile_colour(5, 4, Colour.BLUE)

    evaluator = PositionEvaluator()
    mcts = EnhancedMCTS(Colour.BLUE, evaluator)

    time_budgets = [0.1, 0.5, 1.0, 2.0]

    for budget in time_budgets:
        print(f"\nTime budget: {budget}s")

        start = time.perf_counter()
        move = mcts.search(board, budget)
        actual_time = time.perf_counter() - start

        # Count iterations (approximate from time)
        iterations = len([n for n in mcts.nodes.values() if n.visits > 0])

        print(f"  Actual time: {actual_time:.3f}s")
        print(f"  Nodes created: {len(mcts.nodes)}")
        print(f"  Nodes visited: {iterations}")
        print(f"  Iterations/sec: {iterations/actual_time:.1f}")
        print(f"  Move selected: {move}")

if __name__ == "__main__":
    profile_mcts_performance()
    test_mcts_iterations()
