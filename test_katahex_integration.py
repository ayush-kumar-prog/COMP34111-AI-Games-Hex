#!/usr/bin/env python3
"""
Test script to verify KataHexAgent integration.
Run this after setting up KataHex to ensure everything works.

Usage:
    python3 test_katahex_integration.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.Group12.KataHexAgent import KataHexAgent
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


def test_agent_initialization():
    """Test that KataHexAgent can be initialized."""
    print("=" * 60)
    print("Test 1: Agent Initialization")
    print("=" * 60)

    try:
        agent = KataHexAgent(Colour.RED)
        print("✓ KataHexAgent initialized successfully")
        print(f"  - Using fallback: {agent._use_fallback}")
        print(f"  - Engine initialized: {agent._initialized}")
        return agent
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        sys.exit(1)


def test_first_move(agent):
    """Test that agent can make first move."""
    print("\n" + "=" * 60)
    print("Test 2: First Move Generation")
    print("=" * 60)

    try:
        board = Board(11)
        move = agent.make_move(1, board, None)

        print(f"✓ First move generated: ({move.x}, {move.y})")

        if agent._use_fallback:
            print("  ⚠ Warning: Using fallback agent (KataHex engine not available)")
        else:
            print("  ✓ KataHex engine is working")

        return move
    except Exception as e:
        print(f"✗ Failed to generate first move: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_response_move(agent):
    """Test that agent can respond to opponent's move."""
    print("\n" + "=" * 60)
    print("Test 3: Response Move Generation")
    print("=" * 60)

    try:
        board = Board(11)
        # Simulate opponent playing center
        opp_move = Move(5, 5)
        board.set_tile_colour(5, 5, Colour.BLUE)

        move = agent.make_move(2, board, opp_move)

        print(f"✓ Response move generated: ({move.x}, {move.y})")
        print(f"  Opponent played: ({opp_move.x}, {opp_move.y})")

        # Verify move is legal
        if board.tiles[move.x][move.y].colour is not None:
            print("✗ ERROR: Agent generated illegal move (tile occupied)")
            sys.exit(1)

        print("  ✓ Move is legal")
        return move
    except Exception as e:
        print(f"✗ Failed to generate response move: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_short_game():
    """Test a short 5-move game."""
    print("\n" + "=" * 60)
    print("Test 4: Short Game Simulation (5 moves)")
    print("=" * 60)

    try:
        red_agent = KataHexAgent(Colour.RED)
        board = Board(11)

        print("\nSimulating 5-move game...")
        moves = []

        for turn in range(1, 6):
            if turn == 1:
                move = red_agent.make_move(turn, board, None)
            else:
                # Simulate opponent playing somewhere random
                opp_move = Move(turn-1, 0)
                board.set_tile_colour(opp_move.x, opp_move.y, Colour.BLUE)
                move = red_agent.make_move(turn, board, opp_move)

            board.set_tile_colour(move.x, move.y, Colour.RED)
            moves.append(move)
            print(f"  Turn {turn}: RED plays ({move.x}, {move.y})")

        print(f"\n✓ Successfully simulated {len(moves)} moves")
        return True
    except Exception as e:
        print(f"✗ Short game simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    print("KataHex Integration Test")
    print("=" * 60)
    print("")

    # Check if KataHex files exist
    katahex_dir = os.path.join(os.path.dirname(__file__), "agents", "Group12", "katahex")
    binary_path = os.path.join(katahex_dir, "katahex")
    model_path = os.path.join(katahex_dir, "hex27x3.bin.gz")
    config_path = os.path.join(katahex_dir, "config.cfg")

    print("Pre-flight checks:")
    print(f"  KataHex binary: {'✓ EXISTS' if os.path.exists(binary_path) else '✗ NOT FOUND'}")
    print(f"  Model file:     {'✓ EXISTS' if os.path.exists(model_path) else '✗ NOT FOUND'}")
    print(f"  Config file:    {'✓ EXISTS' if os.path.exists(config_path) else '✗ NOT FOUND'}")
    print("")

    if not os.path.exists(binary_path):
        print("⚠ KataHex binary not found. Agent will use fallback.")
        print("  To set up KataHex, run:")
        print("    cd agents/Group12/katahex")
        print("    ./setup_katahex_cuda.sh")
        print("")

    # Run tests
    agent = test_agent_initialization()
    test_first_move(agent)
    test_response_move(agent)
    test_short_game()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if agent._use_fallback:
        print("⚠ KataHex engine is NOT working (using fallback)")
        print("  This means tournaments will not use the strong KataHex engine.")
        print("  Please run the setup script on CSF3 to enable KataHex.")
    else:
        print("✓ All tests passed!")
        print("  KataHex engine is working correctly.")
        print("  Ready for tournament play.")

    print("")


if __name__ == "__main__":
    main()
