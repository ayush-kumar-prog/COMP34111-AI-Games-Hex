#!/usr/bin/env python3
"""
Quick model evaluation script.
Tests trained models against baseline agents.
"""

import sys
import argparse
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Game import Game


def run_game(agent1_class, agent2_class, verbose=False):
    """Run a single game between two agents.
    Returns: 1 if agent1 wins, 2 if agent2 wins, 0 for draw
    """
    board = Board(11)
    agent1 = agent1_class(Colour.RED)
    agent2 = agent2_class(Colour.BLUE)

    turn = 1
    current_agent = agent1
    current_colour = Colour.RED
    opp_move = None

    # Track which agent is which and their current colors
    agent1_is_current = True
    # Track agent colors (can swap)
    agent1_colour = Colour.RED
    agent2_colour = Colour.BLUE

    while not board.has_ended(Colour.RED) and not board.has_ended(Colour.BLUE):
        move = current_agent.make_move(turn, board, opp_move)

        # Handle swap
        if move.x == -1 and move.y == -1 and turn == 2:
            # Swap colors - agent2 takes over RED with the position agent1 created
            agent1._colour = Colour.BLUE
            agent2._colour = Colour.RED
            agent1_colour = Colour.BLUE
            agent2_colour = Colour.RED
            # After swap, agent2 (now RED) continues the game
            current_agent = agent2
            current_colour = Colour.RED
            agent1_is_current = False
            if verbose:
                print(f"Turn {turn}: SWAP (agent1 now BLUE, agent2 now RED)")
        else:
            # Validate move is legal (cell must be empty)
            if not (0 <= move.x < 11 and 0 <= move.y < 11):
                if verbose:
                    print(f"Turn {turn}: ILLEGAL MOVE (out of bounds) by {'agent1' if agent1_is_current else 'agent2'}")
                # Illegal move = loss for current player
                return 2 if agent1_is_current else 1

            if board.tiles[move.x][move.y].colour is not None:
                if verbose:
                    print(f"Turn {turn}: ILLEGAL MOVE ({move.x}, {move.y}) already occupied by {'agent1' if agent1_is_current else 'agent2'}")
                # Illegal move = loss for current player
                return 2 if agent1_is_current else 1

            board.set_tile_colour(move.x, move.y, current_colour)
            if verbose:
                print(f"Turn {turn}: {'agent1' if agent1_is_current else 'agent2'} ({current_colour.name}) plays ({move.x}, {move.y})")

            # Switch players
            opp_move = move
            if current_agent == agent1:
                current_agent = agent2
                current_colour = agent2_colour
                agent1_is_current = False
            else:
                current_agent = agent1
                current_colour = agent1_colour
                agent1_is_current = True

        turn += 1

        if turn > 200:  # Safety limit
            break

    # Determine winner by color, then map to agent
    if board.has_ended(Colour.RED):
        # RED won - which agent is RED?
        if agent1_colour == Colour.RED:
            return 1  # agent1 wins
        else:
            return 2  # agent2 wins
    elif board.has_ended(Colour.BLUE):
        # BLUE won - which agent is BLUE?
        if agent1_colour == Colour.BLUE:
            return 1  # agent1 wins
        else:
            return 2  # agent2 wins
    return 0  # DRAW


def evaluate(agent1_name, agent2_name, num_games=10):
    """Evaluate two agents against each other."""

    # Import agents dynamically
    if agent1_name == "ultimate":
        from agents.Group12.Group12Agent_ultimate import Group12Agent as Agent1
    elif agent1_name == "tournament":
        from agents.Group12.Group12Agent_tournament import Group12Agent as Agent1
    elif agent1_name == "simple":
        from agents.Group12.Group12Agent_simple import Group12Agent as Agent1
    elif agent1_name == "naive":
        from agents.DefaultAgents.NaiveAgent import NaiveAgent as Agent1
    else:
        raise ValueError(f"Unknown agent: {agent1_name}")

    if agent2_name == "ultimate":
        from agents.Group12.Group12Agent_ultimate import Group12Agent as Agent2
    elif agent2_name == "tournament":
        from agents.Group12.Group12Agent_tournament import Group12Agent as Agent2
    elif agent2_name == "simple":
        from agents.Group12.Group12Agent_simple import Group12Agent as Agent2
    elif agent2_name == "naive":
        from agents.DefaultAgents.NaiveAgent import NaiveAgent as Agent2
    else:
        raise ValueError(f"Unknown agent: {agent2_name}")

    print(f"\n{'='*60}")
    print(f"EVALUATION: {agent1_name} vs {agent2_name}")
    print(f"Games: {num_games}")
    print(f"{'='*60}\n")

    agent1_wins = 0
    agent2_wins = 0

    for game_num in range(num_games):
        # Alternate who plays first
        if game_num % 2 == 0:
            # agent1 starts as RED
            result = run_game(Agent1, Agent2)
            if result == 1:
                agent1_wins += 1
                winner = agent1_name
            elif result == 2:
                agent2_wins += 1
                winner = agent2_name
            else:
                winner = "DRAW"
        else:
            # agent2 starts as RED
            result = run_game(Agent2, Agent1)
            if result == 1:
                agent2_wins += 1
                winner = agent2_name
            elif result == 2:
                agent1_wins += 1
                winner = agent1_name
            else:
                winner = "DRAW"

        print(f"Game {game_num + 1}: {winner} wins")

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"{agent1_name}: {agent1_wins} wins ({100*agent1_wins/num_games:.1f}%)")
    print(f"{agent2_name}: {agent2_wins} wins ({100*agent2_wins/num_games:.1f}%)")
    print(f"{'='*60}\n")

    return agent1_wins, agent2_wins


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hex agents")
    parser.add_argument("--agent1", type=str, default="tournament",
                        choices=["ultimate", "tournament", "simple", "naive"])
    parser.add_argument("--agent2", type=str, default="naive",
                        choices=["ultimate", "tournament", "simple", "naive"])
    parser.add_argument("--games", type=int, default=10)

    args = parser.parse_args()
    evaluate(args.agent1, args.agent2, args.games)


if __name__ == "__main__":
    main()
