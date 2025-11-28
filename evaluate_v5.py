#!/usr/bin/env python3
"""
V5 Model Evaluation Script

Evaluates trained V5 models against baseline agents:
- NaiveAgent (random)
- Group12Agent_simple (heuristic)
- Group12Agent_tournament (MCTS+RAVE, 500 iter)

Usage:
    python evaluate_v5.py --model models/v5_best.pth --games 20
    python evaluate_v5.py --model models/v5_alphazero_final.pth --games 20
"""

import sys
import time
import argparse
import random
from pathlib import Path
from typing import Tuple, List

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


def load_agent(agent_spec: str, colour: Colour):
    """Load an agent by specification."""
    module_name, class_name = agent_spec.rsplit(' ', 1)
    module = __import__(module_name, fromlist=[class_name])
    agent_class = getattr(module, class_name)
    return agent_class(colour)


def play_game(agent1, agent2, verbose: bool = False) -> Tuple[int, int, float]:
    """
    Play a single game between two agents.

    Returns:
        (winner, num_moves, game_time)
        winner: 1 if agent1 wins, 2 if agent2 wins, 0 if draw
    """
    board = Board(11)
    turn = 1
    last_move = None

    # Track whose turn it is
    current_agent = agent1
    current_colour = Colour.RED

    start_time = time.time()

    while True:
        try:
            move = current_agent.make_move(turn, board, last_move)
        except Exception as e:
            if verbose:
                print(f"Error from {current_colour}: {e}")
            # Other player wins
            return (2 if current_colour == Colour.RED else 1), turn, time.time() - start_time

        # Handle swap (turn 2 only)
        if turn == 2 and move.x == -1 and move.y == -1:
            # Swap - agent2 now plays as RED
            if verbose:
                print(f"  Turn {turn}: {current_colour} swaps")
            # Swap the agents' colours
            agent1._colour = Colour.BLUE
            agent2._colour = Colour.RED
            # Reinitialize MCTS if needed
            if hasattr(agent1, 'mcts'):
                agent1.mcts = None
            if hasattr(agent2, 'mcts'):
                agent2.mcts = None
        else:
            # Normal move
            if board.tiles[move.x][move.y].colour is not None:
                if verbose:
                    print(f"  Illegal move by {current_colour}: ({move.x}, {move.y}) already occupied")
                return (2 if current_colour == Colour.RED else 1), turn, time.time() - start_time

            board.set_tile_colour(move.x, move.y, current_colour)

            if verbose:
                print(f"  Turn {turn}: {current_colour} plays ({move.x}, {move.y})")

            # Check for win
            if board.has_ended(Colour.RED):
                winner = 1 if agent1._colour == Colour.RED else 2
                return winner, turn, time.time() - start_time
            if board.has_ended(Colour.BLUE):
                winner = 1 if agent1._colour == Colour.BLUE else 2
                return winner, turn, time.time() - start_time

        last_move = move
        turn += 1

        # Switch players
        if current_agent == agent1:
            current_agent = agent2
            current_colour = Colour.opposite(current_colour)
        else:
            current_agent = agent1
            current_colour = Colour.opposite(current_colour)

        # Safety: max turns
        if turn > 150:
            if verbose:
                print("  Game exceeded 150 turns - draw")
            return 0, turn, time.time() - start_time


def evaluate_agents(agent1_spec: str, agent2_spec: str, num_games: int,
                    verbose: bool = False) -> dict:
    """
    Evaluate agent1 against agent2 over multiple games.

    Returns:
        Dictionary with results
    """
    results = {
        'agent1_wins': 0,
        'agent2_wins': 0,
        'draws': 0,
        'agent1_wins_as_red': 0,
        'agent1_wins_as_blue': 0,
        'total_moves': 0,
        'total_time': 0.0,
        'games': []
    }

    for game_idx in range(num_games):
        # Alternate colours
        if game_idx % 2 == 0:
            agent1 = load_agent(agent1_spec, Colour.RED)
            agent2 = load_agent(agent2_spec, Colour.BLUE)
            agent1_is_red = True
        else:
            agent1 = load_agent(agent1_spec, Colour.BLUE)
            agent2 = load_agent(agent2_spec, Colour.RED)
            agent1_is_red = False

        winner, moves, game_time = play_game(agent1, agent2, verbose=verbose)

        results['total_moves'] += moves
        results['total_time'] += game_time

        if winner == 1:
            results['agent1_wins'] += 1
            if agent1_is_red:
                results['agent1_wins_as_red'] += 1
            else:
                results['agent1_wins_as_blue'] += 1
        elif winner == 2:
            results['agent2_wins'] += 1
        else:
            results['draws'] += 1

        results['games'].append({
            'winner': winner,
            'moves': moves,
            'time': game_time,
            'agent1_colour': 'RED' if agent1_is_red else 'BLUE'
        })

        # Progress
        agent1_short = agent1_spec.split('.')[-1].split()[0]
        agent2_short = agent2_spec.split('.')[-1].split()[0]
        winner_str = agent1_short if winner == 1 else (agent2_short if winner == 2 else 'Draw')
        print(f"  Game {game_idx + 1}/{num_games}: {winner_str} wins "
              f"({moves} moves, {game_time:.1f}s)")

    return results


def print_results(results: dict, agent1_name: str, agent2_name: str):
    """Print evaluation results."""
    total = results['agent1_wins'] + results['agent2_wins'] + results['draws']

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"{agent1_name} vs {agent2_name}")
    print("-"*70)
    print(f"{agent1_name} wins: {results['agent1_wins']}/{total} "
          f"({100*results['agent1_wins']/total:.1f}%)")
    print(f"  As RED:  {results['agent1_wins_as_red']}")
    print(f"  As BLUE: {results['agent1_wins_as_blue']}")
    print(f"{agent2_name} wins: {results['agent2_wins']}/{total} "
          f"({100*results['agent2_wins']/total:.1f}%)")
    print(f"Draws: {results['draws']}")
    print("-"*70)
    print(f"Avg moves per game: {results['total_moves']/total:.1f}")
    print(f"Avg time per game: {results['total_time']/total:.1f}s")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate V5 model')
    parser.add_argument('--model', type=str, default='models/v5_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--games', type=int, default=20,
                       help='Number of games per opponent')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed game info')
    parser.add_argument('--opponents', type=str, nargs='+',
                       default=['naive', 'simple', 'tournament'],
                       help='Opponents to test against')

    args = parser.parse_args()

    # Update model path in V5 agent
    import agents.Group12.Group12Agent_v5 as v5_module

    print("="*70)
    print("V5 MODEL EVALUATION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Games per opponent: {args.games}")
    print("="*70)

    # Agent specifications
    v5_spec = "agents.Group12.Group12Agent_v5 Group12Agent"

    opponents = {
        'naive': ("agents.DefaultAgents.NaiveAgent NaiveAgent", "NaiveAgent"),
        'simple': ("agents.Group12.Group12Agent_simple Group12Agent", "SimpleAgent"),
        'tournament': ("agents.Group12.Group12Agent_tournament Group12Agent", "TournamentMCTS"),
    }

    all_results = {}

    for opp_key in args.opponents:
        if opp_key not in opponents:
            print(f"Unknown opponent: {opp_key}")
            continue

        opp_spec, opp_name = opponents[opp_key]

        print(f"\n{'='*70}")
        print(f"V5 Neural vs {opp_name}")
        print("="*70)

        results = evaluate_agents(v5_spec, opp_spec, args.games, verbose=args.verbose)
        print_results(results, "V5 Neural", opp_name)

        all_results[opp_key] = results

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Opponent':<20} {'Win Rate':<15} {'Games'}")
    print("-"*70)
    for opp_key, results in all_results.items():
        total = results['agent1_wins'] + results['agent2_wins'] + results['draws']
        win_rate = results['agent1_wins'] / total if total > 0 else 0
        _, opp_name = opponents[opp_key]
        print(f"{opp_name:<20} {win_rate:>6.1%}         {total}")
    print("="*70)


if __name__ == '__main__':
    main()
