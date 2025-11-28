#!/usr/bin/env python3
"""
Comprehensive Agent Evaluation Script for Group12 Hex AI.

Runs matches between different agents and produces detailed statistics.

Usage:
    # Quick evaluation (10 games)
    python evaluate_agents.py --games 10

    # Full evaluation (100 games)
    python evaluate_agents.py --games 100 --output results.json

    # Evaluate specific model
    python evaluate_agents.py --model models/hex_model_best.pth --games 50
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class AgentWrapper:
    """Wrapper for different agent types."""

    def __init__(self, agent_type: str, model_path: Optional[str] = None):
        self.agent_type = agent_type
        self.model_path = model_path
        self.agent = None
        self._load_agent()

    def _load_agent(self):
        """Load the appropriate agent."""
        if self.agent_type == 'neural':
            from agents.Group12.Group12Agent_neural import Group12Agent
            self.agent = Group12Agent()
            if self.model_path:
                self.agent.load_model(self.model_path)

        elif self.agent_type == 'full':
            from agents.Group12.Group12Agent import Group12Agent
            self.agent = Group12Agent()

        elif self.agent_type == 'simple':
            from agents.Group12.Group12Agent_simple import Group12Agent
            self.agent = Group12Agent()

        elif self.agent_type == 'naive':
            from agents.DefaultAgents.NaiveAgent import NaiveAgent
            self.agent = NaiveAgent()

        elif self.agent_type == 'random':
            self.agent = RandomAgent()

        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

    def make_move(self, board: Board, colour: Colour, turn: int) -> Move:
        """Get a move from the agent."""
        return self.agent.make_move(turn, board, colour)

    @property
    def name(self) -> str:
        if self.agent_type == 'neural' and self.model_path:
            return f"Neural({Path(self.model_path).stem})"
        return self.agent_type.capitalize()


class RandomAgent:
    """Simple random agent for baseline comparison."""

    def make_move(self, turn: int, board: Board, colour: Colour) -> Move:
        """Make a random legal move."""
        legal_moves = []
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    legal_moves.append(Move(i, j))

        if not legal_moves:
            return Move(-1, -1)  # No moves available

        return random.choice(legal_moves)


def play_game(agent1: AgentWrapper, agent2: AgentWrapper,
              verbose: bool = False) -> Dict:
    """
    Play a single game between two agents.

    Args:
        agent1: First player (RED)
        agent2: Second player (BLUE)
        verbose: Print game progress

    Returns:
        Dictionary with game results
    """
    board = Board(11)
    turn = 0
    move_times = {'RED': [], 'BLUE': []}
    moves = []

    agents = {Colour.RED: agent1, Colour.BLUE: agent2}
    current_colour = Colour.RED

    game_start = time.time()

    while not board.has_ended(Colour.RED) and not board.has_ended(Colour.BLUE):
        turn += 1
        agent = agents[current_colour]
        colour_name = 'RED' if current_colour == Colour.RED else 'BLUE'

        # Get move with timing
        move_start = time.time()
        try:
            move = agent.make_move(board, current_colour, turn)
        except Exception as e:
            if verbose:
                print(f"  {colour_name} ({agent.name}) ERROR: {e}")
            # Agent error = automatic loss
            winner = Colour.opposite(current_colour)
            return {
                'winner': 'RED' if winner == Colour.RED else 'BLUE',
                'winner_agent': agents[winner].name,
                'loser_agent': agent.name,
                'turns': turn,
                'reason': f'error: {str(e)[:50]}',
                'duration': time.time() - game_start,
                'move_times': move_times,
                'moves': moves,
            }

        move_time = time.time() - move_start
        move_times[colour_name].append(move_time)

        # Handle swap
        if turn == 2 and move.x == -1 and move.y == -1:
            if verbose:
                print(f"  Turn {turn}: {colour_name} ({agent.name}) SWAPS")
            moves.append({'turn': turn, 'colour': colour_name, 'action': 'swap'})
            # Swap colors
            agents[Colour.RED], agents[Colour.BLUE] = agents[Colour.BLUE], agents[Colour.RED]
        else:
            # Validate move
            if not (0 <= move.x < 11 and 0 <= move.y < 11):
                if verbose:
                    print(f"  {colour_name} ({agent.name}) ILLEGAL: out of bounds ({move.x}, {move.y})")
                winner = Colour.opposite(current_colour)
                return {
                    'winner': 'RED' if winner == Colour.RED else 'BLUE',
                    'winner_agent': agents[winner].name,
                    'loser_agent': agent.name,
                    'turns': turn,
                    'reason': 'illegal_move_bounds',
                    'duration': time.time() - game_start,
                    'move_times': move_times,
                    'moves': moves,
                }

            if board.tiles[move.x][move.y].colour is not None:
                if verbose:
                    print(f"  {colour_name} ({agent.name}) ILLEGAL: occupied ({move.x}, {move.y})")
                winner = Colour.opposite(current_colour)
                return {
                    'winner': 'RED' if winner == Colour.RED else 'BLUE',
                    'winner_agent': agents[winner].name,
                    'loser_agent': agent.name,
                    'turns': turn,
                    'reason': 'illegal_move_occupied',
                    'duration': time.time() - game_start,
                    'move_times': move_times,
                    'moves': moves,
                }

            # Make move
            board.set_tile_colour(move.x, move.y, current_colour)
            moves.append({
                'turn': turn,
                'colour': colour_name,
                'x': move.x,
                'y': move.y,
                'time': move_time
            })

            if verbose:
                print(f"  Turn {turn}: {colour_name} ({agent.name}) -> ({move.x}, {move.y}) [{move_time:.3f}s]")

        # Switch player
        current_colour = Colour.opposite(current_colour)

        # Safety check: max turns
        if turn > 200:
            if verbose:
                print("  Game exceeded 200 turns - declaring draw")
            return {
                'winner': 'draw',
                'winner_agent': None,
                'loser_agent': None,
                'turns': turn,
                'reason': 'max_turns',
                'duration': time.time() - game_start,
                'move_times': move_times,
                'moves': moves,
            }

    # Determine winner
    if board.has_ended(Colour.RED):
        winner = Colour.RED
    else:
        winner = Colour.BLUE

    winner_name = 'RED' if winner == Colour.RED else 'BLUE'
    game_duration = time.time() - game_start

    if verbose:
        print(f"  Winner: {winner_name} ({agents[winner].name}) in {turn} turns")

    return {
        'winner': winner_name,
        'winner_agent': agents[winner].name,
        'loser_agent': agents[Colour.opposite(winner)].name,
        'turns': turn,
        'reason': 'normal',
        'duration': game_duration,
        'move_times': move_times,
        'moves': moves,
    }


def run_match(agent1_type: str, agent2_type: str,
              num_games: int, model_path: Optional[str] = None,
              verbose: bool = False) -> Dict:
    """
    Run a match between two agent types.

    Each agent plays as both RED and BLUE for fairness.
    """
    print(f"\n{'='*60}")
    print(f"MATCH: {agent1_type} vs {agent2_type}")
    print(f"Games: {num_games} ({num_games//2} as RED, {num_games//2} as BLUE each)")
    print(f"{'='*60}")

    results = {
        'agent1': agent1_type,
        'agent2': agent2_type,
        'model_path': model_path,
        'num_games': num_games,
        'games': [],
        'summary': {
            'agent1_wins': 0,
            'agent2_wins': 0,
            'draws': 0,
            'agent1_wins_as_red': 0,
            'agent1_wins_as_blue': 0,
            'agent2_wins_as_red': 0,
            'agent2_wins_as_blue': 0,
            'total_turns': 0,
            'total_duration': 0,
            'illegal_moves': 0,
            'errors': 0,
        }
    }

    # Create agents
    agent1 = AgentWrapper(agent1_type, model_path if agent1_type == 'neural' else None)
    agent2 = AgentWrapper(agent2_type, model_path if agent2_type == 'neural' else None)

    games_per_color = num_games // 2

    # Agent1 as RED
    print(f"\n--- {agent1.name} as RED vs {agent2.name} as BLUE ---")
    for i in range(games_per_color):
        print(f"Game {i+1}/{games_per_color}...", end=' ')
        game_result = play_game(agent1, agent2, verbose=verbose)
        results['games'].append({**game_result, 'agent1_color': 'RED'})

        if game_result['winner'] == 'RED':
            results['summary']['agent1_wins'] += 1
            results['summary']['agent1_wins_as_red'] += 1
            print(f"{agent1.name} wins ({game_result['turns']} turns)")
        elif game_result['winner'] == 'BLUE':
            results['summary']['agent2_wins'] += 1
            results['summary']['agent2_wins_as_blue'] += 1
            print(f"{agent2.name} wins ({game_result['turns']} turns)")
        else:
            results['summary']['draws'] += 1
            print(f"Draw ({game_result['reason']})")

        results['summary']['total_turns'] += game_result['turns']
        results['summary']['total_duration'] += game_result['duration']

        if 'illegal' in game_result['reason']:
            results['summary']['illegal_moves'] += 1
        if 'error' in game_result['reason']:
            results['summary']['errors'] += 1

    # Agent1 as BLUE
    print(f"\n--- {agent1.name} as BLUE vs {agent2.name} as RED ---")
    for i in range(games_per_color):
        print(f"Game {i+1}/{games_per_color}...", end=' ')
        game_result = play_game(agent2, agent1, verbose=verbose)
        results['games'].append({**game_result, 'agent1_color': 'BLUE'})

        if game_result['winner'] == 'BLUE':
            results['summary']['agent1_wins'] += 1
            results['summary']['agent1_wins_as_blue'] += 1
            print(f"{agent1.name} wins ({game_result['turns']} turns)")
        elif game_result['winner'] == 'RED':
            results['summary']['agent2_wins'] += 1
            results['summary']['agent2_wins_as_red'] += 1
            print(f"{agent2.name} wins ({game_result['turns']} turns)")
        else:
            results['summary']['draws'] += 1
            print(f"Draw ({game_result['reason']})")

        results['summary']['total_turns'] += game_result['turns']
        results['summary']['total_duration'] += game_result['duration']

        if 'illegal' in game_result['reason']:
            results['summary']['illegal_moves'] += 1
        if 'error' in game_result['reason']:
            results['summary']['errors'] += 1

    # Calculate statistics
    total_games = results['summary']['agent1_wins'] + results['summary']['agent2_wins'] + results['summary']['draws']
    results['summary']['agent1_win_rate'] = results['summary']['agent1_wins'] / total_games if total_games > 0 else 0
    results['summary']['agent2_win_rate'] = results['summary']['agent2_wins'] / total_games if total_games > 0 else 0
    results['summary']['avg_turns'] = results['summary']['total_turns'] / total_games if total_games > 0 else 0
    results['summary']['avg_duration'] = results['summary']['total_duration'] / total_games if total_games > 0 else 0

    # Print summary
    print(f"\n{'='*60}")
    print("MATCH SUMMARY")
    print(f"{'='*60}")
    print(f"{agent1.name}: {results['summary']['agent1_wins']} wins ({results['summary']['agent1_win_rate']*100:.1f}%)")
    print(f"  As RED:  {results['summary']['agent1_wins_as_red']}")
    print(f"  As BLUE: {results['summary']['agent1_wins_as_blue']}")
    print(f"{agent2.name}: {results['summary']['agent2_wins']} wins ({results['summary']['agent2_win_rate']*100:.1f}%)")
    print(f"  As RED:  {results['summary']['agent2_wins_as_red']}")
    print(f"  As BLUE: {results['summary']['agent2_wins_as_blue']}")
    print(f"Draws: {results['summary']['draws']}")
    print(f"Average game length: {results['summary']['avg_turns']:.1f} turns")
    print(f"Average game duration: {results['summary']['avg_duration']:.2f}s")
    print(f"Illegal moves: {results['summary']['illegal_moves']}")
    print(f"Errors: {results['summary']['errors']}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Hex AI agents against each other',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test: Neural vs Simple heuristic
    python evaluate_agents.py --games 10

    # Full evaluation with specific model
    python evaluate_agents.py --model models/hex_model_best.pth --games 50

    # Compare all agents
    python evaluate_agents.py --comprehensive --games 20

    # Verbose output (show each move)
    python evaluate_agents.py --games 5 --verbose
        """
    )

    parser.add_argument('--games', type=int, default=10,
                        help='Number of games per match (default: 10)')

    parser.add_argument('--model', type=str, default=None,
                        help='Path to neural network model')

    parser.add_argument('--agent1', type=str, default='neural',
                        choices=['neural', 'full', 'simple', 'naive', 'random'],
                        help='First agent type (default: neural)')

    parser.add_argument('--agent2', type=str, default='simple',
                        choices=['neural', 'full', 'simple', 'naive', 'random'],
                        help='Second agent type (default: simple)')

    parser.add_argument('--comprehensive', action='store_true',
                        help='Run comprehensive evaluation against all agent types')

    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output (show each move)')

    args = parser.parse_args()

    print("=" * 60)
    print("GROUP12 HEX AI - AGENT EVALUATION")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Games per match: {args.games}")
    print("=" * 60)

    all_results = {}

    if args.comprehensive:
        # Run against all agent types
        opponents = ['simple', 'full', 'naive', 'random']
        for opponent in opponents:
            if opponent != args.agent1:
                results = run_match(args.agent1, opponent, args.games,
                                    args.model, args.verbose)
                all_results[f'{args.agent1}_vs_{opponent}'] = results
    else:
        # Single match
        results = run_match(args.agent1, args.agent2, args.games,
                            args.model, args.verbose)
        all_results[f'{args.agent1}_vs_{args.agent2}'] = results

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print(f"\nEvaluation complete!")


if __name__ == '__main__':
    main()
