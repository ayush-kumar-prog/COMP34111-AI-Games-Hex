#!/usr/bin/env python3
"""
Reprocess test results to fix winner parsing.
This script reads the comprehensive test JSON file and re-parses
the winner information from the stored game output.
"""

import json
import re
from pathlib import Path


def parse_winner_from_output(output, p1_name, p2_name):
    """Parse winner from game output using fixed logic"""
    winner = 'UNKNOWN'

    # Pattern 1: "winner,Alice,WIN" or "winner,Bob,WIN"
    winner_match = re.search(r'winner,(\w+),WIN', output)
    if winner_match:
        winner_name = winner_match.group(1)
        # Map agent names (Alice=p2, Bob=p1)
        if winner_name == 'Bob':
            winner = p1_name
        elif winner_name == 'Alice':
            winner = p2_name
    # Pattern 2: "Player Alice has won" or "Player Bob has won"
    elif 'Player Alice has won' in output:
        winner = p2_name
    elif 'Player Bob has won' in output:
        winner = p1_name
    # Pattern 3: "RED wins" or "BLUE wins" (old format)
    elif 'RED wins' in output or 'Player 1 (RED) wins' in output:
        winner = p1_name
    elif 'BLUE wins' in output or 'Player 2 (BLUE) wins' in output:
        winner = p2_name

    return winner


def extract_turn_count(output):
    """Extract actual turn count from output"""
    turn_match = re.search(r'Turn (\d+)', output)
    if turn_match:
        # Get the last turn number
        all_turns = re.findall(r'Turn (\d+)', output)
        if all_turns:
            return int(all_turns[-1])
    return 0


def fix_test_results(input_file, output_file=None):
    """Fix winner parsing in test results"""
    if output_file is None:
        output_file = input_file

    print(f"Loading results from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Found {len(data['games'])} games")

    fixed_count = 0
    for game in data['games']:
        old_winner = game['winner']
        p1_name = game['p1_agent']
        p2_name = game['p2_agent']
        output = game.get('output', '')

        if output and old_winner == 'UNKNOWN':
            new_winner = parse_winner_from_output(output, p1_name, p2_name)
            if new_winner != 'UNKNOWN':
                game['winner'] = new_winner
                # Also fix turn count if it's wrong
                if game.get('turns', 0) <= 1:
                    game['turns'] = extract_turn_count(output)
                fixed_count += 1
                print(f"  Game {game['game_number']}: Fixed winner to {new_winner} (was UNKNOWN)")

    print(f"\nFixed {fixed_count} games")

    # Recompute summary statistics
    valid_games = [g for g in data['games'] if g['winner'] not in ['TIMEOUT', 'ERROR', 'UNKNOWN']]

    if valid_games:
        simple_wins = sum(1 for g in valid_games if g['winner'] == 'Simple')
        full_wins = sum(1 for g in valid_games if g['winner'] == 'Full')

        import math
        n = len(valid_games)
        p = simple_wins / n if n > 0 else 0
        se = math.sqrt(p * (1 - p) / n) if n > 0 else 0
        ci_margin = 1.96 * se

        print(f"\nUpdated Statistics:")
        print(f"  Valid games: {len(valid_games)}/{len(data['games'])}")
        print(f"  Simple wins: {simple_wins} ({simple_wins/n*100:.1f}%)")
        print(f"  Full wins: {full_wins} ({full_wins/n*100:.1f}%)")
        print(f"  95% CI: [{max(0, p-ci_margin)*100:.1f}%, {min(1, p+ci_margin)*100:.1f}%]")
        print(f"  Margin of error: ±{ci_margin*100:.1f}%")

        # Update summary in data
        if 'summary' in data:
            data['summary']['simple_wins'] = simple_wins
            data['summary']['full_wins'] = full_wins
            data['summary']['valid_games'] = len(valid_games)
            data['summary']['simple_win_rate'] = p
            data['summary']['full_win_rate'] = full_wins / n if n > 0 else 0
            if 'confidence_interval_95' not in data['summary']:
                data['summary']['confidence_interval_95'] = {}
            data['summary']['confidence_interval_95']['point_estimate'] = p
            data['summary']['confidence_interval_95']['lower_bound'] = max(0, p - ci_margin)
            data['summary']['confidence_interval_95']['upper_bound'] = min(1, p + ci_margin)
            data['summary']['confidence_interval_95']['margin_of_error'] = ci_margin

    # Save fixed results
    print(f"\nSaving fixed results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print("✅ Done!")


if __name__ == '__main__':
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else 'results_comprehensive_50games.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file

    fix_test_results(input_file, output_file)
