#!/usr/bin/env python3
"""
Comprehensive testing script for Group12 Hex AI agents.
Runs 50+ games with detailed statistics collection.

Usage:
    python3 run_comprehensive_tests.py --games 50 --output results_comprehensive.json
"""

import subprocess
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import re
import sys

class ComprehensiveTestRunner:
    def __init__(self, num_games=50, output_file='results_comprehensive.json'):
        self.num_games = num_games
        self.output_file = output_file
        self.results = {
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'num_games_target': num_games,
                'test_type': 'Simple vs Full comprehensive analysis'
            },
            'games': [],
            'summary': {}
        }

    def run_single_game(self, p1_agent, p1_name, p2_agent, p2_name, game_num):
        """Run a single game and capture detailed results"""
        print(f"\n{'='*60}")
        print(f"Game {game_num}/{self.num_games}: {p1_name} (RED) vs {p2_name} (BLUE)")
        print(f"{'='*60}")

        cmd = [
            'python3', 'Hex.py',
            '-p1', p1_agent,
            '-p2', p2_agent,
            '-v'  # Verbose mode for detailed output
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per game
            )
            elapsed_time = time.time() - start_time

            # Parse output
            output = result.stdout + result.stderr
            game_result = self._parse_game_output(output, p1_name, p2_name, elapsed_time)
            game_result['game_number'] = game_num
            game_result['p1_agent'] = p1_name
            game_result['p2_agent'] = p2_name

            self.results['games'].append(game_result)

            # Print summary
            print(f"Result: {game_result['winner']} wins")
            print(f"Turns: {game_result['turns']}")
            print(f"Time: {elapsed_time:.2f}s")
            print(f"P1 ({p1_name}) time: {game_result.get('p1_time', 'N/A')}")
            print(f"P2 ({p2_name}) time: {game_result.get('p2_time', 'N/A')}")

            return game_result

        except subprocess.TimeoutExpired:
            print(f"TIMEOUT after 600s")
            return {
                'game_number': game_num,
                'p1_agent': p1_name,
                'p2_agent': p2_name,
                'winner': 'TIMEOUT',
                'turns': 0,
                'elapsed_time': 600,
                'error': 'Game timed out'
            }
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return {
                'game_number': game_num,
                'p1_agent': p1_name,
                'p2_agent': p2_name,
                'winner': 'ERROR',
                'turns': 0,
                'elapsed_time': time.time() - start_time,
                'error': str(e)
            }

    def _parse_game_output(self, output, p1_name, p2_name, elapsed_time):
        """Parse game output to extract statistics"""
        result = {
            'output': output[-5000:],  # Last 5000 chars
            'elapsed_time': elapsed_time,
            'winner': 'UNKNOWN',
            'turns': 0,
            'p1_time': 0.0,
            'p2_time': 0.0,
            'final_board': '',
            'termination_reason': ''
        }

        # Extract winner - try multiple patterns
        # Pattern 1: "winner,Alice,WIN" or "winner,Bob,WIN"
        winner_match = re.search(r'winner,(\w+),WIN', output)
        if winner_match:
            winner_name = winner_match.group(1)
            # Map agent names (Alice=p2, Bob=p1)
            if winner_name == 'Bob':
                result['winner'] = p1_name
                result['termination_reason'] = self._extract_termination_reason(output, 'RED')
            elif winner_name == 'Alice':
                result['winner'] = p2_name
                result['termination_reason'] = self._extract_termination_reason(output, 'BLUE')
        # Pattern 2: "Player Alice has won" or "Player Bob has won"
        elif 'Player Alice has won' in output:
            result['winner'] = p2_name
            result['termination_reason'] = self._extract_termination_reason(output, 'BLUE')
        elif 'Player Bob has won' in output:
            result['winner'] = p1_name
            result['termination_reason'] = self._extract_termination_reason(output, 'RED')
        # Pattern 3: "RED wins" or "BLUE wins" (old format)
        elif 'RED wins' in output or 'Player 1 (RED) wins' in output:
            result['winner'] = p1_name
            result['termination_reason'] = self._extract_termination_reason(output, 'RED')
        elif 'BLUE wins' in output or 'Player 2 (BLUE) wins' in output:
            result['winner'] = p2_name
            result['termination_reason'] = self._extract_termination_reason(output, 'BLUE')

        # Extract turn count
        turn_match = re.search(r'Turn (\d+)', output)
        if turn_match:
            result['turns'] = int(turn_match.group(1))

        # Try to extract timing info
        # Look for patterns like "Total time: X.XXs"
        p1_time_match = re.search(rf'{p1_name}.*?time.*?(\d+\.?\d*)s', output, re.IGNORECASE)
        p2_time_match = re.search(rf'{p2_name}.*?time.*?(\d+\.?\d*)s', output, re.IGNORECASE)

        if p1_time_match:
            result['p1_time'] = float(p1_time_match.group(1))
        if p2_time_match:
            result['p2_time'] = float(p2_time_match.group(1))

        return result

    def _extract_termination_reason(self, output, winning_color):
        """Determine why the game ended"""
        if 'illegal move' in output.lower():
            return 'Opponent made illegal move'
        elif 'timeout' in output.lower():
            return 'Opponent timed out'
        elif 'connected' in output.lower() or 'path' in output.lower():
            return f'{winning_color} formed winning connection'
        else:
            return 'Normal termination'

    def run_test_suite(self):
        """Run comprehensive test suite"""
        print(f"\n{'#'*60}")
        print(f"# COMPREHENSIVE TEST SUITE")
        print(f"# Target: {self.num_games} games")
        print(f"# Started: {self.results['metadata']['start_time']}")
        print(f"{'#'*60}\n")

        # Split games evenly: half with Simple as RED, half with Simple as BLUE
        games_per_color = self.num_games // 2

        # Simple as RED (P1) vs Full as BLUE (P2)
        print(f"\n{'='*60}")
        print(f"TEST BATCH 1: Simple (RED) vs Full (BLUE)")
        print(f"Games: {games_per_color}")
        print(f"{'='*60}")

        for i in range(games_per_color):
            self.run_single_game(
                'agents.Group12.Group12Agent_simple Group12Agent',
                'Simple',
                'agents.Group12.Group12Agent Group12Agent',
                'Full',
                i + 1
            )
            self._save_intermediate_results()
            time.sleep(1)  # Brief pause between games

        # Simple as BLUE (P2) vs Full as RED (P1)
        print(f"\n{'='*60}")
        print(f"TEST BATCH 2: Full (RED) vs Simple (BLUE)")
        print(f"Games: {games_per_color}")
        print(f"{'='*60}")

        for i in range(games_per_color):
            self.run_single_game(
                'agents.Group12.Group12Agent Group12Agent',
                'Full',
                'agents.Group12.Group12Agent_simple Group12Agent',
                'Simple',
                games_per_color + i + 1
            )
            self._save_intermediate_results()
            time.sleep(1)

        # Handle odd number of games
        if self.num_games % 2 == 1:
            print(f"\n{'='*60}")
            print(f"TIEBREAKER GAME")
            print(f"{'='*60}")
            self.run_single_game(
                'agents.Group12.Group12Agent_simple Group12Agent',
                'Simple',
                'agents.Group12.Group12Agent Group12Agent',
                'Full',
                self.num_games
            )

        # Compute final statistics
        self._compute_summary()
        self._save_final_results()
        self._print_comprehensive_report()

    def _compute_summary(self):
        """Compute comprehensive statistics"""
        games = self.results['games']
        valid_games = [g for g in games if g['winner'] not in ['TIMEOUT', 'ERROR', 'UNKNOWN']]

        if not valid_games:
            print("WARNING: No valid games completed!")
            return

        # Overall wins
        simple_wins = sum(1 for g in valid_games if g['winner'] == 'Simple')
        full_wins = sum(1 for g in valid_games if g['winner'] == 'Full')

        # Color-separated stats
        simple_as_red = [g for g in valid_games if g['p1_agent'] == 'Simple']
        simple_as_blue = [g for g in valid_games if g['p2_agent'] == 'Simple']

        simple_red_wins = sum(1 for g in simple_as_red if g['winner'] == 'Simple')
        simple_blue_wins = sum(1 for g in simple_as_blue if g['winner'] == 'Simple')

        # Timing statistics
        simple_times = []
        full_times = []

        for g in valid_games:
            if g['p1_agent'] == 'Simple' and g.get('p1_time'):
                simple_times.append(g['p1_time'])
            elif g['p2_agent'] == 'Simple' and g.get('p2_time'):
                simple_times.append(g['p2_time'])

            if g['p1_agent'] == 'Full' and g.get('p1_time'):
                full_times.append(g['p1_time'])
            elif g['p2_agent'] == 'Full' and g.get('p2_time'):
                full_times.append(g['p2_time'])

        # Calculate confidence interval (95%)
        import math
        n = len(valid_games)
        if n > 0:
            p = simple_wins / n
            se = math.sqrt(p * (1 - p) / n)
            ci_margin = 1.96 * se  # 95% confidence
            ci_lower = max(0, p - ci_margin)
            ci_upper = min(1, p + ci_margin)
        else:
            p, ci_lower, ci_upper = 0, 0, 0

        self.results['summary'] = {
            'total_games': len(games),
            'valid_games': len(valid_games),
            'timeouts': sum(1 for g in games if g['winner'] == 'TIMEOUT'),
            'errors': sum(1 for g in games if g['winner'] == 'ERROR'),

            'simple_wins': simple_wins,
            'full_wins': full_wins,
            'simple_win_rate': simple_wins / len(valid_games) if valid_games else 0,
            'full_win_rate': full_wins / len(valid_games) if valid_games else 0,

            'confidence_interval_95': {
                'point_estimate': p,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'margin_of_error': ci_margin
            },

            'by_color': {
                'simple_as_red': {
                    'games': len(simple_as_red),
                    'wins': simple_red_wins,
                    'win_rate': simple_red_wins / len(simple_as_red) if simple_as_red else 0
                },
                'simple_as_blue': {
                    'games': len(simple_as_blue),
                    'wins': simple_blue_wins,
                    'win_rate': simple_blue_wins / len(simple_as_blue) if simple_as_blue else 0
                }
            },

            'timing': {
                'simple_avg_time': sum(simple_times) / len(simple_times) if simple_times else 0,
                'full_avg_time': sum(full_times) / len(full_times) if full_times else 0,
                'simple_max_time': max(simple_times) if simple_times else 0,
                'full_max_time': max(full_times) if full_times else 0
            },

            'average_turns': sum(g['turns'] for g in valid_games) / len(valid_games) if valid_games else 0,
            'average_game_duration': sum(g['elapsed_time'] for g in valid_games) / len(valid_games) if valid_games else 0
        }

        self.results['metadata']['end_time'] = datetime.now().isoformat()

    def _save_intermediate_results(self):
        """Save results after each game"""
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _save_final_results(self):
        """Save final results with summary"""
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {self.output_file}")

    def _print_comprehensive_report(self):
        """Print detailed final report"""
        summary = self.results['summary']

        print(f"\n{'#'*70}")
        print(f"# COMPREHENSIVE TEST RESULTS")
        print(f"{'#'*70}\n")

        print(f"Total Games: {summary['total_games']}")
        print(f"Valid Games: {summary['valid_games']}")
        print(f"Timeouts: {summary['timeouts']}")
        print(f"Errors: {summary['errors']}")

        print(f"\n{'='*70}")
        print(f"OVERALL WIN RATES")
        print(f"{'='*70}")
        print(f"Simple wins: {summary['simple_wins']}/{summary['valid_games']} ({summary['simple_win_rate']*100:.1f}%)")
        print(f"Full wins:   {summary['full_wins']}/{summary['valid_games']} ({summary['full_win_rate']*100:.1f}%)")

        ci = summary['confidence_interval_95']
        print(f"\n95% Confidence Interval for Simple win rate:")
        print(f"  Point estimate: {ci['point_estimate']*100:.1f}%")
        print(f"  Range: [{ci['lower_bound']*100:.1f}%, {ci['upper_bound']*100:.1f}%]")
        print(f"  Margin of error: ±{ci['margin_of_error']*100:.1f}%")

        print(f"\n{'='*70}")
        print(f"BY COLOR ANALYSIS")
        print(f"{'='*70}")

        red_stats = summary['by_color']['simple_as_red']
        blue_stats = summary['by_color']['simple_as_blue']

        print(f"Simple as RED (first player):")
        print(f"  Wins: {red_stats['wins']}/{red_stats['games']} ({red_stats['win_rate']*100:.1f}%)")

        print(f"\nSimple as BLUE (second player):")
        print(f"  Wins: {blue_stats['wins']}/{blue_stats['games']} ({blue_stats['win_rate']*100:.1f}%)")

        # First-player advantage
        if red_stats['games'] > 0 and blue_stats['games'] > 0:
            first_player_advantage = red_stats['win_rate'] - blue_stats['win_rate']
            print(f"\nFirst-player advantage: {first_player_advantage*100:+.1f}%")

        print(f"\n{'='*70}")
        print(f"TIMING ANALYSIS")
        print(f"{'='*70}")

        timing = summary['timing']
        print(f"Simple agent:")
        print(f"  Average time: {timing['simple_avg_time']:.3f}s")
        print(f"  Max time: {timing['simple_max_time']:.3f}s")

        print(f"\nFull agent:")
        print(f"  Average time: {timing['full_avg_time']:.3f}s")
        print(f"  Max time: {timing['full_max_time']:.3f}s")

        speedup = timing['full_avg_time'] / timing['simple_avg_time'] if timing['simple_avg_time'] > 0 else 0
        print(f"\nSimple is {speedup:.1f}x faster than Full")

        print(f"\n{'='*70}")
        print(f"GAME CHARACTERISTICS")
        print(f"{'='*70}")
        print(f"Average turns: {summary['average_turns']:.1f}")
        print(f"Average game duration: {summary['average_game_duration']:.1f}s")

        print(f"\n{'#'*70}")
        print(f"# TEST COMPLETE")
        print(f"{'#'*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive Hex AI tests')
    parser.add_argument('--games', type=int, default=50,
                       help='Number of games to run (default: 50)')
    parser.add_argument('--output', type=str, default='results_comprehensive.json',
                       help='Output JSON file (default: results_comprehensive.json)')

    args = parser.parse_args()

    runner = ComprehensiveTestRunner(num_games=args.games, output_file=args.output)
    runner.run_test_suite()


if __name__ == '__main__':
    main()
