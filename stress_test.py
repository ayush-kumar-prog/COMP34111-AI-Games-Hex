#!/usr/bin/env python3
"""
Comprehensive stress testing to find ALL failure modes.
"""

import sys
import subprocess
import json
import time
from pathlib import Path

def run_game(p1_spec, p2_spec, game_id):
    """Run a single game and capture results."""
    cmd = [
        "python3", "Hex.py",
        "-p1", p1_spec,
        "-p2", p2_spec
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=240  # 4 minute timeout per game
        )

        output = result.stdout + result.stderr

        # Parse result
        winner = None
        reason = None
        times = {}

        for line in output.split('\n'):
            if 'winner,' in line:
                parts = line.split(',')
                if len(parts) >= 3:
                    winner = parts[1]
                    reason = parts[2] if len(parts) > 2 else "UNKNOWN"
            if line.startswith('Alice,'):
                times['Alice'] = int(line.split(',')[1])
            if line.startswith('Bob,'):
                times['Bob'] = int(line.split(',')[1])

        return {
            'game_id': game_id,
            'winner': winner,
            'reason': reason,
            'times': times,
            'output': output,
            'success': result.returncode == 0
        }

    except subprocess.TimeoutExpired:
        return {
            'game_id': game_id,
            'winner': None,
            'reason': 'TIMEOUT',
            'times': {},
            'output': f"Game {game_id} timed out after 240s",
            'success': False
        }
    except Exception as e:
        return {
            'game_id': game_id,
            'winner': None,
            'reason': 'ERROR',
            'times': {},
            'output': str(e),
            'success': False
        }

def stress_test_suite():
    """Run comprehensive stress tests."""

    agents = {
        'Group12_simple': 'agents.Group12.Group12Agent_simple Group12Agent',
        'Group12_full': 'agents.Group12.Group12Agent Group12Agent',
        'NaiveAgent': 'agents.DefaultAgents.NaiveAgent NaiveAgent',
    }

    test_suites = [
        {
            'name': 'Simple vs NaiveAgent (10 games)',
            'p1': 'Group12_simple',
            'p2': 'NaiveAgent',
            'count': 10,
            'swap_colors': True
        },
        {
            'name': 'Full vs NaiveAgent (10 games)',
            'p1': 'Group12_full',
            'p2': 'NaiveAgent',
            'count': 10,
            'swap_colors': True
        },
        {
            'name': 'Full vs Simple (5 games)',
            'p1': 'Group12_full',
            'p2': 'Group12_simple',
            'count': 5,
            'swap_colors': True
        },
        {
            'name': 'Self-play Full vs Full (3 games)',
            'p1': 'Group12_full',
            'p2': 'Group12_full',
            'count': 3,
            'swap_colors': False
        },
    ]

    all_results = []
    failure_modes = []

    print("="*80)
    print("COMPREHENSIVE STRESS TESTING")
    print("="*80)
    print(f"\nTotal test suites: {len(test_suites)}")
    print(f"Available agents: {list(agents.keys())}")
    print()

    for suite in test_suites:
        print(f"\n{'='*80}")
        print(f"TEST SUITE: {suite['name']}")
        print(f"{'='*80}")

        suite_results = []

        for i in range(suite['count']):
            game_id = f"{suite['name']}_game_{i+1}"

            p1_spec = agents[suite['p1']]
            p2_spec = agents[suite['p2']]

            print(f"\n[{i+1}/{suite['count']}] {suite['p1']} vs {suite['p2']}", end=" ... ")
            sys.stdout.flush()

            result = run_game(p1_spec, p2_spec, game_id)
            suite_results.append(result)
            all_results.append(result)

            if result['success']:
                print(f"✓ Winner: {result['winner']} ({result['reason']})")

                # Analyze time
                if result['times']:
                    for player, time_ns in result['times'].items():
                        time_s = time_ns / 1e9
                        if time_s > 170:  # Close to 180s limit
                            failure_modes.append({
                                'type': 'TIME_PRESSURE',
                                'game': game_id,
                                'player': player,
                                'time': time_s,
                                'details': f"{player} used {time_s:.1f}s (close to 180s limit)"
                            })
                            print(f"    ⚠️  {player}: {time_s:.1f}s (TIME PRESSURE!)")
                        elif time_s > 100:
                            print(f"    ⚠️  {player}: {time_s:.1f}s (high time)")
                        else:
                            print(f"       {player}: {time_s:.1f}s")
            else:
                print(f"✗ FAILURE: {result['reason']}")
                failure_modes.append({
                    'type': result['reason'],
                    'game': game_id,
                    'details': result['output'][:500]  # First 500 chars
                })

            # Swap colors if requested
            if suite['swap_colors'] and i < suite['count'] - 1 and (i % 2 == 0):
                # Next game swap p1 and p2
                suite['p1'], suite['p2'] = suite['p2'], suite['p1']

        # Suite summary
        wins = {}
        for r in suite_results:
            if r['winner']:
                wins[r['winner']] = wins.get(r['winner'], 0) + 1

        print(f"\nSuite Results:")
        print(f"  Total games: {len(suite_results)}")
        print(f"  Successful: {sum(1 for r in suite_results if r['success'])}")
        print(f"  Failed: {sum(1 for r in suite_results if not r['success'])}")
        for player, count in wins.items():
            print(f"  {player} wins: {count}")

    # Final analysis
    print(f"\n{'='*80}")
    print("FINAL ANALYSIS")
    print(f"{'='*80}")

    print(f"\nTotal games run: {len(all_results)}")
    print(f"Successful: {sum(1 for r in all_results if r['success'])}")
    print(f"Failed: {sum(1 for r in all_results if not r['success'])}")

    print(f"\nFailure Modes Found: {len(failure_modes)}")

    if failure_modes:
        print("\nDETAILED FAILURE ANALYSIS:")

        # Group by type
        by_type = {}
        for fm in failure_modes:
            fm_type = fm['type']
            if fm_type not in by_type:
                by_type[fm_type] = []
            by_type[fm_type].append(fm)

        for fm_type, instances in by_type.items():
            print(f"\n{fm_type} ({len(instances)} occurrences):")
            for instance in instances:
                print(f"  - {instance['game']}")
                if 'details' in instance:
                    print(f"    {instance['details'][:200]}")

    # Save detailed results
    output_file = f"stress_test_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'all_results': all_results,
            'failure_modes': failure_modes
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    return all_results, failure_modes

if __name__ == "__main__":
    results, failures = stress_test_suite()

    # Exit with error if any failures
    if failures:
        print(f"\n⚠️  Found {len(failures)} failure modes!")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)
