#!/usr/bin/env python3
"""
Test both agents against RandomAgent.
"""

import subprocess
import sys

def run_game(p1_spec, p2_spec):
    """Run a single game."""
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
            timeout=240
        )

        output = result.stdout + result.stderr

        # Parse result
        winner = None
        reason = None
        times = {}

        for line in output.split('\n'):
            if 'winner,' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    winner = parts[1]
                    reason = parts[2] if len(parts) > 2 else "UNKNOWN"
            if line.startswith('Alice,'):
                times['Alice'] = int(line.split(',')[1])
            if line.startswith('Bob,'):
                times['Bob'] = int(line.split(',')[1])

        return {
            'winner': winner,
            'reason': reason,
            'times': times,
            'success': result.returncode == 0
        }

    except subprocess.TimeoutExpired:
        return {
            'winner': None,
            'reason': 'TIMEOUT',
            'times': {},
            'success': False
        }
    except Exception as e:
        return {
            'winner': None,
            'reason': 'ERROR',
            'times': {},
            'success': False
        }

def main():
    """Run tests."""
    agents = {
        'Simple': 'agents.Group12.Group12Agent_simple Group12Agent',
        'Full': 'agents.Group12.Group12Agent Group12Agent',
        'Random': 'agents.DefaultAgents.RandomAgent RandomAgent',
    }

    tests = [
        ('Simple', 'Random', 5),
        ('Full', 'Random', 5),
    ]

    print("=" * 80)
    print("TESTING VS RANDOMAGENT")
    print("=" * 80)

    for agent1, agent2, count in tests:
        print(f"\n{agent1} vs {agent2} ({count} games)")
        print("-" * 80)

        wins = {agent1: 0, agent2: 0}
        times = {agent1: [], agent2: []}

        for i in range(count):
            # Alternate colors
            if i % 2 == 0:
                p1, p2 = agent1, agent2
            else:
                p1, p2 = agent2, agent1

            print(f"[{i+1}/{count}] {p1} (RED) vs {p2} (BLUE)", end=" ... ")
            sys.stdout.flush()

            result = run_game(agents[p1], agents[p2])

            if result['success']:
                # Determine winner
                if result['winner'] == 'Alice':
                    winner = p1
                else:
                    winner = p2

                wins[winner] += 1

                # Record times
                if result['times']:
                    if 'Alice' in result['times']:
                        times[p1].append(result['times']['Alice'] / 1e9)
                    if 'Bob' in result['times']:
                        times[p2].append(result['times']['Bob'] / 1e9)

                print(f"✓ Winner: {winner} ({result['reason']})")
            else:
                print(f"✗ FAILED: {result['reason']}")

        # Summary
        print(f"\nResults:")
        print(f"  {agent1}: {wins[agent1]}/{count} wins ({wins[agent1]/count*100:.0f}%)")
        print(f"  {agent2}: {wins[agent2]}/{count} wins ({wins[agent2]/count*100:.0f}%)")

        if times[agent1]:
            print(f"  {agent1} avg time: {sum(times[agent1])/len(times[agent1]):.2f}s")
        if times[agent2]:
            print(f"  {agent2} avg time: {sum(times[agent2])/len(times[agent2]):.2f}s")

if __name__ == "__main__":
    main()
