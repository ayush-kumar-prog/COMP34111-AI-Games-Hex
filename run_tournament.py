#!/usr/bin/env python3
"""Tournament script for KataHex vs AzaleaAgent."""
import subprocess
import sys

kata_wins = 0
azalea_wins = 0
num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 5

for i in range(num_games):
    if i % 2 == 0:
        p1 = 'agents.Group12.KataHexAgent KataHexAgent'
        p2 = 'agents.Group12.AzaleaAgent AzaleaAgent'
        p1_is_kata = True
    else:
        p1 = 'agents.Group12.AzaleaAgent AzaleaAgent'
        p2 = 'agents.Group12.KataHexAgent KataHexAgent'
        p1_is_kata = False

    try:
        result = subprocess.run(
            ['python3', 'Hex.py', '-p1', p1, '-p2', p2],
            capture_output=True, text=True, timeout=600
        )
        output = result.stderr + result.stdout

        if 'winner,Alice' in output:
            winner = 'p1'
        elif 'winner,Bob' in output:
            winner = 'p2'
        else:
            winner = 'unknown'
            print(f"Game {i+1}: Unknown result")
            continue

        if p1_is_kata:
            if winner == 'p1':
                kata_wins += 1
            else:
                azalea_wins += 1
        else:
            if winner == 'p2':
                kata_wins += 1
            else:
                azalea_wins += 1

        print(f'Game {i+1}/{num_games}: KataHex={kata_wins}, Azalea={azalea_wins}')

    except subprocess.TimeoutExpired:
        print(f"Game {i+1}: Timeout!")
    except Exception as e:
        print(f"Game {i+1}: Error - {e}")

print(f"\n=== FINAL RESULTS ===")
print(f"KataHex: {kata_wins} wins")
print(f"Azalea:  {azalea_wins} wins")
total = kata_wins + azalea_wins
if total > 0:
    print(f"KataHex win rate: {kata_wins/total*100:.1f}%")
