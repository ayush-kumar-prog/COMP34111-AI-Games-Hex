#!/usr/bin/env python3
"""
Tournament: AzaleaAgent vs Group12Agent_tournament (20 games)

Run on CSF3:
    cd ~/scratch/COMP34111-AI-Games-Hex
    git pull origin simulation
    source venv/bin/activate
    python3 tournament_azalea_vs_group12.py
"""

import subprocess
import sys

NUM_GAMES = 20

def run_game(game_num: int) -> dict:
    """Run a single game and return result."""
    # Alternate who plays first
    if game_num % 2 == 0:
        p1 = "agents.Group12.AzaleaAgent AzaleaAgent"
        p2 = "agents.Group12.Group12Agent_tournament Group12Agent_tournament"
        p1_name = "Azalea"
        p2_name = "Group12"
    else:
        p1 = "agents.Group12.Group12Agent_tournament Group12Agent_tournament"
        p2 = "agents.Group12.AzaleaAgent AzaleaAgent"
        p1_name = "Group12"
        p2_name = "Azalea"

    cmd = ["python3", "Hex.py", "-p1", p1, "-p2", p2]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    # Parse result
    output = result.stdout + result.stderr
    winner = None

    # Look for winner in output - game uses player names Alice (P1) and Bob (P2)
    if "Player Alice has won" in output or "winner,Alice" in output:
        winner = p1_name  # Alice is always P1
    elif "Player Bob has won" in output or "winner,Bob" in output:
        winner = p2_name  # Bob is always P2
    elif "RED wins" in output:
        winner = p1_name
    elif "BLUE wins" in output:
        winner = p2_name

    return {
        "game": game_num + 1,
        "p1": p1_name,
        "p2": p2_name,
        "winner": winner,
        "output": output[-500:]  # Last 500 chars
    }


def main():
    print("=" * 70)
    print("TOURNAMENT: AzaleaAgent vs Group12Agent_tournament")
    print(f"Total games: {NUM_GAMES}")
    print("=" * 70)

    azalea_wins = 0
    group12_wins = 0
    results = []

    for i in range(NUM_GAMES):
        print(f"\nGame {i+1}/{NUM_GAMES}...", end=" ", flush=True)
        try:
            result = run_game(i)
            results.append(result)

            if result["winner"] == "Azalea":
                azalea_wins += 1
                print(f"Azalea wins! ({result['p1']} as P1)")
            elif result["winner"] == "Group12":
                group12_wins += 1
                print(f"Group12 wins! ({result['p1']} as P1)")
            else:
                print(f"Unknown result ({result['p1']} as P1)")
                print(f"Output: {result['output']}")

        except subprocess.TimeoutExpired:
            print("TIMEOUT!")
            results.append({"game": i+1, "winner": None, "error": "timeout"})
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"game": i+1, "winner": None, "error": str(e)})

    # Print summary
    print("\n" + "=" * 70)
    print("TOURNAMENT SUMMARY")
    print("=" * 70)
    print(f"Azalea wins:       {azalea_wins}/{NUM_GAMES} ({100*azalea_wins/NUM_GAMES:.1f}%)")
    print(f"Group12 wins:      {group12_wins}/{NUM_GAMES} ({100*group12_wins/NUM_GAMES:.1f}%)")
    print(f"Other:             {NUM_GAMES - azalea_wins - group12_wins}/{NUM_GAMES}")

    # Detailed breakdown
    print("\nDetailed Results:")
    print("-" * 50)
    for r in results:
        if "error" in r:
            print(f"Game {r['game']}: ERROR - {r['error']}")
        else:
            print(f"Game {r['game']}: {r['p1']} vs {r['p2']} -> {r['winner']} wins")


if __name__ == "__main__":
    main()
