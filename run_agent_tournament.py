#!/usr/bin/env python3
"""
Multi-Agent Tournament Script
Runs round-robin matches between all agents and reports results.
"""

import subprocess
import sys
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple

# Define agents to test
AGENTS = {
    "AzaleaAgent": "agents.Group12.AzaleaAgent AzaleaAgent",
    "HexMastUltra2": "agents.Group12.HexMastUltra2 HexMastUltra2",
    "Tournament": "agents.Group12.Group12Agent_tournament Group12Agent",
    "NaiveAgent": "agents.DefaultAgents.NaiveAgent NaiveAgent",
}

# Number of games per matchup (half as RED, half as BLUE)
GAMES_PER_MATCHUP = 4


def run_game(p1_name: str, p1_cmd: str, p2_name: str, p2_cmd: str) -> str:
    """Run a single game and return the winner's name."""
    cmd = [
        "python3", "Hex.py",
        "-p1", p1_cmd,
        "-p1Name", p1_name,
        "-p2", p2_cmd,
        "-p2Name", p2_name,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Parse output to find winner
        for line in result.stderr.split('\n'):
            if 'has won' in line:
                if p1_name in line:
                    return p1_name
                elif p2_name in line:
                    return p2_name

        # Check stdout too
        for line in result.stdout.split('\n'):
            if 'winner,' in line.lower():
                parts = line.split(',')
                if len(parts) >= 2:
                    winner = parts[1].strip()
                    return winner

        return "Unknown"
    except subprocess.TimeoutExpired:
        return "Timeout"
    except Exception as e:
        print(f"Error running game: {e}")
        return "Error"


def main():
    print("=" * 60)
    print("MULTI-AGENT HEX TOURNAMENT")
    print("=" * 60)
    print(f"Agents: {list(AGENTS.keys())}")
    print(f"Games per matchup: {GAMES_PER_MATCHUP}")
    print("=" * 60)

    # Track results
    wins: Dict[str, int] = defaultdict(int)
    losses: Dict[str, int] = defaultdict(int)
    head_to_head: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Get all pairs
    agent_names = list(AGENTS.keys())
    pairs = list(itertools.combinations(agent_names, 2))

    total_games = len(pairs) * GAMES_PER_MATCHUP
    game_num = 0

    for agent1, agent2 in pairs:
        print(f"\n--- {agent1} vs {agent2} ---")

        for i in range(GAMES_PER_MATCHUP):
            game_num += 1

            # Alternate colors
            if i % 2 == 0:
                red_name, red_cmd = agent1, AGENTS[agent1]
                blue_name, blue_cmd = agent2, AGENTS[agent2]
            else:
                red_name, red_cmd = agent2, AGENTS[agent2]
                blue_name, blue_cmd = agent1, AGENTS[agent1]

            print(f"  Game {game_num}/{total_games}: {red_name}(RED) vs {blue_name}(BLUE)...", end=" ", flush=True)

            winner = run_game(red_name, red_cmd, blue_name, blue_cmd)

            if winner == red_name:
                wins[red_name] += 1
                losses[blue_name] += 1
                head_to_head[(agent1, agent2)][red_name] += 1
                print(f"Winner: {red_name} (RED)")
            elif winner == blue_name:
                wins[blue_name] += 1
                losses[red_name] += 1
                head_to_head[(agent1, agent2)][blue_name] += 1
                print(f"Winner: {blue_name} (BLUE)")
            else:
                print(f"Result: {winner}")

    # Print results
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)

    # Calculate win rates
    print("\nOverall Performance:")
    print("-" * 40)
    results = []
    for agent in agent_names:
        total = wins[agent] + losses[agent]
        win_rate = wins[agent] / total * 100 if total > 0 else 0
        results.append((agent, wins[agent], losses[agent], win_rate))

    # Sort by win rate
    results.sort(key=lambda x: x[3], reverse=True)

    print(f"{'Agent':<20} {'Wins':<6} {'Losses':<8} {'Win Rate':<10}")
    print("-" * 40)
    for agent, w, l, wr in results:
        print(f"{agent:<20} {w:<6} {l:<8} {wr:.1f}%")

    # Head-to-head
    print("\nHead-to-Head Results:")
    print("-" * 40)
    for (a1, a2), scores in head_to_head.items():
        print(f"{a1} vs {a2}: {scores[a1]}-{scores[a2]}")

    print("\n" + "=" * 60)
    print(f"TOURNAMENT CHAMPION: {results[0][0]} ({results[0][3]:.1f}% win rate)")
    print("=" * 60)


if __name__ == "__main__":
    main()
