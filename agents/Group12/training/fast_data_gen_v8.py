#!/usr/bin/env python3
"""
Fast Parallel Data Generation for V8 Neural Network
====================================================
Generates training data with:
- Random openings (3-7 moves) for diverse outcomes
- Multiprocessed game generation (12 workers)
- 200 MCTS iterations for quality
- Proper value assignment based on actual game outcomes
- File-based progress monitoring for debugging

Target: 12K games in ~5-6 hours
"""

import os
import sys
import random
import time
import math
import json
import numpy as np
from multiprocessing import Pool, cpu_count, get_start_method, set_start_method
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta

# Force unbuffered output for real-time monitoring
sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

# Configuration
BOARD_SIZE = 11
MCTS_ITERATIONS = 200      # Balanced quality vs speed
EXPLORATION_C = 1.41       # UCB1 exploration constant
RAVE_K = 300               # RAVE equivalence parameter
RANDOM_OPENING_MIN = 3     # Minimum random opening moves
RANDOM_OPENING_MAX = 7     # Maximum random opening moves (more variety)
NUM_GAMES = 12000          # Target games (achievable in 5-6h with 400 iter)
NUM_WORKERS = 12           # Parallel workers
CHECKPOINT_EVERY = 300     # More frequent checkpoints for monitoring
OUTPUT_FILE = "data/expert_games_v8.npz"

# Colors
RED = 1    # First player, connects top-bottom
BLUE = 2   # Second player, connects left-right


class FastBoard:
    """Lightweight board for fast game simulation."""

    def __init__(self, size=BOARD_SIZE):
        self.size = size
        self.board = [[0] * size for _ in range(size)]
        self.current_player = RED
        self.move_count = 0

    def copy(self):
        new_board = FastBoard(self.size)
        new_board.board = [row[:] for row in self.board]
        new_board.current_player = self.current_player
        new_board.move_count = self.move_count
        return new_board

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.size)
                for c in range(self.size) if self.board[r][c] == 0]

    def play(self, row: int, col: int) -> bool:
        if self.board[row][col] != 0:
            return False
        self.board[row][col] = self.current_player
        self.current_player = BLUE if self.current_player == RED else RED
        self.move_count += 1
        return True

    def check_winner(self) -> Optional[int]:
        """Check if there's a winner using flood fill."""
        # Check RED (top to bottom)
        visited = set()
        stack = [(0, c) for c in range(self.size) if self.board[0][c] == RED]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            if r == self.size - 1:
                return RED
            for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr][nc] == RED and (nr, nc) not in visited:
                        stack.append((nr, nc))

        # Check BLUE (left to right)
        visited = set()
        stack = [(r, 0) for r in range(self.size) if self.board[r][0] == BLUE]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            if c == self.size - 1:
                return BLUE
            for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr][nc] == BLUE and (nr, nc) not in visited:
                        stack.append((nr, nc))

        return None

    def is_terminal(self) -> bool:
        return self.check_winner() is not None or len(self.get_legal_moves()) == 0


class MCTSNode:
    """MCTS node with RAVE support."""

    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.wins = 0
        self.rave_visits = 0
        self.rave_wins = 0
        self.untried_moves = None

    def ucb_rave(self, exploration=EXPLORATION_C, rave_k=RAVE_K) -> float:
        if self.visits == 0:
            return float('inf')

        # UCB1 component
        exploit = self.wins / self.visits
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

        # RAVE component
        if self.rave_visits > 0:
            beta = self.rave_visits / (self.visits + self.rave_visits +
                                       4 * self.visits * self.rave_visits / rave_k)
            rave_value = self.rave_wins / self.rave_visits
            return (1 - beta) * exploit + beta * rave_value + explore

        return exploit + explore


class FastMCTS:
    """Fast MCTS with RAVE for data generation."""

    def __init__(self, board: FastBoard, iterations: int = MCTS_ITERATIONS):
        self.root_board = board
        self.iterations = iterations
        self.root = MCTSNode()
        self.root.untried_moves = board.get_legal_moves()

    def search(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Run MCTS and return (policy distribution, best move)."""
        root_player = self.root_board.current_player

        for _ in range(self.iterations):
            node = self.root
            board = self.root_board.copy()
            moves_played = []

            # Selection
            while node.untried_moves is not None and len(node.untried_moves) == 0 and node.children:
                node = max(node.children.values(), key=lambda n: n.ucb_rave())
                board.play(node.move[0], node.move[1])
                moves_played.append((node.move, board.current_player))

            # Expansion
            if node.untried_moves is None:
                node.untried_moves = board.get_legal_moves()

            if node.untried_moves and not board.is_terminal():
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                board.play(move[0], move[1])
                moves_played.append((move, board.current_player))
                child = MCTSNode(move=move, parent=node)
                child.untried_moves = board.get_legal_moves()
                node.children[move] = child
                node = child

            # Simulation (random rollout)
            sim_moves = []
            while not board.is_terminal():
                legal = board.get_legal_moves()
                if not legal:
                    break
                move = random.choice(legal)
                sim_moves.append((move, board.current_player))
                board.play(move[0], move[1])

            # Determine winner
            winner = board.check_winner()
            result = 1 if winner == root_player else 0 if winner else 0.5

            # Backpropagation with RAVE
            all_moves = moves_played + sim_moves
            while node is not None:
                node.visits += 1
                node.wins += result

                # RAVE update
                for move, player in all_moves:
                    if move in node.children:
                        child = node.children[move]
                        child.rave_visits += 1
                        if player != root_player:  # Opponent's perspective
                            child.rave_wins += result

                node = node.parent
                result = 1 - result  # Flip for parent's perspective

        # Create policy from visit counts
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        total_visits = sum(c.visits for c in self.root.children.values())

        if total_visits > 0:
            for move, child in self.root.children.items():
                idx = move[0] * BOARD_SIZE + move[1]
                policy[idx] = child.visits / total_visits

        # Best move by visit count
        if self.root.children:
            best_move = max(self.root.children.items(), key=lambda x: x[1].visits)[0]
        else:
            legal = self.root_board.get_legal_moves()
            best_move = random.choice(legal) if legal else (0, 0)

        return policy, best_move


def encode_board(board: FastBoard) -> np.ndarray:
    """
    8-channel board encoding:
    0: Current player's pieces
    1: Opponent's pieces
    2: Empty cells
    3: Current player's distance to first edge
    4: Current player's distance to second edge
    5: Opponent's distance to first edge
    6: Opponent's distance to second edge
    7: Turn indicator (all 1s if RED, all 0s if BLUE)
    """
    state = np.zeros((8, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    current = board.current_player
    opponent = BLUE if current == RED else RED

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = board.board[r][c]
            if cell == current:
                state[0, r, c] = 1.0
            elif cell == opponent:
                state[1, r, c] = 1.0
            else:
                state[2, r, c] = 1.0

    # Distance channels (normalized)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if current == RED:  # RED connects top-bottom
                state[3, r, c] = r / (BOARD_SIZE - 1)
                state[4, r, c] = (BOARD_SIZE - 1 - r) / (BOARD_SIZE - 1)
                state[5, r, c] = c / (BOARD_SIZE - 1)
                state[6, r, c] = (BOARD_SIZE - 1 - c) / (BOARD_SIZE - 1)
            else:  # BLUE connects left-right
                state[3, r, c] = c / (BOARD_SIZE - 1)
                state[4, r, c] = (BOARD_SIZE - 1 - c) / (BOARD_SIZE - 1)
                state[5, r, c] = r / (BOARD_SIZE - 1)
                state[6, r, c] = (BOARD_SIZE - 1 - r) / (BOARD_SIZE - 1)

    # Turn indicator
    state[7, :, :] = 1.0 if current == RED else 0.0

    return state


def generate_single_game(game_id: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], int, int]:
    """
    Generate one game with random opening + MCTS expert play.
    Returns: (positions, winner, num_moves)
    """
    # Seed with game_id for reproducibility but also add randomness
    random.seed(game_id * 31337 + int.from_bytes(os.urandom(4), 'big') % 1000000)

    board = FastBoard()
    positions = []

    # Random opening phase (3-7 moves for more variety)
    num_random = random.randint(RANDOM_OPENING_MIN, RANDOM_OPENING_MAX)
    for _ in range(num_random):
        legal = board.get_legal_moves()
        if not legal or board.is_terminal():
            break

        move = random.choice(legal)

        # Record position (one-hot policy for random moves)
        state = encode_board(board)
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        policy[move[0] * BOARD_SIZE + move[1]] = 1.0
        positions.append({
            'state': state,
            'policy': policy,
            'player': board.current_player
        })

        board.play(move[0], move[1])

    # Expert MCTS phase
    while not board.is_terminal():
        legal = board.get_legal_moves()
        if not legal:
            break

        # Run MCTS
        mcts = FastMCTS(board, iterations=MCTS_ITERATIONS)
        policy, best_move = mcts.search()

        # Record position
        state = encode_board(board)
        positions.append({
            'state': state,
            'policy': policy,
            'player': board.current_player
        })

        board.play(best_move[0], best_move[1])

    # Determine winner and assign values
    winner = board.check_winner()

    results = []
    for pos in positions:
        if winner is None:
            value = 0.0  # Draw (shouldn't happen in Hex)
        elif pos['player'] == winner:
            value = 1.0  # Win for this player
        else:
            value = -1.0  # Loss for this player

        results.append((pos['state'], pos['policy'], value))

    return results, winner if winner else 0, board.move_count


def worker_init():
    """Initialize worker process with unique random seed."""
    import os
    seed = os.getpid() + int.from_bytes(os.urandom(4), 'big')
    random.seed(seed)
    # Debug: Confirm worker started
    print(f"  [Worker {os.getpid()}] Initialized with seed {seed % 1000000}", flush=True)


STATUS_FILE = "data/v8_datagen_status.json"


def write_status(status_dict):
    """Write current status to JSON file for external monitoring."""
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(status_dict, f, indent=2)
    except Exception as e:
        print(f"  Warning: Could not write status file: {e}", flush=True)


def print_flush(*args, **kwargs):
    """Print with immediate flush for real-time monitoring."""
    print(*args, **kwargs)
    sys.stdout.flush()


def main():
    print_flush("=" * 70)
    print_flush("V8 High-Quality Data Generation")
    print_flush("=" * 70)
    print_flush(f"Start time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_flush(f"Target games:        {NUM_GAMES}")
    print_flush(f"Workers:             {NUM_WORKERS}")
    print_flush(f"MCTS iterations:     {MCTS_ITERATIONS} (high quality)")
    print_flush(f"Random opening:      {RANDOM_OPENING_MIN}-{RANDOM_OPENING_MAX} moves")
    print_flush(f"Output:              {OUTPUT_FILE}")
    print_flush(f"Status file:         {STATUS_FILE}")
    print_flush("=" * 70)

    # Debug multiprocessing setup
    print_flush("\n[DEBUG] Multiprocessing Configuration:")
    print_flush(f"  CPU count:         {cpu_count()}")
    print_flush(f"  Start method:      {get_start_method()}")

    # Try to use 'spawn' for safer multiprocessing on Linux
    try:
        if get_start_method() != 'spawn':
            set_start_method('spawn', force=True)
            print_flush(f"  Changed to:        spawn (safer for SLURM)")
    except RuntimeError:
        print_flush(f"  Note: Could not change start method (already set)")

    print_flush(f"  Workers requested: {NUM_WORKERS}")
    print_flush("")

    start_time = time.time()

    # Create data directory if needed
    os.makedirs("data", exist_ok=True)

    # Initialize status file
    write_status({
        "status": "starting",
        "start_time": datetime.now().isoformat(),
        "target_games": NUM_GAMES,
        "games_completed": 0,
        "positions": 0,
        "red_wins": 0,
        "blue_wins": 0
    })

    all_states = []
    all_policies = []
    all_values = []
    games_completed = 0
    red_wins = 0
    blue_wins = 0
    total_moves = 0

    # Process games in batches for checkpointing
    batch_size = CHECKPOINT_EVERY

    print_flush(f"\n[INFO] Starting Pool with {NUM_WORKERS} workers...")
    print_flush("[INFO] Worker initialization output will appear below:")

    with Pool(NUM_WORKERS, initializer=worker_init) as pool:
        print_flush(f"\n[INFO] Pool created successfully. Starting game generation...\n")

        for batch_start in range(0, NUM_GAMES, batch_size):
            batch_end = min(batch_start + batch_size, NUM_GAMES)
            batch_games = list(range(batch_start, batch_end))

            print_flush(f"[BATCH] Processing games {batch_start} to {batch_end}...")
            batch_start_time = time.time()

            # Generate games in parallel
            results = pool.map(generate_single_game, batch_games)

            # Flatten results and track stats
            for game_positions, winner, num_moves in results:
                for state, policy, value in game_positions:
                    all_states.append(state)
                    all_policies.append(policy)
                    all_values.append(value)
                if winner == RED:
                    red_wins += 1
                elif winner == BLUE:
                    blue_wins += 1
                total_moves += num_moves

            games_completed = batch_end
            batch_time = time.time() - batch_start_time
            total_time = time.time() - start_time

            # Stats
            games_per_sec = len(batch_games) / batch_time
            positions_so_far = len(all_states)
            avg_game_len = total_moves / games_completed if games_completed > 0 else 0

            # Value distribution
            values_arr = np.array(all_values)
            pos_count = (values_arr == 1).sum()
            neg_count = (values_arr == -1).sum()

            # Win rate (this is the key metric for outcome diversity!)
            total_decided = red_wins + blue_wins
            red_rate = 100 * red_wins / total_decided if total_decided > 0 else 0
            blue_rate = 100 * blue_wins / total_decided if total_decided > 0 else 0

            # Print progress
            print_flush(f"  Batch time: {batch_time:.1f}s ({games_per_sec:.2f} games/sec)")
            print_flush(f"  Total positions: {positions_so_far} (avg {avg_game_len:.1f} moves/game)")
            print_flush(f"  Win rates: RED={red_wins} ({red_rate:.1f}%), BLUE={blue_wins} ({blue_rate:.1f}%)")
            print_flush(f"  Value dist: +1={pos_count} ({100*pos_count/len(values_arr):.1f}%), "
                  f"-1={neg_count} ({100*neg_count/len(values_arr):.1f}%)")

            # Update status file (can be read at any time to monitor progress)
            eta_seconds = (total_time / games_completed) * (NUM_GAMES - games_completed) if games_completed > 0 else 0
            write_status({
                "status": "running",
                "last_update": datetime.now().isoformat(),
                "target_games": NUM_GAMES,
                "games_completed": games_completed,
                "positions": positions_so_far,
                "red_wins": red_wins,
                "blue_wins": blue_wins,
                "red_rate_pct": round(red_rate, 1),
                "blue_rate_pct": round(blue_rate, 1),
                "games_per_sec": round(games_per_sec, 2),
                "avg_game_len": round(avg_game_len, 1),
                "elapsed_seconds": round(total_time, 0),
                "eta_seconds": round(eta_seconds, 0),
                "eta_readable": str(timedelta(seconds=int(eta_seconds)))
            })

            # Checkpoint
            if games_completed % 1000 == 0 or games_completed == NUM_GAMES:
                checkpoint_file = f"data/expert_games_v8_checkpoint_{games_completed}.npz"
                np.savez_compressed(
                    checkpoint_file,
                    states=np.array(all_states, dtype=np.float32),
                    policies=np.array(all_policies, dtype=np.float32),
                    values=np.array(all_values, dtype=np.float32),
                    games_done=games_completed
                )
                print_flush(f"  [CHECKPOINT] Saved: {checkpoint_file}")

            # ETA
            if games_completed > 0:
                games_remaining = NUM_GAMES - games_completed
                eta_seconds = (total_time / games_completed) * games_remaining
                print_flush(f"  ETA: {timedelta(seconds=int(eta_seconds))}")

    # Final save
    total_time = time.time() - start_time

    print_flush("\n" + "=" * 70)
    print_flush("GENERATION COMPLETE")
    print_flush("=" * 70)

    # Convert to arrays
    states = np.array(all_states, dtype=np.float32)
    policies = np.array(all_policies, dtype=np.float32)
    values = np.array(all_values, dtype=np.float32)

    print_flush(f"Total games:      {games_completed}")
    print_flush(f"Total positions:  {len(states)}")
    print_flush(f"Avg game length:  {total_moves / games_completed:.1f} moves")
    print_flush(f"Total time:       {timedelta(seconds=int(total_time))}")
    print_flush(f"Games/second:     {games_completed / total_time:.2f}")
    print_flush(f"States shape:     {states.shape}")
    print_flush(f"Policies shape:   {policies.shape}")
    print_flush(f"Values shape:     {values.shape}")

    # Win rate (KEY METRIC!)
    total_decided = red_wins + blue_wins
    print_flush(f"\nGame outcomes (should be ~50/50 for good value training):")
    print_flush(f"  RED wins:   {red_wins} ({100 * red_wins / total_decided:.1f}%)")
    print_flush(f"  BLUE wins:  {blue_wins} ({100 * blue_wins / total_decided:.1f}%)")

    # Value distribution
    print_flush(f"\nValue distribution:")
    print_flush(f"  +1 (wins):   {(values == 1).sum()} ({100*(values == 1).mean():.1f}%)")
    print_flush(f"  -1 (losses): {(values == -1).sum()} ({100*(values == -1).mean():.1f}%)")
    print_flush(f"  0 (draws):   {(values == 0).sum()} ({100*(values == 0).mean():.1f}%)")

    # Save final
    np.savez_compressed(
        OUTPUT_FILE,
        states=states,
        policies=policies,
        values=values,
        games_done=games_completed
    )
    print_flush(f"\nSaved to: {OUTPUT_FILE}")
    print_flush("=" * 70)

    # Write final status
    write_status({
        "status": "completed",
        "end_time": datetime.now().isoformat(),
        "target_games": NUM_GAMES,
        "games_completed": games_completed,
        "positions": len(states),
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "red_rate_pct": round(100 * red_wins / total_decided, 1) if total_decided > 0 else 0,
        "blue_rate_pct": round(100 * blue_wins / total_decided, 1) if total_decided > 0 else 0,
        "total_time_seconds": round(total_time, 0),
        "output_file": OUTPUT_FILE
    })


if __name__ == '__main__':
    main()
