"""
Batched GPU Training V4 - Maximum GPU Utilization

Key insight: Run multiple games simultaneously, batch ALL neural network calls.

Architecture:
- Run N games in parallel (N=64-128)
- Each game does MCTS, but neural network calls go to a central batcher
- Batcher collects positions, makes ONE GPU call, distributes results
- GPU utilization: 60-90% (vs 1% in V2)

Expected speedup: 20-50x over V2
"""

import os
import sys
import random
import pickle
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TrainingExample:
    state: np.ndarray
    policy: np.ndarray
    value: float


class FastBoard:
    """Fast board using numpy array."""

    __slots__ = ['tiles', 'size']
    NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    def __init__(self, size: int = 11):
        self.size = size
        self.tiles = np.zeros((size, size), dtype=np.int8)

    def copy(self) -> 'FastBoard':
        new = FastBoard.__new__(FastBoard)
        new.size = self.size
        new.tiles = self.tiles.copy()
        return new

    def set_tile(self, x: int, y: int, colour: int):
        self.tiles[x, y] = colour

    def is_empty(self, x: int, y: int) -> bool:
        return self.tiles[x, y] == 0

    def get_empty(self) -> List[Tuple[int, int]]:
        return list(zip(*np.where(self.tiles == 0)))

    def has_won(self, colour: int) -> bool:
        n = self.size
        if colour == 1:  # RED: top-bottom
            starts = [(0, j) for j in range(n) if self.tiles[0, j] == colour]
            goal = lambda x, y: x == n - 1
        else:  # BLUE: left-right
            starts = [(i, 0) for i in range(n) if self.tiles[i, 0] == colour]
            goal = lambda x, y: y == n - 1

        if not starts:
            return False

        visited = set()
        stack = list(starts)

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            if goal(x, y):
                return True

            for dx, dy in self.NEIGHBORS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    if self.tiles[nx, ny] == colour and (nx, ny) not in visited:
                        stack.append((nx, ny))

        return False


class BoardEncoder:
    """Encode board for neural network."""

    def __init__(self, size: int = 11):
        self.size = size
        # Pre-compute distance matrices
        self._precompute_distances()

    def _precompute_distances(self):
        n = self.size
        self.dist_top = np.zeros((n, n), dtype=np.float32)
        self.dist_bottom = np.zeros((n, n), dtype=np.float32)
        self.dist_left = np.zeros((n, n), dtype=np.float32)
        self.dist_right = np.zeros((n, n), dtype=np.float32)
        self.dist_center = np.zeros((n, n), dtype=np.float32)

        center = n // 2
        for i in range(n):
            for j in range(n):
                self.dist_top[i, j] = i / (n - 1)
                self.dist_bottom[i, j] = (n - 1 - i) / (n - 1)
                self.dist_left[i, j] = j / (n - 1)
                self.dist_right[i, j] = (n - 1 - j) / (n - 1)
                self.dist_center[i, j] = 1.0 - (abs(i - center) + abs(j - center)) / (n - 1)

    def encode(self, board: FastBoard, colour: int) -> np.ndarray:
        n = self.size
        state = np.zeros((8, n, n), dtype=np.float32)

        opp = 3 - colour
        state[0] = (board.tiles == colour).astype(np.float32)
        state[1] = (board.tiles == opp).astype(np.float32)
        state[2] = (board.tiles == 0).astype(np.float32)

        if colour == 1:  # RED
            state[3] = self.dist_top
            state[4] = self.dist_bottom
            state[5] = self.dist_left
            state[6] = self.dist_right
        else:  # BLUE
            state[3] = self.dist_left
            state[4] = self.dist_right
            state[5] = self.dist_top
            state[6] = self.dist_bottom

        state[7] = self.dist_center

        return state


class BatchedInferenceEngine:
    """
    Batches neural network inference for maximum GPU utilization.

    V4.2 FIXES:
    - Added timeout to event.wait() to prevent deadlocks
    - Added exception handling in inference loop
    - Added health check for inference thread
    - Added fallback for failed inferences
    """

    def __init__(self, network, device='cuda', max_batch_size=256,
                 max_wait_ms=5):
        self.network = network.to(device)
        self.network.eval()
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        # Request queue: (request_id, position_tensor)
        self.request_queue = queue.Queue()
        # Results: request_id -> (policy, value)
        self.results = {}
        self.results_lock = threading.Lock()
        self.result_events = {}  # request_id -> Event

        # Stats
        self.total_inferences = 0
        self.total_batches = 0
        self.failed_inferences = 0
        self.last_inference_time = time.time()

        # Start inference thread
        self.running = True
        self.inference_error = None
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def is_healthy(self) -> bool:
        """Check if inference engine is working."""
        if not self.inference_thread.is_alive():
            return False
        if self.inference_error is not None:
            return False
        # Check if inference is happening (within last 30 seconds)
        if time.time() - self.last_inference_time > 30:
            return False
        return True

    def request_inference(self, position: np.ndarray, timeout: float = 30.0) -> Tuple[np.ndarray, float]:
        """Submit position for inference, wait for result with timeout."""
        request_id = id(threading.current_thread()) + random.randint(0, 1000000)

        # Create event to wait on
        event = threading.Event()
        with self.results_lock:
            self.result_events[request_id] = event

        # Submit request
        self.request_queue.put((request_id, position))

        # Wait for result WITH TIMEOUT (FIX #1)
        got_result = event.wait(timeout=timeout)

        if not got_result:
            # Timeout - return uniform policy and neutral value
            self.failed_inferences += 1
            with self.results_lock:
                if request_id in self.result_events:
                    del self.result_events[request_id]
                if request_id in self.results:
                    del self.results[request_id]
            # Return fallback: uniform policy, neutral value
            n = position.shape[1]  # board size
            return np.ones(n * n) / (n * n), 0.0

        # Get result
        with self.results_lock:
            if request_id in self.results:
                policy, value = self.results.pop(request_id)
            else:
                # Result missing - return fallback
                policy = np.ones(121) / 121
                value = 0.0
            if request_id in self.result_events:
                del self.result_events[request_id]

        return policy, value

    def _inference_loop(self):
        """Background thread that batches and processes requests."""
        print("Inference thread started.", flush=True)

        while self.running:
            batch_requests = []

            # Collect requests (wait for first, then grab more if available)
            try:
                first = self.request_queue.get(timeout=0.5)
                batch_requests.append(first)
            except queue.Empty:
                continue

            # Grab more requests without blocking (up to batch size)
            deadline = time.time() + self.max_wait_ms / 1000
            while len(batch_requests) < self.max_batch_size:
                try:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    req = self.request_queue.get(timeout=max(0.001, remaining))
                    batch_requests.append(req)
                except queue.Empty:
                    break

            if not batch_requests:
                continue

            # Process batch WITH EXCEPTION HANDLING (FIX #2)
            try:
                request_ids = [r[0] for r in batch_requests]
                positions = np.stack([r[1] for r in batch_requests])

                positions_tensor = torch.from_numpy(positions).to(self.device)

                with torch.no_grad():
                    policy_logits, values = self.network(positions_tensor)
                    policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                    values = values.cpu().numpy().flatten()

                # Distribute results
                with self.results_lock:
                    for i, req_id in enumerate(request_ids):
                        self.results[req_id] = (policies[i], values[i])
                        if req_id in self.result_events:
                            self.result_events[req_id].set()

                self.total_inferences += len(batch_requests)
                self.total_batches += 1
                self.last_inference_time = time.time()

            except Exception as e:
                # Log error but don't crash - set events with fallback values
                print(f"Inference error: {e}", flush=True)
                self.inference_error = str(e)

                # Set fallback results for all requests in batch
                with self.results_lock:
                    for req_id in request_ids:
                        self.results[req_id] = (np.ones(121) / 121, 0.0)
                        if req_id in self.result_events:
                            self.result_events[req_id].set()

                self.failed_inferences += len(batch_requests)

        print("Inference thread stopped.", flush=True)

    def stop(self):
        self.running = False
        self.inference_thread.join(timeout=2)

    def get_stats(self):
        avg_batch = self.total_inferences / max(1, self.total_batches)
        return {
            'total_inferences': self.total_inferences,
            'total_batches': self.total_batches,
            'avg_batch_size': avg_batch,
            'failed_inferences': self.failed_inferences,
            'healthy': self.is_healthy()
        }


class PureAlphaZeroMCTS:
    """
    Pure AlphaZero MCTS - NO ROLLOUTS, uses network value directly.

    This maximizes GPU utilization by eliminating CPU-bound rollouts.
    Every leaf evaluation is a neural network call (batched on GPU).
    """

    EXPLORATION = 1.5
    DIRICHLET_ALPHA = 0.3
    NOISE_WEIGHT = 0.25

    def __init__(self, inference_engine: BatchedInferenceEngine,
                 encoder: BoardEncoder, num_sims: int = 100):
        self.engine = inference_engine
        self.encoder = encoder
        self.num_sims = num_sims

    def get_move(self, board: FastBoard, colour: int) -> Tuple[np.ndarray, float]:
        """Run MCTS and return (policy, value)."""
        n = board.size
        empty = board.get_empty()

        if not empty:
            return np.zeros(n * n), 0.0

        if len(empty) == 1:
            policy = np.zeros(n * n)
            policy[empty[0][0] * n + empty[0][1]] = 1.0
            return policy, 0.0

        # Get prior from neural network
        state = self.encoder.encode(board, colour)
        prior, root_value = self.engine.request_inference(state)

        # Mask illegal moves
        legal_mask = np.zeros(n * n)
        for x, y in empty:
            legal_mask[x * n + y] = 1.0
        prior = prior * legal_mask
        if prior.sum() > 0:
            prior /= prior.sum()
        else:
            prior = legal_mask / legal_mask.sum()

        # Add Dirichlet noise for exploration
        legal_indices = [x * n + y for x, y in empty]
        noise = np.random.dirichlet([self.DIRICHLET_ALPHA] * len(empty))
        noisy_prior = prior.copy()
        for i, idx in enumerate(legal_indices):
            noisy_prior[idx] = (1 - self.NOISE_WEIGHT) * prior[idx] + self.NOISE_WEIGHT * noise[i]
        noisy_prior = noisy_prior * legal_mask
        noisy_prior /= noisy_prior.sum()

        # MCTS tree: store visits, values, and child priors
        visits = np.zeros(n * n)
        total_values = np.zeros(n * n)

        for _ in range(self.num_sims):
            # UCB selection at root
            best_score = -float('inf')
            best_idx = legal_indices[0]
            total_visits = visits.sum() + 1

            for idx in legal_indices:
                if visits[idx] == 0:
                    # Prioritize unvisited nodes
                    score = noisy_prior[idx] * 10 + random.random() * 0.01
                else:
                    q = total_values[idx] / visits[idx]
                    u = self.EXPLORATION * noisy_prior[idx] * np.sqrt(total_visits) / (1 + visits[idx])
                    score = q + u

                if score > best_score:
                    best_score = score
                    best_idx = idx

            # Make move
            x, y = best_idx // n, best_idx % n
            sim_board = board.copy()
            sim_board.set_tile(x, y, colour)

            # Check for terminal state
            if sim_board.has_won(colour):
                leaf_value = 1.0
            elif sim_board.has_won(3 - colour):
                leaf_value = 0.0
            else:
                # PURE ALPHAZERO: Use network value instead of rollout!
                # This is a GPU call (batched with other games)
                child_state = self.encoder.encode(sim_board, 3 - colour)
                _, opponent_value = self.engine.request_inference(child_state)
                # Opponent's value, so negate for our perspective
                leaf_value = 1.0 - (opponent_value + 1) / 2  # Convert [-1,1] to [0,1] and flip

            visits[best_idx] += 1
            total_values[best_idx] += leaf_value

        # Return normalized visits as policy
        if visits.sum() > 0:
            return visits / visits.sum(), root_value
        return noisy_prior, root_value


# Keep SimpleMCTS as fallback with hybrid mode
class SimpleMCTS:
    """MCTS with configurable evaluation: network-only or hybrid with short rollouts."""

    EXPLORATION = 1.5
    DIRICHLET_ALPHA = 0.3
    NOISE_WEIGHT = 0.25

    def __init__(self, inference_engine: BatchedInferenceEngine,
                 encoder: BoardEncoder, num_sims: int = 100,
                 use_rollouts: bool = False, rollout_depth: int = 10):
        self.engine = inference_engine
        self.encoder = encoder
        self.num_sims = num_sims
        self.use_rollouts = use_rollouts
        self.rollout_depth = rollout_depth

    def get_move(self, board: FastBoard, colour: int) -> Tuple[np.ndarray, float]:
        """Run MCTS and return (policy, value)."""
        n = board.size
        empty = board.get_empty()

        if not empty:
            return np.zeros(n * n), 0.0

        if len(empty) == 1:
            policy = np.zeros(n * n)
            policy[empty[0][0] * n + empty[0][1]] = 1.0
            return policy, 0.0

        # Get prior from neural network
        state = self.encoder.encode(board, colour)
        prior, value = self.engine.request_inference(state)

        # Mask illegal moves
        legal_mask = np.zeros(n * n)
        for x, y in empty:
            legal_mask[x * n + y] = 1.0
        prior = prior * legal_mask
        if prior.sum() > 0:
            prior /= prior.sum()
        else:
            prior = legal_mask / legal_mask.sum()

        # Add Dirichlet noise
        legal_indices = [x * n + y for x, y in empty]
        noise = np.random.dirichlet([self.DIRICHLET_ALPHA] * len(empty))
        noisy_prior = prior.copy()
        for i, idx in enumerate(legal_indices):
            noisy_prior[idx] = (1 - self.NOISE_WEIGHT) * prior[idx] + self.NOISE_WEIGHT * noise[i]
        noisy_prior = noisy_prior * legal_mask
        noisy_prior /= noisy_prior.sum()

        # Run MCTS simulations
        visits = np.zeros(n * n)
        values = np.zeros(n * n)

        for _ in range(self.num_sims):
            # UCB selection
            best_score = -float('inf')
            best_idx = legal_indices[0]
            total_visits = visits.sum() + 1

            for idx in legal_indices:
                if visits[idx] == 0:
                    score = noisy_prior[idx] * 10 + random.random() * 0.01
                else:
                    q = values[idx] / visits[idx]
                    u = self.EXPLORATION * noisy_prior[idx] * np.sqrt(total_visits) / (1 + visits[idx])
                    score = q + u

                if score > best_score:
                    best_score = score
                    best_idx = idx

            # Simulate
            x, y = best_idx // n, best_idx % n
            sim_board = board.copy()
            sim_board.set_tile(x, y, colour)

            if sim_board.has_won(colour):
                result = 1.0
            elif sim_board.has_won(3 - colour):
                result = 0.0
            elif self.use_rollouts:
                result = self._rollout(sim_board, 3 - colour, colour)
            else:
                # Pure network evaluation
                child_state = self.encoder.encode(sim_board, 3 - colour)
                _, opp_value = self.engine.request_inference(child_state)
                result = 1.0 - (opp_value + 1) / 2

            visits[best_idx] += 1
            values[best_idx] += result

        # Return normalized visits as policy
        if visits.sum() > 0:
            return visits / visits.sum(), value
        return noisy_prior, value

    def _rollout(self, board: FastBoard, current: int, our_colour: int) -> float:
        """Short random rollout (only used in hybrid mode)."""
        for _ in range(self.rollout_depth):
            if board.has_won(1):
                return 1.0 if our_colour == 1 else 0.0
            if board.has_won(2):
                return 1.0 if our_colour == 2 else 0.0

            empty = board.get_empty()
            if not empty:
                break

            x, y = random.choice(empty)
            board.set_tile(x, y, current)
            current = 3 - current

        if board.has_won(our_colour):
            return 1.0
        elif board.has_won(3 - our_colour):
            return 0.0
        return 0.5


def game_worker(worker_id: int, game_queue: queue.Queue, result_queue: queue.Queue,
                mcts, num_games: int):
    """Worker that plays games using shared MCTS (with batched inference)."""
    try:
        for game_idx in range(num_games):
            examples = []
            board = FastBoard(11)
            current = 1  # RED starts
            move_num = 0

            while not board.has_won(1) and not board.has_won(2):
                empty = board.get_empty()
                if not empty or move_num > 150:
                    break

                # Get move from MCTS (uses batched inference)
                state = mcts.encoder.encode(board, current)
                policy, value = mcts.get_move(board, current)

                examples.append(TrainingExample(
                    state=state,
                    policy=policy,
                    value=0.0
                ))

                # Sample move with temperature
                temp = 1.0 if move_num < 15 else 0.3
                policy_temp = policy ** (1.0 / temp)
                if policy_temp.sum() > 0:
                    policy_temp /= policy_temp.sum()
                    move_idx = np.random.choice(121, p=policy_temp)
                else:
                    move_idx = random.choice([x * 11 + y for x, y in empty])

                x, y = move_idx // 11, move_idx % 11
                board.set_tile(x, y, current)
                current = 3 - current
                move_num += 1

            # Fill outcomes
            winner = 1 if board.has_won(1) else (2 if board.has_won(2) else 0)
            for i, ex in enumerate(examples):
                player = 1 if i % 2 == 0 else 2
                ex.value = 1.0 if player == winner else -1.0

            result_queue.put(examples)
    except Exception as e:
        print(f"Worker {worker_id} error: {e}", flush=True)
        import traceback
        traceback.print_exc()


def get_gpu_utilization():
    """Get current GPU utilization percentage."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0])
    except:
        pass
    return -1


class BatchedTrainer:
    """Trainer with batched GPU inference for maximum GPU utilization."""

    def __init__(self, network, device='cuda', lr=0.001):
        self.network = network.to(device)
        self.device = device
        self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-5)
        self.best_loss = float('inf')
        self.epoch = 0

    def generate_games(self, num_games: int, num_workers: int,
                       mcts_sims: int) -> List[TrainingExample]:
        """Generate games with PURE ALPHAZERO batched inference (no rollouts)."""
        import sys

        print(f"Generating {num_games} games with {num_workers} workers...", flush=True)
        print(f"MCTS simulations: {mcts_sims} (pure network evaluation, no rollouts)", flush=True)

        # Create batched inference engine
        print("Creating batched inference engine...", flush=True)
        engine = BatchedInferenceEngine(
            self.network,
            device=self.device,
            max_batch_size=256,  # Reduced for stability
            max_wait_ms=10  # More time to collect batches
        )
        print("Inference engine created.", flush=True)

        encoder = BoardEncoder()
        print("Board encoder created.", flush=True)

        # Use PURE ALPHAZERO MCTS - no rollouts, 100% GPU utilization potential
        mcts = PureAlphaZeroMCTS(engine, encoder, num_sims=mcts_sims)
        print("MCTS created.", flush=True)

        # Queues for workers
        result_queue = queue.Queue()

        # Start workers
        games_per_worker = max(1, num_games // num_workers)
        actual_games = games_per_worker * num_workers
        workers = []

        print(f"Starting {num_workers} workers, {games_per_worker} games each...", flush=True)
        for i in range(num_workers):
            t = threading.Thread(
                target=game_worker,
                args=(i, None, result_queue, mcts, games_per_worker),
                daemon=True
            )
            t.start()
            workers.append(t)
        print(f"All {num_workers} workers started.", flush=True)
        sys.stdout.flush()

        # Collect results
        all_examples = []
        games_done = 0
        total_games = actual_games

        start_time = time.time()
        last_print = start_time
        gpu_samples = []

        print(f"Waiting for {total_games} games to complete...", flush=True)

        stall_count = 0
        last_games_done = 0

        while games_done < total_games:
            try:
                examples = result_queue.get(timeout=5)
                all_examples.extend(examples)
                games_done += 1
                stall_count = 0  # Reset stall counter on progress

                # Progress update with GPU monitoring
                now = time.time()
                if now - last_print > 10 or games_done == 1:
                    elapsed = now - start_time
                    rate = games_done / elapsed if elapsed > 0 else 0
                    eta = (total_games - games_done) / rate if rate > 0 else 0
                    stats = engine.get_stats()
                    gpu_util = get_gpu_utilization()
                    if gpu_util >= 0:
                        gpu_samples.append(gpu_util)
                    gpu_str = f", GPU: {gpu_util}%" if gpu_util >= 0 else ""
                    health_str = "" if stats.get('healthy', True) else " [UNHEALTHY]"
                    print(f"  Progress: {games_done}/{total_games} games "
                          f"({rate:.1f} games/s, ETA: {eta:.0f}s, "
                          f"batch: {stats['avg_batch_size']:.0f}{gpu_str}{health_str})", flush=True)
                    last_print = now
            except queue.Empty:
                stall_count += 1

                # Check if workers are still alive
                alive = sum(1 for t in workers if t.is_alive())
                if alive == 0 and games_done < total_games:
                    print(f"WARNING: All workers died! Got {games_done}/{total_games} games", flush=True)
                    break

                # Check engine health
                if not engine.is_healthy():
                    print(f"WARNING: Inference engine unhealthy! Error: {engine.inference_error}", flush=True)
                    print(f"Got {games_done}/{total_games} games before failure", flush=True)
                    break

                # Stall detection - if no progress for 60 seconds, something is wrong
                if stall_count >= 12:  # 12 * 5 seconds = 60 seconds
                    print(f"WARNING: No progress for 60 seconds! Stalled at {games_done}/{total_games}", flush=True)
                    stats = engine.get_stats()
                    print(f"Engine stats: {stats}", flush=True)
                    print(f"Workers alive: {alive}/{num_workers}", flush=True)
                    # Don't break yet, give it more time but warn
                    stall_count = 0  # Reset to avoid spam

        # Wait for workers
        for t in workers:
            t.join(timeout=5)

        # Print stats
        stats = engine.get_stats()
        elapsed = time.time() - start_time
        avg_gpu = sum(gpu_samples) / len(gpu_samples) if gpu_samples else -1
        print(f"Generated {len(all_examples)} examples in {elapsed:.1f}s", flush=True)
        print(f"Inference: {stats['total_inferences']} calls, "
              f"{stats['total_batches']} batches, "
              f"avg batch: {stats['avg_batch_size']:.1f}", flush=True)
        if avg_gpu >= 0:
            print(f"Average GPU utilization: {avg_gpu:.1f}%", flush=True)

        engine.stop()
        return all_examples

    def train_epoch(self, examples: List[TrainingExample], batch_size=512):
        """Train one epoch."""
        self.network.train()
        random.shuffle(examples)

        total_loss = 0
        total_policy = 0
        total_value = 0
        n_batches = 0

        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]

            states = torch.stack([torch.from_numpy(e.state) for e in batch]).to(self.device)
            policies = torch.stack([torch.from_numpy(e.policy) for e in batch]).to(self.device)
            values = torch.tensor([e.value for e in batch], dtype=torch.float32).to(self.device)

            # Augmentation
            if random.random() > 0.5:
                states = torch.rot90(states, k=2, dims=[2, 3])
                policies = policies.view(-1, 11, 11)
                policies = torch.rot90(policies, k=2, dims=[1, 2])
                policies = policies.view(-1, 121)

            policy_logits, value_pred = self.network(states)

            policy_loss = -torch.mean(torch.sum(policies * policy_logits, dim=1))
            value_loss = nn.functional.mse_loss(value_pred.squeeze(), values)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy += policy_loss.item()
            total_value += value_loss.item()
            n_batches += 1

        self.epoch += 1
        return {
            'loss': total_loss / n_batches,
            'policy': total_policy / n_batches,
            'value': total_value / n_batches
        }

    def train_full(self, num_iterations, games_per_iter, epochs_per_iter,
                   mcts_sims, num_workers, model_dir):
        """Full training loop."""

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("BATCHED GPU TRAINING V4")
        print(f"{'='*70}")
        print(f"Iterations: {num_iterations}")
        print(f"Games/iter: {games_per_iter}")
        print(f"MCTS sims: {mcts_sims}")
        print(f"Workers: {num_workers}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")

        for iteration in range(num_iterations):
            iter_start = time.time()

            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*70}")

            # Self-play
            print(f"\n[1/3] Self-play with batched inference...")
            examples = self.generate_games(games_per_iter, num_workers, mcts_sims)

            # Save self-play data
            with open(model_path / f"selfplay_v4_iter{iteration+1}.pkl", 'wb') as f:
                pickle.dump(examples, f)

            # Train
            print(f"\n[2/3] Training...")
            for epoch in range(epochs_per_iter):
                metrics = self.train_epoch(examples)
                print(f"  Epoch {epoch+1}/{epochs_per_iter}: "
                      f"Loss={metrics['loss']:.4f} "
                      f"Policy={metrics['policy']:.4f} "
                      f"Value={metrics['value']:.4f}")

            self.scheduler.step()

            # Save
            print(f"\n[3/3] Saving...")
            torch.save({
                'epoch': self.epoch,
                'model': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_loss': self.best_loss
            }, model_path / f"hex_v4_iter{iteration+1}.pth")

            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                torch.save({
                    'epoch': self.epoch,
                    'model': self.network.state_dict(),
                }, model_path / "hex_v4_best.pth")
                print("New best model!")

            iter_time = time.time() - iter_start
            print(f"\nIteration time: {iter_time/60:.1f} min")
            print(f"Remaining: {(num_iterations - iteration - 1) * iter_time / 3600:.1f} hours")

        # Export numpy
        weights = {k: v.cpu().numpy() for k, v in self.network.state_dict().items()}
        np.savez_compressed(model_path / "hex_v4_numpy.npz", **weights)

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--games', type=int, default=400)
    parser.add_argument('--mcts-sims', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--model-dir', type=str, default='models')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    from agents.Group12.neural.hex_network_v2 import create_hex_network_v2

    network = create_hex_network_v2(
        num_res_blocks=15,
        num_channels=256,
        device=device
    )

    trainer = BatchedTrainer(network, device=device)
    trainer.train_full(
        num_iterations=args.iterations,
        games_per_iter=args.games,
        epochs_per_iter=args.epochs,
        mcts_sims=args.mcts_sims,
        num_workers=args.workers,
        model_dir=args.model_dir
    )


if __name__ == '__main__':
    main()
