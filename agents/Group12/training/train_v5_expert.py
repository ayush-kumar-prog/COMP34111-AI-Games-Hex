"""
V5 Expert Iteration Training Script

Two-phase training approach:
1. Supervised pre-training on expert MCTS data
2. Self-play reinforcement learning with neural-guided MCTS

Usage:
    # Phase 1: Supervised pre-training
    python -m agents.Group12.training.train_v5_expert \
        --phase supervised \
        --data data/expert_games_v5.npz \
        --epochs 50 \
        --output models/v5_supervised.pth

    # Phase 2: Self-play RL (Expert Iteration)
    python -m agents.Group12.training.train_v5_expert \
        --phase selfplay \
        --resume models/v5_supervised.pth \
        --iterations 100 \
        --games-per-iter 1000 \
        --mcts-sims 800 \
        --output models/v5_expert_final.pth

    # Phase 2: Pure AlphaZero (from scratch)
    python -m agents.Group12.training.train_v5_expert \
        --phase selfplay \
        --iterations 150 \
        --games-per-iter 800 \
        --mcts-sims 800 \
        --output models/v5_alphazero_final.pth
"""

import os
import sys
import random
import pickle
import time
import threading
import queue
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.Group12.neural.hex_network_v2 import HexNeuralNetworkV2, create_hex_network_v2


# ============================================================================
# Training Data Structures
# ============================================================================

@dataclass
class TrainingExample:
    """A single training example."""
    state: np.ndarray   # (8, 11, 11)
    policy: np.ndarray  # (121,)
    value: float        # [-1, +1]


class HexDataset(Dataset):
    """PyTorch dataset for Hex training data."""

    def __init__(self, examples: List[TrainingExample], augment: bool = True):
        self.examples = examples
        self.augment = augment

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        state = torch.from_numpy(ex.state)
        policy = torch.from_numpy(ex.policy)
        value = torch.tensor(ex.value, dtype=torch.float32)

        # Data augmentation: 180-degree rotation (Hex symmetry)
        if self.augment and random.random() > 0.5:
            state = torch.rot90(state, k=2, dims=[1, 2])
            policy = policy.view(11, 11)
            policy = torch.rot90(policy, k=2, dims=[0, 1])
            policy = policy.view(-1)

        return state, policy, value


# ============================================================================
# Fast Board for Self-Play
# ============================================================================

class FastBoard:
    """Fast numpy-based board for self-play."""

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


# ============================================================================
# Board Encoder
# ============================================================================

class BoardEncoder:
    """Encode board for neural network (8 channels)."""

    def __init__(self, size: int = 11):
        self.size = size
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


# ============================================================================
# Batched Inference Engine (from V4, improved)
# ============================================================================

class BatchedInferenceEngine:
    """Batches neural network inference for maximum GPU utilization."""

    def __init__(self, network, device='cuda', max_batch_size=512, max_wait_ms=5):
        self.network = network.to(device)
        self.network.eval()
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self.request_queue = queue.Queue()
        self.results = {}
        self.results_lock = threading.Lock()
        self.result_events = {}

        self.total_inferences = 0
        self.total_batches = 0
        self.failed_inferences = 0
        self.last_inference_time = time.time()

        self.running = True
        self.inference_error = None
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def is_healthy(self) -> bool:
        if not self.inference_thread.is_alive():
            return False
        if self.inference_error is not None:
            return False
        if time.time() - self.last_inference_time > 60:
            return False
        return True

    def request_inference(self, position: np.ndarray, timeout: float = 60.0) -> Tuple[np.ndarray, float]:
        request_id = id(threading.current_thread()) + random.randint(0, 10000000)

        event = threading.Event()
        with self.results_lock:
            self.result_events[request_id] = event

        self.request_queue.put((request_id, position))

        got_result = event.wait(timeout=timeout)

        if not got_result:
            self.failed_inferences += 1
            with self.results_lock:
                self.result_events.pop(request_id, None)
                self.results.pop(request_id, None)
            n = position.shape[1]
            return np.ones(n * n) / (n * n), 0.0

        with self.results_lock:
            if request_id in self.results:
                policy, value = self.results.pop(request_id)
            else:
                policy = np.ones(121) / 121
                value = 0.0
            self.result_events.pop(request_id, None)

        return policy, value

    def _inference_loop(self):
        while self.running:
            batch_requests = []

            try:
                first = self.request_queue.get(timeout=0.5)
                batch_requests.append(first)
            except queue.Empty:
                continue

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

            try:
                request_ids = [r[0] for r in batch_requests]
                positions = np.stack([r[1] for r in batch_requests])

                positions_tensor = torch.from_numpy(positions).to(self.device)

                with torch.no_grad():
                    policy_logits, values = self.network(positions_tensor)
                    policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                    values = values.cpu().numpy().flatten()

                with self.results_lock:
                    for i, req_id in enumerate(request_ids):
                        self.results[req_id] = (policies[i], values[i])
                        if req_id in self.result_events:
                            self.result_events[req_id].set()

                self.total_inferences += len(batch_requests)
                self.total_batches += 1
                self.last_inference_time = time.time()

            except Exception as e:
                print(f"Inference error: {e}", flush=True)
                self.inference_error = str(e)

                with self.results_lock:
                    for req_id in request_ids:
                        self.results[req_id] = (np.ones(121) / 121, 0.0)
                        if req_id in self.result_events:
                            self.result_events[req_id].set()

                self.failed_inferences += len(batch_requests)

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


# ============================================================================
# Neural MCTS for Self-Play (High Quality - 800 sims)
# ============================================================================

class NeuralMCTS:
    """
    Neural-guided MCTS for self-play.
    Uses 800 simulations per move for high-quality training data.
    """

    EXPLORATION = 1.5
    DIRICHLET_ALPHA = 0.3
    NOISE_WEIGHT = 0.25

    def __init__(self, inference_engine: BatchedInferenceEngine,
                 encoder: BoardEncoder, num_sims: int = 800):
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

        # MCTS tree
        visits = np.zeros(n * n)
        total_values = np.zeros(n * n)

        for _ in range(self.num_sims):
            # UCB selection
            best_score = -float('inf')
            best_idx = legal_indices[0]
            total_visits = visits.sum() + 1

            for idx in legal_indices:
                if visits[idx] == 0:
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

            # Evaluate
            if sim_board.has_won(colour):
                leaf_value = 1.0
            elif sim_board.has_won(3 - colour):
                leaf_value = 0.0
            else:
                # Neural network evaluation
                child_state = self.encoder.encode(sim_board, 3 - colour)
                _, opponent_value = self.engine.request_inference(child_state)
                leaf_value = 1.0 - (opponent_value + 1) / 2

            visits[best_idx] += 1
            total_values[best_idx] += leaf_value

        # Return normalized visits as policy
        if visits.sum() > 0:
            return visits / visits.sum(), root_value
        return noisy_prior, root_value


# ============================================================================
# Self-Play Game Worker
# ============================================================================

def game_worker(worker_id: int, result_queue: queue.Queue,
                mcts: NeuralMCTS, num_games: int):
    """Worker that plays games using neural MCTS."""
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

                # Get move from MCTS
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
                if winner == 0:
                    ex.value = 0.0
                elif player == winner:
                    ex.value = 1.0
                else:
                    ex.value = -1.0

            result_queue.put(examples)
    except Exception as e:
        print(f"Worker {worker_id} error: {e}", flush=True)
        import traceback
        traceback.print_exc()


# ============================================================================
# GPU Utilization Monitor
# ============================================================================

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


# ============================================================================
# Supervised Trainer
# ============================================================================

class SupervisedTrainer:
    """Trainer for supervised learning phase."""

    def __init__(self, network, device='cuda', lr=0.001):
        self.network = network.to(device)
        self.device = device
        self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = None
        self.best_loss = float('inf')
        self.epoch = 0

    def train(self, train_data: List[TrainingExample], val_data: List[TrainingExample],
              epochs: int, batch_size: int = 512, model_dir: str = 'models'):
        """Train on supervised data."""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # Create data loaders
        train_dataset = HexDataset(train_data, augment=True)
        val_dataset = HexDataset(val_data, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )

        print(f"\n{'='*70}")
        print("SUPERVISED PRE-TRAINING")
        print(f"{'='*70}")
        print(f"Training examples: {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*70}\n")

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_policy, train_value = self._train_epoch(train_loader)

            # Validate
            val_loss, val_policy, val_value, val_accuracy = self._validate(val_loader)

            self.scheduler.step()
            self.epoch += 1

            epoch_time = time.time() - epoch_start
            lr = self.scheduler.get_last_lr()[0]

            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.1%} | "
                  f"LR: {lr:.6f}")

            # Save checkpoint
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save({
                    'epoch': self.epoch,
                    'model': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                }, model_path / 'v5_supervised_best.pth')
                print(f"  -> New best model!")

            # Periodic save
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': self.epoch,
                    'model': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                }, model_path / f'v5_supervised_epoch{epoch+1}.pth')

        # Final save
        torch.save({
            'epoch': self.epoch,
            'model': self.network.state_dict(),
        }, model_path / 'v5_supervised_final.pth')

        print(f"\n{'='*70}")
        print("SUPERVISED TRAINING COMPLETE")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"{'='*70}")

    def _train_epoch(self, loader):
        self.network.train()
        total_loss = 0
        total_policy = 0
        total_value = 0
        n_batches = 0

        for states, policies, values in loader:
            states = states.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)

            policy_logits, value_pred = self.network(states)

            # Policy loss (cross-entropy)
            policy_loss = -torch.mean(torch.sum(policies * F.log_softmax(policy_logits, dim=1), dim=1))
            # Value loss (MSE)
            value_loss = F.mse_loss(value_pred.squeeze(-1), values)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy += policy_loss.item()
            total_value += value_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_policy / n_batches, total_value / n_batches

    def _validate(self, loader):
        self.network.eval()
        total_loss = 0
        total_policy = 0
        total_value = 0
        correct_top1 = 0
        total = 0
        n_batches = 0

        with torch.no_grad():
            for states, policies, values in loader:
                states = states.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)

                policy_logits, value_pred = self.network(states)

                policy_loss = -torch.mean(torch.sum(policies * F.log_softmax(policy_logits, dim=1), dim=1))
                value_loss = F.mse_loss(value_pred.squeeze(-1), values)
                loss = policy_loss + value_loss

                # Top-1 accuracy
                pred_moves = policy_logits.argmax(dim=1)
                target_moves = policies.argmax(dim=1)
                correct_top1 += (pred_moves == target_moves).sum().item()
                total += states.size(0)

                total_loss += loss.item()
                total_policy += policy_loss.item()
                total_value += value_loss.item()
                n_batches += 1

        accuracy = correct_top1 / total if total > 0 else 0
        return (total_loss / n_batches, total_policy / n_batches,
                total_value / n_batches, accuracy)


# ============================================================================
# Self-Play Trainer
# ============================================================================

class SelfPlayTrainer:
    """Trainer for self-play reinforcement learning phase."""

    def __init__(self, network, device='cuda', lr=0.0005):
        self.network = network.to(device)
        self.device = device
        self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )
        self.best_loss = float('inf')
        self.epoch = 0

        # Replay buffer
        self.replay_buffer = deque(maxlen=500000)

    def train_iteration(self, num_games: int, num_workers: int, mcts_sims: int,
                       epochs_per_iter: int, batch_size: int = 512) -> Dict:
        """Run one iteration of self-play + training."""

        # Generate games
        examples = self._generate_games(num_games, num_workers, mcts_sims)

        # Add to replay buffer
        self.replay_buffer.extend(examples)

        # Train on replay buffer
        train_examples = list(self.replay_buffer)
        metrics = self._train_epochs(train_examples, epochs_per_iter, batch_size)

        return metrics

    def _generate_games(self, num_games: int, num_workers: int,
                       mcts_sims: int) -> List[TrainingExample]:
        """Generate games with batched neural MCTS."""
        print(f"Generating {num_games} games with {num_workers} workers...")
        print(f"MCTS simulations: {mcts_sims}")

        # Create batched inference engine
        engine = BatchedInferenceEngine(
            self.network,
            device=self.device,
            max_batch_size=512,
            max_wait_ms=10
        )

        encoder = BoardEncoder()
        mcts = NeuralMCTS(engine, encoder, num_sims=mcts_sims)

        # Result queue
        result_queue = queue.Queue()

        # Start workers
        games_per_worker = max(1, num_games // num_workers)
        actual_games = games_per_worker * num_workers
        workers = []

        for i in range(num_workers):
            t = threading.Thread(
                target=game_worker,
                args=(i, result_queue, mcts, games_per_worker),
                daemon=True
            )
            t.start()
            workers.append(t)

        # Collect results
        all_examples = []
        games_done = 0

        start_time = time.time()
        last_print = start_time
        gpu_samples = []

        while games_done < actual_games:
            try:
                examples = result_queue.get(timeout=10)
                all_examples.extend(examples)
                games_done += 1

                now = time.time()
                if now - last_print > 30 or games_done == 1:
                    elapsed = now - start_time
                    rate = games_done / elapsed if elapsed > 0 else 0
                    eta = (actual_games - games_done) / rate if rate > 0 else 0
                    stats = engine.get_stats()
                    gpu_util = get_gpu_utilization()
                    if gpu_util >= 0:
                        gpu_samples.append(gpu_util)
                    gpu_str = f", GPU: {gpu_util}%" if gpu_util >= 0 else ""
                    print(f"  Progress: {games_done}/{actual_games} games "
                          f"({rate:.1f}/s, ETA: {eta:.0f}s, "
                          f"batch: {stats['avg_batch_size']:.0f}{gpu_str})")
                    last_print = now
            except queue.Empty:
                alive = sum(1 for t in workers if t.is_alive())
                if alive == 0 and games_done < actual_games:
                    print(f"WARNING: All workers died! Got {games_done}/{actual_games}")
                    break
                if not engine.is_healthy():
                    print(f"WARNING: Inference engine unhealthy!")
                    break

        # Wait for workers
        for t in workers:
            t.join(timeout=5)

        stats = engine.get_stats()
        elapsed = time.time() - start_time
        avg_gpu = sum(gpu_samples) / len(gpu_samples) if gpu_samples else -1
        print(f"Generated {len(all_examples)} examples in {elapsed:.1f}s")
        if avg_gpu >= 0:
            print(f"Average GPU utilization: {avg_gpu:.1f}%")

        engine.stop()
        return all_examples

    def _train_epochs(self, examples: List[TrainingExample],
                     epochs: int, batch_size: int) -> Dict:
        """Train for multiple epochs."""
        self.network.train()

        for epoch in range(epochs):
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

                policy_loss = -torch.mean(torch.sum(policies * F.log_softmax(policy_logits, dim=1), dim=1))
                value_loss = F.mse_loss(value_pred.squeeze(-1), values)
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

            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Loss={total_loss/n_batches:.4f} "
                  f"Policy={total_policy/n_batches:.4f} "
                  f"Value={total_value/n_batches:.4f}")

        self.scheduler.step()

        return {
            'loss': total_loss / n_batches,
            'policy': total_policy / n_batches,
            'value': total_value / n_batches
        }

    def save(self, path: str):
        torch.save({
            'epoch': self.epoch,
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }, path)


# ============================================================================
# Main Training Functions
# ============================================================================

def load_expert_data(data_path: str) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """Load and split expert data."""
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)

    examples = []
    for i in range(len(data['states'])):
        examples.append(TrainingExample(
            state=data['states'][i],
            policy=data['policies'][i],
            value=data['values'][i]
        ))

    print(f"Loaded {len(examples)} examples")

    # Split 90/10
    random.shuffle(examples)
    split = int(len(examples) * 0.9)
    train_data = examples[:split]
    val_data = examples[split:]

    return train_data, val_data


def run_supervised_training(args):
    """Run supervised pre-training phase."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    train_data, val_data = load_expert_data(args.data)

    # Create network
    network = create_hex_network_v2(
        num_res_blocks=15,
        num_channels=256,
        device=device
    )

    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        network.load_state_dict(checkpoint['model'])

    # Train
    trainer = SupervisedTrainer(network, device=device)
    trainer.train(train_data, val_data, epochs=args.epochs,
                 batch_size=args.batch_size, model_dir=Path(args.output).parent)


def run_selfplay_training(args):
    """Run self-play RL phase."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create network
    network = create_hex_network_v2(
        num_res_blocks=15,
        num_channels=256,
        device=device
    )

    # Resume if specified (for Expert Iteration, start from supervised model)
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        network.load_state_dict(checkpoint['model'])
    else:
        print("Starting from random initialization (Pure AlphaZero)")

    # Create trainer
    trainer = SelfPlayTrainer(network, device=device)

    model_dir = Path(args.output).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("SELF-PLAY REINFORCEMENT LEARNING")
    print(f"{'='*70}")
    print(f"Iterations: {args.iterations}")
    print(f"Games/iter: {args.games_per_iter}")
    print(f"MCTS sims: {args.mcts_sims}")
    print(f"Workers: {args.workers}")
    print(f"{'='*70}\n")

    for iteration in range(args.iterations):
        iter_start = time.time()

        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'='*70}")

        # Self-play + training
        metrics = trainer.train_iteration(
            num_games=args.games_per_iter,
            num_workers=args.workers,
            mcts_sims=args.mcts_sims,
            epochs_per_iter=args.epochs_per_iter,
            batch_size=args.batch_size
        )

        # Save checkpoint
        trainer.save(model_dir / f'v5_iter{iteration+1}.pth')

        if metrics['loss'] < trainer.best_loss:
            trainer.best_loss = metrics['loss']
            trainer.save(model_dir / 'v5_best.pth')
            print("New best model!")

        iter_time = time.time() - iter_start
        remaining_iters = args.iterations - iteration - 1
        eta_hours = remaining_iters * iter_time / 3600

        print(f"\nIteration time: {iter_time/60:.1f} min")
        print(f"Remaining: {eta_hours:.1f} hours")
        print(f"Replay buffer: {len(trainer.replay_buffer)} examples")

    # Final save
    trainer.save(args.output)

    # Export numpy weights
    weights = {k: v.cpu().numpy() for k, v in network.state_dict().items()}
    np.savez_compressed(model_dir / 'v5_final_numpy.npz', **weights)

    print(f"\n{'='*70}")
    print("SELF-PLAY TRAINING COMPLETE!")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='V5 Expert Iteration Training')
    parser.add_argument('--phase', type=str, required=True,
                       choices=['supervised', 'selfplay'],
                       help='Training phase')

    # Data args
    parser.add_argument('--data', type=str, default='data/expert_games_v5.npz',
                       help='Path to expert data (supervised phase)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Training args
    parser.add_argument('--epochs', type=int, default=50,
                       help='Epochs for supervised training')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Iterations for self-play')
    parser.add_argument('--games-per-iter', type=int, default=1000,
                       help='Games per self-play iteration')
    parser.add_argument('--mcts-sims', type=int, default=800,
                       help='MCTS simulations per move')
    parser.add_argument('--epochs-per-iter', type=int, default=10,
                       help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Training batch size')
    parser.add_argument('--workers', type=int, default=64,
                       help='Number of self-play workers')

    # Output args
    parser.add_argument('--output', type=str, default='models/v5_final.pth',
                       help='Output model path')

    args = parser.parse_args()

    print("="*70)
    print("V5 EXPERT ITERATION TRAINING")
    print("="*70)
    print(f"Phase: {args.phase}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    if args.phase == 'supervised':
        run_supervised_training(args)
    else:
        run_selfplay_training(args)


if __name__ == '__main__':
    main()
