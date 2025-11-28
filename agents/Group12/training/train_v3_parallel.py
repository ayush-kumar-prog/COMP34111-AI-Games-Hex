"""
Parallel Self-Play Training V3 - Optimized for GPU Utilization

Key improvements over V2:
1. Parallel self-play using torch.multiprocessing (8 workers)
2. Reduced MCTS simulations (150 vs 400) but more games
3. Batched inference where possible
4. Faster board copying (avoid deepcopy)
5. More aggressive exploration in early training

Expected speedup: 10-15x over sequential V2
"""

import os
import sys
import copy
import random
import pickle
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm

# Set multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


@dataclass
class TrainingExample:
    """Single training example."""
    state: np.ndarray
    policy: np.ndarray
    value: float


class FastBoard:
    """Faster board implementation avoiding deepcopy."""

    __slots__ = ['tiles', 'size']

    def __init__(self, size: int = 11):
        self.size = size
        # Use numpy array instead of nested objects
        # 0 = empty, 1 = RED, 2 = BLUE
        self.tiles = np.zeros((size, size), dtype=np.int8)

    def copy(self) -> 'FastBoard':
        """Fast copy using numpy."""
        new_board = FastBoard.__new__(FastBoard)
        new_board.size = self.size
        new_board.tiles = self.tiles.copy()
        return new_board

    def set_tile(self, x: int, y: int, colour: int):
        """Set tile colour (1=RED, 2=BLUE)."""
        self.tiles[x, y] = colour

    def get_tile(self, x: int, y: int) -> int:
        return self.tiles[x, y]

    def is_empty(self, x: int, y: int) -> bool:
        return self.tiles[x, y] == 0

    def get_empty_positions(self) -> List[Tuple[int, int]]:
        """Get all empty positions."""
        return list(zip(*np.where(self.tiles == 0)))

    def has_won(self, colour: int) -> bool:
        """Check if colour has won using flood fill."""
        n = self.size

        if colour == 1:  # RED: top to bottom
            # Start from top row
            starts = [(0, j) for j in range(n) if self.tiles[0, j] == colour]
            target_row = n - 1
            check_win = lambda x, y: x == target_row
        else:  # BLUE: left to right
            starts = [(i, 0) for i in range(n) if self.tiles[i, 0] == colour]
            target_col = n - 1
            check_win = lambda x, y: y == target_col

        if not starts:
            return False

        visited = set()
        stack = starts.copy()

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            if check_win(x, y):
                return True

            # Check neighbors
            for dx, dy in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    if self.tiles[nx, ny] == colour and (nx, ny) not in visited:
                        stack.append((nx, ny))

        return False


class FastBoardEncoder:
    """Fast board encoder for neural network."""

    def __init__(self, board_size: int = 11):
        self.board_size = board_size

    def encode(self, board: FastBoard, colour: int) -> np.ndarray:
        """Encode board to 8-channel tensor."""
        n = self.board_size
        state = np.zeros((8, n, n), dtype=np.float32)

        # Channel 0: Our stones
        # Channel 1: Opponent stones
        # Channel 2: Empty cells
        opp = 3 - colour  # 1->2, 2->1

        state[0] = (board.tiles == colour).astype(np.float32)
        state[1] = (board.tiles == opp).astype(np.float32)
        state[2] = (board.tiles == 0).astype(np.float32)

        # Channel 3-4: Distance to our edges
        # Channel 5-6: Distance to opponent edges
        for i in range(n):
            for j in range(n):
                if colour == 1:  # RED: top-bottom
                    state[3, i, j] = i / (n - 1)
                    state[4, i, j] = (n - 1 - i) / (n - 1)
                    state[5, i, j] = j / (n - 1)
                    state[6, i, j] = (n - 1 - j) / (n - 1)
                else:  # BLUE: left-right
                    state[3, i, j] = j / (n - 1)
                    state[4, i, j] = (n - 1 - j) / (n - 1)
                    state[5, i, j] = i / (n - 1)
                    state[6, i, j] = (n - 1 - i) / (n - 1)

        # Channel 7: Center distance (strategic importance)
        center = n // 2
        for i in range(n):
            for j in range(n):
                dist = abs(i - center) + abs(j - center)
                state[7, i, j] = 1.0 - dist / (n - 1)

        return state


class SimpleMCTS:
    """Simplified MCTS optimized for speed."""

    EXPLORATION = 1.5
    DIRICHLET_ALPHA = 0.3
    NOISE_WEIGHT = 0.25

    def __init__(self, num_simulations: int = 150):
        self.num_simulations = num_simulations

    def search(self, board: FastBoard, colour: int,
               policy_prior: np.ndarray, value_prior: float) -> np.ndarray:
        """Run MCTS and return visit distribution."""
        n = board.size

        # Get legal moves
        empty = board.get_empty_positions()
        if not empty:
            return np.zeros(n * n)

        if len(empty) == 1:
            result = np.zeros(n * n)
            result[empty[0][0] * n + empty[0][1]] = 1.0
            return result

        # Initialize visit counts and values
        visits = np.zeros(n * n)
        values = np.zeros(n * n)

        # Add Dirichlet noise to prior
        legal_indices = [x * n + y for x, y in empty]
        noise = np.random.dirichlet([self.DIRICHLET_ALPHA] * len(empty))

        noisy_prior = policy_prior.copy()
        for i, idx in enumerate(legal_indices):
            noisy_prior[idx] = (1 - self.NOISE_WEIGHT) * policy_prior[idx] + \
                               self.NOISE_WEIGHT * noise[i]

        # Normalize
        prior_sum = sum(noisy_prior[idx] for idx in legal_indices)
        if prior_sum > 0:
            for idx in legal_indices:
                noisy_prior[idx] /= prior_sum

        # Run simulations
        for _ in range(self.num_simulations):
            # UCB selection
            best_score = -float('inf')
            best_idx = legal_indices[0]

            total_visits = visits.sum() + 1

            for idx in legal_indices:
                if visits[idx] == 0:
                    score = noisy_prior[idx] * 10 + random.random() * 0.01
                else:
                    q = values[idx] / visits[idx]
                    u = self.EXPLORATION * noisy_prior[idx] * \
                        np.sqrt(total_visits) / (1 + visits[idx])
                    score = q + u

                if score > best_score:
                    best_score = score
                    best_idx = idx

            # Simulate from this move
            x, y = best_idx // n, best_idx % n
            sim_board = board.copy()
            sim_board.set_tile(x, y, colour)

            # Quick rollout
            result = self._rollout(sim_board, 3 - colour, colour)

            # Update stats
            visits[best_idx] += 1
            values[best_idx] += result

        # Return normalized visit counts
        visit_sum = visits.sum()
        if visit_sum > 0:
            return visits / visit_sum
        return noisy_prior

    def _rollout(self, board: FastBoard, current: int, our_colour: int,
                 max_depth: int = 30) -> float:
        """Fast random rollout."""
        for _ in range(max_depth):
            if board.has_won(1):  # RED won
                return 1.0 if our_colour == 1 else 0.0
            if board.has_won(2):  # BLUE won
                return 1.0 if our_colour == 2 else 0.0

            empty = board.get_empty_positions()
            if not empty:
                break

            # Biased random: prefer center
            if random.random() < 0.3 and len(empty) > 1:
                center = board.size // 2
                empty.sort(key=lambda p: abs(p[0] - center) + abs(p[1] - center))
                x, y = empty[0]
            else:
                x, y = random.choice(empty)

            board.set_tile(x, y, current)
            current = 3 - current

        # Evaluate final position
        if board.has_won(our_colour):
            return 1.0
        elif board.has_won(3 - our_colour):
            return 0.0
        return 0.5


def worker_generate_games(worker_id: int, num_games: int,
                          model_state_dict: dict, model_config: dict,
                          num_simulations: int, device: str) -> List[TrainingExample]:
    """Worker function to generate self-play games."""

    # Import here to avoid issues with multiprocessing
    from agents.Group12.neural.hex_network_v2 import create_hex_network_v2

    # Create local model
    network = create_hex_network_v2(
        num_res_blocks=model_config['num_res_blocks'],
        num_channels=model_config['num_channels'],
        device=device
    )
    network.load_state_dict(model_state_dict)
    network.eval()

    encoder = FastBoardEncoder()
    mcts = SimpleMCTS(num_simulations=num_simulations)

    all_examples = []

    for game_idx in range(num_games):
        examples = generate_single_game(network, encoder, mcts, device)
        all_examples.extend(examples)

        if (game_idx + 1) % 10 == 0:
            print(f"Worker {worker_id}: {game_idx + 1}/{num_games} games", flush=True)

    return all_examples


def generate_single_game(network, encoder: FastBoardEncoder,
                         mcts: SimpleMCTS, device: str) -> List[TrainingExample]:
    """Generate a single self-play game."""
    board = FastBoard(11)
    examples = []
    current = 1  # RED starts
    move_number = 0

    while not board.has_won(1) and not board.has_won(2):
        empty = board.get_empty_positions()
        if not empty:
            break

        # Encode and get network prediction
        state = encoder.encode(board, current)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

        with torch.no_grad():
            policy_logits, value = network(state_tensor)
            policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
            value_pred = value[0, 0].item()

        # Run MCTS
        mcts_policy = mcts.search(board, current, policy, value_pred)

        # Save example
        examples.append(TrainingExample(
            state=state,
            policy=mcts_policy,
            value=0.0  # Filled later
        ))

        # Select move with temperature
        temp = 1.0 if move_number < 20 else 0.5
        policy_temp = mcts_policy ** (1.0 / temp)
        policy_temp /= policy_temp.sum()

        move_idx = np.random.choice(121, p=policy_temp)
        x, y = move_idx // 11, move_idx % 11

        board.set_tile(x, y, current)
        current = 3 - current
        move_number += 1

        if move_number > 150:  # Safety limit
            break

    # Fill in outcomes
    winner = 1 if board.has_won(1) else (2 if board.has_won(2) else 0)
    for i, ex in enumerate(examples):
        player = 1 if i % 2 == 0 else 2
        ex.value = 1.0 if player == winner else -1.0

    return examples


class ParallelHexTrainer:
    """Trainer with parallel self-play."""

    def __init__(self, network, learning_rate: float = 0.001,
                 weight_decay: float = 1e-4, device: str = 'cuda'):
        self.network = network.to(device)
        self.device = device

        self.optimizer = optim.Adam(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )

        self.epoch = 0
        self.best_loss = float('inf')

        # Model config for workers
        self.model_config = {
            'num_res_blocks': 15,
            'num_channels': 256
        }

    def parallel_self_play(self, num_games: int, num_workers: int,
                           num_simulations: int) -> List[TrainingExample]:
        """Generate self-play games in parallel."""

        games_per_worker = num_games // num_workers

        # Prepare model state dict for workers
        model_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}

        # Determine device for workers
        # Use CPU for workers to avoid GPU memory contention
        worker_device = 'cpu'  # Workers use CPU, main process uses GPU for training

        print(f"Starting {num_workers} workers, {games_per_worker} games each...")
        print(f"Workers using: {worker_device}")

        # Use process pool
        all_examples = []

        with mp.Pool(processes=num_workers) as pool:
            worker_fn = partial(
                worker_generate_games,
                num_games=games_per_worker,
                model_state_dict=model_state_dict,
                model_config=self.model_config,
                num_simulations=num_simulations,
                device=worker_device
            )

            results = pool.map(worker_fn, range(num_workers))

            for worker_examples in results:
                all_examples.extend(worker_examples)

        return all_examples

    def train_epoch(self, examples: List[TrainingExample],
                    batch_size: int = 256) -> Dict[str, float]:
        """Train for one epoch."""
        self.network.train()
        random.shuffle(examples)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]

            states = torch.stack([
                torch.from_numpy(e.state) for e in batch
            ]).to(self.device)

            policies = torch.stack([
                torch.from_numpy(e.policy) for e in batch
            ]).to(self.device)

            values = torch.tensor(
                [e.value for e in batch], dtype=torch.float32
            ).to(self.device)

            # Data augmentation: 180-degree rotation
            if random.random() > 0.5:
                states = torch.rot90(states, k=2, dims=[2, 3])
                policies = policies.view(-1, 11, 11)
                policies = torch.rot90(policies, k=2, dims=[1, 2])
                policies = policies.view(-1, 121)

            # Forward
            policy_logits, value_pred = self.network(states)

            # Losses
            policy_loss = -torch.mean(torch.sum(policies * policy_logits, dim=1))
            value_loss = nn.functional.mse_loss(value_pred.squeeze(), values)
            loss = policy_loss + value_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        self.epoch += 1

        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches
        }

    def train_full(self, num_iterations: int, games_per_iter: int,
                   epochs_per_iter: int, num_simulations: int,
                   num_workers: int, model_dir: str):
        """Full training loop."""

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("PARALLEL ALPHAZERO TRAINING V3")
        print(f"{'='*70}")
        print(f"Iterations: {num_iterations}")
        print(f"Games/iteration: {games_per_iter}")
        print(f"Workers: {num_workers}")
        print(f"MCTS simulations: {num_simulations}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")

        for iteration in range(num_iterations):
            iter_start = time.time()

            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*70}")

            # 1. Parallel self-play
            print(f"\n[1/3] Generating {games_per_iter} self-play games...")
            sp_start = time.time()

            examples = self.parallel_self_play(
                games_per_iter, num_workers, num_simulations
            )

            sp_time = time.time() - sp_start
            print(f"Generated {len(examples)} examples in {sp_time/60:.1f} minutes")

            # Save self-play data
            with open(model_path / f"selfplay_iter{iteration+1}.pkl", 'wb') as f:
                pickle.dump(examples, f)

            # 2. Training
            print(f"\n[2/3] Training network...")
            for epoch in range(epochs_per_iter):
                metrics = self.train_epoch(examples)
                print(f"  Epoch {epoch+1}/{epochs_per_iter}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"Policy={metrics['policy_loss']:.4f}, "
                      f"Value={metrics['value_loss']:.4f}")

            self.scheduler.step()

            # 3. Save checkpoint
            print(f"\n[3/3] Saving checkpoint...")
            self.save_checkpoint(model_path / f"hex_model_v3_iter{iteration+1}.pth")

            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                self.save_checkpoint(model_path / "hex_model_v3_best.pth")
                print("New best model!")

            iter_time = time.time() - iter_start
            print(f"\nIteration time: {iter_time/60:.1f} minutes")

            # Estimate remaining time
            remaining = (num_iterations - iteration - 1) * iter_time
            print(f"Estimated remaining: {remaining/3600:.1f} hours")

        # Export to numpy
        print("\nExporting to NumPy format...")
        self._export_numpy(model_path / "hex_model_v3_numpy.npz")

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")

    def save_checkpoint(self, path):
        """Save checkpoint."""
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'model_config': self.model_config
        }, path)

    def load_checkpoint(self, path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))

    def _export_numpy(self, path):
        """Export to NumPy."""
        weights = {}
        for name, param in self.network.state_dict().items():
            weights[name] = param.cpu().numpy()
        np.savez_compressed(path, **weights)
        print(f"Saved: {path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--games', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--simulations', type=int, default=150)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    from agents.Group12.neural.hex_network_v2 import create_hex_network_v2

    network = create_hex_network_v2(
        num_res_blocks=15,
        num_channels=256,
        device=device
    )

    trainer = ParallelHexTrainer(network, device=device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.train_full(
        num_iterations=args.iterations,
        games_per_iter=args.games,
        epochs_per_iter=args.epochs,
        num_simulations=args.simulations,
        num_workers=args.workers,
        model_dir=args.model_dir
    )


if __name__ == '__main__':
    main()
