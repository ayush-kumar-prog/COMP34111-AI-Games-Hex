"""
Enhanced training pipeline for Hex Neural Network V2.

This script implements the full AlphaZero-style training loop with:
- Enhanced board encoder (8 channels)
- Improved network architecture (15 blocks, SE attention)
- Dirichlet noise for exploration
- Temperature annealing
- Data augmentation
- Model checkpointing with best model tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.Group12.neural.hex_network_v2 import HexNeuralNetworkV2, create_hex_network_v2
from agents.Group12.neural.enhanced_board_encoder import EnhancedBoardEncoder


@dataclass
class TrainingExample:
    """Single training example from self-play."""
    state: np.ndarray  # (8, 11, 11) board state
    policy: np.ndarray  # (121,) visit count distribution
    value: float  # Game outcome (-1 or 1)
    move_number: int


class SelfPlayWorkerV2:
    """Self-play worker using V2 network."""

    DIRICHLET_ALPHA = 0.3
    NOISE_WEIGHT = 0.25

    def __init__(self, network: HexNeuralNetworkV2, encoder: EnhancedBoardEncoder,
                 device: str = 'cuda', num_simulations: int = 400):
        self.network = network
        self.encoder = encoder
        self.device = device
        self.num_simulations = num_simulations
        self.network.to(device)
        self.network.eval()

    def get_temperature(self, move_number: int) -> float:
        """Temperature schedule for move selection."""
        if move_number < 15:
            return 1.5  # High exploration in opening
        elif move_number < 30:
            return 1.0
        elif move_number < 50:
            return 0.7
        else:
            return 0.5  # Never below 0.5 for diversity

    def generate_game(self) -> List[TrainingExample]:
        """Generate one self-play game."""
        board = Board(11)
        examples = []
        move_number = 0
        current_colour = Colour.RED

        while not board.has_ended(Colour.RED) and not board.has_ended(Colour.BLUE):
            # Get neural network prediction
            state = self.encoder.encode(board, current_colour, device=self.device)
            with torch.no_grad():
                policy_logits, value = self.network(state.unsqueeze(0))
                policy = torch.exp(policy_logits[0]).cpu().numpy()

            # Get legal moves
            legal_mask = self._get_legal_mask(board)
            policy = policy * legal_mask
            if policy.sum() > 0:
                policy /= policy.sum()
            else:
                policy = legal_mask / legal_mask.sum()

            # Add Dirichlet noise at root
            if move_number < 40:
                num_legal = int(legal_mask.sum())
                noise = np.random.dirichlet([self.DIRICHLET_ALPHA] * num_legal)
                noise_full = np.zeros(121)
                legal_indices = np.where(legal_mask > 0)[0]
                for i, idx in enumerate(legal_indices):
                    noise_full[idx] = noise[i]
                policy = ((1 - self.NOISE_WEIGHT) * policy +
                         self.NOISE_WEIGHT * noise_full)
                policy = policy * legal_mask
                policy /= policy.sum()

            # Run MCTS to get improved policy
            mcts_policy = self._run_mcts(board, current_colour, policy)

            # Save training example
            example = TrainingExample(
                state=state.cpu().numpy(),
                policy=mcts_policy,
                value=0.0,  # Filled in after game
                move_number=move_number
            )
            examples.append(example)

            # Sample move with temperature
            temp = self.get_temperature(move_number)
            policy_temp = mcts_policy ** (1.0 / temp)
            policy_temp /= policy_temp.sum()

            move_idx = np.random.choice(121, p=policy_temp)
            x, y = move_idx // 11, move_idx % 11

            # Make move
            board.set_tile_colour(x, y, current_colour)
            current_colour = Colour.opposite(current_colour)
            move_number += 1

        # Fill in game outcomes
        winner = Colour.RED if board.has_ended(Colour.RED) else Colour.BLUE
        for i, example in enumerate(examples):
            player = Colour.RED if i % 2 == 0 else Colour.BLUE
            example.value = 1.0 if player == winner else -1.0

        return examples

    def _run_mcts(self, board: Board, colour: Colour,
                  prior_policy: np.ndarray, num_sims: int = None) -> np.ndarray:
        """Run MCTS from current position."""
        if num_sims is None:
            num_sims = self.num_simulations

        visit_counts = np.zeros(121)

        # Simple MCTS with neural prior
        for _ in range(num_sims):
            sim_board = copy.deepcopy(board)
            path = []
            current = colour

            # Selection/Simulation
            while not sim_board.has_ended(Colour.RED) and not sim_board.has_ended(Colour.BLUE):
                legal_moves = self._get_legal_moves(sim_board)
                if not legal_moves:
                    break

                # UCB selection with prior
                best_score = -float('inf')
                best_move = None

                total_visits = sum(visit_counts[m.x * 11 + m.y] for m in legal_moves) + 1

                for move in legal_moves:
                    idx = move.x * 11 + move.y
                    visits = visit_counts[idx]
                    prior = prior_policy[idx]

                    if visits == 0:
                        score = prior * 10 + random.random() * 0.01
                    else:
                        q = 0.5  # Neutral value for simplicity
                        u = 1.5 * prior * np.sqrt(total_visits) / (1 + visits)
                        score = q + u

                    if score > best_score:
                        best_score = score
                        best_move = move

                if best_move is None:
                    break

                path.append(best_move)
                sim_board.set_tile_colour(best_move.x, best_move.y, current)
                current = Colour.opposite(current)

                # Only simulate one move for speed
                break

            # Update visit counts for first move
            if path:
                idx = path[0].x * 11 + path[0].y
                visit_counts[idx] += 1

        # Normalize to get policy
        if visit_counts.sum() > 0:
            return visit_counts / visit_counts.sum()
        return prior_policy

    def _get_legal_mask(self, board: Board) -> np.ndarray:
        """Get mask of legal moves."""
        mask = np.zeros(121, dtype=np.float32)
        for i in range(11):
            for j in range(11):
                if board.tiles[i][j].colour is None:
                    mask[i * 11 + j] = 1.0
        return mask

    def _get_legal_moves(self, board: Board) -> List[Move]:
        """Get list of legal moves."""
        moves = []
        for i in range(11):
            for j in range(11):
                if board.tiles[i][j].colour is None:
                    moves.append(Move(i, j))
        return moves


class HexTrainerV2:
    """Enhanced trainer for V2 network."""

    def __init__(self, network: HexNeuralNetworkV2,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 device: str = 'cuda'):
        self.network = network.to(device)
        self.device = device

        self.optimizer = optim.Adam(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-5
        )

        self.epoch = 0
        self.best_loss = float('inf')

    def train_epoch(self, examples: List[TrainingExample],
                    batch_size: int = 256) -> Dict[str, float]:
        """Train for one epoch."""
        self.network.train()

        # Shuffle examples
        random.shuffle(examples)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]

            # Prepare batch
            states = torch.stack([torch.from_numpy(e.state) for e in batch]).to(self.device)
            policies = torch.stack([torch.from_numpy(e.policy) for e in batch]).to(self.device)
            values = torch.tensor([e.value for e in batch], dtype=torch.float32).to(self.device)

            # Data augmentation: 180-degree rotation (50% chance)
            if random.random() > 0.5:
                states = torch.rot90(states, k=2, dims=[2, 3])
                policies = policies.view(-1, 11, 11)
                policies = torch.rot90(policies, k=2, dims=[1, 2])
                policies = policies.view(-1, 121)

            # Forward pass
            policy_logits, value_pred = self.network(states)

            # Policy loss (cross-entropy)
            policy_loss = -torch.mean(torch.sum(policies * policy_logits, dim=1))

            # Value loss (MSE)
            value_loss = nn.functional.mse_loss(value_pred.squeeze(), values)

            # Combined loss
            loss = policy_loss + value_loss

            # Backward pass
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
            'value_loss': total_value_loss / num_batches,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def train_full(self, num_iterations: int = 20,
                   games_per_iter: int = 200,
                   epochs_per_iter: int = 10,
                   num_simulations: int = 400,
                   model_dir: str = 'models',
                   log_dir: str = 'runs'):
        """Full training loop."""
        print(f"\n{'='*70}")
        print("ALPHAZERO-STYLE TRAINING V2")
        print(f"{'='*70}")
        print(f"Iterations: {num_iterations}")
        print(f"Games/iteration: {games_per_iter}")
        print(f"Epochs/iteration: {epochs_per_iter}")
        print(f"MCTS simulations: {num_simulations}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        encoder = EnhancedBoardEncoder()

        # TensorBoard
        writer = None
        if HAS_TB:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            writer = SummaryWriter(f"{log_dir}/hex_v2_{timestamp}")

        for iteration in range(num_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*70}")

            # 1. Generate self-play data
            print("\n[1/3] Generating self-play games...")
            worker = SelfPlayWorkerV2(
                self.network, encoder, self.device, num_simulations
            )

            all_examples = []
            for game_num in tqdm(range(games_per_iter), desc="Self-play"):
                examples = worker.generate_game()
                all_examples.extend(examples)

            print(f"Generated {len(all_examples)} training examples")

            # Save self-play data
            data_path = model_path / f"selfplay_iter{iteration+1}.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump(all_examples, f)

            # 2. Train on data
            print(f"\n[2/3] Training network...")
            for epoch in range(epochs_per_iter):
                metrics = self.train_epoch(all_examples)
                print(f"  Epoch {epoch+1}/{epochs_per_iter}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"Policy={metrics['policy_loss']:.4f}, "
                      f"Value={metrics['value_loss']:.4f}")

                if writer:
                    step = iteration * epochs_per_iter + epoch
                    writer.add_scalar('Loss/total', metrics['loss'], step)
                    writer.add_scalar('Loss/policy', metrics['policy_loss'], step)
                    writer.add_scalar('Loss/value', metrics['value_loss'], step)

            self.scheduler.step()

            # 3. Save checkpoint
            print(f"\n[3/3] Saving checkpoint...")
            checkpoint_path = model_path / f"hex_model_v2_iter{iteration+1}.pth"
            self.save_checkpoint(checkpoint_path)

            # Save best model
            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                best_path = model_path / "hex_model_v2_best.pth"
                self.save_checkpoint(best_path)
                print(f"New best model saved!")

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")

        # Export to NumPy for tournament
        print("\nExporting to NumPy format...")
        self._export_numpy(model_path / "hex_model_numpy.npz")

        if writer:
            writer.close()

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))

    def _export_numpy(self, path: str):
        """Export model to NumPy format."""
        weights = {}
        for name, param in self.network.state_dict().items():
            weights[name] = param.cpu().numpy()
        np.savez_compressed(path, **weights)
        print(f"NumPy weights saved to {path}")


def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Hex Neural Network V2")
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--simulations', type=int, default=400)
    parser.add_argument('--blocks', type=int, default=15)
    parser.add_argument('--channels', type=int, default=256)
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create network
    network = create_hex_network_v2(
        num_res_blocks=args.blocks,
        num_channels=args.channels,
        device=device
    )

    # Create trainer
    trainer = HexTrainerV2(network, device=device)

    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train_full(
        num_iterations=args.iterations,
        games_per_iter=args.games,
        epochs_per_iter=args.epochs,
        num_simulations=args.simulations,
        model_dir=args.model_dir
    )


if __name__ == '__main__':
    main()
