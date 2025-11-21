"""
Training loop for neural Hex AI.
Implements AlphaZero-style training with policy and value losses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

from agents.Group12.neural.hex_network import HexNeuralNetwork
from agents.Group12.neural.board_encoder import BoardEncoder
from .dataset import HexDataset
from .self_play import SelfPlayManager


class HexTrainer:
    """
    Trainer for neural Hex AI using AlphaZero methodology.
    """

    def __init__(self, network: HexNeuralNetwork,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 device: str = 'cuda'):
        """
        Initialize trainer.

        Args:
            network: Neural network to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            device: 'cuda' or 'cpu'
        """
        self.network = network
        self.device = device
        self.network.to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.5
        )

        # TensorBoard logging
        self.writer = None

        # Training statistics
        self.epoch = 0
        self.global_step = 0

    def train_epoch(self, dataset: HexDataset, batch_size: int = 256) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataset: Training dataset
            batch_size: Batch size

        Returns:
            Dictionary of losses and metrics
        """
        self.network.train()

        dataloader = dataset.get_dataloader(batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")

        for states, policy_targets, value_targets in pbar:
            # Move to GPU
            states = states.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)

            # Forward pass
            policy_logits, value_pred = self.network(states)

            # Policy loss: Cross-entropy between predicted and target
            # policy_logits are already log-probabilities
            policy_loss = -torch.mean(torch.sum(policy_targets * policy_logits, dim=1))

            # Value loss: MSE between predicted and target
            value_loss = nn.functional.mse_loss(value_pred.squeeze(), value_targets.squeeze())

            # Combined loss
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'policy': f'{policy_loss.item():.4f}',
                'value': f'{value_loss.item():.4f}'
            })

            # TensorBoard logging (every 10 batches)
            if self.writer and self.global_step % 10 == 0:
                self.writer.add_scalar('Loss/batch', loss.item(), self.global_step)
                self.writer.add_scalar('PolicyLoss/batch', policy_loss.item(), self.global_step)
                self.writer.add_scalar('ValueLoss/batch', value_loss.item(), self.global_step)

        # Epoch metrics
        metrics = {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        self.epoch += 1

        return metrics

    def train(self, num_iterations: int = 10,
             games_per_iteration: int = 100,
             epochs_per_iteration: int = 10,
             batch_size: int = 256,
             num_simulations: int = 800,
             model_dir: str = 'models',
             log_dir: str = 'runs'):
        """
        Full AlphaZero-style training loop.

        Args:
            num_iterations: Number of self-play → train iterations
            games_per_iteration: Self-play games to generate per iteration
            epochs_per_iteration: Training epochs per iteration
            batch_size: Training batch size
            num_simulations: MCTS simulations per move in self-play
            model_dir: Directory to save models
            log_dir: TensorBoard log directory
        """
        print(f"\n{'='*70}")
        print(f"ALPHAZERO-STYLE TRAINING")
        print(f"{'='*70}")
        print(f"Iterations: {num_iterations}")
        print(f"Games per iteration: {games_per_iteration}")
        print(f"Epochs per iteration: {epochs_per_iteration}")
        print(f"Batch size: {batch_size}")
        print(f"MCTS simulations: {num_simulations}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")

        # Setup
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"hex_training_{timestamp}"
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")

        # Create self-play manager
        encoder = BoardEncoder()
        selfplay_manager = SelfPlayManager(self.network, encoder, self.device)

        # Training loop
        for iteration in range(num_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*70}\n")

            # 1. Generate self-play data
            print(f"[1/3] Generating self-play data...")
            data_filepath = selfplay_manager.generate_training_data(
                num_games=games_per_iteration,
                num_simulations=num_simulations
            )

            # 2. Load dataset
            print(f"\n[2/3] Loading training dataset...")
            dataset = HexDataset.load_from_file(data_filepath)

            # Print statistics
            stats = dataset.get_statistics()
            print(f"Dataset statistics:")
            print(f"  Examples: {stats['num_examples']}")
            print(f"  RED wins: {stats['red_wins']}")
            print(f"  BLUE wins: {stats['blue_wins']}")
            print(f"  Mean value: {stats['mean_value']:.3f}")
            print(f"  Policy entropy: {stats['policy_entropy']:.3f}")

            # 3. Train on data
            print(f"\n[3/3] Training network...")
            for epoch in range(epochs_per_iteration):
                metrics = self.train_epoch(dataset, batch_size)

                print(f"  Epoch {epoch + 1}/{epochs_per_iteration}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"Policy={metrics['policy_loss']:.4f}, "
                      f"Value={metrics['value_loss']:.4f}, "
                      f"LR={metrics['learning_rate']:.6f}")

                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('Loss/epoch', metrics['loss'], self.epoch)
                    self.writer.add_scalar('PolicyLoss/epoch', metrics['policy_loss'], self.epoch)
                    self.writer.add_scalar('ValueLoss/epoch', metrics['value_loss'], self.epoch)
                    self.writer.add_scalar('LearningRate', metrics['learning_rate'], self.epoch)

            # Step learning rate scheduler
            self.scheduler.step()

            # 4. Save checkpoint
            checkpoint_path = model_path / f"hex_model_iter{iteration + 1}.pth"
            self.save_checkpoint(checkpoint_path)
            print(f"\n✅ Checkpoint saved: {checkpoint_path}")

            # 5. Save best model (based on iteration)
            if iteration == num_iterations - 1:
                best_path = model_path / "hex_model_best.pth"
                self.save_checkpoint(best_path)
                print(f"✅ Best model saved: {best_path}")

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total iterations: {num_iterations}")
        print(f"Total epochs: {self.epoch}")
        print(f"Final model: {model_path / 'hex_model_best.pth'}")
        print(f"TensorBoard: tensorboard --logdir={log_dir}")
        print(f"{'='*70}\n")

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        print(f"Checkpoint loaded from {filepath}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Global step: {self.global_step}")


def quick_training_test():
    """Quick test of training infrastructure."""
    print("Testing Training Infrastructure...")

    from agents.Group12.neural.hex_network import create_hex_network

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create network
    network = create_hex_network(device=device, num_res_blocks=2)  # Smaller for testing

    # Create trainer
    trainer = HexTrainer(network, device=device)

    print("\n✅ Trainer created successfully!")
    print(f"   Device: {device}")
    print(f"   Optimizer: Adam")
    print(f"   Scheduler: StepLR")
    print(f"   Ready for training!")


if __name__ == '__main__':
    quick_training_test()
