#!/usr/bin/env python3
"""
Training script for Group12 Neural Hex AI.
Runs full AlphaZero-style training pipeline.

Usage:
    python3 train_neural_agent.py --iterations 10 --games 1000 --device cuda
"""

import argparse
import torch
from pathlib import Path

from agents.Group12.neural.hex_network import create_hex_network
from agents.Group12.training.trainer import HexTrainer


def main():
    parser = argparse.ArgumentParser(description='Train neural Hex AI')

    # Training parameters
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of training iterations (default: 10)')
    parser.add_argument('--games', type=int, default=100,
                       help='Self-play games per iteration (default: 100)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs per iteration (default: 10)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Training batch size (default: 256)')
    parser.add_argument('--simulations', type=int, default=800,
                       help='MCTS simulations per move (default: 800)')

    # Network architecture
    parser.add_argument('--res-blocks', type=int, default=10,
                       help='Number of ResNet blocks (default: 10)')
    parser.add_argument('--channels', type=int, default=256,
                       help='Number of channels (default: 256)')

    # Optimization
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')

    # Directories
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Model save directory (default: models)')
    parser.add_argument('--log-dir', type=str, default='runs',
                       help='TensorBoard log directory (default: runs)')

    args = parser.parse_args()

    # Verify CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("\n" + "="*70)
    print("GROUP12 NEURAL HEX AI TRAINING")
    print("="*70)
    print(f"Configuration:")
    print(f"  Device: {args.device}")
    if args.device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Games/iteration: {args.games}")
    print(f"  Epochs/iteration: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  MCTS simulations: {args.simulations}")
    print(f"  Network: {args.res_blocks} ResBlocks, {args.channels} channels")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print("="*70 + "\n")

    # Create network
    print("Creating neural network...")
    network = create_hex_network(
        board_size=11,
        num_res_blocks=args.res_blocks,
        num_channels=args.channels,
        device=args.device
    )

    # Create trainer
    print("\nCreating trainer...")
    trainer = HexTrainer(
        network=network,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device
    )

    # Run training
    print("\nStarting training...")
    print("This will take 12-24 hours on GPU.")
    print("Monitor progress with: tensorboard --logdir=runs\n")

    try:
        trainer.train(
            num_iterations=args.iterations,
            games_per_iteration=args.games,
            epochs_per_iteration=args.epochs,
            batch_size=args.batch_size,
            num_simulations=args.simulations,
            model_dir=args.model_dir,
            log_dir=args.log_dir
        )

        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Best model saved to: {args.model_dir}/hex_model_best.pth")
        print(f"TensorBoard logs: {args.log_dir}")
        print("\nTo use the trained agent:")
        print("  python3 Hex.py -p1 \"agents.Group12.Group12Agent_neural Group12Agent\"")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Partial models saved in: {args.model_dir}")
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
