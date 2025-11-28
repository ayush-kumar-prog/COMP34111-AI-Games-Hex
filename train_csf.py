#!/usr/bin/env python3
"""
CSF (Computational Shared Facility) Training Script for Group12 Hex AI.

This script is optimized for University of Manchester's CSF with:
- A100/V100 GPU support
- SLURM job integration
- Checkpoint recovery
- Comprehensive logging
- Large-scale training configuration

Usage:
    # Local testing (quick)
    python train_csf.py --mode test

    # Full training on CSF
    python train_csf.py --mode full --device cuda

    # Resume from checkpoint
    python train_csf.py --mode full --resume models/hex_model_iter5.pth
"""

import os
import sys
import argparse
import torch
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.Group12.training.trainer import HexTrainer
from agents.Group12.neural.hex_network import HexNeuralNetwork


# ============================================================================
# TRAINING CONFIGURATIONS
# ============================================================================

CONFIGS = {
    # Quick test - verify pipeline works (5-10 minutes)
    'test': {
        'num_iterations': 2,
        'games_per_iteration': 10,
        'epochs_per_iteration': 2,
        'batch_size': 64,
        'num_simulations': 50,
        'learning_rate': 0.001,
        'num_res_blocks': 4,
        'num_channels': 64,
    },

    # Development - reasonable training (2-4 hours on GPU)
    'dev': {
        'num_iterations': 5,
        'games_per_iteration': 100,
        'epochs_per_iteration': 5,
        'batch_size': 256,
        'num_simulations': 100,
        'learning_rate': 0.001,
        'num_res_blocks': 6,
        'num_channels': 128,
    },

    # Standard - good training (8-12 hours on A100)
    'standard': {
        'num_iterations': 10,
        'games_per_iteration': 200,
        'epochs_per_iteration': 5,
        'batch_size': 256,
        'num_simulations': 200,
        'learning_rate': 0.001,
        'num_res_blocks': 10,
        'num_channels': 256,
    },

    # Full - comprehensive training (24-48 hours on A100)
    'full': {
        'num_iterations': 20,
        'games_per_iteration': 500,
        'epochs_per_iteration': 5,
        'batch_size': 512,
        'num_simulations': 400,
        'learning_rate': 0.001,
        'num_res_blocks': 15,
        'num_channels': 256,
    },

    # Maximum - championship-level (72+ hours on A100)
    'max': {
        'num_iterations': 30,
        'games_per_iteration': 1000,
        'epochs_per_iteration': 5,
        'batch_size': 512,
        'num_simulations': 800,
        'learning_rate': 0.0005,
        'num_res_blocks': 20,
        'num_channels': 256,
    },
}


def print_system_info():
    """Print system and GPU information."""
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")

    # Check for CSF environment
    if os.environ.get('SGE_ROOT') or os.environ.get('SLURM_JOB_ID'):
        print("\nCSF Environment Detected:")
        print(f"  Job ID: {os.environ.get('SLURM_JOB_ID', os.environ.get('JOB_ID', 'N/A'))}")
        print(f"  Node: {os.environ.get('SLURM_NODELIST', os.environ.get('HOSTNAME', 'N/A'))}")

    print("=" * 70)
    print()


def estimate_training_time(config: dict, device: str) -> str:
    """Estimate training time based on configuration."""
    total_games = config['num_iterations'] * config['games_per_iteration']

    # Rough estimates based on simulations and device
    if 'cuda' in device:
        # GPU: ~30-60 seconds per game with 100-200 sims
        secs_per_game = config['num_simulations'] * 0.5
    else:
        # CPU: ~5-10x slower
        secs_per_game = config['num_simulations'] * 3.0

    total_seconds = total_games * secs_per_game
    hours = total_seconds / 3600

    if hours < 1:
        return f"{total_seconds / 60:.0f} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        return f"{hours / 24:.1f} days"


def train(args):
    """Main training function."""
    # Get configuration
    config = CONFIGS[args.mode]

    # Override with command line arguments if provided
    if args.iterations:
        config['num_iterations'] = args.iterations
    if args.games:
        config['games_per_iteration'] = args.games
    if args.simulations:
        config['num_simulations'] = args.simulations
    if args.batch_size:
        config['batch_size'] = args.batch_size

    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Print info
    print_system_info()

    print("=" * 70)
    print("GROUP12 HEX AI - NEURAL NETWORK TRAINING")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    print(f"Estimated training time: {estimate_training_time(config, device)}")
    print("=" * 70)
    print()

    # Create output directories
    model_dir = Path(args.output) / 'models'
    log_dir = Path(args.output) / 'runs'
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create network
    print("Creating neural network...")
    network = HexNeuralNetwork(
        board_size=11,
        num_res_blocks=config['num_res_blocks'],
        num_channels=config['num_channels']
    )

    num_params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {num_params:,}")
    print(f"Architecture: {config['num_res_blocks']} ResNet blocks, {config['num_channels']} channels")
    print()

    # Create trainer
    trainer = HexTrainer(
        network=network,
        learning_rate=config['learning_rate'],
        weight_decay=1e-4,
        label_smoothing=0.1,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print()

    # Run training
    start_time = time.time()

    try:
        trainer.train(
            num_iterations=config['num_iterations'],
            games_per_iteration=config['games_per_iteration'],
            epochs_per_iteration=config['epochs_per_iteration'],
            batch_size=config['batch_size'],
            num_simulations=config['num_simulations'],
            model_dir=str(model_dir),
            log_dir=str(log_dir)
        )

        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Total time: {elapsed / 3600:.2f} hours")
        print(f"Final model: {model_dir / 'hex_model_best.pth'}")
        print("=" * 70)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print("TRAINING INTERRUPTED")
        print("=" * 70)
        print(f"Elapsed time: {elapsed / 3600:.2f} hours")
        print(f"Partial models saved in: {model_dir}")
        print("To resume: python train_csf.py --resume <checkpoint.pth>")
        print("=" * 70)

    except Exception as e:
        print()
        print("=" * 70)
        print(f"ERROR: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Train Group12 Hex AI Neural Network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (5-10 minutes)
    python train_csf.py --mode test

    # Development training (2-4 hours)
    python train_csf.py --mode dev --device cuda

    # Full training for tournament (24-48 hours)
    python train_csf.py --mode full --device cuda --output /scratch/models

    # Resume interrupted training
    python train_csf.py --mode full --resume models/hex_model_iter10.pth
        """
    )

    parser.add_argument('--mode', type=str, default='dev',
                        choices=['test', 'dev', 'standard', 'full', 'max'],
                        help='Training mode (default: dev)')

    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')

    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for models and logs')

    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file')

    # Override specific parameters
    parser.add_argument('--iterations', type=int, default=None,
                        help='Override number of iterations')
    parser.add_argument('--games', type=int, default=None,
                        help='Override games per iteration')
    parser.add_argument('--simulations', type=int, default=None,
                        help='Override MCTS simulations per move')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (larger for A100)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
