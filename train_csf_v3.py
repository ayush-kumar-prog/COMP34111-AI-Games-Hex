#!/usr/bin/env python3
"""
CSF Training Script V3 - Parallel Self-Play

Key improvements:
- 8 parallel workers for self-play
- Reduced MCTS simulations (150 vs 400) but faster
- More iterations (30 vs 20)
- FastBoard avoiding deepcopy
- Expected 10-15x speedup over V2

Usage:
    python train_csf_v3.py --mode test      # Quick test
    python train_csf_v3.py --mode fast      # Fast training (4-6 hours)
    python train_csf_v3.py --mode standard  # Standard (8-12 hours)
    python train_csf_v3.py --mode full      # Full (18-24 hours)
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.Group12.neural.hex_network_v2 import create_hex_network_v2
from agents.Group12.training.train_v3_parallel import ParallelHexTrainer


# Optimized configurations for parallel training
CONFIGS = {
    # Quick test (10-15 minutes)
    'test': {
        'num_iterations': 2,
        'games_per_iteration': 32,  # 8 workers × 4 games
        'epochs_per_iteration': 2,
        'num_simulations': 50,
        'num_workers': 8,
        'num_res_blocks': 6,
        'num_channels': 128,
    },

    # Fast training (4-6 hours) - good for experimentation
    'fast': {
        'num_iterations': 15,
        'games_per_iteration': 200,
        'epochs_per_iteration': 5,
        'num_simulations': 100,
        'num_workers': 8,
        'num_res_blocks': 12,
        'num_channels': 192,
    },

    # Standard training (8-12 hours) - good balance
    'standard': {
        'num_iterations': 25,
        'games_per_iteration': 300,
        'epochs_per_iteration': 8,
        'num_simulations': 150,
        'num_workers': 10,
        'num_res_blocks': 15,
        'num_channels': 256,
    },

    # Full training (18-24 hours) - best quality
    'full': {
        'num_iterations': 40,
        'games_per_iteration': 400,
        'epochs_per_iteration': 10,
        'num_simulations': 150,
        'num_workers': 10,
        'num_res_blocks': 15,
        'num_channels': 256,
    },

    # Maximum (36-48 hours) - competition level
    'max': {
        'num_iterations': 60,
        'games_per_iteration': 500,
        'epochs_per_iteration': 10,
        'num_simulations': 200,
        'num_workers': 10,
        'num_res_blocks': 15,
        'num_channels': 256,
    },
}


def print_system_info():
    """Print system information."""
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    print(f"CPU cores: {os.cpu_count()}")

    if os.environ.get('SLURM_JOB_ID'):
        print(f"\nSLURM Job: {os.environ.get('SLURM_JOB_ID')}")
        print(f"Node: {os.environ.get('SLURM_NODELIST')}")
        print(f"CPUs allocated: {os.environ.get('SLURM_CPUS_ON_NODE', 'N/A')}")

    print("=" * 70 + "\n")


def estimate_time(config: dict) -> str:
    """Estimate training time."""
    # With parallel workers, much faster
    games = config['num_iterations'] * config['games_per_iteration']
    workers = config['num_workers']
    sims = config['num_simulations']

    # Rough estimate: 5-10 seconds per game with 150 sims on CPU workers
    secs_per_game = sims * 0.05  # Much faster than before
    total_game_time = (games / workers) * secs_per_game

    # Training time (relatively fast)
    training_time = config['num_iterations'] * config['epochs_per_iteration'] * 30

    total_seconds = total_game_time + training_time
    hours = total_seconds / 3600

    if hours < 1:
        return f"{total_seconds / 60:.0f} minutes"
    else:
        return f"{hours:.1f} hours"


def train(args):
    """Main training function."""
    config = CONFIGS[args.mode]

    # Apply overrides
    if args.iterations:
        config['num_iterations'] = args.iterations
    if args.games:
        config['games_per_iteration'] = args.games
    if args.simulations:
        config['num_simulations'] = args.simulations
    if args.workers:
        config['num_workers'] = args.workers

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print_system_info()

    print("=" * 70)
    print("GROUP12 HEX AI - PARALLEL TRAINING V3")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    print(f"Estimated time: {estimate_time(config)}")
    print("=" * 70 + "\n")

    # Create directories
    model_dir = Path(args.output) / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create network
    print("Creating neural network...")
    network = create_hex_network_v2(
        num_res_blocks=config['num_res_blocks'],
        num_channels=config['num_channels'],
        device=device
    )

    # Create trainer
    trainer = ParallelHexTrainer(network, device=device)
    trainer.model_config = {
        'num_res_blocks': config['num_res_blocks'],
        'num_channels': config['num_channels']
    }

    # Resume if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    start_time = time.time()

    try:
        trainer.train_full(
            num_iterations=config['num_iterations'],
            games_per_iter=config['games_per_iteration'],
            epochs_per_iter=config['epochs_per_iteration'],
            num_simulations=config['num_simulations'],
            num_workers=config['num_workers'],
            model_dir=str(model_dir)
        )

        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Total time: {elapsed / 3600:.2f} hours")
        print(f"Models saved to: {model_dir}")
        print("=" * 70)

        # Copy to agent directory
        agent_models = PROJECT_ROOT / 'agents' / 'Group12' / 'models'
        agent_models.mkdir(exist_ok=True)
        numpy_src = model_dir / 'hex_model_v3_numpy.npz'
        if numpy_src.exists():
            import shutil
            shutil.copy(numpy_src, agent_models / 'hex_model_numpy.npz')
            print(f"Copied to: {agent_models / 'hex_model_numpy.npz'}")

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("INTERRUPTED - Checkpoints saved")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Parallel Hex AI Training V3')
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['test', 'fast', 'standard', 'full', 'max'])
    parser.add_argument('--output', type=str, default='.')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--games', type=int, default=None)
    parser.add_argument('--simulations', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
