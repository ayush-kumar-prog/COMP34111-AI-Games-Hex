#!/usr/bin/env python3
"""
CSF Training Script V4 - Batched GPU Inference

Key improvements over V3:
- ALL neural network calls batched on GPU
- Threading-based parallelism (not multiprocessing) to share GPU
- Expected GPU utilization: 60-90% (vs 1% in V2)
- Expected speedup: 20-50x over V2

Usage:
    python train_csf_v4.py --mode test      # Quick test (10-15 min)
    python train_csf_v4.py --mode fast      # Fast training (2-3 hours)
    python train_csf_v4.py --mode standard  # Standard (4-6 hours)
    python train_csf_v4.py --mode full      # Full (8-12 hours)
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
from agents.Group12.training.train_v4_batched import BatchedTrainer


# V4 configurations - Pure AlphaZero with maximum GPU utilization
# Reduced workers to avoid threading issues on CSF3
CONFIGS = {
    # Quick test (10-15 minutes)
    'test': {
        'num_iterations': 3,
        'games_per_iteration': 64,
        'epochs_per_iteration': 3,
        'mcts_sims': 30,
        'num_workers': 16,  # Reduced from 64
        'num_res_blocks': 6,
        'num_channels': 128,
    },

    # Fast training (1-2 hours) - good for experimentation
    'fast': {
        'num_iterations': 15,
        'games_per_iteration': 300,
        'epochs_per_iteration': 5,
        'mcts_sims': 50,
        'num_workers': 32,  # Reduced from 128
        'num_res_blocks': 12,
        'num_channels': 192,
    },

    # Standard training (3-5 hours) - good balance
    'standard': {
        'num_iterations': 25,
        'games_per_iteration': 400,
        'epochs_per_iteration': 8,
        'mcts_sims': 60,
        'num_workers': 32,  # Reduced from 128
        'num_res_blocks': 15,
        'num_channels': 256,
    },

    # Full training (6-10 hours) - best quality
    'full': {
        'num_iterations': 40,
        'games_per_iteration': 600,
        'epochs_per_iteration': 10,
        'mcts_sims': 80,
        'num_workers': 32,  # Reduced from 128
        'num_res_blocks': 15,
        'num_channels': 256,
    },

    # Maximum (12-18 hours) - competition level
    'max': {
        'num_iterations': 60,
        'games_per_iteration': 800,
        'epochs_per_iteration': 10,
        'mcts_sims': 100,
        'num_workers': 32,  # Reduced from 128
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

        # GPU memory info
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    print(f"CPU cores: {os.cpu_count()}")

    if os.environ.get('SLURM_JOB_ID'):
        print(f"\nSLURM Job: {os.environ.get('SLURM_JOB_ID')}")
        print(f"Node: {os.environ.get('SLURM_NODELIST')}")
        print(f"CPUs allocated: {os.environ.get('SLURM_CPUS_ON_NODE', 'N/A')}")

    print("=" * 70 + "\n")


def estimate_time(config: dict) -> str:
    """Estimate training time with pure AlphaZero batched inference."""
    games = config['num_iterations'] * config['games_per_iteration']
    workers = config['num_workers']
    sims = config['mcts_sims']

    # Pure AlphaZero: each sim = 2 GPU calls, batched across workers
    # With 128 workers and batch size 512, very fast
    # ~40 moves per game, each move = sims GPU batches
    moves_per_game = 40
    gpu_calls_per_game = moves_per_game * sims * 2
    total_gpu_calls = games * gpu_calls_per_game

    # A100 can do ~5000 inferences/sec with batch 512
    inferences_per_sec = 5000
    gpu_time = total_gpu_calls / inferences_per_sec

    # Add overhead for worker coordination
    overhead_factor = 1.5
    game_time = gpu_time * overhead_factor

    # Training time (GPU training is fast)
    training_time = config['num_iterations'] * config['epochs_per_iteration'] * 15

    total_seconds = game_time + training_time
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
    if args.mcts_sims:
        config['mcts_sims'] = args.mcts_sims
    if args.workers:
        config['num_workers'] = args.workers

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print_system_info()

    print("=" * 70)
    print("GROUP12 HEX AI - BATCHED GPU TRAINING V4")
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
    print()
    print("Key V4 improvements:")
    print("  - Batched GPU inference (60-90% utilization)")
    print("  - Threading-based workers share GPU")
    print("  - 20-50x faster than V2")
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

    param_count = sum(p.numel() for p in network.parameters())
    print(f"Parameters: {param_count:,}")

    # Create trainer
    trainer = BatchedTrainer(network, device=device)

    # Resume if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        network.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            trainer.epoch = checkpoint['epoch']
        print(f"Resumed at epoch {trainer.epoch}")

    # Train
    start_time = time.time()

    try:
        trainer.train_full(
            num_iterations=config['num_iterations'],
            games_per_iter=config['games_per_iteration'],
            epochs_per_iter=config['epochs_per_iteration'],
            mcts_sims=config['mcts_sims'],
            num_workers=config['num_workers'],
            model_dir=str(model_dir)
        )

        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Total time: {elapsed / 3600:.2f} hours")
        print(f"Models saved to: {model_dir}")
        print()
        print("Model files:")
        for f in sorted(model_dir.glob("*")):
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name}: {size_mb:.1f} MB")
        print("=" * 70)

        # Copy to agent directory
        agent_models = PROJECT_ROOT / 'agents' / 'Group12' / 'models'
        agent_models.mkdir(exist_ok=True)
        numpy_src = model_dir / 'hex_v4_numpy.npz'
        if numpy_src.exists():
            import shutil
            dst = agent_models / 'hex_model_numpy.npz'
            shutil.copy(numpy_src, dst)
            print(f"Copied to: {dst}")

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
    parser = argparse.ArgumentParser(description='Batched GPU Hex AI Training V4')
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['test', 'fast', 'standard', 'full', 'max'])
    parser.add_argument('--output', type=str, default='.')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--games', type=int, default=None)
    parser.add_argument('--mcts-sims', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
