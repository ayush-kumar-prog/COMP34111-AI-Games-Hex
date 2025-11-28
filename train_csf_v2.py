#!/usr/bin/env python3
"""
CSF Training Script V2 - Ultimate Hex AI Training

This script trains the enhanced neural network (V2) with:
- 8-channel board encoding (Hex-specific features)
- 15 residual blocks with SE attention
- Dilated convolutions for larger receptive field
- Data augmentation (180-degree rotation)
- Cosine annealing learning rate schedule
- Automatic NumPy weight export for tournament

Usage:
    # Local testing
    python train_csf_v2.py --mode test

    # Full training on CSF (24-48 hours)
    python train_csf_v2.py --mode full --device cuda

    # Resume from checkpoint
    python train_csf_v2.py --mode full --resume models/hex_model_v2_iter5.pth
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.Group12.neural.hex_network_v2 import HexNeuralNetworkV2, create_hex_network_v2
from agents.Group12.neural.enhanced_board_encoder import EnhancedBoardEncoder
from agents.Group12.training.train_v2 import HexTrainerV2


# ============================================================================
# TRAINING CONFIGURATIONS
# ============================================================================

CONFIGS = {
    # Quick test - verify pipeline (5-10 minutes)
    'test': {
        'num_iterations': 2,
        'games_per_iteration': 10,
        'epochs_per_iteration': 2,
        'batch_size': 64,
        'num_simulations': 50,
        'learning_rate': 0.001,
        'num_res_blocks': 6,
        'num_channels': 128,
    },

    # Development - reasonable training (2-4 hours on GPU)
    'dev': {
        'num_iterations': 5,
        'games_per_iteration': 100,
        'epochs_per_iteration': 5,
        'batch_size': 256,
        'num_simulations': 100,
        'learning_rate': 0.001,
        'num_res_blocks': 10,
        'num_channels': 192,
    },

    # Standard - good training (8-12 hours on A100)
    'standard': {
        'num_iterations': 10,
        'games_per_iteration': 200,
        'epochs_per_iteration': 5,
        'batch_size': 256,
        'num_simulations': 200,
        'learning_rate': 0.001,
        'num_res_blocks': 12,
        'num_channels': 256,
    },

    # Full - comprehensive training (24-48 hours on A100)
    'full': {
        'num_iterations': 20,
        'games_per_iteration': 400,
        'epochs_per_iteration': 10,
        'batch_size': 512,
        'num_simulations': 400,
        'learning_rate': 0.001,
        'num_res_blocks': 15,
        'num_channels': 256,
    },

    # Maximum - championship-level (72+ hours on A100)
    'max': {
        'num_iterations': 30,
        'games_per_iteration': 800,
        'epochs_per_iteration': 10,
        'batch_size': 512,
        'num_simulations': 600,
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
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {memory_gb:.2f} GB")
            print(f"    Compute: {props.major}.{props.minor}")

    # CSF environment detection
    if os.environ.get('SLURM_JOB_ID'):
        print("\nCSF/SLURM Environment:")
        print(f"  Job ID: {os.environ.get('SLURM_JOB_ID')}")
        print(f"  Node: {os.environ.get('SLURM_NODELIST')}")
        print(f"  GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")

    print("=" * 70 + "\n")


def estimate_training_time(config: dict, device: str) -> str:
    """Estimate training time."""
    total_games = config['num_iterations'] * config['games_per_iteration']

    if 'cuda' in device:
        # A100 GPU estimate
        secs_per_game = config['num_simulations'] * 0.3
    else:
        secs_per_game = config['num_simulations'] * 2.0

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
    config = CONFIGS[args.mode]

    # Override with command line args
    if args.iterations:
        config['num_iterations'] = args.iterations
    if args.games:
        config['games_per_iteration'] = args.games
    if args.simulations:
        config['num_simulations'] = args.simulations
    if args.batch_size:
        config['batch_size'] = args.batch_size

    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print_system_info()

    print("=" * 70)
    print("GROUP12 HEX AI - NEURAL NETWORK TRAINING V2")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    print(f"Estimated time: {estimate_training_time(config, device)}")
    print("=" * 70 + "\n")

    # Create directories
    model_dir = Path(args.output) / 'models'
    log_dir = Path(args.output) / 'runs'
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create network
    print("Creating V2 neural network...")
    network = create_hex_network_v2(
        num_res_blocks=config['num_res_blocks'],
        num_channels=config['num_channels'],
        device=device
    )

    # Create trainer
    trainer = HexTrainerV2(
        network=network,
        learning_rate=config['learning_rate'],
        weight_decay=1e-4,
        device=device
    )

    # Resume if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Training
    start_time = time.time()

    try:
        trainer.train_full(
            num_iterations=config['num_iterations'],
            games_per_iter=config['games_per_iteration'],
            epochs_per_iter=config['epochs_per_iteration'],
            num_simulations=config['num_simulations'],
            model_dir=str(model_dir),
            log_dir=str(log_dir)
        )

        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED!")
        print("=" * 70)
        print(f"Total time: {elapsed / 3600:.2f} hours")
        print(f"PyTorch model: {model_dir / 'hex_model_v2_best.pth'}")
        print(f"NumPy weights: {model_dir / 'hex_model_numpy.npz'}")
        print("=" * 70)

        # Copy NumPy weights to agent directory
        agent_models = PROJECT_ROOT / 'agents' / 'Group12' / 'models'
        agent_models.mkdir(exist_ok=True)
        numpy_src = model_dir / 'hex_model_numpy.npz'
        numpy_dst = agent_models / 'hex_model_numpy.npz'
        if numpy_src.exists():
            import shutil
            shutil.copy(numpy_src, numpy_dst)
            print(f"Copied to: {numpy_dst}")

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING INTERRUPTED")
        print("=" * 70)
        print(f"Elapsed: {elapsed / 3600:.2f} hours")
        print(f"Models in: {model_dir}")
        print("Resume with: --resume <checkpoint.pth>")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"ERROR: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Train Group12 Hex AI V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test
    python train_csf_v2.py --mode test

    # Full training on CSF A100
    python train_csf_v2.py --mode full --device cuda

    # Resume training
    python train_csf_v2.py --mode full --resume models/hex_model_v2_iter10.pth
        """
    )

    parser.add_argument('--mode', type=str, default='dev',
                        choices=['test', 'dev', 'standard', 'full', 'max'])
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--output', type=str, default='.')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--games', type=int, default=None)
    parser.add_argument('--simulations', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
