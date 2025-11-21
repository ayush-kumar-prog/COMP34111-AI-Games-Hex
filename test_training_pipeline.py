#!/usr/bin/env python3
"""
Quick test of training pipeline on CPU.
Verifies all components work before GPU training.

This runs a mini training cycle:
- 2 iterations
- 10 games per iteration
- 2 epochs per iteration
- 50 MCTS simulations (fast)

Expected time: 5-10 minutes on CPU
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.Group12.neural.hex_network import create_hex_network
from agents.Group12.training.trainer import HexTrainer


def test_training_pipeline():
    """Test training pipeline with minimal parameters."""

    print("\n" + "="*70)
    print("TRAINING PIPELINE VERIFICATION (CPU)")
    print("="*70)
    print("Purpose: Verify all components work before GPU training")
    print("Parameters: Minimal (2 iterations, 10 games, 50 MCTS sims)")
    print("Expected time: 5-10 minutes")
    print("="*70 + "\n")

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if device == 'cpu':
        print("⚠️  Running on CPU (slow but functional)")
        print("   For production training, use GPU with:")
        print("   docker run --runtime=nvidia --gpus all ...")

    print()

    # Create small network for testing
    print("[1/3] Creating neural network (small version for testing)...")
    network = create_hex_network(
        board_size=11,
        num_res_blocks=2,  # Small: 2 instead of 10
        num_channels=64,    # Small: 64 instead of 256
        device=device
    )
    print(f"✅ Network created with {network.get_num_parameters():,} parameters")

    # Create trainer
    print("\n[2/3] Creating trainer...")
    trainer = HexTrainer(
        network=network,
        learning_rate=0.01,  # Higher LR for fast test
        weight_decay=1e-4,
        device=device
    )
    print("✅ Trainer created")

    # Run mini training
    print("\n[3/3] Running training cycle...")
    print("This will take 5-10 minutes on CPU")
    print("Press Ctrl+C to interrupt if needed\n")

    try:
        trainer.train(
            num_iterations=2,        # Just 2 iterations
            games_per_iteration=10,  # Just 10 games
            epochs_per_iteration=2,  # Just 2 epochs
            batch_size=32,           # Smaller batch
            num_simulations=50,      # Fast MCTS (50 instead of 800)
            model_dir='models/test',
            log_dir='runs/test'
        )

        print("\n" + "="*70)
        print("✅ TRAINING PIPELINE VERIFICATION COMPLETE!")
        print("="*70)
        print("All components working correctly:")
        print("  ✅ Neural network")
        print("  ✅ Board encoder")
        print("  ✅ Neural MCTS")
        print("  ✅ Self-play generation")
        print("  ✅ Dataset loading")
        print("  ✅ Training loop")
        print("  ✅ Checkpoint saving")
        print("\nReady for full GPU training!")
        print("Use: python3 train_neural_agent.py --device cuda")
        print("="*70 + "\n")

        return True

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Partial results saved in models/test/")
        return False

    except Exception as e:
        print(f"\n\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_training_pipeline()
    sys.exit(0 if success else 1)
