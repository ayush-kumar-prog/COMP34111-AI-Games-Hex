#!/usr/bin/env python3
"""
Test V4.2 Batched Training Locally

This script tests the fixed V4.2 training on CPU to verify:
1. BatchedInferenceEngine doesn't deadlock
2. Workers complete games successfully
3. Training loop works end-to-end

Usage:
    python test_v4_local.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np


def test_batched_inference_engine():
    """Test the BatchedInferenceEngine in isolation."""
    print("=" * 60)
    print("TEST 1: BatchedInferenceEngine")
    print("=" * 60)

    from agents.Group12.training.train_v4_batched import BatchedInferenceEngine, BoardEncoder
    from agents.Group12.neural.hex_network_v2 import create_hex_network_v2

    device = 'cpu'  # Use CPU for local testing

    print("Creating network...", flush=True)
    network = create_hex_network_v2(num_res_blocks=2, num_channels=32, device=device)
    print(f"Network created on {device}", flush=True)

    print("Creating inference engine...", flush=True)
    engine = BatchedInferenceEngine(network, device=device, max_batch_size=8, max_wait_ms=50)
    print(f"Engine created, healthy: {engine.is_healthy()}", flush=True)

    encoder = BoardEncoder()

    # Test single inference
    print("\nTesting single inference...", flush=True)
    from agents.Group12.training.train_v4_batched import FastBoard
    board = FastBoard(11)
    state = encoder.encode(board, 1)

    start = time.time()
    policy, value = engine.request_inference(state, timeout=5.0)
    elapsed = time.time() - start

    print(f"Inference completed in {elapsed:.3f}s", flush=True)
    print(f"Policy shape: {policy.shape}, Value: {value:.4f}", flush=True)
    print(f"Policy sum: {policy.sum():.4f}", flush=True)

    # Test concurrent inferences
    print("\nTesting concurrent inferences (10 threads)...", flush=True)
    import threading
    results = []
    errors = []

    def worker(worker_id):
        try:
            for i in range(5):
                board = FastBoard(11)
                # Make some random moves
                for _ in range(worker_id % 5):
                    empty = board.get_empty()
                    if empty:
                        x, y = empty[0]
                        board.set_tile(x, y, 1)
                state = encoder.encode(board, 1)
                policy, value = engine.request_inference(state, timeout=10.0)
                results.append((worker_id, i, policy.sum()))
        except Exception as e:
            errors.append((worker_id, str(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    elapsed = time.time() - start

    print(f"Completed {len(results)} inferences in {elapsed:.2f}s", flush=True)
    print(f"Rate: {len(results)/elapsed:.1f} inferences/s", flush=True)
    print(f"Errors: {len(errors)}", flush=True)
    if errors:
        for err in errors[:3]:
            print(f"  Error: {err}", flush=True)

    stats = engine.get_stats()
    print(f"Engine stats: {stats}", flush=True)

    engine.stop()

    if len(results) == 50 and len(errors) == 0:
        print("\n[PASS] BatchedInferenceEngine test passed!", flush=True)
        return True
    else:
        print("\n[FAIL] BatchedInferenceEngine test failed!", flush=True)
        return False


def test_game_worker():
    """Test that game workers can complete games."""
    print("\n" + "=" * 60)
    print("TEST 2: Game Worker")
    print("=" * 60)

    from agents.Group12.training.train_v4_batched import (
        BatchedInferenceEngine, BoardEncoder, PureAlphaZeroMCTS,
        FastBoard, game_worker, TrainingExample
    )
    from agents.Group12.neural.hex_network_v2 import create_hex_network_v2
    import queue

    device = 'cpu'

    print("Creating network and engine...", flush=True)
    network = create_hex_network_v2(num_res_blocks=2, num_channels=32, device=device)
    engine = BatchedInferenceEngine(network, device=device, max_batch_size=8, max_wait_ms=50)
    encoder = BoardEncoder()
    mcts = PureAlphaZeroMCTS(engine, encoder, num_sims=5)  # Very few sims for speed

    print("Starting 2 workers, 2 games each...", flush=True)
    result_queue = queue.Queue()

    import threading
    workers = []
    for i in range(2):
        t = threading.Thread(target=game_worker, args=(i, None, result_queue, mcts, 2), daemon=True)
        t.start()
        workers.append(t)

    # Collect results
    games_done = 0
    total_games = 4
    all_examples = []
    start = time.time()

    while games_done < total_games:
        try:
            examples = result_queue.get(timeout=60)
            all_examples.extend(examples)
            games_done += 1
            print(f"  Game {games_done}/{total_games} completed, {len(examples)} examples", flush=True)
        except queue.Empty:
            alive = sum(1 for t in workers if t.is_alive())
            if alive == 0:
                print("All workers died!", flush=True)
                break
            print(f"Waiting... {alive} workers alive", flush=True)

    elapsed = time.time() - start

    for t in workers:
        t.join(timeout=5)

    engine.stop()

    print(f"\nCompleted {games_done} games in {elapsed:.1f}s", flush=True)
    print(f"Total examples: {len(all_examples)}", flush=True)

    if games_done == 4:
        print("\n[PASS] Game worker test passed!", flush=True)
        return True
    else:
        print("\n[FAIL] Game worker test failed!", flush=True)
        return False


def test_full_iteration():
    """Test a full training iteration."""
    print("\n" + "=" * 60)
    print("TEST 3: Full Training Iteration")
    print("=" * 60)

    from agents.Group12.training.train_v4_batched import BatchedTrainer
    from agents.Group12.neural.hex_network_v2 import create_hex_network_v2
    import tempfile

    device = 'cpu'

    print("Creating network...", flush=True)
    network = create_hex_network_v2(num_res_blocks=2, num_channels=32, device=device)

    print("Creating trainer...", flush=True)
    trainer = BatchedTrainer(network, device=device)

    print("\nGenerating 8 games with 4 workers...", flush=True)
    start = time.time()
    examples = trainer.generate_games(num_games=8, num_workers=4, mcts_sims=5)
    elapsed = time.time() - start

    print(f"Generated {len(examples)} examples in {elapsed:.1f}s", flush=True)

    if len(examples) > 0:
        print("\nTraining for 2 epochs...", flush=True)
        for epoch in range(2):
            metrics = trainer.train_epoch(examples, batch_size=32)
            print(f"  Epoch {epoch+1}: Loss={metrics['loss']:.4f}", flush=True)

        print("\n[PASS] Full iteration test passed!", flush=True)
        return True
    else:
        print("\n[FAIL] No examples generated!", flush=True)
        return False


def main():
    print("=" * 60)
    print("V4.2 LOCAL TEST SUITE")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: cpu (local testing)")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("BatchedInferenceEngine", test_batched_inference_engine()))
    results.append(("Game Worker", test_game_worker()))
    results.append(("Full Iteration", test_full_iteration()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests passed! V4.2 is ready for deployment.")
        return 0
    else:
        print("Some tests failed. Review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
