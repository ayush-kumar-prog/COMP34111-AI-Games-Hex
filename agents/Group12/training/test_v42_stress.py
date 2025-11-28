#!/usr/bin/env python3
"""
V4.2 Stress Test

Higher concurrency test to verify deadlock fixes under load.
"""

import os
import sys
import time
import queue
import random
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn


class SmallTestNetwork(nn.Module):
    """Small network for testing."""

    def __init__(self, in_channels=8, num_channels=32, num_res_blocks=2, board_size=11):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.policy_fc = nn.Linear(num_channels * board_size * board_size, board_size * board_size)
        self.value_fc = nn.Linear(num_channels * board_size * board_size, 1)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        p = self.policy_fc(x)
        v = torch.tanh(self.value_fc(x))
        return p, v.squeeze(-1)


from agents.Group12.training.train_v4_batched import BatchedInferenceEngine


def stress_test_concurrent_inference():
    """Stress test with many concurrent threads."""
    print("="*70)
    print("STRESS TEST: High Concurrency Inference")
    print("="*70)

    network = SmallTestNetwork()
    engine = BatchedInferenceEngine(
        network=network,
        device='cpu',
        max_batch_size=64,
        max_wait_ms=10
    )

    time.sleep(0.5)

    # Test parameters
    num_threads = 32
    inferences_per_thread = 50
    total_expected = num_threads * inferences_per_thread

    print(f"\nRunning {num_threads} threads x {inferences_per_thread} inferences = {total_expected} total")

    completed = []
    errors = []
    lock = threading.Lock()

    def worker(worker_id):
        local_count = 0
        try:
            for _ in range(inferences_per_thread):
                pos = np.random.randn(8, 11, 11).astype(np.float32)
                policy, value = engine.request_inference(pos, timeout=30.0)

                if policy.shape == (121,) and isinstance(value, (float, np.floating)):
                    local_count += 1
                else:
                    with lock:
                        errors.append(f"Worker {worker_id}: Bad output")
                    return

            with lock:
                completed.append((worker_id, local_count))
        except Exception as e:
            with lock:
                errors.append(f"Worker {worker_id}: {e}")

    threads = []
    start_time = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)

    # Wait with timeout
    timeout = 120
    deadline = time.time() + timeout

    for t in threads:
        remaining = deadline - time.time()
        if remaining > 0:
            t.join(timeout=remaining)

    elapsed = time.time() - start_time

    # Check results
    alive = sum(1 for t in threads if t.is_alive())
    total_completed = sum(c[1] for c in completed)

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Threads completed: {len(completed)}/{num_threads}")
    print(f"  Inferences completed: {total_completed}/{total_expected}")
    print(f"  Threads still alive: {alive}")
    print(f"  Errors: {len(errors)}")

    if errors:
        for err in errors[:5]:
            print(f"    - {err}")

    stats = engine.get_stats()
    print(f"\nEngine stats:")
    print(f"  Total inferences: {stats['total_inferences']}")
    print(f"  Total batches: {stats['total_batches']}")
    print(f"  Avg batch size: {stats['avg_batch_size']:.2f}")
    print(f"  Failed: {stats['failed_inferences']}")
    print(f"  Throughput: {total_completed/elapsed:.1f} inferences/s")

    engine.stop()

    if alive > 0:
        print("\n>>> STRESS TEST: FAIL (threads deadlocked)")
        return False

    if total_completed < total_expected * 0.95:
        print(f"\n>>> STRESS TEST: FAIL (only {total_completed}/{total_expected} completed)")
        return False

    print("\n>>> STRESS TEST: PASS")
    return True


def stress_test_rapid_start_stop():
    """Test rapid creation and destruction of engines."""
    print("\n" + "="*70)
    print("STRESS TEST: Rapid Engine Creation/Destruction")
    print("="*70)

    iterations = 10

    print(f"\nCreating and destroying {iterations} engines...")

    for i in range(iterations):
        network = SmallTestNetwork()
        engine = BatchedInferenceEngine(
            network=network,
            device='cpu',
            max_batch_size=32,
            max_wait_ms=5
        )

        # Do a few inferences
        for _ in range(5):
            pos = np.random.randn(8, 11, 11).astype(np.float32)
            policy, value = engine.request_inference(pos, timeout=5.0)

        engine.stop()
        print(f"  Engine {i+1}/{iterations} OK")

    print("\n>>> STRESS TEST: PASS")
    return True


def stress_test_timeout_flood():
    """Test behavior when engine is flooded with requests."""
    print("\n" + "="*70)
    print("STRESS TEST: Request Flood with Slow Processing")
    print("="*70)

    # Create a very slow network
    class SlowNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 32, 3, padding=1)
            self.fc_p = nn.Linear(32 * 11 * 11, 121)
            self.fc_v = nn.Linear(32 * 11 * 11, 1)

        def forward(self, x):
            # Add artificial delay
            time.sleep(0.05 * len(x))  # 50ms per sample
            x = torch.relu(self.conv(x))
            x = x.view(x.size(0), -1)
            return self.fc_p(x), torch.tanh(self.fc_v(x)).squeeze(-1)

    network = SlowNetwork()
    engine = BatchedInferenceEngine(
        network=network,
        device='cpu',
        max_batch_size=16,
        max_wait_ms=5
    )

    time.sleep(0.5)

    # Flood with requests
    num_threads = 20
    inferences = 10
    timeout_count = [0]
    success_count = [0]
    lock = threading.Lock()

    print(f"\nFlooding with {num_threads} x {inferences} = {num_threads * inferences} requests...")
    print(f"Using 2s timeout per request...")

    def flood_worker():
        for _ in range(inferences):
            pos = np.random.randn(8, 11, 11).astype(np.float32)
            start = time.time()
            policy, value = engine.request_inference(pos, timeout=2.0)
            elapsed = time.time() - start

            with lock:
                if elapsed >= 1.9:  # Near timeout
                    timeout_count[0] += 1
                else:
                    success_count[0] += 1

    threads = []
    start_time = time.time()

    for _ in range(num_threads):
        t = threading.Thread(target=flood_worker)
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=60)

    elapsed = time.time() - start_time

    alive = sum(1 for t in threads if t.is_alive())

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Successful: {success_count[0]}")
    print(f"  Timeouts: {timeout_count[0]}")
    print(f"  Threads alive: {alive}")

    stats = engine.get_stats()
    print(f"\nEngine stats:")
    print(f"  Total inferences: {stats['total_inferences']}")
    print(f"  Failed: {stats['failed_inferences']}")

    engine.stop()

    if alive > 0:
        print("\n>>> STRESS TEST: FAIL (threads deadlocked)")
        return False

    print("\n>>> STRESS TEST: PASS (all threads returned, timeouts handled)")
    return True


def main():
    print("\n" + "="*70)
    print(" V4.2 STRESS TESTS")
    print("="*70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    results.append(("High Concurrency", stress_test_concurrent_inference()))
    results.append(("Rapid Start/Stop", stress_test_rapid_start_stop()))
    results.append(("Timeout Flood", stress_test_timeout_flood()))

    print("\n" + "="*70)
    print(" STRESS TEST SUMMARY")
    print("="*70)

    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1

    print(f"\n  Total: {passed}/{len(results)} passed")

    if passed == len(results):
        print("\n>>> ALL STRESS TESTS PASSED - V4.2 is robust")
    else:
        print("\n>>> SOME TESTS FAILED - review before deployment")

    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
