"""
Export PyTorch model weights to NumPy format for tournament inference.

This script converts trained PyTorch models (.pth) to NumPy arrays (.npz)
that can be loaded by the NumPy inference engine without PyTorch.

Usage:
    python export_weights.py models/hex_model_best.pth models/hex_model_numpy.npz
"""

import sys
import argparse
from pathlib import Path

import numpy as np

# Try to import torch (only needed for export, not inference)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. Cannot export weights.")


def export_model(pytorch_path: str, numpy_path: str, verbose: bool = True):
    """
    Export PyTorch model to NumPy format.

    Args:
        pytorch_path: Path to .pth PyTorch model
        numpy_path: Path to save .npz NumPy weights
        verbose: Print progress
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to export weights")

    if verbose:
        print(f"Loading PyTorch model from {pytorch_path}...")

    # Load checkpoint
    checkpoint = torch.load(pytorch_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if verbose:
                print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume it's a raw state dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Convert to numpy
    numpy_weights = {}
    total_params = 0

    if verbose:
        print(f"\nConverting {len(state_dict)} weight tensors...")

    for name, param in state_dict.items():
        numpy_weights[name] = param.numpy().astype(np.float32)
        total_params += param.numel()

        if verbose:
            print(f"  {name}: {list(param.shape)}")

    # Save as .npz
    np.savez_compressed(numpy_path, **numpy_weights)

    # Report
    if verbose:
        file_size = Path(numpy_path).stat().st_size / 1024 / 1024
        print(f"\nExport complete!")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Output file: {numpy_path}")
        print(f"  File size: {file_size:.2f} MB")


def verify_export(pytorch_path: str, numpy_path: str, rtol: float = 1e-5):
    """
    Verify that exported weights match original.

    Args:
        pytorch_path: Original PyTorch model path
        numpy_path: Exported NumPy weights path
        rtol: Relative tolerance for comparison
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to verify weights")

    print(f"\nVerifying export...")

    # Load both
    checkpoint = torch.load(pytorch_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    numpy_weights = np.load(numpy_path)

    # Compare
    all_match = True
    for name in state_dict.keys():
        torch_arr = state_dict[name].numpy()
        numpy_arr = numpy_weights[name]

        if not np.allclose(torch_arr, numpy_arr, rtol=rtol):
            print(f"  MISMATCH: {name}")
            all_match = False
        else:
            max_diff = np.max(np.abs(torch_arr - numpy_arr))
            if max_diff > 0:
                print(f"  OK: {name} (max diff: {max_diff:.2e})")

    if all_match:
        print("All weights verified successfully!")
    else:
        print("WARNING: Some weights don't match!")

    numpy_weights.close()


def test_inference_equivalence(pytorch_path: str, numpy_path: str,
                               num_tests: int = 5):
    """
    Test that NumPy inference produces same results as PyTorch.

    Args:
        pytorch_path: PyTorch model path
        numpy_path: NumPy weights path
        num_tests: Number of random inputs to test
    """
    if not HAS_TORCH:
        print("Skipping inference test (PyTorch not available)")
        return

    print(f"\nTesting inference equivalence ({num_tests} random inputs)...")

    # Import the network classes
    from agents.Group12.neural.hex_network_v2 import HexNeuralNetworkV2
    from agents.Group12.neural.numpy_inference import NumpyHexNetwork

    # Load PyTorch model
    pytorch_model = HexNeuralNetworkV2(num_res_blocks=15)
    checkpoint = torch.load(pytorch_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()

    # Load NumPy model
    numpy_model = NumpyHexNetwork(numpy_path)

    # Test on random inputs
    max_policy_diff = 0
    max_value_diff = 0

    for i in range(num_tests):
        # Random input
        x_np = np.random.randn(1, 8, 11, 11).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # PyTorch inference
        with torch.no_grad():
            policy_torch, value_torch = pytorch_model(x_torch)
            policy_torch = torch.exp(policy_torch).numpy()
            value_torch = value_torch.numpy()

        # NumPy inference
        policy_numpy, value_numpy = numpy_model.forward(x_np)

        # Compare
        policy_diff = np.max(np.abs(policy_torch - policy_numpy))
        value_diff = np.abs(value_torch[0, 0] - value_numpy[0, 0])

        max_policy_diff = max(max_policy_diff, policy_diff)
        max_value_diff = max(max_value_diff, value_diff)

        print(f"  Test {i+1}: policy_diff={policy_diff:.6f}, value_diff={value_diff:.6f}")

    print(f"\nMax differences:")
    print(f"  Policy: {max_policy_diff:.6f}")
    print(f"  Value: {max_value_diff:.6f}")

    if max_policy_diff < 1e-4 and max_value_diff < 1e-4:
        print("Inference equivalence verified!")
    else:
        print("WARNING: Differences exceed threshold!")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to NumPy format"
    )
    parser.add_argument('pytorch_path', type=str,
                        help='Path to PyTorch model (.pth)')
    parser.add_argument('numpy_path', type=str,
                        help='Path to save NumPy weights (.npz)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify exported weights match')
    parser.add_argument('--test-inference', action='store_true',
                        help='Test inference equivalence')

    args = parser.parse_args()

    # Export
    export_model(args.pytorch_path, args.numpy_path)

    # Optional verification
    if args.verify:
        verify_export(args.pytorch_path, args.numpy_path)

    # Optional inference test
    if args.test_inference:
        test_inference_equivalence(args.pytorch_path, args.numpy_path)


if __name__ == '__main__':
    main()
