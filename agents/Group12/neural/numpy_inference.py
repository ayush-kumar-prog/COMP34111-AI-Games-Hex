"""
Pure NumPy neural network inference engine.

This module implements the forward pass of the HexNeuralNetworkV2
using only NumPy (and scipy for convolutions), allowing inference
in environments without PyTorch (like the tournament Docker).

The key insight is that once trained, a neural network only needs
matrix multiplications, convolutions, and simple element-wise operations
- all of which NumPy can do efficiently.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path

# Try to import scipy for efficient convolutions, fall back to pure numpy
try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class NumpyConv2d:
    """2D convolution layer implemented in NumPy."""

    def __init__(self, weight: np.ndarray, bias: Optional[np.ndarray] = None,
                 padding: int = 1, dilation: int = 1):
        """
        Initialize convolution layer.

        Args:
            weight: Convolution kernels (out_channels, in_channels, kH, kW)
            bias: Optional bias (out_channels,)
            padding: Padding size
            dilation: Dilation factor
        """
        self.weight = weight
        self.bias = bias
        self.padding = padding
        self.dilation = dilation

        self.out_channels = weight.shape[0]
        self.in_channels = weight.shape[1]
        self.kernel_size = weight.shape[2]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input (batch, in_channels, H, W)

        Returns:
            Output (batch, out_channels, H_out, W_out)
        """
        batch_size = x.shape[0]
        h_in, w_in = x.shape[2], x.shape[3]

        # Handle dilation by dilating the kernel
        if self.dilation > 1:
            weight = self._dilate_kernel(self.weight, self.dilation)
            # For dilated conv, self.padding already accounts for the larger receptive field
            # Don't multiply again - just use self.padding directly
            padding = self.padding
        else:
            weight = self.weight
            padding = self.padding

        # Pad input
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding),
                                  (padding, padding)), mode='constant')
        else:
            x_padded = x

        h_out = h_in  # Same padding maintains spatial dimensions
        w_out = w_in

        # Output tensor
        output = np.zeros((batch_size, self.out_channels, h_out, w_out),
                         dtype=np.float32)

        # Convolution via correlation
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    if HAS_SCIPY:
                        output[b, oc] += signal.correlate2d(
                            x_padded[b, ic], weight[oc, ic],
                            mode='valid'
                        )
                    else:
                        output[b, oc] += self._convolve2d(
                            x_padded[b, ic], weight[oc, ic]
                        )

        # Add bias
        if self.bias is not None:
            output += self.bias.reshape(1, -1, 1, 1)

        return output

    def _dilate_kernel(self, kernel: np.ndarray, dilation: int) -> np.ndarray:
        """Dilate kernel by inserting zeros."""
        out_c, in_c, kh, kw = kernel.shape
        new_kh = kh + (kh - 1) * (dilation - 1)
        new_kw = kw + (kw - 1) * (dilation - 1)
        dilated = np.zeros((out_c, in_c, new_kh, new_kw), dtype=kernel.dtype)
        dilated[:, :, ::dilation, ::dilation] = kernel
        return dilated

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Pure numpy 2D convolution (correlation)."""
        kh, kw = kernel.shape
        h_out = image.shape[0] - kh + 1
        w_out = image.shape[1] - kw + 1
        output = np.zeros((h_out, w_out), dtype=np.float32)

        for i in range(h_out):
            for j in range(w_out):
                output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)

        return output


class NumpyBatchNorm2d:
    """Batch normalization layer implemented in NumPy."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray,
                 running_mean: np.ndarray, running_var: np.ndarray,
                 eps: float = 1e-5):
        """
        Initialize batch norm layer.

        Args:
            weight: Scale parameters (gamma)
            bias: Shift parameters (beta)
            running_mean: Running mean from training
            running_var: Running variance from training
            eps: Small constant for numerical stability
        """
        self.weight = weight
        self.bias = bias
        self.running_mean = running_mean
        self.running_var = running_var
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass (inference mode using running statistics).

        Args:
            x: Input (batch, channels, H, W)

        Returns:
            Normalized output
        """
        # Normalize using running statistics
        mean = self.running_mean.reshape(1, -1, 1, 1)
        var = self.running_var.reshape(1, -1, 1, 1)
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return weight * x_norm + bias


class NumpyLinear:
    """Linear (fully connected) layer implemented in NumPy."""

    def __init__(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        """
        Initialize linear layer.

        Args:
            weight: Weight matrix (out_features, in_features)
            bias: Optional bias (out_features,)
        """
        self.weight = weight
        self.bias = bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input (batch, in_features)

        Returns:
            Output (batch, out_features)
        """
        output = np.dot(x, self.weight.T)
        if self.bias is not None:
            output += self.bias
        return output


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax activation (numerically stable)."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Log softmax activation (numerically stable)."""
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation."""
    return np.tanh(x)


class NumpySqueezeExcitation:
    """Squeeze-and-Excitation block in NumPy."""

    def __init__(self, fc1_weight: np.ndarray, fc2_weight: np.ndarray):
        """
        Initialize SE block.

        Args:
            fc1_weight: First FC layer weight
            fc2_weight: Second FC layer weight
        """
        self.fc1 = NumpyLinear(fc1_weight, None)
        self.fc2 = NumpyLinear(fc2_weight, None)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input (batch, channels, H, W)

        Returns:
            Output with channel attention applied
        """
        batch, channels, h, w = x.shape

        # Global average pooling
        y = np.mean(x, axis=(2, 3))  # (batch, channels)

        # FC layers with ReLU and Sigmoid
        y = relu(self.fc1(y))
        y = 1 / (1 + np.exp(-self.fc2(y)))  # Sigmoid

        # Reshape and multiply
        y = y.reshape(batch, channels, 1, 1)
        return x * y


class NumpyResBlock:
    """Residual block in NumPy."""

    def __init__(self, conv1: NumpyConv2d, bn1: NumpyBatchNorm2d,
                 conv2: NumpyConv2d, bn2: NumpyBatchNorm2d,
                 se: Optional[NumpySqueezeExcitation] = None):
        """Initialize residual block."""
        self.conv1 = conv1
        self.bn1 = bn1
        self.conv2 = conv2
        self.bn2 = bn2
        self.se = se

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with residual connection."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)

        out = out + residual
        out = relu(out)

        return out


class NumpyHexNetwork:
    """
    Complete Hex neural network implemented in pure NumPy.

    This class loads weights from a trained PyTorch model and
    performs inference using only NumPy operations.
    """

    def __init__(self, weights_path: str, board_size: int = 11):
        """
        Initialize network from saved weights.

        Args:
            weights_path: Path to .npz weights file
            board_size: Board size (default 11)
        """
        self.board_size = board_size

        # Load weights
        print(f"Loading NumPy weights from {weights_path}...")
        weights = np.load(weights_path)
        self.weights = {k: weights[k] for k in weights.files}
        weights.close()

        # Build network layers
        self._build_network()

        print(f"NumPy Hex Network initialized:")
        print(f"  Board size: {board_size}x{board_size}")
        print(f"  Residual blocks: {len(self.res_blocks)}")

    def _build_network(self):
        """Build network layers from loaded weights."""
        w = self.weights

        # Input convolution
        self.conv_input = NumpyConv2d(w['conv_input.weight'], None, padding=1)
        self.bn_input = NumpyBatchNorm2d(
            w['bn_input.weight'], w['bn_input.bias'],
            w['bn_input.running_mean'], w['bn_input.running_var']
        )

        # Residual blocks
        self.res_blocks = []
        block_idx = 0
        while f'res_blocks.{block_idx}.conv1.weight' in w:
            # Determine dilation (blocks 5 and 10 use dilation=2)
            dilation = 2 if block_idx in [5, 10] else 1
            padding = dilation

            conv1 = NumpyConv2d(
                w[f'res_blocks.{block_idx}.conv1.weight'], None,
                padding=padding, dilation=dilation
            )
            bn1 = NumpyBatchNorm2d(
                w[f'res_blocks.{block_idx}.bn1.weight'],
                w[f'res_blocks.{block_idx}.bn1.bias'],
                w[f'res_blocks.{block_idx}.bn1.running_mean'],
                w[f'res_blocks.{block_idx}.bn1.running_var']
            )

            conv2 = NumpyConv2d(
                w[f'res_blocks.{block_idx}.conv2.weight'], None, padding=1
            )
            bn2 = NumpyBatchNorm2d(
                w[f'res_blocks.{block_idx}.bn2.weight'],
                w[f'res_blocks.{block_idx}.bn2.bias'],
                w[f'res_blocks.{block_idx}.bn2.running_mean'],
                w[f'res_blocks.{block_idx}.bn2.running_var']
            )

            # SE block (if present)
            se = None
            if f'res_blocks.{block_idx}.se.fc.0.weight' in w:
                se = NumpySqueezeExcitation(
                    w[f'res_blocks.{block_idx}.se.fc.0.weight'],
                    w[f'res_blocks.{block_idx}.se.fc.2.weight']
                )

            self.res_blocks.append(NumpyResBlock(conv1, bn1, conv2, bn2, se))
            block_idx += 1

        # Policy head
        self.policy_conv = NumpyConv2d(w['policy_conv.weight'], None, padding=0)
        self.policy_bn = NumpyBatchNorm2d(
            w['policy_bn.weight'], w['policy_bn.bias'],
            w['policy_bn.running_mean'], w['policy_bn.running_var']
        )
        self.policy_fc = NumpyLinear(w['policy_fc.weight'], w['policy_fc.bias'])

        # Value head
        self.value_conv = NumpyConv2d(w['value_conv.weight'], None, padding=0)
        self.value_bn = NumpyBatchNorm2d(
            w['value_bn.weight'], w['value_bn.bias'],
            w['value_bn.running_mean'], w['value_bn.running_var']
        )
        self.value_fc1 = NumpyLinear(w['value_fc1.weight'], w['value_fc1.bias'])
        self.value_fc2 = NumpyLinear(w['value_fc2.weight'], w['value_fc2.bias'])

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the network.

        Args:
            x: Input (batch, 8, board_size, board_size)

        Returns:
            policy: Move probabilities (batch, board_size^2)
            value: Position evaluation (batch, 1)
        """
        # Input convolution
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = relu(x)

        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = relu(policy)
        policy = policy.reshape(policy.shape[0], -1)
        policy = self.policy_fc(policy)
        policy = softmax(policy, axis=1)

        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = relu(value)
        value = value.reshape(value.shape[0], -1)
        value = self.value_fc1(value)
        value = relu(value)
        value = self.value_fc2(value)
        value = tanh(value)

        return policy, value

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value for a single state.

        Args:
            state: Board state (8, board_size, board_size)

        Returns:
            policy: Move probabilities (board_size^2,)
            value: Position evaluation scalar
        """
        if state.ndim == 3:
            state = state[np.newaxis, ...]

        policy, value = self.forward(state)
        return policy[0], float(value[0, 0])


def export_pytorch_to_numpy(pytorch_model_path: str, output_path: str):
    """
    Export PyTorch model weights to NumPy format.

    Args:
        pytorch_model_path: Path to .pth PyTorch model
        output_path: Path to save .npz NumPy weights
    """
    import torch

    print(f"Loading PyTorch model from {pytorch_model_path}...")
    state_dict = torch.load(pytorch_model_path, map_location='cpu')

    # Handle checkpoint format
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Convert to numpy
    numpy_weights = {}
    for name, param in state_dict.items():
        numpy_weights[name] = param.numpy()
        print(f"  {name}: {param.shape}")

    # Save
    np.savez(output_path, **numpy_weights)
    print(f"Saved NumPy weights to {output_path}")

    # Print file size
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"File size: {size_mb:.1f} MB")


def test_numpy_inference():
    """Test NumPy inference engine."""
    print("Testing NumPy Inference Engine...")

    # Test individual components
    print("\n1. Testing Conv2d...")
    conv = NumpyConv2d(np.random.randn(32, 8, 3, 3).astype(np.float32),
                       None, padding=1)
    x = np.random.randn(1, 8, 11, 11).astype(np.float32)
    y = conv(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (1, 32, 11, 11), "Conv2d shape mismatch"

    print("\n2. Testing BatchNorm2d...")
    bn = NumpyBatchNorm2d(
        np.ones(32, dtype=np.float32),
        np.zeros(32, dtype=np.float32),
        np.zeros(32, dtype=np.float32),
        np.ones(32, dtype=np.float32)
    )
    y_bn = bn(y)
    print(f"   Input: {y.shape} -> Output: {y_bn.shape}")
    assert y_bn.shape == y.shape, "BatchNorm shape mismatch"

    print("\n3. Testing Linear...")
    linear = NumpyLinear(
        np.random.randn(121, 256).astype(np.float32),
        np.zeros(121, dtype=np.float32)
    )
    z = np.random.randn(1, 256).astype(np.float32)
    z_out = linear(z)
    print(f"   Input: {z.shape} -> Output: {z_out.shape}")
    assert z_out.shape == (1, 121), "Linear shape mismatch"

    print("\n4. Testing activations...")
    test_input = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    print(f"   ReLU: {relu(test_input)}")
    print(f"   Softmax sum: {softmax(test_input).sum():.6f}")
    print(f"   Tanh range: [{tanh(test_input).min():.3f}, {tanh(test_input).max():.3f}]")

    print("\nNumPy Inference Engine tests passed!")


if __name__ == '__main__':
    test_numpy_inference()
