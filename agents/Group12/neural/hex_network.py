"""
AlphaZero-style neural network for Hex.
Uses ResNet architecture with policy and value heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HexResBlock(nn.Module):
    """
    Residual block for Hex neural network.
    Uses batch normalization and ReLU activations.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Output tensor with same shape as input
        """
        residual = x

        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)

        # Add residual connection
        out += residual
        out = F.relu(out)

        return out


class HexNeuralNetwork(nn.Module):
    """
    AlphaZero-style neural network for Hex.

    Architecture:
    - Input: 11×11×5 tensor (board state with 5 feature channels)
    - Conv tower with ResNet blocks
    - Policy head: Outputs move probabilities (121 outputs)
    - Value head: Outputs position evaluation (-1 to 1)

    Input channels:
    0: Our stones (1 for our stone, 0 otherwise)
    1: Opponent stones (1 for opponent stone, 0 otherwise)
    2: Empty cells (1 for empty, 0 otherwise)
    3: Legal moves mask (1 for legal move, 0 otherwise)
    4: Normalized edge distances (continuous 0-1)
    """

    def __init__(self, board_size: int = 11, num_res_blocks: int = 10, num_channels: int = 256):
        """
        Initialize Hex neural network.

        Args:
            board_size: Size of Hex board (default 11×11)
            num_res_blocks: Number of residual blocks (default 10)
            num_channels: Number of channels in residual tower (default 256)
        """
        super().__init__()
        self.board_size = board_size
        self.num_channels = num_channels

        # Input processing: 5 input channels → num_channels
        self.conv_input = nn.Conv2d(5, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            HexResBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy head: Predicts move probabilities
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, board_size * board_size)

        # Value head: Predicts position evaluation
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (batch_size, 5, board_size, board_size)

        Returns:
            policy_logits: Log probabilities for each move (batch_size, board_size²)
            value: Position evaluation in [-1, 1] (batch_size, 1)
        """
        # Input convolution
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)

        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        policy_logits = F.log_softmax(policy, dim=1)

        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)  # Flatten
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)

        return policy_logits, value

    def predict(self, board_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Predict policy and value for a single board state.

        Args:
            board_state: Board state tensor (5, board_size, board_size)

        Returns:
            policy: Move probabilities (board_size²,)
            value: Position evaluation in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            if board_state.dim() == 3:
                board_state = board_state.unsqueeze(0)

            policy_logits, value = self.forward(board_state)

            # Convert to probabilities and extract single values
            policy = torch.exp(policy_logits[0])
            value = value[0].item()

        return policy, value

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, filepath: str):
        """Save model weights."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str, device='cuda'):
        """Load model weights."""
        self.load_state_dict(torch.load(filepath, map_location=device))


def create_hex_network(board_size: int = 11,
                       num_res_blocks: int = 10,
                       num_channels: int = 256,
                       device: str = 'cuda') -> HexNeuralNetwork:
    """
    Factory function to create and initialize a Hex neural network.

    Args:
        board_size: Size of Hex board
        num_res_blocks: Number of residual blocks
        num_channels: Number of channels in conv layers
        device: Device to place network on ('cuda' or 'cpu')

    Returns:
        Initialized HexNeuralNetwork on specified device
    """
    network = HexNeuralNetwork(board_size, num_res_blocks, num_channels)
    network = network.to(device)

    print(f"Created Hex Neural Network:")
    print(f"  Board size: {board_size}×{board_size}")
    print(f"  Residual blocks: {num_res_blocks}")
    print(f"  Channels: {num_channels}")
    print(f"  Parameters: {network.get_num_parameters():,}")
    print(f"  Device: {device}")

    return network


if __name__ == '__main__':
    # Test network creation and forward pass
    print("Testing Hex Neural Network...")

    # Create network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = create_hex_network(device=device)

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 5, 11, 11).to(device)

    # Forward pass
    policy_logits, value = network(dummy_input)

    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Policy shape: {policy_logits.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")

    # Test single prediction
    single_state = torch.randn(5, 11, 11).to(device)
    policy, value = network.predict(single_state)
    print(f"\nSingle prediction test:")
    print(f"  Policy sum: {policy.sum().item():.3f} (should be ~1.0)")
    print(f"  Value: {value:.3f} (should be in [-1, 1])")

    print("\n✅ Network tests passed!")
