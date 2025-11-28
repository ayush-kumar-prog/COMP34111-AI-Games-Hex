"""
Enhanced AlphaZero-style neural network for Hex (Version 2).

Improvements over v1:
- 8 input channels (enhanced features)
- 15 residual blocks (deeper)
- Optional dilated convolutions for larger receptive field
- Squeeze-and-excitation blocks for channel attention
- Data augmentation support (180-degree rotation symmetry)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Learns to weight channels based on global context.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        # Global average pooling
        y = self.avg_pool(x).view(batch, channels)
        # Channel attention weights
        y = self.fc(y).view(batch, channels, 1, 1)
        return x * y.expand_as(x)


class HexResBlockV2(nn.Module):
    """
    Enhanced residual block with optional SE attention.
    """

    def __init__(self, num_channels: int, use_se: bool = True,
                 dilation: int = 1):
        super().__init__()

        # Adjust padding for dilation
        padding = dilation

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

        # Squeeze-and-excitation
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        out = out + residual
        out = F.relu(out, inplace=True)

        return out


class HexNeuralNetworkV2(nn.Module):
    """
    Enhanced AlphaZero-style neural network for Hex.

    Architecture:
    - Input: 11x11x8 tensor (enhanced board encoding)
    - Initial convolution
    - 15 residual blocks with SE attention
    - Some blocks use dilated convolutions for larger receptive field
    - Policy head: 121 outputs (11x11 board)
    - Value head: Single scalar in [-1, 1]

    Input channels (8 total):
    0: Our stones
    1: Opponent stones
    2: Empty cells
    3: Our distance to start edge
    4: Our distance to end edge
    5: Opponent distance to start edge
    6: Opponent distance to end edge
    7: Bridge pattern indicators
    """

    def __init__(self, board_size: int = 11, num_res_blocks: int = 15,
                 num_channels: int = 256, input_channels: int = 8,
                 use_se: bool = True):
        """
        Initialize enhanced Hex neural network.

        Args:
            board_size: Size of Hex board (default 11x11)
            num_res_blocks: Number of residual blocks (default 15)
            num_channels: Channels in residual tower (default 256)
            input_channels: Number of input channels (default 8)
            use_se: Use squeeze-and-excitation blocks (default True)
        """
        super().__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.input_channels = input_channels

        # Input processing
        self.conv_input = nn.Conv2d(input_channels, num_channels,
                                    kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower with varying dilation
        # Use dilation > 1 for some blocks to increase receptive field
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            # Use dilation=2 for blocks 5, 10 to increase receptive field
            dilation = 2 if i in [5, 10] else 1
            self.res_blocks.append(
                HexResBlockV2(num_channels, use_se=use_se, dilation=dilation)
            )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size,
                                   board_size * board_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (batch_size, 8, board_size, board_size)

        Returns:
            policy_logits: Log probabilities (batch_size, board_size^2)
            value: Position evaluation in [-1, 1] (batch_size, 1)
        """
        # Input convolution
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x, inplace=True)

        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy, inplace=True)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy_logits = F.log_softmax(policy, dim=1)

        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value, inplace=True)
        value = value.view(value.size(0), -1)
        value = self.value_fc1(value)
        value = F.relu(value, inplace=True)
        value = self.value_fc2(value)
        value = torch.tanh(value)

        return policy_logits, value

    def predict(self, board_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Predict policy and value for a single board state.

        Args:
            board_state: Board state tensor (8, board_size, board_size)

        Returns:
            policy: Move probabilities (board_size^2,)
            value: Position evaluation in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            if board_state.dim() == 3:
                board_state = board_state.unsqueeze(0)

            policy_logits, value = self.forward(board_state)
            policy = torch.exp(policy_logits[0])
            value = value[0].item()

        return policy, value

    def predict_batch(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict policy and value for a batch of states.

        Args:
            states: Batch of states (batch_size, 8, board_size, board_size)

        Returns:
            policies: Move probabilities (batch_size, board_size^2)
            values: Position evaluations (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            policy_logits, values = self.forward(states)
            policies = torch.exp(policy_logits)
            values = values.squeeze(-1)

        return policies, values

    def augment_for_training(self, state: torch.Tensor,
                            policy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation for training.

        Hex has 180-degree rotational symmetry.

        Args:
            state: Board state (C, H, W)
            policy: Policy vector (H*W,)

        Returns:
            Augmented state and policy
        """
        if torch.rand(1).item() > 0.5:
            # Apply 180-degree rotation
            state = torch.rot90(state, k=2, dims=[1, 2])
            policy_2d = policy.view(self.board_size, self.board_size)
            policy_2d = torch.rot90(policy_2d, k=2, dims=[0, 1])
            policy = policy_2d.view(-1)

        return state, policy

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, filepath: str):
        """Save model weights."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str, device='cuda'):
        """Load model weights."""
        self.load_state_dict(torch.load(filepath, map_location=device))

    def export_weights_numpy(self) -> dict:
        """
        Export all weights as numpy arrays for NumPy inference.

        Returns:
            Dictionary of layer_name -> numpy array
        """
        weights = {}
        for name, param in self.state_dict().items():
            weights[name] = param.cpu().numpy()
        return weights


def create_hex_network_v2(board_size: int = 11,
                          num_res_blocks: int = 15,
                          num_channels: int = 256,
                          device: str = 'cuda') -> HexNeuralNetworkV2:
    """
    Factory function to create enhanced Hex neural network.

    Args:
        board_size: Size of Hex board
        num_res_blocks: Number of residual blocks
        num_channels: Number of channels
        device: Device to place network on

    Returns:
        Initialized HexNeuralNetworkV2
    """
    network = HexNeuralNetworkV2(board_size, num_res_blocks, num_channels)
    network = network.to(device)

    print(f"Created Enhanced Hex Neural Network V2:")
    print(f"  Board size: {board_size}x{board_size}")
    print(f"  Input channels: 8")
    print(f"  Residual blocks: {num_res_blocks}")
    print(f"  Channels: {num_channels}")
    print(f"  Parameters: {network.get_num_parameters():,}")
    print(f"  Device: {device}")
    print(f"  Features: SE attention, dilated convolutions")

    return network


if __name__ == '__main__':
    print("Testing Hex Neural Network V2...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = create_hex_network_v2(device=device)

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 8, 11, 11).to(device)

    policy_logits, value = network(dummy_input)

    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Policy shape: {policy_logits.shape}")
    print(f"  Value shape: {value.shape}")

    # Test single prediction
    single_state = torch.randn(8, 11, 11).to(device)
    policy, value = network.predict(single_state)
    print(f"\nSingle prediction:")
    print(f"  Policy sum: {policy.sum().item():.3f}")
    print(f"  Value: {value:.3f}")

    # Test weight export
    weights = network.export_weights_numpy()
    print(f"\nWeight export:")
    print(f"  Number of weight tensors: {len(weights)}")
    print(f"  First layer shape: {weights['conv_input.weight'].shape}")

    print("\nHex Neural Network V2 tests passed!")
