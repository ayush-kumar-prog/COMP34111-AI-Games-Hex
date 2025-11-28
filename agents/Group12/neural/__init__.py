"""
Neural network components for GPU-accelerated Hex AI.

V1 Components (original):
- HexNeuralNetwork: 5-channel input, 10 residual blocks
- BoardEncoder: Basic 5-channel encoding

V2 Components (enhanced):
- HexNeuralNetworkV2: 8-channel input, 15 blocks, SE attention, dilated convs
- EnhancedBoardEncoder: Hex-specific features (edge distances, bridges)
- NumpyHexNetwork: Pure NumPy inference for tournament (no PyTorch)
"""

# V1 (original)
from .hex_network import HexNeuralNetwork, HexResBlock
from .board_encoder import BoardEncoder

# V2 (enhanced)
from .hex_network_v2 import HexNeuralNetworkV2, create_hex_network_v2
from .enhanced_board_encoder import EnhancedBoardEncoder

__all__ = [
    # V1
    'HexNeuralNetwork', 'HexResBlock', 'BoardEncoder',
    # V2
    'HexNeuralNetworkV2', 'create_hex_network_v2', 'EnhancedBoardEncoder',
]
