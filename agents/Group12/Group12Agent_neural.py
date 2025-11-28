"""
Group12 Neural Hex AI Agent - V4 Championship Agent
Uses trained neural network (V2 architecture) + MCTS for superhuman Hex play.
Trained on 10,000 self-play games using AlphaZero-style training on A100 GPU.
"""

import torch
import numpy as np
from typing import Optional
from pathlib import Path

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.Group12.neural.hex_network_v2 import create_hex_network_v2
from agents.Group12.algorithms.mcts_neural import NeuralMCTS


class V4BoardEncoder:
    """
    8-channel board encoder matching V4 training.

    Channels:
    0: Our pieces
    1: Opponent pieces
    2: Empty cells
    3-4: Distance to our goal edges
    5-6: Distance to opponent goal edges
    7: Distance to center
    """

    def __init__(self, size: int = 11):
        self.size = size
        self._precompute_distances()

    def _precompute_distances(self):
        n = self.size
        self.dist_top = np.zeros((n, n), dtype=np.float32)
        self.dist_bottom = np.zeros((n, n), dtype=np.float32)
        self.dist_left = np.zeros((n, n), dtype=np.float32)
        self.dist_right = np.zeros((n, n), dtype=np.float32)
        self.dist_center = np.zeros((n, n), dtype=np.float32)

        center = n // 2
        for i in range(n):
            for j in range(n):
                self.dist_top[i, j] = i / (n - 1)
                self.dist_bottom[i, j] = (n - 1 - i) / (n - 1)
                self.dist_left[i, j] = j / (n - 1)
                self.dist_right[i, j] = (n - 1 - j) / (n - 1)
                self.dist_center[i, j] = 1.0 - (abs(i - center) + abs(j - center)) / (n - 1)

    def encode(self, board, colour, device='cpu') -> torch.Tensor:
        """Encode board state for neural network."""
        n = self.size
        state = np.zeros((8, n, n), dtype=np.float32)

        # Get colour as int (1=RED, 2=BLUE)
        if hasattr(colour, 'value'):
            colour_int = 1 if colour.value == 1 else 2
        else:
            colour_int = colour

        opp_int = 3 - colour_int

        # Extract board tiles
        for i in range(n):
            for j in range(n):
                tile = board.tiles[i][j]
                if tile.colour is not None:
                    if tile.colour.value == colour_int:
                        state[0, i, j] = 1.0
                    else:
                        state[1, i, j] = 1.0
                else:
                    state[2, i, j] = 1.0

        # Distance planes based on colour
        if colour_int == 1:  # RED (top-bottom)
            state[3] = self.dist_top
            state[4] = self.dist_bottom
            state[5] = self.dist_left
            state[6] = self.dist_right
        else:  # BLUE (left-right)
            state[3] = self.dist_left
            state[4] = self.dist_right
            state[5] = self.dist_top
            state[6] = self.dist_bottom

        state[7] = self.dist_center

        # Convert to tensor
        tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        return tensor

    def decode_policy(self, policy_tensor: torch.Tensor, board) -> dict:
        """
        Convert policy tensor to dictionary of (move -> probability).

        Args:
            policy_tensor: Tensor of shape (board_size²,) with move probabilities
            board: Current board state (to filter legal moves)

        Returns:
            Dictionary mapping (x, y) tuples to probabilities
        """
        policy_dict = {}
        n = self.size

        for i in range(n):
            for j in range(n):
                if board.tiles[i][j].colour is None:  # Legal move
                    idx = i * n + j
                    if hasattr(policy_tensor, 'item'):
                        prob = policy_tensor[idx].item()
                    else:
                        prob = float(policy_tensor[idx])
                    policy_dict[(i, j)] = prob

        # Normalize to sum to 1.0
        total = sum(policy_dict.values())
        if total > 0:
            policy_dict = {k: v/total for k, v in policy_dict.items()}

        return policy_dict

    def create_move_mask(self, board) -> torch.Tensor:
        """
        Create mask for legal moves.

        Args:
            board: Current board state

        Returns:
            Tensor of shape (board_size²,) with 1 for legal moves, 0 for illegal
        """
        n = self.size
        mask = torch.zeros(n * n)

        for i in range(n):
            for j in range(n):
                if board.tiles[i][j].colour is None:
                    idx = i * n + j
                    mask[idx] = 1.0

        return mask

    def encode_batch(self, boards: list, colours: list, device: str = 'cpu') -> torch.Tensor:
        """Encode multiple board states as a batch."""
        batch = []
        for board, colour in zip(boards, colours):
            state = self.encode(board, colour, device='cpu')
            batch.append(state.squeeze(0))  # Remove batch dimension

        batch_tensor = torch.stack(batch)
        return batch_tensor.to(device)


class Group12Agent(AgentBase):
    """
    Championship-level Hex AI using neural network + MCTS.

    Performance targets:
    - 10,000-50,000 MCTS iterations per move
    - 95%+ win rate vs heuristic agents
    - <2 seconds per move (GPU accelerated)
    - Top 1-3 tournament placement
    """

    # Time management
    TOTAL_TIME_LIMIT = 3 * 60 * 10**9  # 3 minutes in nanoseconds
    TIME_BUFFER = 0.05  # 5% safety buffer
    MAX_TIME_PER_MOVE = 2 * 10**9  # 2 seconds max per move

    # MCTS configuration (adjust based on hardware)
    # GPU: 800/1600/800, CPU: 100/200/100
    MCTS_SIMULATIONS_OPENING = 100  # Fewer sims for opening (faster)
    MCTS_SIMULATIONS_MIDGAME = 200  # More sims for critical positions
    MCTS_SIMULATIONS_ENDGAME = 100  # Fewer sims when game is decided

    # Nash equilibrium swap threshold
    SWAP_THRESHOLD = 0.52

    # V4 model architecture (trained on CSF3 A100)
    V4_NUM_RES_BLOCKS = 15
    V4_NUM_CHANNELS = 256

    def __init__(self, colour: Colour = None, model_path: str = None,
                 num_res_blocks: int = None, num_channels: int = None):
        """
        Initialize neural agent with V4 trained model.

        Args:
            colour: Our colour (can be set later by game engine)
            model_path: Path to trained model weights (default: V4 model)
            num_res_blocks: Network depth (default: 15 for V4)
            num_channels: Network width (default: 256 for V4)
        """
        super().__init__(colour)

        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Group12 Neural Agent V4 initialized on {self.device}")

        # Default to V4 model path
        if model_path is None:
            # Try multiple paths
            possible_paths = [
                Path(__file__).parent / 'models' / 'hex_model_numpy.npz',
                Path('agents/Group12/models/hex_model_numpy.npz'),
                Path('models/hex_model_numpy.npz'),
            ]
            for p in possible_paths:
                if p.exists():
                    model_path = str(p)
                    break
            else:
                model_path = str(possible_paths[0])  # Use first as default

        self.model_path = model_path

        # Use V4 architecture
        blocks = num_res_blocks or self.V4_NUM_RES_BLOCKS
        channels = num_channels or self.V4_NUM_CHANNELS

        # Create V2 network architecture
        self.network = create_hex_network_v2(
            num_res_blocks=blocks,
            num_channels=channels,
            device=self.device
        )

        # Load weights
        if Path(model_path).exists():
            self._load_v4_weights(model_path)
        else:
            print(f"WARNING: Model not found at {model_path}")
            print("Using untrained network!")

        self.network.eval()

        # Create V4 encoder (8 channels)
        self.encoder = V4BoardEncoder()

        # Game state tracking
        self.total_time_used = 0
        self.moves_made = 0

    def _load_v4_weights(self, model_path: str):
        """
        Load V4 model weights from NumPy .npz file.

        Args:
            model_path: Path to .npz weights file
        """
        print(f"Loading V4 weights from: {model_path}")

        # Load NumPy weights
        weights = np.load(model_path)
        state_dict = {}
        for key in weights.files:
            state_dict[key] = torch.from_numpy(weights[key])
        weights.close()

        # Load into network
        self.network.load_state_dict(state_dict)
        param_count = sum(p.numel() for p in self.network.parameters())
        print(f"V4 model loaded: {param_count:,} parameters")

    def _load_model_legacy(self, model_path: str, num_res_blocks: int = None, num_channels: int = None):
        """
        Legacy method for loading old .pth checkpoint files.
        Kept for backwards compatibility.
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Training checkpoint format
            state_dict = checkpoint['model_state_dict']
            print(f"Loading training checkpoint (epoch {checkpoint.get('epoch', '?')})")
        else:
            # Raw state dict
            state_dict = checkpoint

        # Auto-detect architecture from state dict
        if num_res_blocks is None or num_channels is None:
            # Try to infer from the state dict keys
            detected_blocks = 0
            detected_channels = None

            for key in state_dict.keys():
                # Detect number of residual blocks
                if key.startswith('res_blocks.'):
                    # Extract block number from key like 'res_blocks.5.conv1.weight'
                    try:
                        block_num = int(key.split('.')[1])
                        detected_blocks = max(detected_blocks, block_num + 1)
                    except (ValueError, IndexError):
                        pass

                # Detect channel count from first conv layer in res_blocks
                if detected_channels is None and 'res_blocks.0.conv1.weight' == key:
                    detected_channels = state_dict[key].shape[0]

            # Use detected values or sensible defaults
            blocks = num_res_blocks if num_res_blocks is not None else (detected_blocks if detected_blocks > 0 else 4)
            channels = num_channels if num_channels is not None else (detected_channels if detected_channels else 64)
            print(f"Auto-detected architecture: {blocks} blocks, {channels} channels")
        else:
            blocks = num_res_blocks
            channels = num_channels

        # Create network with detected/specified architecture
        self.network = HexNeuralNetwork(
            board_size=11,
            num_res_blocks=blocks,
            num_channels=channels
        )

        # Load weights
        self.network.load_state_dict(state_dict)
        print(f"Loaded model from: {model_path}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")

    def load_model(self, model_path: str):
        """Public method to load a different model."""
        self._load_model(model_path)
        self.network.to(self.device)
        self.network.eval()

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """
        Make a move using neural network + MCTS.

        Args:
            turn: Current turn number
            board: Current board state
            opp_move: Opponent's last move

        Returns:
            Best move found
        """
        from time import perf_counter_ns as time
        start_time = time()

        # Handle swap decision (turn 2 only)
        if turn == 2 and opp_move is not None:
            if self._should_swap_neural(board, opp_move):
                return Move(-1, -1)

        # Determine number of simulations based on game phase
        empty_cells = self._count_empty_cells(board)
        total_cells = board.size * board.size

        if turn <= 5:
            # Opening: Faster moves
            num_simulations = self.MCTS_SIMULATIONS_OPENING
        elif empty_cells < 30:
            # Endgame: Moderate simulations
            num_simulations = self.MCTS_SIMULATIONS_ENDGAME
        else:
            # Midgame: Maximum simulations
            num_simulations = self.MCTS_SIMULATIONS_MIDGAME

        # Create MCTS
        mcts = NeuralMCTS(self.network, self.encoder, self.colour, self.device)

        # Search with neural network guidance
        move, policy_target = mcts.search(board, num_simulations=num_simulations)

        # Update time tracking
        time_used = time() - start_time
        self.total_time_used += time_used
        self.moves_made += 1

        return move

    def _should_swap_neural(self, board: Board, opp_move: Move) -> bool:
        """
        Use neural network to evaluate swap decision.

        Args:
            board: Current board (after opponent's first move)
            opp_move: Opponent's opening move

        Returns:
            True if should swap, False otherwise
        """
        # Encode position
        state = self.encoder.encode(board, self.colour, device=self.device)

        # Get neural network evaluation
        with torch.no_grad():
            _, value = self.network.predict(state)

        # Swap if opponent's position is too strong
        # value > 0 means good for us (opponent played weak move)
        # value < 0 means bad for us (opponent played strong move) → swap!

        # Also consider geometric position (fallback to heuristic)
        x, y = opp_move.x, opp_move.y
        center = board.size // 2
        dist_from_center = abs(x - center) + abs(y - center)

        # Combine neural evaluation with geometric heuristic
        neural_swap = value < -0.3  # Neural network says position is strong for opponent
        geometric_swap = dist_from_center <= 1  # Center or adjacent (traditionally strong)

        # Swap if either condition is met
        return neural_swap or geometric_swap

    def _count_empty_cells(self, board: Board) -> int:
        """Count empty cells on board."""
        count = 0
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    count += 1
        return count


if __name__ == '__main__':
    print("Group12 Neural Agent - Championship Level")
    print("=" * 70)

    # Test initialization
    try:
        agent = Group12Agent(Colour.RED, model_path='models/hex_model_best.pth')
        print("✅ Agent initialized successfully")
        print(f"   Device: {agent.device}")
        print(f"   Network parameters: {agent.network.get_num_parameters():,}")
        print(f"   MCTS simulations: {agent.MCTS_SIMULATIONS_MIDGAME}")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")

    print("=" * 70)
