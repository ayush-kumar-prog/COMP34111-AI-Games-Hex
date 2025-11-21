"""
Group12 Neural Hex AI Agent - GPU-Accelerated Championship Agent
Uses trained neural network + MCTS for superhuman Hex play.
"""

import torch
from typing import Optional
from pathlib import Path

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.Group12.neural.hex_network import HexNeuralNetwork
from agents.Group12.neural.board_encoder import BoardEncoder
from agents.Group12.algorithms.mcts_neural import NeuralMCTS


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

    # MCTS configuration
    MCTS_SIMULATIONS_OPENING = 800  # Fewer sims for opening (faster)
    MCTS_SIMULATIONS_MIDGAME = 1600  # More sims for critical positions
    MCTS_SIMULATIONS_ENDGAME = 800  # Fewer sims when game is decided

    # Nash equilibrium swap threshold
    SWAP_THRESHOLD = 0.52

    def __init__(self, colour: Colour, model_path: str = 'models/hex_model_best.pth'):
        """
        Initialize neural agent.

        Args:
            colour: Our colour
            model_path: Path to trained model weights
        """
        super().__init__(colour)

        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Group12 Neural Agent initialized on {self.device}")

        # Load neural network
        self.network = HexNeuralNetwork(board_size=11, num_res_blocks=10, num_channels=256)

        if Path(model_path).exists():
            self.network.load(model_path, device=self.device)
            print(f"Loaded model from: {model_path}")
        else:
            print(f"WARNING: Model not found at {model_path}")
            print(f"Using untrained network (will play randomly)")

        self.network.to(self.device)
        self.network.eval()

        # Create encoder
        self.encoder = BoardEncoder()

        # Game state tracking
        self.total_time_used = 0
        self.moves_made = 0

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
