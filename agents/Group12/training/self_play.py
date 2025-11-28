"""
Self-play infrastructure for generating training data.
Neural agent plays against itself to create high-quality training examples.
"""

import torch
import numpy as np
import copy
import random
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime
import json
import pickle
from tqdm import tqdm
import multiprocessing as mp

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.Group12.neural.hex_network import HexNeuralNetwork
from agents.Group12.neural.board_encoder import BoardEncoder
from agents.Group12.algorithms.mcts_neural import NeuralMCTS


@dataclass
class TrainingExample:
    """
    Single training example from self-play.
    """
    state: np.ndarray  # (5, 11, 11) board state
    policy: np.ndarray  # (121,) visit count distribution
    value: float  # Game outcome (-1, 0, or 1)
    move_number: int  # Which move in the game


class SelfPlayWorker:
    """
    Worker that generates self-play games.
    """

    def __init__(self, network: HexNeuralNetwork, encoder: BoardEncoder,
                 device: str = 'cuda', num_simulations: int = 800):
        """
        Initialize self-play worker.

        Args:
            network: Neural network for move selection
            encoder: Board state encoder
            device: 'cuda' or 'cpu'
            num_simulations: MCTS simulations per move
        """
        self.network = network
        self.encoder = encoder
        self.device = device
        self.num_simulations = num_simulations

        # Move network to device
        self.network.to(device)

    def get_temperature(self, move_number: int) -> float:
        """
        Get temperature for move selection based on game phase.

        CRITICAL: Never drop below 0.5 to maintain exploration diversity.
        This prevents the policy entropy from collapsing to zero.

        Args:
            move_number: Current move number in the game

        Returns:
            Temperature value (higher = more exploration)
        """
        if move_number < 15:
            return 1.5  # High exploration in opening
        elif move_number < 30:
            return 1.0  # Moderate exploration in early midgame
        elif move_number < 50:
            return 0.7  # Reduced but still meaningful exploration
        else:
            return 0.5  # Minimum exploration (NEVER go below this!)

    def generate_game(self, temperature: float = 1.0,
                     temperature_threshold: int = 30) -> List[TrainingExample]:
        """
        Generate one self-play game.

        Args:
            temperature: Base temperature for move selection (now dynamically adjusted)
            temperature_threshold: Deprecated - using dynamic temperature instead

        Returns:
            List of training examples from the game
        """
        board = Board(11)
        examples = []
        move_number = 0

        current_colour = Colour.RED

        while not board.has_ended(Colour.RED) and not board.has_ended(Colour.BLUE):
            # Create MCTS for current position
            mcts = NeuralMCTS(self.network, self.encoder,
                            current_colour, self.device)

            # Run MCTS search - always add noise for exploration
            move, policy_target = mcts.search(board, self.num_simulations,
                                            add_noise=(move_number < 40))

            # Save training example (value will be filled in later)
            state = self.encoder.encode(board, current_colour, device='cpu')
            example = TrainingExample(
                state=state.numpy(),
                policy=policy_target,
                value=0.0,  # Placeholder
                move_number=move_number
            )
            examples.append(example)

            # Get dynamic temperature based on game phase
            current_temp = self.get_temperature(move_number)

            # ALWAYS sample with temperature (never pure argmax!)
            # Apply temperature to policy
            policy_temp = policy_target ** (1.0 / current_temp)

            # Safety check: handle zero/NaN sums (untrained networks)
            policy_sum = policy_temp.sum()
            if policy_sum > 1e-10 and not np.isnan(policy_sum):
                policy_temp /= policy_sum
            else:
                # Fallback to uniform distribution over legal moves
                legal_moves = []
                for i in range(11):
                    for j in range(11):
                        if board.tiles[i][j].colour is None:
                            legal_moves.append(i * 11 + j)
                policy_temp = np.zeros(121)
                for idx in legal_moves:
                    policy_temp[idx] = 1.0 / len(legal_moves)

            # Sample move from distribution
            flat_idx = np.random.choice(len(policy_temp), p=policy_temp)
            x, y = flat_idx // 11, flat_idx % 11

            # Verify it's legal
            if board.tiles[x][y].colour is None:
                move = Move(x, y)
            # else: use MCTS best move (already set)

            # Make move
            board.set_tile_colour(move.x, move.y, current_colour)

            # Switch players
            current_colour = Colour.opposite(current_colour)
            move_number += 1

        # Determine winner
        if board.has_ended(Colour.RED):
            winner = Colour.RED
            outcome = 1.0  # RED wins
        else:
            winner = Colour.BLUE
            outcome = -1.0  # BLUE wins (from RED's perspective)

        # Fill in game outcomes
        for i, example in enumerate(examples):
            # Value is from the perspective of the player who made the move
            if i % 2 == 0:  # RED's move
                example.value = outcome
            else:  # BLUE's move
                example.value = -outcome

        return examples

    def generate_games(self, num_games: int, verbose: bool = True) -> List[TrainingExample]:
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to generate
            verbose: Show progress bar

        Returns:
            List of all training examples
        """
        all_examples = []

        iterator = range(num_games)
        if verbose:
            iterator = tqdm(iterator, desc="Generating self-play games")

        for game_num in iterator:
            examples = self.generate_game()
            all_examples.extend(examples)

            if verbose and (game_num + 1) % 10 == 0:
                avg_length = len(all_examples) / (game_num + 1)
                print(f"  Games: {game_num + 1}, "
                      f"Examples: {len(all_examples)}, "
                      f"Avg game length: {avg_length:.1f}")

        return all_examples


class SelfPlayManager:
    """
    Manages parallel self-play game generation across multiple workers.
    """

    def __init__(self, network: HexNeuralNetwork, encoder: BoardEncoder,
                 device: str = 'cuda', num_workers: int = 1):
        """
        Initialize self-play manager.

        Args:
            network: Neural network
            encoder: Board encoder
            device: 'cuda' or 'cpu'
            num_workers: Number of parallel workers (1 for single GPU)
        """
        self.network = network
        self.encoder = encoder
        self.device = device
        self.num_workers = num_workers

    def generate_training_data(self, num_games: int,
                               num_simulations: int = 800,
                               output_dir: str = 'data/self_play') -> str:
        """
        Generate training data from self-play.

        Args:
            num_games: Total number of games to generate
            num_simulations: MCTS simulations per move
            output_dir: Directory to save data

        Returns:
            Path to saved data file
        """
        print(f"\n{'='*70}")
        print(f"SELF-PLAY DATA GENERATION")
        print(f"{'='*70}")
        print(f"Games: {num_games}")
        print(f"MCTS simulations: {num_simulations}")
        print(f"Workers: {self.num_workers}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")

        # Create worker
        worker = SelfPlayWorker(self.network, self.encoder,
                               self.device, num_simulations)

        # Generate games
        all_examples = worker.generate_games(num_games, verbose=True)

        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total examples: {len(all_examples)}")
        print(f"Average game length: {len(all_examples) / num_games:.1f} moves")
        print(f"{'='*70}\n")

        # Save data
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selfplay_{num_games}games_{timestamp}.pkl"
        filepath = output_path / filename

        with open(filepath, 'wb') as f:
            pickle.dump(all_examples, f)

        print(f"Data saved to: {filepath}")
        print(f"File size: {filepath.stat().st_size / 1024 / 1024:.1f} MB\n")

        # Save metadata
        metadata = {
            'num_games': num_games,
            'num_examples': len(all_examples),
            'num_simulations': num_simulations,
            'device': self.device,
            'timestamp': timestamp,
            'avg_game_length': len(all_examples) / num_games
        }

        meta_filepath = output_path / f"selfplay_{num_games}games_{timestamp}_meta.json"
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        return str(filepath)

    def load_training_data(self, filepath: str) -> List[TrainingExample]:
        """Load training data from file."""
        with open(filepath, 'rb') as f:
            examples = pickle.load(f)
        print(f"Loaded {len(examples)} training examples from {filepath}")
        return examples


def merge_training_data(filepaths: List[str], output_path: str):
    """
    Merge multiple training data files into one.

    Args:
        filepaths: List of data file paths
        output_path: Output file path
    """
    all_examples = []

    for filepath in filepaths:
        with open(filepath, 'rb') as f:
            examples = pickle.load(f)
            all_examples.extend(examples)
            print(f"Loaded {len(examples)} examples from {filepath}")

    print(f"\nTotal examples: {len(all_examples)}")

    with open(output_path, 'wb') as f:
        pickle.dump(all_examples, f)

    print(f"Merged data saved to: {output_path}")


def test_self_play():
    """Test self-play game generation."""
    print("Testing Self-Play Infrastructure...")

    from agents.Group12.neural.hex_network import create_hex_network

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create network (untrained, just for testing)
    network = create_hex_network(device=device)
    encoder = BoardEncoder()

    # Create self-play worker
    worker = SelfPlayWorker(network, encoder, device, num_simulations=100)

    print("\n✅ Generating test game...")
    examples = worker.generate_game()

    print(f"\n✅ Self-play test complete!")
    print(f"   Game length: {len(examples)} moves")
    print(f"   Example state shape: {examples[0].state.shape}")
    print(f"   Example policy shape: {examples[0].policy.shape}")
    print(f"   Example value: {examples[0].value}")
    print(f"   Final outcome: {'RED wins' if examples[-1].value > 0 else 'BLUE wins'}")


if __name__ == '__main__':
    test_self_play()
