"""
Expert Data Generator for V5 Training

Generates high-quality training data using the existing MCTS+RAVE tournament agent
through self-play. This creates supervised learning data for Phase 1 of Expert Iteration.

Key differences from V4:
- Uses full MCTS+RAVE (500 iterations) instead of neural-guided MCTS (30-100)
- CPU-only execution (no GPU needed for data generation)
- Multiprocessing for parallel game generation
- Outputs state, policy (visit distribution), and game outcome

Usage:
    python -m agents.Group12.training.expert_data_gen \
        --num-games 20000 \
        --workers 12 \
        --output data/expert_games_v5.npz
"""

import os
import sys
import math
import random
import copy
import argparse
import pickle
import time
import multiprocessing as mp
from pathlib import Path

# CRITICAL: Use 'spawn' instead of 'fork' to avoid deadlocks on Linux
# fork() can cause issues with certain libraries and shared state
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from time import perf_counter_ns

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


# ============================================================================
# Board Encoder (matches V2 network input format)
# ============================================================================

class V5BoardEncoder:
    """
    8-channel board encoder for V5 training.
    Matches the V2 network architecture input format.

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

    def encode(self, board: Board, colour: Colour) -> np.ndarray:
        """Encode board state to 8-channel numpy array."""
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

        return state


# ============================================================================
# Simplified MCTS Node for Data Generation
# ============================================================================

@dataclass
class ExpertMCTSNode:
    """Node in MCTS tree with RAVE statistics for expert data generation."""
    board: Board
    colour: Colour  # Colour to play next
    move: Optional[Move] = None
    parent: Optional['ExpertMCTSNode'] = None

    # Initialize as None to save memory, created on demand
    children: List['ExpertMCTSNode'] = None
    untried_moves: List[Move] = None

    # MCTS statistics
    visits: int = 0
    wins: float = 0.0

    # RAVE statistics (lazy initialization)
    amaf_visits: Dict[Tuple[int, int], int] = None
    amaf_wins: Dict[Tuple[int, int], float] = None

    is_terminal: bool = False
    winner: Optional[Colour] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.amaf_visits is None:
            self.amaf_visits = {}
        if self.amaf_wins is None:
            self.amaf_wins = {}


# ============================================================================
# Expert MCTS (Matches Tournament Agent Quality)
# ============================================================================

class ExpertMCTS:
    """
    MCTS with RAVE for expert data generation.
    This is a simplified version of TournamentMCTS optimized for data generation.
    """

    EXPLORATION = math.sqrt(2)
    RAVE_K = 300
    NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    def __init__(self, colour: Colour, max_iterations: int = 500):
        self.colour = colour
        self.max_iterations = max_iterations

    def search(self, board: Board) -> Tuple[Move, np.ndarray]:
        """
        Run MCTS search and return (best_move, policy_distribution).

        Returns:
            best_move: The selected move
            policy: Normalized visit counts as policy target (121,)
        """
        root = self._create_root(board)
        n = board.size

        # Special cases
        if not root.untried_moves and not root.children:
            policy = np.zeros(n * n, dtype=np.float32)
            return Move(5, 5), policy

        if len(root.untried_moves) == 1 and not root.children:
            move = root.untried_moves[0]
            policy = np.zeros(n * n, dtype=np.float32)
            policy[move.x * n + move.y] = 1.0
            return move, policy

        # Run MCTS iterations
        for _ in range(self.max_iterations):
            # Selection
            node = self._select(root)

            # Expansion
            if node.untried_moves and not node.is_terminal:
                node = self._expand(node)

            # Simulation
            result, moves_played = self._simulate(node)

            # Backpropagation
            self._backpropagate(node, result)

            # Update RAVE
            self._update_rave(node, moves_played, result)

        # Build policy from visit counts
        policy = np.zeros(n * n, dtype=np.float32)
        for child in root.children:
            if child.move:
                idx = child.move.x * n + child.move.y
                policy[idx] = child.visits

        # Normalize policy
        if policy.sum() > 0:
            policy = policy / policy.sum()

        # Select best move by visit count
        best_move = self._best_move(root)

        return best_move, policy

    def _create_root(self, board: Board) -> ExpertMCTSNode:
        root = ExpertMCTSNode(board=copy.deepcopy(board), colour=self.colour)
        root.untried_moves = self._get_legal_moves(board)
        self._check_terminal(root)
        return root

    def _select(self, node: ExpertMCTSNode) -> ExpertMCTSNode:
        while not node.is_terminal:
            if node.untried_moves:
                return node
            if not node.children:
                return node
            node = self._best_child(node)
        return node

    def _best_child(self, node: ExpertMCTSNode) -> ExpertMCTSNode:
        best_score = -float('inf')
        best = None

        for child in node.children:
            score = self._ucb1_rave(child, node)
            if score > best_score:
                best_score = score
                best = child

        return best if best else node.children[0]

    def _ucb1_rave(self, child: ExpertMCTSNode, parent: ExpertMCTSNode) -> float:
        if child.visits == 0:
            return float('inf')

        # UCB1 value
        q = child.wins / child.visits
        if child.colour != self.colour:
            q = 1 - q

        # RAVE value
        move_key = (child.move.x, child.move.y) if child.move else None
        if move_key and move_key in parent.amaf_visits and parent.amaf_visits[move_key] > 0:
            rave_q = parent.amaf_wins[move_key] / parent.amaf_visits[move_key]
            if child.colour != self.colour:
                rave_q = 1 - rave_q

            # Weight RAVE vs UCB1
            beta = math.sqrt(self.RAVE_K / (3 * parent.visits + self.RAVE_K))
            q = (1 - beta) * q + beta * rave_q

        # Exploration bonus
        explore = self.EXPLORATION * math.sqrt(math.log(parent.visits) / child.visits)

        return q + explore

    def _expand(self, node: ExpertMCTSNode) -> ExpertMCTSNode:
        # Select move (prioritize center, avoid corners)
        move = self._select_expansion_move(node)
        node.untried_moves.remove(move)

        # Create child
        child_board = copy.deepcopy(node.board)
        child_board.set_tile_colour(move.x, move.y, node.colour)

        child = ExpertMCTSNode(
            board=child_board,
            colour=Colour.opposite(node.colour),
            move=move,
            parent=node
        )
        child.untried_moves = self._get_legal_moves(child_board)
        self._check_terminal(child)

        node.children.append(child)
        return child

    def _select_expansion_move(self, node: ExpertMCTSNode) -> Move:
        """Select move for expansion, prioritizing center."""
        n = node.board.size
        center = n // 2

        scored = []
        for move in node.untried_moves:
            dist = abs(move.x - center) + abs(move.y - center)
            # Penalize corners
            if (move.x, move.y) in [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]:
                score = -100
            else:
                score = -dist
            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [m for _, m in scored[:3]]
        return random.choice(top)

    def _simulate(self, node: ExpertMCTSNode) -> Tuple[float, List[Tuple[int, int]]]:
        if node.is_terminal:
            return (1.0 if node.winner == self.colour else 0.0), []

        sim_board = copy.deepcopy(node.board)
        current = node.colour
        moves_played = []
        n = sim_board.size

        for _ in range(60):
            # Check win every 5 moves
            if len(moves_played) % 5 == 0:
                if sim_board.has_ended(Colour.RED) or sim_board.has_ended(Colour.BLUE):
                    break

            # Get legal moves
            legal = []
            for i in range(n):
                for j in range(n):
                    if sim_board.tiles[i][j].colour is None:
                        legal.append((i, j))

            if not legal:
                break

            # Simulation policy: 30% smart, 70% random
            if random.random() < 0.3:
                center = n // 2
                scored = []
                for x, y in legal:
                    if (x, y) in [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]:
                        score = -50
                    else:
                        score = -(abs(x - center) + abs(y - center))
                        if current == Colour.RED:
                            score += max(0, 5 - min(x, n-1-x))
                        else:
                            score += max(0, 5 - min(y, n-1-y))
                    scored.append((score, x, y))
                scored.sort(key=lambda t: t[0], reverse=True)
                x, y = scored[0][1], scored[0][2]
            else:
                x, y = random.choice(legal)

            sim_board.set_tile_colour(x, y, current)
            moves_played.append((x, y))
            current = Colour.opposite(current)

        # Determine result
        if sim_board.has_ended(self.colour):
            return 1.0, moves_played
        elif sim_board.has_ended(Colour.opposite(self.colour)):
            return 0.0, moves_played
        else:
            return 0.5, moves_played

    def _backpropagate(self, node: ExpertMCTSNode, result: float):
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent

    def _update_rave(self, node: ExpertMCTSNode, moves: List[Tuple[int, int]], result: float):
        current = node.parent
        colour = Colour.opposite(node.colour)

        while current:
            for mx, my in moves:
                if current.board.tiles[mx][my].colour is None:
                    key = (mx, my)
                    if key not in current.amaf_visits:
                        current.amaf_visits[key] = 0
                        current.amaf_wins[key] = 0.0

                    current.amaf_visits[key] += 1
                    adj_result = result if colour == self.colour else (1 - result)
                    current.amaf_wins[key] += adj_result

            current = current.parent
            colour = Colour.opposite(colour)

    def _best_move(self, root: ExpertMCTSNode) -> Move:
        if not root.children:
            return root.untried_moves[0] if root.untried_moves else Move(5, 5)
        best = max(root.children, key=lambda c: c.visits)
        return best.move

    def _get_legal_moves(self, board: Board) -> List[Move]:
        moves = []
        n = board.size
        for i in range(n):
            for j in range(n):
                if board.tiles[i][j].colour is None:
                    moves.append(Move(i, j))
        return moves

    def _check_terminal(self, node: ExpertMCTSNode):
        if node.board.has_ended(Colour.RED):
            node.is_terminal = True
            node.winner = Colour.RED
        elif node.board.has_ended(Colour.BLUE):
            node.is_terminal = True
            node.winner = Colour.BLUE


# ============================================================================
# Training Example
# ============================================================================

@dataclass
class TrainingExample:
    """A single training example for the neural network."""
    state: np.ndarray   # (8, 11, 11) board encoding
    policy: np.ndarray  # (121,) normalized visit distribution
    value: float        # {-1, +1} game outcome from this player's perspective


# ============================================================================
# Game Generation Worker
# ============================================================================

def generate_single_game(args: Tuple[int, int, int]) -> List[TrainingExample]:
    """
    Generate a single game through self-play.

    Args:
        args: (game_id, mcts_iterations, random_seed)

    Returns:
        List of TrainingExample from this game
    """
    game_id, mcts_iterations, seed = args
    random.seed(seed)
    np.random.seed(seed)

    encoder = V5BoardEncoder()
    examples = []

    # Initialize board
    board = Board(11)
    current_colour = Colour.RED
    move_num = 0

    while not board.has_ended(Colour.RED) and not board.has_ended(Colour.BLUE):
        # Check for moves
        empty_cells = sum(1 for i in range(11) for j in range(11)
                        if board.tiles[i][j].colour is None)
        if empty_cells == 0 or move_num > 150:
            break

        # Create MCTS for current player
        mcts = ExpertMCTS(current_colour, max_iterations=mcts_iterations)

        # Get move and policy
        move, policy = mcts.search(board)

        # Encode state before move
        state = encoder.encode(board, current_colour)

        # Store example (value will be filled in after game ends)
        examples.append(TrainingExample(
            state=state,
            policy=policy,
            value=0.0  # Placeholder
        ))

        # Make move
        board.set_tile_colour(move.x, move.y, current_colour)
        current_colour = Colour.opposite(current_colour)
        move_num += 1

    # Determine winner and fill in values
    if board.has_ended(Colour.RED):
        winner = Colour.RED
    elif board.has_ended(Colour.BLUE):
        winner = Colour.BLUE
    else:
        winner = None

    # Fill in values from each player's perspective
    for i, ex in enumerate(examples):
        player = Colour.RED if i % 2 == 0 else Colour.BLUE
        if winner is None:
            ex.value = 0.0  # Draw (rare in Hex)
        elif player == winner:
            ex.value = 1.0
        else:
            ex.value = -1.0

    return examples


def generate_games_parallel(num_games: int, num_workers: int,
                           mcts_iterations: int = 500,
                           checkpoint_interval: int = 100,
                           output_path: str = None) -> List[TrainingExample]:
    """
    Generate games in parallel using multiprocessing.

    IMPORTANT: Saves checkpoints every checkpoint_interval games to avoid
    losing work if the job crashes or times out.

    Args:
        num_games: Total number of games to generate
        num_workers: Number of parallel workers
        mcts_iterations: MCTS iterations per move
        checkpoint_interval: Save checkpoint every N games
        output_path: Output path (used for checkpoint naming)

    Returns:
        List of all training examples
    """
    print(f"Generating {num_games} expert games with {num_workers} workers...")
    print(f"MCTS iterations: {mcts_iterations}")
    print(f"Checkpointing every {checkpoint_interval} games")

    all_examples = []
    base_seed = int(time.time())

    # Create task list
    tasks = [(i, mcts_iterations, base_seed + i) for i in range(num_games)]

    start_time = time.time()
    games_done = 0
    last_checkpoint = 0

    # Checkpoint path
    if output_path:
        checkpoint_dir = os.path.dirname(output_path) or '.'
        checkpoint_base = os.path.splitext(os.path.basename(output_path))[0]
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_base}_checkpoint.npz")
    else:
        checkpoint_path = "data/checkpoint.npz"

    # Use multiprocessing pool
    with mp.Pool(processes=num_workers) as pool:
        # Use imap for progress tracking
        for examples in pool.imap_unordered(generate_single_game, tasks):
            all_examples.extend(examples)
            games_done += 1

            # Progress report every 100 games
            if games_done % 100 == 0 or games_done == num_games:
                elapsed = time.time() - start_time
                rate = games_done / elapsed if elapsed > 0 else 0
                eta = (num_games - games_done) / rate if rate > 0 else 0
                print(f"  Progress: {games_done}/{num_games} games "
                      f"({rate:.2f} games/s, ETA: {eta:.0f}s)")

            # CHECKPOINT: Save every checkpoint_interval games
            if games_done - last_checkpoint >= checkpoint_interval:
                print(f"  [CHECKPOINT] Saving {len(all_examples)} examples at game {games_done}...")
                try:
                    # Save checkpoint
                    states = np.array([ex.state for ex in all_examples])
                    policies = np.array([ex.policy for ex in all_examples])
                    values = np.array([ex.value for ex in all_examples])
                    np.savez_compressed(
                        checkpoint_path,
                        states=states,
                        policies=policies,
                        values=values,
                        games_done=games_done
                    )
                    ckpt_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
                    print(f"  [CHECKPOINT] Saved to {checkpoint_path} ({ckpt_size:.1f} MB)")
                    last_checkpoint = games_done
                except Exception as e:
                    print(f"  [CHECKPOINT] Warning: Failed to save checkpoint: {e}")

    elapsed = time.time() - start_time
    print(f"\nGenerated {len(all_examples)} examples from {num_games} games "
          f"in {elapsed/60:.1f} minutes")
    print(f"Average examples per game: {len(all_examples)/num_games:.1f}")

    return all_examples


def save_examples(examples: List[TrainingExample], output_path: str):
    """Save training examples to .npz file."""
    states = np.array([ex.state for ex in examples])
    policies = np.array([ex.policy for ex in examples])
    values = np.array([ex.value for ex in examples])

    np.savez_compressed(
        output_path,
        states=states,
        policies=policies,
        values=values
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {len(examples)} examples to {output_path} ({file_size:.1f} MB)")


def load_examples(input_path: str) -> List[TrainingExample]:
    """Load training examples from .npz file."""
    data = np.load(input_path)
    examples = []
    for i in range(len(data['states'])):
        examples.append(TrainingExample(
            state=data['states'][i],
            policy=data['policies'][i],
            value=data['values'][i]
        ))
    return examples


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate expert training data')
    parser.add_argument('--num-games', type=int, default=20000,
                       help='Number of games to generate')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--mcts-iters', type=int, default=500,
                       help='MCTS iterations per move')
    parser.add_argument('--output', type=str, default='data/expert_games_v5.npz',
                       help='Output file path')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='Save checkpoint every N games')

    args = parser.parse_args()

    # Default workers to CPU count
    if args.workers is None:
        args.workers = mp.cpu_count()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("EXPERT DATA GENERATION V5")
    print("="*70)
    print(f"Games: {args.num_games}")
    print(f"Workers: {args.workers}")
    print(f"MCTS iterations: {args.mcts_iters}")
    print(f"Output: {args.output}")
    print("="*70)

    # Generate games
    examples = generate_games_parallel(
        num_games=args.num_games,
        num_workers=args.workers,
        mcts_iterations=args.mcts_iters,
        checkpoint_interval=args.checkpoint_interval,
        output_path=args.output
    )

    # Save
    save_examples(examples, args.output)

    # Print statistics
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    values = [ex.value for ex in examples]
    print(f"Total examples: {len(examples)}")
    print(f"Win examples: {sum(1 for v in values if v > 0)}")
    print(f"Loss examples: {sum(1 for v in values if v < 0)}")
    print(f"Draw examples: {sum(1 for v in values if v == 0)}")


if __name__ == '__main__':
    main()
