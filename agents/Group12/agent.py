import math
import random
import time
from copy import deepcopy
import sys
import os

# Add src to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from AgentBase import AgentBase
from Board import Board
from Colour import Colour
from Move import Move

class MCTSNode:
    """Node in the Monte Carlo Tree Search"""
    def __init__(self, board, colour, move=None, parent=None):
        self.board = board
        self.colour = colour
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = self.get_legal_moves()
        
    def get_legal_moves(self):
        """Get all legal moves for current board state"""
        moves = []
        for x in range(11):
            for y in range(11):
                if self.board.tiles[x][y].colour == Colour.EMPTY:
                    moves.append((x, y))
        random.shuffle(moves)  # Randomize to avoid bias
        return moves
    
    def is_terminal(self):
        """Check if the game is over"""
        return self.board.has_ended(self.colour) or self.board.has_ended(self.colour.opposite())
    
    def is_fully_expanded(self):
        """Check if all children have been expanded"""
        return len(self.untried_moves) == 0
    
    def best_child(self, c_param=1.41):
        """Select best child using UCB1 formula"""
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float('inf')
            else:
                exploit = child.wins / child.visits
                explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
                weight = exploit + explore
            choices_weights.append(weight)
        return self.children[choices_weights.index(max(choices_weights))]
    
    def expand(self):
        """Expand a random untried move"""
        move = self.untried_moves.pop()
        new_board = deepcopy(self.board)
        new_board.set_tile_colour(move[0], move[1], self.colour)
        child_node = MCTSNode(new_board, self.colour.opposite(), move, self)
        self.children.append(child_node)
        return child_node
    
    def rollout(self):
        """Simulate a random game from this state"""
        current_board = deepcopy(self.board)
        current_colour = self.colour
        
        # Quick simulation with random moves
        moves_made = 0
        max_moves = 121  # 11x11 board
        
        while not current_board.has_ended(Colour.RED) and not current_board.has_ended(Colour.BLUE):
            if moves_made >= max_moves:
                break
            
            # Get legal moves
            legal_moves = []
            for x in range(11):
                for y in range(11):
                    if current_board.tiles[x][y].colour == Colour.EMPTY:
                        legal_moves.append((x, y))
            
            if not legal_moves:
                break
            
            # Make random move
            move = random.choice(legal_moves)
            current_board.set_tile_colour(move[0], move[1], current_colour)
            current_colour = current_colour.opposite()
            moves_made += 1
        
        # Determine winner
        if current_board.has_ended(self.colour):
            return 1  # Win
        elif current_board.has_ended(self.colour.opposite()):
            return 0  # Loss
        else:
            return 0.5  # Draw (shouldn't happen in Hex)
    
    def backpropagate(self, result):
        """Backpropagate the result up the tree"""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1 - result)


class MCTSAgent(AgentBase):
    """Monte Carlo Tree Search agent for Hex"""
    
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.colour = colour
        self.opponent_colour = colour.opposite()
        self.time_limit = 4.5  # Leave buffer before 5 minute timeout
        self.total_time_used = 0
        
    def make_move(self, turn: int, board: Board, opp_move: Move) -> Move:
        """Make a move using MCTS"""
        start_time = time.time()
        
        # Handle swap rule for second move
        if turn == 2 and self.colour == Colour.BLUE:
            # Consider swapping if opponent's move is good
            if opp_move and self.should_swap(opp_move):
                return Move(-1, -1)
        
        # Use MCTS to find best move
        best_move = self.mcts_search(board, time_limit=self.get_time_for_move(turn))
        
        elapsed = time.time() - start_time
        self.total_time_used += elapsed
        
        return Move(best_move[0], best_move[1])
    
    def get_time_for_move(self, turn):
        """Allocate time per move based on game stage"""
        if turn <= 10:
            return 2.0  # More time for opening
        elif turn <= 50:
            return 1.5  # Mid game
        else:
            return 1.0  # End game - faster moves
    
    def should_swap(self, opp_move):
        """Decide whether to swap based on opponent's opening move"""
        # Center positions are generally strong
        x, y = opp_move.x, opp_move.y
        center = 5
        distance_from_center = abs(x - center) + abs(y - center)
        
        # Swap if move is close to center (within 3 tiles)
        return distance_from_center <= 3
    
    def mcts_search(self, board, time_limit=2.0):
        """Perform MCTS search"""
        root = MCTSNode(deepcopy(board), self.colour)
        end_time = time.time() + time_limit
        iterations = 0
        
        while time.time() < end_time:
            node = root
            
            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()
            
            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # Simulation
            result = node.rollout()
            
            # Backpropagation
            node.backpropagate(result)
            
            iterations += 1
        
        # Choose move with most visits (most robust)
        if not root.children:
            # Fallback to random move if no children expanded
            legal_moves = root.get_legal_moves()
            return legal_moves[0] if legal_moves else (5, 5)
        
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move