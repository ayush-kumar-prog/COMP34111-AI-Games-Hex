#!/usr/bin/env python3
"""
Test script for HexHexAgent.
Run on CSF3 after git pull:
    cd ~/scratch/COMP34111-AI-Games-Hex
    git pull origin simulation
    source venv/bin/activate
    python3 test_hexhex.py
"""

import sys
import torch

# Test 1: Inspect checkpoint structure
print("=" * 60)
print("TEST 1: Inspect checkpoint structure")
print("=" * 60)

model_path = "agents/Group12/models/hexhex_11x11.pt"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
print(f"Checkpoint keys: {checkpoint.keys()}")

if "config" in checkpoint:
    config = checkpoint["config"]
    print(f"Config type: {type(config)}")
    print(f"Board size: {config.getint('DEFAULT', 'board_size')}")
    print(f"Layers: {config.getint('DEFAULT', 'layers')}")
    print(f"Channels: {config.getint('DEFAULT', 'intermediate_channels')}")
    print(f"Reach: {config.getint('DEFAULT', 'reach')}")
    print(f"Switch model: {config.getboolean('DEFAULT', 'switch_model')}")
    print(f"Rotation model: {config.getboolean('DEFAULT', 'rotation_model')}")
else:
    print("WARNING: No config in checkpoint, using defaults")

# Test 2: Load agent
print("\n" + "=" * 60)
print("TEST 2: Load HexHexAgent")
print("=" * 60)

sys.path.insert(0, ".")
from agents.Group12.HexHexAgent import HexHexAgent
from src.Colour import Colour
from src.Board import Board
from src.Move import Move

agent_red = HexHexAgent(Colour.RED)
print(f"RED agent loaded successfully!")
print(f"Board size: {agent_red.board_size}")
print(f"Device: {agent_red.device}")

agent_blue = HexHexAgent(Colour.BLUE)
print(f"BLUE agent loaded successfully!")

# Test 3: Make a move on empty board
print("\n" + "=" * 60)
print("TEST 3: Make move on empty board")
print("=" * 60)

board = Board(11)
move = agent_red.make_move(1, board, None)
print(f"RED's opening move: ({move.x}, {move.y})")

# Test 4: BLUE's response
print("\n" + "=" * 60)
print("TEST 4: BLUE's response (may swap)")
print("=" * 60)

board.set_tile_colour(move.x, move.y, Colour.RED)
blue_move = agent_blue.make_move(2, board, move)
if blue_move.x == -1:
    print("BLUE decides to SWAP!")
else:
    print(f"BLUE plays: ({blue_move.x}, {blue_move.y})")

# Test 5: Run a quick game
print("\n" + "=" * 60)
print("TEST 5: Quick self-play game")
print("=" * 60)

board = Board(11)
agent_red = HexHexAgent(Colour.RED)
agent_blue = HexHexAgent(Colour.BLUE)

turn = 1
last_move = None
max_turns = 121

while turn <= max_turns:
    if turn % 2 == 1:  # RED's turn
        current_agent = agent_red
        current_color = Colour.RED
    else:  # BLUE's turn
        current_agent = agent_blue
        current_color = Colour.BLUE

    move = current_agent.make_move(turn, board, last_move)

    if move.x == -1 and move.y == -1:  # Swap
        print(f"Turn {turn}: {current_color} SWAPS")
        # Handle swap logic (simplified)
        last_move = move
        turn += 1
        continue

    print(f"Turn {turn}: {current_color} plays ({move.x}, {move.y})")
    board.set_tile_colour(move.x, move.y, current_color)

    # Check if game over (simplified - just count empty spaces)
    empty_count = sum(1 for i in range(11) for j in range(11) if board._tiles[i][j].colour is None)
    if empty_count == 0:
        print("Board full!")
        break

    last_move = move
    turn += 1

    if turn > 20:  # Stop after 20 moves for quick test
        print(f"... (stopping after 20 moves)")
        break

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
