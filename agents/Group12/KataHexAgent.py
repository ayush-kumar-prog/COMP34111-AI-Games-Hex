"""
KataHexAgent - Wrapper for KataHex, the strongest available Hex engine.

KataHex is based on KataGo, an AlphaZero-style engine trained for Hex.
It communicates via GTP (Go Text Protocol) subprocess.

If KataHex binary is not available, falls back to AzaleaAgent.

Setup Instructions:
1. SSH to CSF3: ssh r36859ak@csf3.itservices.manchester.ac.uk
2. cd ~/scratch/COMP34111-AI-Games-Hex/agents/Group12/katahex
3. chmod +x setup_katahex.sh && ./setup_katahex.sh
4. The script will compile KataHex and download the model

References:
- https://github.com/selinger/katahex
- https://www.hexwiki.net/index.php/KataHex
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import Optional, List, Tuple

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# Directory where this agent file is located
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to KataHex binary, model, and config
KATAHEX_BINARY = os.path.join(AGENT_DIR, "katahex", "katahex")
KATAHEX_MODEL = os.path.join(AGENT_DIR, "katahex", "hex27x3.bin.gz")
KATAHEX_CONFIG = os.path.join(AGENT_DIR, "katahex", "config.cfg")

# Try to import AzaleaAgent for fallback
_FALLBACK_AGENT = None
try:
    from agents.Group12.AzaleaAgent import AzaleaAgent
    _FALLBACK_AGENT = AzaleaAgent
except ImportError:
    pass


class KataHexAgent(AgentBase):
    """
    Agent using KataHex engine for move selection via GTP protocol.

    KataHex is approximately 300 Elo stronger than other available Hex engines
    and can defeat top-level human players.

    GTP Protocol for Hex:
    - boardsize <n>: Set board size to n×n
    - clear_board: Clear the board
    - play <color> <vertex>: Play a move (color: black/white, vertex: a1-k11)
    - genmove <color>: Generate and play the best move
    - swap-pieces: Swap colors (for swap rule)

    Coordinate mapping:
    - Our Board: (row, col) where row=0 is TOP, col=0 is LEFT
    - GTP Hex: column letter (a-k) + row number (1-11)
    - a1 is top-left corner

    Color mapping:
    - RED (connects TOP-BOTTOM) = BLACK in GTP
    - BLUE (connects LEFT-RIGHT) = WHITE in GTP
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.board_size = 11
        self.process: Optional[subprocess.Popen] = None
        self.move_history: List[Tuple[str, str]] = []  # [(gtp_color, gtp_coord), ...]
        self._lock = threading.Lock()
        self._initialized = False
        self._fallback_agent = None
        self._use_fallback = False

        # Start KataHex subprocess
        if not self._start_engine():
            self._setup_fallback()

    def _setup_fallback(self) -> None:
        """Set up fallback agent when KataHex isn't available."""
        self._use_fallback = True
        if _FALLBACK_AGENT is not None:
            print("[KataHex] Engine not available, using AzaleaAgent as fallback")
            self._fallback_agent = _FALLBACK_AGENT(self.colour)
        else:
            print("[KataHex] WARNING: Engine not available and no fallback agent!")

    def _start_engine(self) -> bool:
        """Start the KataHex GTP subprocess."""
        if not os.path.exists(KATAHEX_BINARY):
            print(f"[KataHex] ERROR: Binary not found at {KATAHEX_BINARY}")
            return False

        if not os.path.exists(KATAHEX_MODEL):
            print(f"[KataHex] ERROR: Model not found at {KATAHEX_MODEL}")
            return False

        if not os.path.exists(KATAHEX_CONFIG):
            print(f"[KataHex] ERROR: Config not found at {KATAHEX_CONFIG}")
            return False

        try:
            cmd = [
                KATAHEX_BINARY,
                "gtp",
                "-config", KATAHEX_CONFIG,
                "-model", KATAHEX_MODEL
            ]

            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )

            # Initialize board
            self._send_command(f"boardsize {self.board_size}")
            self._send_command("clear_board")

            self._initialized = True
            print(f"[KataHex] Engine started successfully")
            return True

        except Exception as e:
            print(f"[KataHex] ERROR starting engine: {e}")
            return False

    def _send_command(self, cmd: str, timeout: float = 30.0) -> str:
        """
        Send a GTP command and get the response.

        GTP Response format:
        - Success: "= <response>\n\n"
        - Error: "? <error message>\n\n"
        """
        if not self.process or self.process.poll() is not None:
            print(f"[KataHex] Process not running, cannot send: {cmd}")
            return ""

        with self._lock:
            try:
                # Send command
                self.process.stdin.write(cmd + "\n")
                self.process.stdin.flush()

                # Read response until we get an empty line
                response_lines = []
                while True:
                    line = self.process.stdout.readline()
                    if line == "\n" or line == "":
                        break
                    response_lines.append(line.rstrip("\n"))

                response = "\n".join(response_lines)

                # Parse response
                if response.startswith("="):
                    return response[1:].strip()
                elif response.startswith("?"):
                    print(f"[KataHex] GTP error for '{cmd}': {response[1:].strip()}")
                    return ""
                else:
                    return response.strip()

            except Exception as e:
                print(f"[KataHex] Error sending command '{cmd}': {e}")
                return ""

    def _coord_to_gtp(self, row: int, col: int) -> str:
        """
        Convert our (row, col) to GTP vertex string.

        Our coordinates: (row, col) where (0,0) is top-left
        GTP coordinates: letter (a-k for columns) + number (1-11 for rows)

        Note: In standard Hex GTP, row 1 is at the top.
        """
        # Column: 0 -> 'a', 1 -> 'b', ..., 10 -> 'k'
        col_letter = chr(ord('a') + col)
        # Row: 0 -> 1, 1 -> 2, ..., 10 -> 11
        row_number = row + 1
        return f"{col_letter}{row_number}"

    def _gtp_to_coord(self, vertex: str) -> Tuple[int, int]:
        """
        Convert GTP vertex string to our (row, col).

        Handles formats like "a1", "k11", "D5", etc.
        """
        vertex = vertex.lower().strip()

        # Handle special cases
        if vertex in ("pass", "resign"):
            return (-1, -1)
        if vertex == "swap-pieces":
            return (-1, -1)  # Swap

        # Parse column letter and row number
        col_letter = vertex[0]
        row_str = vertex[1:]

        col = ord(col_letter) - ord('a')
        row = int(row_str) - 1

        return (row, col)

    def _colour_to_gtp(self, colour: Colour) -> str:
        """
        Convert our Colour to GTP color string.

        RED (top-bottom) = black
        BLUE (left-right) = white
        """
        return "black" if colour == Colour.RED else "white"

    def _sync_board(self, board: Board, opp_move: Optional[Move]) -> None:
        """
        Sync the board state with KataHex.

        Strategy: Track move history and replay if needed, or just play the opponent's move.
        """
        if opp_move is not None and not opp_move.is_swap():
            # Play opponent's move
            opp_gtp_color = self._colour_to_gtp(self.opp_colour())
            opp_gtp_coord = self._coord_to_gtp(opp_move.x, opp_move.y)

            response = self._send_command(f"play {opp_gtp_color} {opp_gtp_coord}")
            self.move_history.append((opp_gtp_color, opp_gtp_coord))

        elif opp_move is not None and opp_move.is_swap():
            # Handle swap - opponent chose to swap colors
            # In GTP, this is "swap-pieces" or we need to resync
            self._send_command("swap-pieces")
            self.move_history.append(("swap", "swap-pieces"))

    def _should_swap(self, board: Board, opp_move: Move) -> bool:
        """
        Decide whether to swap based on opponent's opening.

        KataHex should be able to make this decision, but as a fallback,
        we use heuristics for strong central openings.
        """
        if opp_move is None:
            return False

        # Strong central openings are worth swapping
        cx, cy = self.board_size // 2, self.board_size // 2
        dist = abs(opp_move.x - cx) + abs(opp_move.y - cy)

        # Swap if opponent played close to center
        return dist <= 2

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """Select the best move using KataHex engine."""
        self.board_size = board.size

        # Use fallback agent if KataHex isn't available
        if self._use_fallback:
            if self._fallback_agent is not None:
                return self._fallback_agent.make_move(turn, board, opp_move)
            else:
                return self._fallback_move(board)

        # If engine not running, try to start it
        if not self._initialized:
            if not self._start_engine():
                self._setup_fallback()
                if self._fallback_agent is not None:
                    return self._fallback_agent.make_move(turn, board, opp_move)
                return self._fallback_move(board)

        # Handle swap decision for BLUE on turn 2
        if (turn == 2 and self.colour == Colour.BLUE and
            opp_move is not None and not opp_move.is_swap()):
            if self._should_swap(board, opp_move):
                return Move(-1, -1)

        # Sync opponent's move with engine
        self._sync_board(board, opp_move)

        # Generate our move
        my_gtp_color = self._colour_to_gtp(self.colour)
        response = self._send_command(f"genmove {my_gtp_color}", timeout=60.0)

        if not response:
            print("[KataHex] No response from genmove, using fallback")
            return self._fallback_move(board)

        # Parse response
        response = response.strip().lower()

        if response == "swap-pieces":
            return Move(-1, -1)

        if response in ("resign", "pass"):
            # Shouldn't happen in Hex, but handle it
            return self._fallback_move(board)

        try:
            row, col = self._gtp_to_coord(response)
            if 0 <= row < self.board_size and 0 <= col < self.board_size:
                # Verify move is legal
                if board.tiles[row][col].colour is None:
                    self.move_history.append((my_gtp_color, response))
                    return Move(row, col)
                else:
                    print(f"[KataHex] Illegal move returned: {response} -> ({row}, {col})")
                    return self._fallback_move(board)
            else:
                print(f"[KataHex] Out of bounds: {response} -> ({row}, {col})")
                return self._fallback_move(board)
        except Exception as e:
            print(f"[KataHex] Error parsing move '{response}': {e}")
            return self._fallback_move(board)

    def _fallback_move(self, board: Board) -> Move:
        """Return the first legal move as fallback."""
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    return Move(i, j)
        return Move(0, 0)

    def __del__(self):
        """Clean up subprocess on deletion."""
        if self.process and self.process.poll() is None:
            try:
                self._send_command("quit")
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except:
                try:
                    self.process.kill()
                except:
                    pass


# For tournament compatibility
KataHexAgentClass = KataHexAgent
