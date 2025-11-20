# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Hex game framework for COMP34111 AI and Games coursework. The codebase provides a complete game engine for running Hex matches between AI agents, including tournament support with multiprocessing. All agents must run within a Docker container with strict resource constraints (8 CPUs, 8GB memory, 3 minutes per player).

## Development Commands

### Docker Setup

Build the Docker image:
```bash
docker build --build-arg UID=$UID -t hex .
```

Run the container:
```bash
docker run --cpus=8 --memory=8G -v "$(pwd)":/home/hex --name hex --rm -it hex /bin/bash
```

For GPU access (CUDA 12.3.0, TensorFlow 2.19.0, PyTorch 2.5.1+cu121):
```bash
docker run --runtime=nvidia --cpus=8 --memory=8G -v "$(pwd)":/home/hex --name hex --rm -it hex /bin/bash
```

Re-enter an existing container:
```bash
docker start -i hex
```

### Running Games

Run a single game with default agents:
```bash
python3 Hex.py
```

Run a game with custom agents:
```bash
python3 Hex.py -p1 "agents.GroupX.AgentFile AgentClassName" -p1Name "PlayerName" -p2 "agents.GroupY.AgentFile AgentClassName" -p2Name "PlayerName"
```

Additional options:
- `-v` or `--verbose`: Enable verbose logging
- `-b` or `--board_size`: Specify board size (default: 11)
- `-l` or `--log`: Save moves to log file (default: game.log if flag present)

Run the test suite:
```bash
python3 -m unittest discover
```

Run tournament (all agents play each other):
```bash
python3 HexTournament.py
```

Run partial tournament (specific groups vs all others):
```bash
python3 HexTournament.py -p path/to/group_numbers.txt
```

## Architecture

### Game Engine Core

**Game Flow (src/Game.py:105-182)**
- The `Game` class orchestrates matches with turn-based execution
- Each turn: deepcopy board → call agent's `make_move()` → validate move → apply move → check win condition
- Anti-cheat: Asserts board/turn/player state unchanged after agent move (src/Game.py:152-156)
- Time tracking: Per-player cumulative move time with 3-minute limit (MAXIMUM_TIME = 3 * 60 * 10^9 nanoseconds)
- Win conditions: WIN (connected path), BAD_MOVE (illegal move), TIMEOUT (exceeded time limit)

**Board Representation (src/Board.py)**
- 2D array of `Tile` objects with `Colour` (RED/BLUE/None)
- RED connects top-to-bottom (vertical), BLUE connects left-to-right (horizontal)
- Win detection: DFS traversal from starting edges to ending edges (src/Board.py:90-118)
- Tiles track visited status during DFS, cleared after each check

**Moves (src/Move.py)**
- Regular move: `Move(x, y)` where x,y are board coordinates
- Swap move (pie rule): `Move(-1, -1)` - only valid on turn 2
- Swap effect: Players exchange colours and agents (src/Game.py:188-203)

### Agent System

**Agent Contract (src/AgentBase.py)**
- All agents must inherit from `AgentBase` abstract class
- Required implementation: `make_move(turn: int, board: Board, opp_move: Move | None) -> Move`
- Agent receives a deepcopy of board - modifications don't affect game state
- First player receives `opp_move=None`, subsequent turns receive opponent's last move
- After swap, the game engine updates agent's `colour` property automatically

**Agent Types**
- **NaiveAgent**: Random valid moves, always swaps on turn 2 (example implementation)
- **ExternalAgent**: Subprocess-based agents (e.g., Java) via stdin/stdout protocol
  - Protocol: `START;;board;turn;` or `CHANGE;x,y;board;turn;` or `SWAP;;board;turn;`
  - Response: `x,y` coordinates
- Custom agents: Place in `agents/GroupX/` with corresponding `cmd.txt` file

**Agent Loading for Tournaments**
- Each agent directory (`agents/GroupX/`) must contain `cmd.txt`
- Format: `agents.GroupX.FileName ClassName` (single line)
- Tournament discovers agents via glob pattern `agents/Group*/cmd.txt`
- Group number extracted from path and validated against agent module path

### Tournament System (HexTournament.py)

**Execution**
- Uses multiprocessing.Pool for parallel game execution
- Each game runs in separate process with 6-minute timeout (2x player time limit)
- Results written incrementally to CSV: `game_results_{timestamp}.csv`
- Errors logged to: `error_game_list_{timestamp}.log`
- Individual game logs: `all_game_logs_{timestamp}/{player1}_vs_{player2}.log`

**Statistics Export**
- Aggregates: wins, win rate, average move time, loss types (illegal/timeout/regular)
- Output: `game_stat_{timestamp}.csv`

## Key Constraints & Validation

**Time Limits**
- Per-player total: 3 minutes (cumulative across all moves)
- Tournament timeout: 6 minutes per game
- Time measured in nanoseconds via `perf_counter_ns()`

**Move Validation (src/Game.py:273-291)**
- Must be `Move` object (exact type check, not subclass)
- Coordinates in bounds `[0, board_size)` and target tile empty
- OR swap move `(-1, -1)` only on turn 2

**Anti-Cheat Mechanisms**
- Board/turn/player state immutability checks via deepcopy comparison
- Move time must be positive (end > start)
- Agent hash based on source code (`inspect.getsource()`)

## File Structure

- `Hex.py`: Single-game runner with argparse CLI
- `HexTournament.py`: Multi-game tournament orchestrator
- `src/`: Core game engine
  - `Game.py`: Main game loop and validation
  - `Board.py`: Board state and win detection
  - `AgentBase.py`: Abstract agent interface
  - `Move.py`, `Colour.py`, `Tile.py`, `Player.py`, `EndState.py`: Data structures
- `agents/`: Agent implementations organized by group
  - `DefaultAgents/`: Reference implementations (NaiveAgent, ExternalAgent)
  - `GroupX/`: Student agent directories (must contain cmd.txt)
- `test/`: Unit tests (unittest framework)
