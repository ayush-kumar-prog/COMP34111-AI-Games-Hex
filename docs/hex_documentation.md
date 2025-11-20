# Hex Game Engine Documentation

## 1. Submission checklist
This is provided to help you check you have included everything in your submission. Make sure you test your submission and read this documentation carefully. Pay particular attention to the Docker instructions to ensure that your code will run in the test environment.

1. Correctly formatted cmd.txt file specifying your Python agent
2. A Python class implementing `AgentBase` - either your own implementation or the `ExternalAgent`
3. Any other files required to run your agent in the Docker container, such as extra Python files, your agent implementation in an alternative programming language, and compiled files if relevant (e.g. .class files for Java). *We will not compile your code, you should do this before submission in the specified Docker environment and include both the source code and the compiled files in your submission.*

## 2. General Information
- The rules for Hex can be found [here](https://en.wikipedia.org/wiki/Hex_(board_game)).
- The board is 11x11 and it is represented by a 0-indexed two-dimensional matrix with rows that fall diagonally to the right.
- Position (x, y) has neighbours: left (x−1, y), right (x+1, y), above (x, y−1), below (x, y+1) and diagonally (x−1, y+1) and (x+1, y−1).

### Hex Board Neighbor Visualization:
```
    0   1   2   3   4
a(0) 0   0   N   N   0
b(1)  0   N   X   N   0
c(2)   0   N   N   0   0
```

- Red aims to connect top and bottom sides, while Blue connects horizontally. Red moves first.
- The pie rule is implemented
- Each agent has a maximum total of 5 minutes per match (3 minutes in code implementation)
- Your agent's performance will be evaluated by a combination of win rate (75%) and move speed (25%).

### Possible outcomes:
- **Win** -- one agent has connected their sides of the board
- **Timeout** – one agent has used up all the time allocated to it (3 minutes). The opposing agent wins.
- **Illegal move** – one agent tries to occupy an already occupied tile, play a move outside the board or does not return a `Move` object. The opposing agent wins.

## 3. Implementation

### Python agent
An agent template is provided at `agents/DefaultAgents/NaiveAgent.py`. The class has two methods: `__init__` and `make_move`.

When the game starts, the engine will call the `__init__` method in your Python agent, which will tell you which colour your agent will be playing as and allow you to perform any setup steps.

When it is your turn to move, the game engine will call the `make_move` method, which should always return a `Move` object. The method receives:
- `turn`: The current turn number
- `board`: A `Board` object specifying the current game state
- `opp_move`: A `Move` object specifying the opponent's last move (or None for first move)

The `Move` object can be used to make:
- **Normal move**: `Move(x, y)` where x and y are the coordinates
- **Swap move**: `Move(-1, -1)` only valid on turn 2

### Alternative Languages
The provided `agents/DefaultAgents/ExternalAgent.py` class can launch an external process in any language that communicates via stdin/stdout. See the `NaiveAgent.java` example.

#### Message Format
- **Commands**: `START`, `SWAP`, or `CHANGE`
- **Format**: `COMMAND;MOVE;BOARD;TURN;\n`
- **Response**: `x,y\n` where x,y are coordinates (or -1,-1 for swap)

Examples:
- `START;;000,000,000;1;\n` - Opening move request
- `SWAP;;0R0,000,000;3;\n` - Opponent swapped
- `CHANGE;0,1;000,R00,000;2;\n` - Opponent played at (0,1)

## 4. Running the Game

### From Terminal
```bash
python3 Hex.py                    # Two default agents
python3 Hex.py -p1 "agents.Group0.MyAgent MyGoodAgent"  # Custom player 1
python3 Hex.py --help              # See all options
```

### Arguments:
- `-p1/--player1`: Specify player 1 agent (format: "agents.GroupX.File Class")
- `-p2/--player2`: Specify player 2 agent
- `-p1Name/--player1Name`: Player 1 name
- `-p2Name/--player2Name`: Player 2 name
- `-v/--verbose`: Enable verbose logging
- `-b/--board_size`: Board size (default 11)
- `-l/--log`: Save moves to log file

### Docker Environment
Build:
```bash
docker build --build-arg UID=$UID -t hex .
```

Run:
```bash
docker run --cpus=8 --memory=8G -v $(pwd):/home/hex --name hex --rm -it hex /bin/bash
```

GPU support:
```bash
docker run --runtime=nvidia --cpus=8 --memory=8G -v $(pwd):/home/hex --name hex --rm -it hex /bin/bash
```

## 5. Log Format
CSV format with moves and timing:
```csv
1,Alice,RED(x=7, y=4),4125
2,Bob,REDSWAP(),1584
...
Bob,11376
Alice,12126
winner,Alice,BAD_MOVE
```

## 6. cmd.txt Format
**Critical**: The cmd.txt file must contain ONLY the agent path and class name:
```
agents.Group12.Group12Agent Group12Agent
```

No additional text, messages, or instructions should be included.