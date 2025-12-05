# KataHex Setup Guide

KataHex is the strongest available Hex engine (~300 Elo above alternatives).
This guide explains how to set it up on CSF3.

## Quick Setup on CSF3

```bash
# 1. SSH to CSF3
ssh r36859ak@csf3.itservices.manchester.ac.uk

# 2. Load required modules
module load tools/gcc/9.3.0
module load libs/eigen/3.4.0
module load tools/cmake/3.18.0

# 3. Navigate to project
cd ~/scratch/COMP34111-AI-Games-Hex

# 4. Pull latest code
git pull origin simulation

# 5. Run setup script
cd agents/Group12/katahex
chmod +x setup_katahex.sh
./setup_katahex.sh
```

## Manual Setup Steps

If the setup script fails, follow these steps:

### Step 1: Clone KataHex

```bash
cd agents/Group12/katahex
git clone --branch Hex2024 --depth 1 https://github.com/selinger/katahex.git katahex-src
```

### Step 2: Build KataHex

```bash
cd katahex-src
mkdir build && cd build
cmake ../cpp -DUSE_BACKEND=EIGEN -DMAX_BOARD_LEN=19 -DCMAKE_BUILD_TYPE=Release
make -j4
cp katahex ../../
```

### Step 3: Download Model

```bash
cd agents/Group12/katahex
wget https://github.com/hzyhhzy/KataGo/releases/download/Hex_20240812/hex27x3.bin.gz
```

### Step 4: Test

```bash
echo -e "boardsize 11\nclear_board\ngenmove black\nquit" | \
    ./katahex gtp -config config.cfg -model hex27x3.bin.gz
```

## Testing KataHexAgent

```bash
cd ~/scratch/COMP34111-AI-Games-Hex
source venv/bin/activate

# Test single game
python3 Hex.py -p1 "agents.Group12.KataHexAgent KataHexAgent" \
               -p2 "agents.Group12.AzaleaAgent AzaleaAgent"
```

## Tournament Test (20 games)

```bash
python3 -c '
import subprocess
kata_wins = 0
azalea_wins = 0
for i in range(20):
    if i % 2 == 0:
        p1 = "agents.Group12.KataHexAgent KataHexAgent"
        p2 = "agents.Group12.AzaleaAgent AzaleaAgent"
        p1_is_kata = True
    else:
        p1 = "agents.Group12.AzaleaAgent AzaleaAgent"
        p2 = "agents.Group12.KataHexAgent KataHexAgent"
        p1_is_kata = False

    result = subprocess.run(
        ["python3", "Hex.py", "-p1", p1, "-p2", p2],
        capture_output=True, text=True, timeout=600
    )

    # Parse result from stderr
    output = result.stderr
    if "RED wins" in output:
        winner = "p1"
    elif "BLUE wins" in output:
        winner = "p2"
    else:
        winner = "unknown"

    if p1_is_kata:
        if winner == "p1": kata_wins += 1
        else: azalea_wins += 1
    else:
        if winner == "p2": kata_wins += 1
        else: azalea_wins += 1

    print(f"Game {i+1}: KataHex={kata_wins}, Azalea={azalea_wins}")

print(f"\nFinal: KataHex {kata_wins} - Azalea {azalea_wins}")
'
```

## Files

- `katahex` - Compiled binary (built from source)
- `hex27x3.bin.gz` - Neural network weights (~100MB)
- `config.cfg` - Configuration for tournament play
- `setup_katahex.sh` - Automated setup script

## Troubleshooting

### "command not found: cmake"
Load the cmake module: `module load tools/cmake/3.18.0`

### "Eigen not found"
Load eigen module: `module load libs/eigen/3.4.0`

### "GLIBC not found" (on tournament Docker)
The binary needs to be statically linked. Rebuild with:
```bash
cmake ../cpp -DUSE_BACKEND=EIGEN -DCMAKE_EXE_LINKER_FLAGS="-static"
```

### GTP timeout
Increase timeout in agent or reduce maxVisits in config.cfg.

## Alternative: Use Fallback

If KataHex won't work in the tournament environment, the agent
automatically falls back to AzaleaAgent (which achieves 100% win
rate against MCTS agents).

## Resources

- [KataHex on HexWiki](https://www.hexwiki.net/index.php/KataHex)
- [selinger/katahex GitHub](https://github.com/selinger/katahex)
- [HZY's KataGo releases](https://github.com/hzyhhzy/KataGo/releases)
