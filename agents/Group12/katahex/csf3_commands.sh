#!/bin/bash
# Copy-paste these commands on CSF3 to set up KataHex
# Run each section one at a time

# === STEP 1: Load modules ===
module load tools/gcc/9.3.0
module load libs/eigen/3.4.0
module load tools/cmake/3.18.0

# === STEP 2: Navigate and update ===
cd ~/scratch/COMP34111-AI-Games-Hex
git pull origin simulation
cd agents/Group12/katahex

# === STEP 3: Clone KataHex source ===
git clone --branch Hex2024 --depth 1 https://github.com/selinger/katahex.git katahex-src

# === STEP 4: Build KataHex ===
cd katahex-src
mkdir -p build && cd build
cmake ../cpp \
    -DUSE_BACKEND=EIGEN \
    -DMAX_BOARD_LEN=19 \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_AVX2=1 \
    -DNO_GIT_REVISION=1
make -j8
cp katahex ../../
cd ../..

# === STEP 5: Download model ===
wget -O hex27x3.bin.gz https://github.com/hzyhhzy/KataGo/releases/download/Hex_20240812/hex27x3.bin.gz

# === STEP 6: Test GTP ===
echo "Testing KataHex..."
echo -e "boardsize 11\nclear_board\ngenmove black\nquit" | \
    ./katahex gtp -config config.cfg -model hex27x3.bin.gz

# === STEP 7: Test agent ===
cd ~/scratch/COMP34111-AI-Games-Hex
source venv/bin/activate
python3 -c "
from agents.Group12.KataHexAgent import KataHexAgent
from src.Colour import Colour
agent = KataHexAgent(Colour.RED)
print(f'Using fallback: {agent._use_fallback}')
"

# === STEP 8: Run tournament test (20 games) ===
echo "Running 20-game tournament..."
python3 << 'TOURNAMENT_EOF'
import subprocess
import sys

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

    try:
        result = subprocess.run(
            ["python3", "Hex.py", "-p1", p1, "-p2", p2],
            capture_output=True, text=True, timeout=600
        )
        output = result.stderr + result.stdout

        if "RED wins" in output or "Player 1 wins" in output:
            winner = "p1"
        elif "BLUE wins" in output or "Player 2 wins" in output:
            winner = "p2"
        else:
            winner = "unknown"
            print(f"Game {i+1}: Unknown result")
            print(output[-500:])
            continue

        if p1_is_kata:
            if winner == "p1":
                kata_wins += 1
            else:
                azalea_wins += 1
        else:
            if winner == "p2":
                kata_wins += 1
            else:
                azalea_wins += 1

        print(f"Game {i+1}/20: KataHex={kata_wins}, Azalea={azalea_wins}")

    except subprocess.TimeoutExpired:
        print(f"Game {i+1}: Timeout!")
    except Exception as e:
        print(f"Game {i+1}: Error - {e}")

print(f"\n=== FINAL RESULTS ===")
print(f"KataHex: {kata_wins} wins")
print(f"Azalea:  {azalea_wins} wins")
print(f"Win rate: {kata_wins/(kata_wins+azalea_wins)*100:.1f}%" if kata_wins+azalea_wins > 0 else "N/A")
TOURNAMENT_EOF

echo "Done!"
