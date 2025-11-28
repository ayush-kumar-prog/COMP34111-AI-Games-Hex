#!/bin/bash
#SBATCH --job-name=hex_v5_10k
#SBATCH --partition=multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=v5_data_gen_10k_%j.out
#SBATCH --error=v5_data_gen_10k_%j.err

# V5 Expert Data Generation - 10K games (faster, safer)
# Expected runtime: 5-6 hours

echo "=================================================="
echo "V5 EXPERT DATA GENERATION (10K GAMES)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start: $(date)"
echo "=================================================="

cd ~/scratch/COMP34111-AI-Games-Hex || { echo "ERROR: Project directory not found!"; exit 1; }
echo "Working directory: $(pwd)"

mkdir -p data

echo ""
echo "========== ENVIRONMENT SETUP =========="

PYTHON_CMD=""

# Try Anaconda first (we know this works)
echo "Loading Anaconda..."
module load apps/binapps/anaconda3/2023.09 2>/dev/null
if command -v python &> /dev/null; then
    if python -c "import numpy" 2>/dev/null; then
        PYTHON_CMD="python"
        echo "Found working Python with numpy via Anaconda"
    fi
fi

# Fallback to system python + pip install
if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &> /dev/null; then
        python3 -m pip install --user numpy 2>&1 | tail -5
        if python3 -c "import numpy" 2>/dev/null; then
            PYTHON_CMD="python3"
        fi
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Could not find Python with numpy!"
    exit 1
fi

echo ""
echo "========== PYTHON ENVIRONMENT =========="
echo "Using: $PYTHON_CMD"
$PYTHON_CMD --version

echo ""
echo "========== STARTING DATA GENERATION =========="
echo "Config: 10000 games (reduced for reliability), 12 workers, 500 MCTS iterations"
echo "Start: $(date)"
echo ""

# Run with unbuffered output so we see progress
$PYTHON_CMD -u -m agents.Group12.training.expert_data_gen \
    --num-games 10000 \
    --workers 12 \
    --mcts-iters 500 \
    --output data/expert_games_v5_10k.npz

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "DATA GENERATION COMPLETE"
else
    echo "DATA GENERATION FAILED (exit code: $EXIT_CODE)"
fi
echo "=================================================="
echo "End: $(date)"
ls -lh data/expert_games_v5_10k.npz 2>/dev/null || echo "Output file not found!"
