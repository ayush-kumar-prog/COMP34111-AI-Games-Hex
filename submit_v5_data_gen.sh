#!/bin/bash
#SBATCH --job-name=hex_v5_data
#SBATCH --partition=multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=v5_data_gen_%j.out
#SBATCH --error=v5_data_gen_%j.err

# V5 Expert Data Generation - CPU-only job
# Generates 20,000 games using MCTS+RAVE (500 iterations)
# Expected runtime: 8-12 hours

echo "=================================================="
echo "V5 EXPERT DATA GENERATION"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start: $(date)"
echo "=================================================="

# Change to project directory
cd ~/scratch/COMP34111-AI-Games-Hex || { echo "ERROR: Project directory not found!"; exit 1; }
echo "Working directory: $(pwd)"

# Create data directory
mkdir -p data

echo ""
echo "========== ENVIRONMENT SETUP =========="

# Try to find a working Python with numpy
PYTHON_CMD=""

# Option 1: Try loading python/3.13.1 module and check if it works
echo "Trying module python/3.13.1..."
module load python/3.13.1 2>/dev/null
if command -v python3 &> /dev/null; then
    if python3 -c "import numpy" 2>/dev/null; then
        PYTHON_CMD="python3"
        echo "Found working Python with numpy via module"
    fi
fi

# Option 2: Try Anaconda if available
if [ -z "$PYTHON_CMD" ]; then
    echo "Trying Anaconda..."
    module load apps/binapps/anaconda3/2023.09 2>/dev/null || module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null
    if command -v python &> /dev/null; then
        if python -c "import numpy" 2>/dev/null; then
            PYTHON_CMD="python"
            echo "Found working Python with numpy via Anaconda"
        fi
    fi
fi

# Option 3: Check system Python and install numpy if needed
if [ -z "$PYTHON_CMD" ]; then
    echo "Checking system Python..."
    if command -v python3 &> /dev/null; then
        echo "System Python3 found: $(python3 --version)"
        if python3 -c "import numpy" 2>/dev/null; then
            PYTHON_CMD="python3"
            echo "NumPy already available"
        else
            echo "Installing numpy to user space..."
            python3 -m pip install --user numpy 2>&1 | tail -5
            if python3 -c "import numpy" 2>/dev/null; then
                PYTHON_CMD="python3"
                echo "NumPy installed successfully"
            fi
        fi
    fi
fi

# Option 4: Last resort - try 'python' command
if [ -z "$PYTHON_CMD" ]; then
    echo "Trying 'python' command..."
    if command -v python &> /dev/null; then
        echo "Python found: $(python --version)"
        if python -c "import numpy" 2>/dev/null; then
            PYTHON_CMD="python"
        else
            echo "Installing numpy..."
            python -m pip install --user numpy 2>&1 | tail -5
            if python -c "import numpy" 2>/dev/null; then
                PYTHON_CMD="python"
            fi
        fi
    fi
fi

# Final check
if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Could not find a working Python with numpy!"
    echo ""
    echo "Available modules:"
    module avail 2>&1 | grep -i python | head -20
    echo ""
    echo "Python paths:"
    which python python3 2>/dev/null
    exit 1
fi

echo ""
echo "========== PYTHON ENVIRONMENT =========="
echo "Using: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""
echo "NumPy version:"
$PYTHON_CMD -c "import numpy; print(f'NumPy {numpy.__version__}')"
echo ""
echo "Python path:"
$PYTHON_CMD -c "import sys; print(sys.executable)"

# Verify our module can be imported
echo ""
echo "========== MODULE IMPORT TEST =========="
$PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
try:
    from agents.Group12.training.expert_data_gen import ExpertMCTS, V5BoardEncoder
    print('SUCCESS: expert_data_gen imports correctly')
except Exception as e:
    print(f'FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Module import failed. Check the error above."
    exit 1
fi

# Run data generation
echo ""
echo "========== STARTING DATA GENERATION =========="
echo "Config: 20000 games, 12 workers, 500 MCTS iterations"
echo "Start: $(date)"
echo ""

$PYTHON_CMD -m agents.Group12.training.expert_data_gen \
    --num-games 20000 \
    --workers 12 \
    --mcts-iters 500 \
    --output data/expert_games_v5.npz

EXIT_CODE=$?

# Check output
echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "DATA GENERATION COMPLETE"
else
    echo "DATA GENERATION FAILED (exit code: $EXIT_CODE)"
fi
echo "=================================================="
echo "End: $(date)"
ls -lh data/expert_games_v5.npz 2>/dev/null || echo "Output file not found!"
