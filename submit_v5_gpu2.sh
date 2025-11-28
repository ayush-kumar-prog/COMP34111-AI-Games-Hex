#!/bin/bash
#SBATCH --job-name=hex_v5_alphazero
#SBATCH --partition=gpu
#SBATCH -G a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=v5_alphazero_%j.out
#SBATCH --error=v5_alphazero_%j.err

# V5 Pure AlphaZero Training - GPU 2 (EXPERIMENTAL)
# Tests hypothesis: V4 failed due to low MCTS sims, not architecture
# Pure self-play from scratch with 800 MCTS sims
# Expected runtime: ~4 days

echo "=================================================="
echo "V5 PURE ALPHAZERO TRAINING (GPU 2 - EXPERIMENTAL)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start: $(date)"
echo "Working directory: $(pwd)"
echo "=================================================="

# Change to project directory FIRST
cd ~/scratch/COMP34111-AI-Games-Hex || { echo "ERROR: Project directory not found!"; exit 1; }

# Create directories
mkdir -p models

# Load modules
echo ""
echo "Loading modules..."
module load cuda/12.3 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || {
    echo "WARNING: Could not load CUDA module"
}
module load python/3.13.1 || {
    echo "ERROR: Could not load python/3.13.1"
    exit 1
}
echo "Modules loaded successfully"

# Install PyTorch if needed
echo ""
echo "Checking PyTorch..."
python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null || {
    echo "Installing PyTorch..."
    pip install --user torch==2.5.1
}

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Pure AlphaZero: Self-play from scratch
echo ""
echo "=================================================="
echo "PURE ALPHAZERO SELF-PLAY (FROM SCRATCH)"
echo "=================================================="
echo "Start: $(date)"
echo "Config: 150 iterations, 800 games/iter, 800 MCTS sims"
echo ""
echo "This tests if V4's failure was due to low MCTS sims (30-100)"
echo "With 800 sims, pure AlphaZero should work!"
echo ""

python -m agents.Group12.training.train_v5_expert \
    --phase selfplay \
    --iterations 150 \
    --games-per-iter 800 \
    --mcts-sims 800 \
    --epochs-per-iter 10 \
    --workers 64 \
    --batch-size 512 \
    --output models/v5_alphazero_final.pth

# Final status
echo ""
echo "=================================================="
echo "V5 PURE ALPHAZERO TRAINING COMPLETE"
echo "=================================================="
echo "End: $(date)"
echo ""
echo "Model files:"
ls -lh models/v5_*.pth 2>/dev/null
echo ""
echo "NumPy weights:"
ls -lh models/v5_*_numpy.npz 2>/dev/null

# Copy best model to agent directory
if [ -f "models/v5_best.pth" ]; then
    echo ""
    echo "Copying best model to agent directory..."
    mkdir -p agents/Group12/models
    cp models/v5_best.pth agents/Group12/models/hex_v5_alphazero.pth
    echo "Done: agents/Group12/models/hex_v5_alphazero.pth"
fi
