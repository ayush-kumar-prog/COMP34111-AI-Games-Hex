#!/bin/bash
#SBATCH --job-name=hex_v5_expert
#SBATCH --partition=gpu
#SBATCH -G a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=v5_expert_%j.out
#SBATCH --error=v5_expert_%j.err

# V5 Expert Iteration Training - GPU 1 (PRIMARY)
# Phase 1: Supervised pre-training (2-4 hours)
# Phase 2: Self-play RL with 800 MCTS sims (3.5 days)
# Expected total runtime: ~4 days

echo "=================================================="
echo "V5 EXPERT ITERATION TRAINING (GPU 1 - PRIMARY)"
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

# Check data exists
if [ ! -f "data/expert_games_v5.npz" ]; then
    echo "ERROR: Expert data not found at data/expert_games_v5.npz"
    echo "Please run data generation first: sbatch submit_v5_data_gen.sh"
    exit 1
fi
echo ""
echo "Expert data found:"
ls -lh data/expert_games_v5.npz

# Phase 1: Supervised Pre-training
echo ""
echo "=================================================="
echo "PHASE 1: SUPERVISED PRE-TRAINING"
echo "=================================================="
echo "Start: $(date)"
echo ""

python -m agents.Group12.training.train_v5_expert \
    --phase supervised \
    --data data/expert_games_v5.npz \
    --epochs 50 \
    --batch-size 512 \
    --output models/v5_supervised.pth

echo ""
echo "Phase 1 complete: $(date)"

# Phase 2: Self-play RL
echo ""
echo "=================================================="
echo "PHASE 2: SELF-PLAY REINFORCEMENT LEARNING"
echo "=================================================="
echo "Start: $(date)"
echo "Config: 100 iterations, 1000 games/iter, 800 MCTS sims"
echo ""

python -m agents.Group12.training.train_v5_expert \
    --phase selfplay \
    --resume models/v5_supervised_best.pth \
    --iterations 100 \
    --games-per-iter 1000 \
    --mcts-sims 800 \
    --epochs-per-iter 10 \
    --workers 64 \
    --batch-size 512 \
    --output models/v5_expert_final.pth

# Final status
echo ""
echo "=================================================="
echo "V5 EXPERT ITERATION TRAINING COMPLETE"
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
    cp models/v5_best.pth agents/Group12/models/hex_v5_best.pth
    echo "Done: agents/Group12/models/hex_v5_best.pth"
fi
