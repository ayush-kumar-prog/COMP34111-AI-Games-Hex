#!/bin/bash --login
#SBATCH --job-name=hex_v3
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH -t 2-0
#SBATCH --output=logs/hex_v3_%j.out
#SBATCH --error=logs/hex_v3_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r36859ak@student.manchester.ac.uk

# ============================================================================
# Group12 Hex AI V3 - PARALLEL Self-Play Training
# ============================================================================
#
# Key improvements over V2:
# - 8-10 parallel workers for self-play (10x faster!)
# - Reduced MCTS simulations (150 vs 400) but more iterations
# - FastBoard implementation avoiding Python deepcopy
# - Expected total time: 8-24 hours instead of 80+ hours
#
# Usage:
#   sbatch submit_csf_v3.sh              # Default: standard mode (8-12 hours)
#   sbatch submit_csf_v3.sh test         # Quick test (15 minutes)
#   sbatch submit_csf_v3.sh fast         # Fast (4-6 hours)
#   sbatch submit_csf_v3.sh standard     # Standard (8-12 hours)
#   sbatch submit_csf_v3.sh full         # Full (18-24 hours)
#   sbatch submit_csf_v3.sh max          # Maximum (36-48 hours)
#
# ============================================================================

set -e

echo "=============================================="
echo "GROUP12 HEX AI V3 - PARALLEL TRAINING"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start: $(date)"
echo "=============================================="

# Create logs directory
mkdir -p ~/logs logs

# Load modules
module purge
module load libs/cuda/12.4.1

# Setup Python environment
echo ""
echo "Setting up Python environment..."

if module avail apps/binapps/anaconda3 2>&1 | grep -q anaconda3; then
    module load apps/binapps/anaconda3/2023.09
    echo "Anaconda loaded"

    # Use existing hex_v2 environment (has PyTorch)
    if conda info --envs 2>/dev/null | grep -q "hex_v2"; then
        echo "Activating hex_v2 environment..."
        source activate hex_v2
    else
        echo "Creating hex_v3 environment..."
        conda create -n hex_v3 python=3.10 -y
        source activate hex_v3
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install numpy scipy tqdm tensorboard
    fi
else
    pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install --user numpy scipy tqdm tensorboard
fi

# Verify GPU
echo ""
echo "Verifying GPU..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Training mode
MODE=${1:-standard}

# Project directory
PROJECT_DIR=~/scratch/COMP34111-AI-Games-Hex
OUTPUT_DIR="$PROJECT_DIR/output_v3_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "V3 Parallel Training Configuration"
echo "=============================================="
echo "Mode: $MODE"
echo "Project: $PROJECT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Workers: 10 (parallel self-play)"
echo "=============================================="

# Change to project directory
cd "$PROJECT_DIR"

# Verify files exist
echo ""
echo "Verifying files..."
ls -la train_csf_v3.py
ls -la agents/Group12/training/train_v3_parallel.py

# Run V3 training
echo ""
echo "Starting V3 parallel training..."
echo "This should be 10-15x faster than V2!"
echo ""

python train_csf_v3.py \
    --mode "$MODE" \
    --output "$OUTPUT_DIR"

# Copy results
echo ""
echo "Organizing results..."

if [ -f "$OUTPUT_DIR/models/hex_model_v3_best.pth" ]; then
    mkdir -p "$PROJECT_DIR/models"
    cp "$OUTPUT_DIR/models/hex_model_v3_best.pth" "$PROJECT_DIR/models/"
    echo "Best model copied to models/"
fi

if [ -f "$OUTPUT_DIR/models/hex_model_v3_numpy.npz" ]; then
    mkdir -p "$PROJECT_DIR/agents/Group12/models"
    cp "$OUTPUT_DIR/models/hex_model_v3_numpy.npz" "$PROJECT_DIR/agents/Group12/models/hex_model_numpy.npz"
    echo "NumPy weights copied to agents/Group12/models/"
fi

echo ""
echo "=============================================="
echo "V3 Training Complete!"
echo "=============================================="
echo "End: $(date)"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Model files:"
ls -lah "$OUTPUT_DIR/models/" 2>/dev/null || echo "No models directory"
echo "=============================================="
