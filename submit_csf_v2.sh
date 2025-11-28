#!/bin/bash --login
#SBATCH --job-name=hex_v2
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH -t 4-0
#SBATCH --output=logs/hex_v2_%j.out
#SBATCH --error=logs/hex_v2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r36859ak@student.manchester.ac.uk

# ============================================================================
# Group12 Hex AI V2 - CSF3 A100 Training (Ultimate Bot)
# ============================================================================
#
# This script trains the enhanced V2 neural network with:
# - 8-channel Hex-specific board encoding
# - 15 residual blocks with SE attention
# - Dilated convolutions for larger receptive field
# - Automatic NumPy export for tournament
#
# Usage:
#   sbatch submit_csf_v2.sh              # Default: standard mode
#   sbatch submit_csf_v2.sh test         # Quick test (10 minutes)
#   sbatch submit_csf_v2.sh dev          # Development (2-4 hours)
#   sbatch submit_csf_v2.sh standard     # Standard (8-12 hours)
#   sbatch submit_csf_v2.sh full         # Full training (24-48 hours)
#   sbatch submit_csf_v2.sh max          # Maximum (72+ hours)
#
# ============================================================================

set -e

echo "=============================================="
echo "GROUP12 HEX AI V2 - CSF3 A100 TRAINING"
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

    if conda info --envs 2>/dev/null | grep -q "hex_v2"; then
        echo "Activating hex_v2 environment..."
        source activate hex_v2
    else
        echo "Creating hex_v2 environment..."
        conda create -n hex_v2 python=3.10 -y
        source activate hex_v2
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

# Use project directory directly (more reliable than NVMe copy)
PROJECT_DIR=~/scratch/COMP34111-AI-Games-Hex
OUTPUT_DIR="$PROJECT_DIR/output_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Training V2 Configuration"
echo "=============================================="
echo "Mode: $MODE"
echo "Project dir: $PROJECT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Change to project directory
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"
echo "Files present:"
ls -la train_csf_v2.py agents/Group12/neural/*.py 2>&1 | head -20

# Run V2 training
echo ""
echo "Starting V2 training..."
python train_csf_v2.py \
    --mode "$MODE" \
    --device cuda \
    --output "$OUTPUT_DIR"

# Copy trained models to main locations
echo ""
echo "Organizing results..."

# Copy best model to main models directory
if [ -f "$OUTPUT_DIR/models/hex_model_v2_best.pth" ]; then
    cp "$OUTPUT_DIR/models/hex_model_v2_best.pth" "$PROJECT_DIR/models/"
    echo "Best PyTorch model copied to models/"
fi

# Copy NumPy weights to agent directory for tournament
if [ -f "$OUTPUT_DIR/models/hex_model_numpy.npz" ]; then
    mkdir -p "$PROJECT_DIR/agents/Group12/models"
    cp "$OUTPUT_DIR/models/hex_model_numpy.npz" "$PROJECT_DIR/agents/Group12/models/"
    echo "NumPy weights copied to agents/Group12/models/"
fi

echo ""
echo "Output directory contents:"
ls -la "$OUTPUT_DIR/models/" 2>/dev/null || echo "No models directory yet"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "End: $(date)"
echo "Results: $RESULTS_DIR"
echo ""
echo "To use the trained model:"
echo "  The NumPy weights are automatically copied to:"
echo "  agents/Group12/models/hex_model_numpy.npz"
echo ""
echo "  Update cmd.txt to use the ultimate agent:"
echo "  agents.Group12.Group12Agent_ultimate Group12Agent"
echo "=============================================="
