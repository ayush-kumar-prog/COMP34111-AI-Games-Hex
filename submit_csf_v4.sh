#!/bin/bash --login
#SBATCH --job-name=hex_v4
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH -t 1-0
#SBATCH --output=logs/hex_v4_%j.out
#SBATCH --error=logs/hex_v4_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r36859ak@student.manchester.ac.uk

# ============================================================================
# Group12 Hex AI V4 - BATCHED GPU Training
# ============================================================================
#
# Key V4 improvements:
# - ALL neural network calls batched on GPU
# - Threading-based parallelism shares GPU inference
# - Expected GPU utilization: 60-90% (vs 1% in V2, vs CPU-bound in V3)
# - Expected speedup: 20-50x over V2
#
# Usage:
#   sbatch submit_csf_v4.sh              # Default: standard mode (4-6 hours)
#   sbatch submit_csf_v4.sh test         # Quick test (15 minutes)
#   sbatch submit_csf_v4.sh fast         # Fast (2-3 hours)
#   sbatch submit_csf_v4.sh standard     # Standard (4-6 hours)
#   sbatch submit_csf_v4.sh full         # Full (8-12 hours)
#   sbatch submit_csf_v4.sh max          # Maximum (16-24 hours)
#
# ============================================================================

set -e

echo "=============================================="
echo "GROUP12 HEX AI V4 - BATCHED GPU TRAINING"
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
        echo "Creating hex_v4 environment..."
        conda create -n hex_v4 python=3.10 -y
        source activate hex_v4
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install numpy scipy tqdm
    fi
else
    pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install --user numpy scipy tqdm
fi

# Verify GPU
echo ""
echo "Verifying GPU..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Training mode
MODE=${1:-standard}

# Project directory
PROJECT_DIR=~/scratch/COMP34111-AI-Games-Hex
OUTPUT_DIR="$PROJECT_DIR/output_v4_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "V4 Batched GPU Training Configuration"
echo "=============================================="
echo "Mode: $MODE"
echo "Project: $PROJECT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""
echo "V4 Key Features:"
echo "  - Batched GPU inference for ALL neural network calls"
echo "  - Threading-based workers share single GPU"
echo "  - Expected 60-90% GPU utilization"
echo "  - 20-50x faster than V2"
echo "=============================================="

# Change to project directory
cd "$PROJECT_DIR"

# Verify files exist
echo ""
echo "Verifying files..."
ls -la train_csf_v4.py
ls -la agents/Group12/training/train_v4_batched.py

# Run V4 training
echo ""
echo "Starting V4 batched GPU training..."
echo "This should maximize A100 GPU utilization!"
echo ""

# Use unbuffered Python output (-u flag) so logs appear in real-time
export PYTHONUNBUFFERED=1
python -u train_csf_v4.py \
    --mode "$MODE" \
    --output "$OUTPUT_DIR" \
    --workers 32

# Copy results
echo ""
echo "Organizing results..."

if [ -f "$OUTPUT_DIR/models/hex_v4_best.pth" ]; then
    mkdir -p "$PROJECT_DIR/models"
    cp "$OUTPUT_DIR/models/hex_v4_best.pth" "$PROJECT_DIR/models/"
    echo "Best model copied to models/"
fi

if [ -f "$OUTPUT_DIR/models/hex_v4_numpy.npz" ]; then
    mkdir -p "$PROJECT_DIR/agents/Group12/models"
    cp "$OUTPUT_DIR/models/hex_v4_numpy.npz" "$PROJECT_DIR/agents/Group12/models/hex_model_numpy.npz"
    echo "NumPy weights copied to agents/Group12/models/"
fi

echo ""
echo "=============================================="
echo "V4 Training Complete!"
echo "=============================================="
echo "End: $(date)"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Model files:"
ls -lah "$OUTPUT_DIR/models/" 2>/dev/null || echo "No models directory"
echo "=============================================="
