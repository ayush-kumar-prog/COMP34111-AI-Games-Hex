#!/bin/bash --login
#SBATCH --job-name=hex_train
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH -t 4-0
#SBATCH --output=logs/hex_train_%j.out
#SBATCH --error=logs/hex_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r36859ak@student.manchester.ac.uk

# ============================================================================
# Group12 Hex AI - CSF3 A100 Training Job Script (University of Manchester)
# ============================================================================
#
# NOTE: V100 GPUs are NO LONGER AVAILABLE (as of Oct 2025)
# This script uses A100 (80GB) GPUs instead.
#
# A100 GPU Specifications:
#   - 80GB HBM2e memory
#   - Max 12 CPUs per GPU
#   - 4-day max runtime
#
# Usage (run from ~/scratch):
#   cd ~/scratch/COMP34111-AI-Games-Hex
#   sbatch submit_csf.sh              # Default: standard mode
#   sbatch submit_csf.sh test         # Quick test (10 minutes)
#   sbatch submit_csf.sh dev          # Development (2-4 hours)
#   sbatch submit_csf.sh standard     # Standard (8-12 hours)
#   sbatch submit_csf.sh full         # Full training (24-48 hours)
#
# Monitor:
#   squeue                            # Check job status
#   gpustat -j <jobid>                # Monitor GPU usage
#   tail -f logs/hex_train_*.out      # Watch output
#   scancel <job_id>                  # Cancel job
#
# Login:
#   ssh r36859ak@csf3.itservices.manchester.ac.uk
#
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "GROUP12 HEX AI - CSF3 A100 TRAINING"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start time: $(date)"
echo "=============================================="

# Create logs directory
mkdir -p ~/logs
mkdir -p logs

# Load required modules for CSF3
module purge
module load libs/cuda/12.4.1

# Setup Python environment
echo ""
echo "Setting up Python environment..."

# Try to load anaconda
if module avail apps/binapps/anaconda3 2>&1 | grep -q anaconda3; then
    module load apps/binapps/anaconda3/2023.09
    echo "Anaconda loaded"

    # Create or activate conda environment
    if conda info --envs 2>/dev/null | grep -q "hex_ai"; then
        echo "Activating existing hex_ai conda environment..."
        source activate hex_ai
    else
        echo "Creating new hex_ai conda environment..."
        conda create -n hex_ai python=3.10 -y
        source activate hex_ai
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install numpy tqdm tensorboard
    fi
else
    echo "Anaconda not found, using pip --user"
    pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install --user numpy tqdm tensorboard
fi

# Verify GPU
echo ""
echo "Verifying GPU setup..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Set training mode (default: standard)
MODE=${1:-standard}

# Use fast NVMe storage ($TMPDIR) for training
WORK_DIR="$TMPDIR/hex_training"
OUTPUT_DIR="$WORK_DIR/output"
mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Training Configuration"
echo "=============================================="
echo "Mode: $MODE"
echo "Work directory: $WORK_DIR (fast NVMe)"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

# Copy training code to fast NVMe storage
echo ""
echo "Copying training code to fast NVMe storage..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp -r "$SCRIPT_DIR"/* "$WORK_DIR/" 2>/dev/null || cp -r ~/scratch/COMP34111-AI-Games-Hex/* "$WORK_DIR/"
cd "$WORK_DIR"

# Run training
echo ""
echo "Starting training..."
python train_csf.py \
    --mode "$MODE" \
    --device cuda \
    --output "$OUTPUT_DIR"

# Copy results back to scratch
echo ""
echo "Copying results back to scratch..."
RESULTS_DIR=~/scratch/hex_results_${SLURM_JOB_ID}
mkdir -p "$RESULTS_DIR"
cp -r "$OUTPUT_DIR/models" "$RESULTS_DIR/" 2>/dev/null || echo "No models directory"
cp -r "$OUTPUT_DIR/runs" "$RESULTS_DIR/" 2>/dev/null || echo "No runs directory"

# Copy best model to main models directory
if [ -f "$OUTPUT_DIR/models/hex_model_best.pth" ]; then
    mkdir -p ~/scratch/COMP34111-AI-Games-Hex/models
    cp "$OUTPUT_DIR/models/hex_model_best.pth" ~/scratch/COMP34111-AI-Games-Hex/models/
    echo "Best model copied to ~/scratch/COMP34111-AI-Games-Hex/models/"
fi

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "End time: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To use trained model:"
echo "  cp $RESULTS_DIR/models/hex_model_best.pth models/"
echo "=============================================="
