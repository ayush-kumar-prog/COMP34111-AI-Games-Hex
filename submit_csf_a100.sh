#!/bin/bash --login
#SBATCH --job-name=hex_a100_train
#SBATCH -p gpuA
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH -t 4-0
#SBATCH --output=logs/hex_train_a100_%j.out
#SBATCH --error=logs/hex_train_a100_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r36859ak@student.manchester.ac.uk

# ============================================================================
# Group12 Hex AI - CSF3 A100 Training Job Script (University of Manchester)
# ============================================================================
#
# IMPORTANT: A100 access is RESTRICTED!
#   - You need to justify why V100 (16GB) is insufficient
#   - Request via: https://ri.itservices.manchester.ac.uk/csf3/batch/gpu-jobs/
#
# A100 GPU Specifications:
#   - 80GB HBM2e memory (5x more than V100)
#   - 312 TFLOPS tensor performance
#   - Max 12 CPUs per GPU
#   - Ideal for: Large batch sizes, big models
#
# Usage (run from ~/scratch):
#   cd ~/scratch/COMP34111-AI-Games-Hex
#   sbatch submit_csf_a100.sh              # Default: full mode
#   sbatch submit_csf_a100.sh standard     # Standard training
#   sbatch submit_csf_a100.sh full         # Full training (recommended for A100)
#   sbatch submit_csf_a100.sh max          # Maximum training
#
# Monitor:
#   squeue                            # Check job status
#   gpustat -j <jobid>                # Monitor GPU usage
#   tail -f logs/hex_train_a100_*.out # Watch output
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
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NGPUS GPU(s))"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start time: $(date)"
echo "=============================================="

# Create logs directory
mkdir -p ~/logs

# Load required modules for CSF3
module purge
module load libs/cuda/12.4.1
module load compilers/gcc/6.4.0

# Setup Python environment
echo ""
echo "Setting up Python environment..."

if module avail apps/binapps/anaconda3 2>&1 | grep -q anaconda3; then
    module load apps/binapps/anaconda3/2023.09

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
    echo "Using system Python with pip --user"
    pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install --user numpy tqdm tensorboard
fi

# Verify GPU
echo ""
echo "Verifying A100 GPU setup..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Set training mode (default: full for A100 - take advantage of the power!)
MODE=${1:-full}

# Use fast NVMe storage ($TMPDIR) - A100 nodes have 1.6TB NVMe
WORK_DIR="$TMPDIR/hex_training"
OUTPUT_DIR="$WORK_DIR/output"
mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "A100 Training Configuration"
echo "=============================================="
echo "Mode: $MODE"
echo "Work directory: $WORK_DIR (fast NVMe)"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: 512 (optimized for A100 80GB)"
echo "=============================================="

# Copy training code to fast NVMe storage
echo ""
echo "Copying training code to fast NVMe storage..."
cp -r ~/scratch/COMP34111-AI-Games-Hex/* "$WORK_DIR/"
cd "$WORK_DIR"

# Run training with larger batch size for A100
echo ""
echo "Starting A100-optimized training..."
python train_csf.py \
    --mode "$MODE" \
    --device cuda \
    --output "$OUTPUT_DIR" \
    --batch_size 512

# Copy results back to scratch
echo ""
echo "Copying results back to scratch..."
RESULTS_DIR=~/scratch/hex_results_a100_${SLURM_JOB_ID}
mkdir -p "$RESULTS_DIR"
cp -r "$OUTPUT_DIR/models" "$RESULTS_DIR/" 2>/dev/null || echo "No models directory"
cp -r "$OUTPUT_DIR/runs" "$RESULTS_DIR/" 2>/dev/null || echo "No runs directory"

# Copy best model to main models directory
if [ -f "$OUTPUT_DIR/models/hex_model_best.pth" ]; then
    cp "$OUTPUT_DIR/models/hex_model_best.pth" ~/scratch/COMP34111-AI-Games-Hex/models/
    echo "Best model copied to ~/scratch/COMP34111-AI-Games-Hex/models/"
fi

echo ""
echo "=============================================="
echo "A100 Training complete!"
echo "=============================================="
echo "End time: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To copy to home directory (for backup):"
echo "  cp -r $RESULTS_DIR ~/hex_results_a100_${SLURM_JOB_ID}"
echo ""
echo "To use trained model:"
echo "  cp $RESULTS_DIR/models/hex_model_best.pth models/"
echo "=============================================="
