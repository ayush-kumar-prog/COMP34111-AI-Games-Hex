#!/bin/bash --login
#SBATCH --job-name=hex_v3_full
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH -t 2-0
#SBATCH --output=logs/hex_v3_full_%j.out
#SBATCH --error=logs/hex_v3_full_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r36859ak@student.manchester.ac.uk

# ============================================================================
# Group12 Hex AI V3 FULL - Maximum Quality Training (18-24 hours)
# ============================================================================
#
# Configuration:
#   - 40 iterations (2x standard)
#   - 400 games per iteration
#   - 150 MCTS simulations
#   - 10 parallel workers
#   - Expected time: 18-24 hours
#
# This produces the highest quality model.
#
# ============================================================================

set -e

echo "=============================================="
echo "GROUP12 HEX AI V3 - FULL MODE (BEST QUALITY)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start: $(date)"
echo "=============================================="

mkdir -p ~/logs logs

# Load modules
module purge
module load libs/cuda/12.4.1

# Setup Python
echo "Setting up Python..."
if module avail apps/binapps/anaconda3 2>&1 | grep -q anaconda3; then
    module load apps/binapps/anaconda3/2023.09
    if conda info --envs 2>/dev/null | grep -q "hex_v2"; then
        source activate hex_v2
        echo "Using hex_v2 environment"
    else
        echo "Creating environment..."
        conda create -n hex_v2 python=3.10 -y
        source activate hex_v2
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install numpy scipy tqdm
    fi
fi

# Verify GPU
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Setup directories
PROJECT_DIR=~/scratch/COMP34111-AI-Games-Hex
OUTPUT_DIR="$PROJECT_DIR/output_v3_full_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

echo ""
echo "=============================================="
echo "V3 FULL Configuration (MAXIMUM QUALITY)"
echo "=============================================="
echo "Iterations: 40"
echo "Games/iter: 400"
echo "MCTS sims: 150"
echo "Workers: 10"
echo "Expected: 18-24 hours"
echo "=============================================="

# Run training
python train_csf_v3.py \
    --mode full \
    --output "$OUTPUT_DIR"

# Copy results
echo "Copying results..."
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/agents/Group12/models"

if [ -f "$OUTPUT_DIR/models/hex_model_v3_best.pth" ]; then
    cp "$OUTPUT_DIR/models/hex_model_v3_best.pth" "$PROJECT_DIR/models/hex_model_v3_full.pth"
fi

if [ -f "$OUTPUT_DIR/models/hex_model_v3_numpy.npz" ]; then
    cp "$OUTPUT_DIR/models/hex_model_v3_numpy.npz" "$PROJECT_DIR/agents/Group12/models/hex_model_v3_full.npz"
fi

echo ""
echo "=============================================="
echo "V3 FULL Complete!"
echo "=============================================="
echo "End: $(date)"
echo "Output: $OUTPUT_DIR"
ls -lah "$OUTPUT_DIR/models/" 2>/dev/null || echo "No models"
echo "=============================================="
