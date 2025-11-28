#!/bin/bash --login
#SBATCH --job-name=hex_sanity
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 0-0:15
#SBATCH --output=sanity_test_%j.out
#SBATCH --error=sanity_test_%j.err

# ============================================================================
# Group12 Hex AI - CSF3 Sanity Test (~5-10 minutes)
# ============================================================================
# Verifies: GPU access, CUDA, PyTorch, training code
# ============================================================================

echo "=============================================="
echo "CSF3 SANITY TEST - $(date)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# Test 1: Environment
echo ""
echo "TEST 1: Environment"
echo "-------------------"
echo "User: $(whoami)"
echo "Home: $HOME"
echo "TMPDIR: $TMPDIR"

# Test 2: Modules
echo ""
echo "TEST 2: Loading CUDA"
echo "--------------------"
module purge
module load libs/cuda/12.4.1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi failed"

# Test 3: Python/PyTorch
echo ""
echo "TEST 3: Python/PyTorch"
echo "----------------------"

# Load anaconda if available
module load apps/binapps/anaconda3/2023.09 2>/dev/null || echo "Anaconda module not found"

python3 << 'PYTHON_TEST'
import sys
print(f"Python: {sys.version.split()[0]}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Quick test
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        print("GPU tensor test: PASSED")
    else:
        print("ERROR: CUDA not available!")
except ImportError:
    print("PyTorch not installed - will install on first training job")
except Exception as e:
    print(f"Error: {e}")
PYTHON_TEST

# Test 4: Training code check
echo ""
echo "TEST 4: Training Code"
echo "---------------------"
PROJ_DIR=~/scratch/COMP34111-AI-Games-Hex
if [ -d "$PROJ_DIR" ]; then
    echo "Project found at: $PROJ_DIR"
    ls -la "$PROJ_DIR"/*.py 2>/dev/null | head -5 || echo "No .py files in root"
    ls -la "$PROJ_DIR"/agents/Group12/*.py 2>/dev/null | head -3 || echo "No agent files"
else
    echo "Project NOT FOUND at $PROJ_DIR"
    echo "Please copy project: scp -r . r36859ak@csf3.itservices.manchester.ac.uk:~/scratch/COMP34111-AI-Games-Hex/"
fi

echo ""
echo "=============================================="
echo "SANITY TEST COMPLETE"
echo "=============================================="
echo ""
echo "NEXT STEPS:"
echo "1. If GPU shows ~80GB memory (A100) - you're ready!"
echo "2. If PyTorch missing - it will auto-install on training"
echo "3. Submit training: sbatch submit_csf.sh dev"
echo "=============================================="
