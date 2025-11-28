#!/bin/bash
#SBATCH --job-name=debug_test
#SBATCH --partition=multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Debug script to identify CSF configuration
echo "========== DEBUG INFO =========="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "PWD at start: $(pwd)"
echo "HOME: $HOME"
echo ""

echo "========== PATH CHECK =========="
echo "Checking ~/scratch..."
ls -la ~/scratch 2>&1 | head -5
echo ""
echo "Checking ~/scratch/COMP34111-AI-Games-Hex..."
ls ~/scratch/COMP34111-AI-Games-Hex 2>&1 | head -5
echo ""

echo "========== MODULE CHECK =========="
echo "Available Python modules:"
module avail python 2>&1 | grep -i python | head -10
echo ""

echo "Trying to load Python..."
module load python/3.11.5 && echo "SUCCESS: python/3.11.5" || \
module load python/3.11 && echo "SUCCESS: python/3.11" || \
module load python3 && echo "SUCCESS: python3" || \
module load apps/binapps/anaconda3/2023.09 && echo "SUCCESS: anaconda3" || \
echo "FAILED: No Python module worked"

echo ""
echo "Python version:"
python --version 2>&1 || python3 --version 2>&1 || echo "Python not found"

echo ""
echo "========== NUMPY CHECK =========="
python -c "import numpy; print(f'NumPy {numpy.__version__}')" 2>&1 || echo "NumPy not available"

echo ""
echo "========== PROJECT CHECK =========="
cd ~/scratch/COMP34111-AI-Games-Hex 2>&1 && echo "cd SUCCESS" || echo "cd FAILED"
echo "PWD after cd: $(pwd)"
ls -la src/ 2>&1 | head -3
python -c "from src.Board import Board; print('Import SUCCESS')" 2>&1 || echo "Import FAILED"

echo ""
echo "========== DEBUG COMPLETE =========="
