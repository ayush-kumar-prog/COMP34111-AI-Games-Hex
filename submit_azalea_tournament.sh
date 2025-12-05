#!/bin/bash
#SBATCH --job-name=azalea_tournament
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=azalea_tournament_%j.out
#SBATCH --error=azalea_tournament_%j.err

echo "========================================"
echo "AzaleaAgent Tournament Test (GPU)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================"

cd ~/scratch/COMP34111-AI-Games-Hex
source venv/bin/activate

echo ""
echo "Python: $(which python3)"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# Run tournament
python3 run_agent_tournament.py

echo ""
echo "========================================"
echo "End time: $(date)"
echo "========================================"
