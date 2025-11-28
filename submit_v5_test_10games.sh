#!/bin/bash
#SBATCH --job-name=hex_test_10
#SBATCH --partition=multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=v5_test_10_%j.out
#SBATCH --error=v5_test_10_%j.err

# Quick test: 10 games, 1 worker, 500 MCTS iterations
# Should complete in 10-30 minutes if my estimates are correct

echo "=================================================="
echo "V5 TEST: 10 GAMES ONLY"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=================================================="

cd ~/scratch/COMP34111-AI-Games-Hex || exit 1

module load apps/binapps/anaconda3/2023.09 2>/dev/null
PYTHON_CMD="python"

echo "Python: $($PYTHON_CMD --version)"
echo ""
echo "Starting 10-game test with UNBUFFERED output (-u flag)..."
echo "This will show us exact timing per game."
echo ""

# Use -u for unbuffered output so we see progress immediately
$PYTHON_CMD -u -m agents.Group12.training.expert_data_gen \
    --num-games 10 \
    --workers 1 \
    --mcts-iters 500 \
    --output data/test_10games.npz

echo ""
echo "=================================================="
echo "TEST COMPLETE"
echo "=================================================="
echo "End: $(date)"
ls -lh data/test_10games.npz 2>/dev/null
