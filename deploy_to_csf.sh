#!/bin/bash
# ============================================================================
# Group12 Hex AI - Deploy to CSF3 and Run Sanity Test
# ============================================================================
# Run this script in your terminal - it will prompt for your password
# Usage: ./deploy_to_csf.sh
# ============================================================================

set -e

CSF_USER="r36859ak"
CSF_HOST="csf3.itservices.manchester.ac.uk"
PROJECT_NAME="COMP34111-AI-Games-Hex"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "DEPLOYING TO CSF3"
echo "=============================================="
echo "User: $CSF_USER"
echo "Host: $CSF_HOST"
echo "Project: $PROJECT_NAME"
echo ""
echo "You will be prompted for your university password."
echo "=============================================="
echo ""

# Step 1: Copy project to CSF
echo "Step 1: Copying project to CSF scratch..."
echo "Command: scp -r $SCRIPT_DIR $CSF_USER@$CSF_HOST:~/scratch/"
scp -r "$SCRIPT_DIR" "$CSF_USER@$CSF_HOST:~/scratch/"

echo ""
echo "Step 2: Submitting sanity test..."
ssh "$CSF_USER@$CSF_HOST" << 'REMOTE_COMMANDS'
cd ~/scratch/COMP34111-AI-Games-Hex
echo "Current directory: $(pwd)"
echo "Files:"
ls -la *.sh 2>/dev/null | head -5

echo ""
echo "Submitting sanity test..."
sbatch sanity_test_csf.sh

echo ""
echo "Job queue:"
squeue -u $USER

echo ""
echo "=============================================="
echo "DEPLOYMENT COMPLETE!"
echo "=============================================="
echo ""
echo "Monitor with: ssh $USER@csf3.itservices.manchester.ac.uk 'squeue; cat ~/scratch/COMP34111-AI-Games-Hex/sanity_test_*.out'"
REMOTE_COMMANDS

echo ""
echo "=============================================="
echo "DONE! Check your email or run:"
echo "  ssh $CSF_USER@$CSF_HOST 'cat ~/scratch/$PROJECT_NAME/sanity_test_*.out'"
echo "=============================================="
