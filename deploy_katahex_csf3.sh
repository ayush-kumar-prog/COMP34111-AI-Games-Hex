#!/bin/bash
# CSF3 KataHex Deployment Script
# Run this script AFTER SSH-ing into CSF3
#
# Usage:
#   1. SSH to CSF3: ssh r36859ak@csf3.itservices.manchester.ac.uk
#   2. cd ~/scratch/COMP34111-AI-Games-Hex
#   3. bash deploy_katahex_csf3.sh

set -e

echo "========================================"
echo "KataHex CSF3 Deployment"
echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Step 1: Pull latest code
echo "=== Step 1: Pulling latest code ==="
git fetch origin simulation
git checkout simulation
git pull origin simulation
echo "✓ Code updated"
echo ""

# Step 2: Navigate to katahex directory
echo "=== Step 2: Navigating to katahex directory ==="
cd ~/scratch/COMP34111-AI-Games-Hex/agents/Group12/katahex
echo "Working directory: $(pwd)"
echo ""

# Step 3: Make setup script executable
echo "=== Step 3: Preparing setup script ==="
chmod +x setup_katahex_cuda.sh
echo "✓ Setup script is executable"
echo ""

# Step 4: Run setup script
echo "=== Step 4: Running CUDA setup script ==="
echo "This will:"
echo "  - Clone KataHex repository"
echo "  - Compile with CUDA backend for A100 GPU"
echo "  - Download pretrained model (~100MB)"
echo "  - Test GTP communication"
echo ""
read -p "Press Enter to continue..."

./setup_katahex_cuda.sh

echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Test integration:"
echo "     cd ~/scratch/COMP34111-AI-Games-Hex"
echo "     source venv/bin/activate"
echo "     python3 test_katahex_integration.py"
echo ""
echo "  2. Run tournament:"
echo "     python3 tournament_azalea_vs_katahex.py"
echo ""
