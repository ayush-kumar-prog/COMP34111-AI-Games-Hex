#!/bin/bash
# ============================================================================
# Deploy V4.2 Batched GPU Training to CSF3 (FIXED VERSION)
# ============================================================================
#
# V4.2 FIXES:
# - Added timeout to event.wait() to prevent deadlocks
# - Added exception handling in inference loop
# - Added health check for inference thread
# - Added stall detection (warns after 60s no progress)
#
# Usage:
#   ./deploy_v4_to_csf.sh
#
# ============================================================================

set -e

CSF_USER="r36859ak"
CSF_HOST="csf3.itservices.manchester.ac.uk"
CSF_DIR="~/scratch/COMP34111-AI-Games-Hex"

echo "=============================================="
echo "Deploying V4 Batched GPU Training to CSF3"
echo "=============================================="
echo ""

# Files to upload
FILES=(
    "train_csf_v4.py"
    "submit_csf_v4.sh"
    "agents/Group12/training/train_v4_batched.py"
    "agents/Group12/neural/hex_network_v2.py"
)

echo "Files to upload:"
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  + $f"
    else
        echo "  ! $f (NOT FOUND)"
        exit 1
    fi
done

echo ""
echo "Uploading to ${CSF_USER}@${CSF_HOST}:${CSF_DIR}/"
echo ""

# Upload main files
for f in "${FILES[@]}"; do
    echo "Uploading $f..."
    scp "$f" "${CSF_USER}@${CSF_HOST}:${CSF_DIR}/$f"
done

echo ""
echo "=============================================="
echo "Upload Complete!"
echo "=============================================="
echo ""
echo "Now SSH into CSF3 and run these commands:"
echo ""
echo "  ssh ${CSF_USER}@${CSF_HOST}"
echo "  cd ${CSF_DIR}"
echo "  chmod +x submit_csf_v4.sh"
echo ""
echo "  # Cancel any broken V4 and V3 jobs:"
echo "  squeue -u ${CSF_USER}"
echo "  scancel 9108069  # Old broken V4"
echo "  scancel 9106039 9106040  # V3 jobs"
echo ""
echo "  # Submit fixed V4:"
echo "  sbatch submit_csf_v4.sh standard"
echo ""
echo "  # Monitor (should see output immediately now):"
echo "  tail -f logs/hex_v4_*.out"
echo ""
echo "V4.1 Fixes:"
echo "  - Unbuffered Python output (real-time logs)"
echo "  - Reduced workers (32 instead of 128)"
echo "  - Better error handling and progress reporting"
echo "=============================================="
