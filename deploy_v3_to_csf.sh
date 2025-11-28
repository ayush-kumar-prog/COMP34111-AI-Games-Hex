#!/bin/bash
# ============================================================================
# Deploy V3 Parallel Training to CSF3
# ============================================================================
#
# This script uploads all V3 training files to CSF3.
# After running this, SSH into CSF3 and submit the jobs.
#
# Usage:
#   ./deploy_v3_to_csf.sh
#
# ============================================================================

set -e

CSF_USER="r36859ak"
CSF_HOST="csf3.itservices.manchester.ac.uk"
CSF_DIR="~/scratch/COMP34111-AI-Games-Hex"

echo "=============================================="
echo "Deploying V3 Training Files to CSF3"
echo "=============================================="
echo ""

# Files to upload
FILES=(
    "train_csf_v3.py"
    "submit_csf_v3_standard.sh"
    "submit_csf_v3_full.sh"
    "agents/Group12/training/train_v3_parallel.py"
)

echo "Files to upload:"
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ $f (NOT FOUND)"
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
echo "Now SSH into CSF3 and submit the jobs:"
echo ""
echo "  ssh ${CSF_USER}@${CSF_HOST}"
echo "  cd ${CSF_DIR}"
echo "  chmod +x submit_csf_v3_standard.sh submit_csf_v3_full.sh"
echo ""
echo "  # Submit both jobs:"
echo "  sbatch submit_csf_v3_standard.sh    # 8-12 hours"
echo "  sbatch submit_csf_v3_full.sh        # 18-24 hours"
echo ""
echo "  # Monitor:"
echo "  squeue -u ${CSF_USER}"
echo "=============================================="
