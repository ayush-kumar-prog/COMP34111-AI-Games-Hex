#!/bin/bash
# Prepare codebase for Google Colab upload
# Version 3: Fixed temperature schedule, label smoothing, gradient clipping

echo "=================================="
echo "Preparing Codebase for Google Colab"
echo "VERSION 3 - FIXED TRAINING"
echo "=================================="
echo ""
echo "Key fixes in this version:"
echo "- Dynamic temperature (0.5-1.5, never drops to 0)"
echo "- Label smoothing (0.1) prevents overconfidence"
echo "- Gradient clipping (max_norm=1.0)"
echo "- L2 regularization (1e-4)"
echo "- Reduced epochs (10->5) prevents overfitting"
echo ""

# Get the current directory name
CURRENT_DIR=$(basename "$PWD")
ZIP_NAME="COMP34111-AI-Games-Hex-for-Colab-FIXED-v3.zip"

echo "Creating ZIP file: $ZIP_NAME"
echo ""

# Create ZIP, excluding unnecessary files
zip -r "$ZIP_NAME" . \
    -x "*.git/*" \
    -x "*.pyc" \
    -x "__pycache__/*" \
    -x "*.log" \
    -x "*.json" \
    -x "docker_build.log" \
    -x "comprehensive_test.log" \
    -x "gpu_training.log" \
    -x "stress_test_results*.json" \
    -x ".DS_Store" \
    -x "venv/*" \
    -x "models/*" \
    -x "runs/*"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS!"
    echo ""
    echo "ZIP created: $ZIP_NAME"
    echo "Size: $(du -h "$ZIP_NAME" | cut -f1)"
    echo ""
    echo "Next steps:"
    echo "1. Upload this ZIP to Google Colab"
    echo "2. Open Group12_Hex_Neural_Training.ipynb in Colab"
    echo "3. Run all cells"
    echo "4. Download trained models after 2-4 hours"
    echo ""
else
    echo "❌ Error creating ZIP file"
    exit 1
fi
