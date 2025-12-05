#!/bin/bash
# KataHex Setup Script for CSF3
# This script compiles KataHex from source and downloads the pretrained model.
#
# Prerequisites (CSF3 should have these):
# - cmake (>= 3.10)
# - gcc/g++ (>= 9.0)
# - libeigen3-dev
# - git
#
# Usage:
#   cd agents/Group12/katahex
#   chmod +x setup_katahex.sh
#   ./setup_katahex.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== KataHex Setup Script ==="
echo "Working directory: $SCRIPT_DIR"

# Check for required tools
echo ""
echo "Checking prerequisites..."

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found. Please install cmake."
    exit 1
fi

if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ not found. Please install g++."
    exit 1
fi

# Check Eigen availability (may need to load module on CSF3)
# On CSF3: module load libs/eigen/3.4.0
if ! pkg-config --exists eigen3 2>/dev/null; then
    echo "WARNING: eigen3 not found in pkg-config. Will try to build anyway."
    echo "On CSF3, try: module load libs/eigen/3.4.0"
fi

# Step 1: Clone KataHex repository if not exists
if [ ! -d "katahex-src" ]; then
    echo ""
    echo "=== Step 1: Cloning KataHex repository ==="
    git clone --branch Hex2024 --depth 1 https://github.com/selinger/katahex.git katahex-src
else
    echo "=== Step 1: KataHex source already exists ==="
fi

# Step 2: Build KataHex
echo ""
echo "=== Step 2: Building KataHex ==="
cd katahex-src

# Create build directory
mkdir -p build
cd build

# Configure with Eigen backend, 19x19 max board size
echo "Configuring with cmake..."
cmake ../cpp \
    -DUSE_BACKEND=EIGEN \
    -DMAX_BOARD_LEN=19 \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_AVX2=1 \
    -DNO_GIT_REVISION=1

# Build (use multiple cores for speed)
echo "Building..."
make -j$(nproc)

# Copy binary to katahex directory
cp katahex "$SCRIPT_DIR/katahex"
echo "Binary copied to: $SCRIPT_DIR/katahex"

cd "$SCRIPT_DIR"

# Step 3: Download pretrained model
echo ""
echo "=== Step 3: Downloading pretrained model ==="

MODEL_FILE="hex27x3.bin.gz"
MODEL_URL="https://github.com/hzyhhzy/KataGo/releases/download/Hex_20240812/hex27x3.bin.gz"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading model from: $MODEL_URL"

    # Try wget first, then curl
    if command -v wget &> /dev/null; then
        wget -O "$MODEL_FILE" "$MODEL_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$MODEL_FILE" "$MODEL_URL"
    else
        echo "ERROR: Neither wget nor curl found."
        echo "Please manually download from: $MODEL_URL"
        echo "And place it in: $SCRIPT_DIR/$MODEL_FILE"
        exit 1
    fi

    echo "Model downloaded: $MODEL_FILE"
else
    echo "Model already exists: $MODEL_FILE"
fi

# Step 4: Verify setup
echo ""
echo "=== Step 4: Verifying setup ==="

if [ -x "$SCRIPT_DIR/katahex" ]; then
    echo "✓ KataHex binary: OK"
else
    echo "✗ KataHex binary: MISSING or not executable"
fi

if [ -f "$SCRIPT_DIR/$MODEL_FILE" ]; then
    MODEL_SIZE=$(ls -lh "$SCRIPT_DIR/$MODEL_FILE" | awk '{print $5}')
    echo "✓ Model file: OK ($MODEL_SIZE)"
else
    echo "✗ Model file: MISSING"
fi

if [ -f "$SCRIPT_DIR/config.cfg" ]; then
    echo "✓ Config file: OK"
else
    echo "✗ Config file: MISSING"
fi

# Step 5: Test basic GTP communication
echo ""
echo "=== Step 5: Testing GTP communication ==="

if [ -x "$SCRIPT_DIR/katahex" ] && [ -f "$SCRIPT_DIR/$MODEL_FILE" ]; then
    echo "Running quick test..."

    # Test basic GTP commands
    RESULT=$(echo -e "boardsize 11\nclear_board\nquit" | "$SCRIPT_DIR/katahex" gtp -config "$SCRIPT_DIR/config.cfg" -model "$SCRIPT_DIR/$MODEL_FILE" 2>&1)

    if echo "$RESULT" | grep -q "^="; then
        echo "✓ GTP communication: OK"
    else
        echo "✗ GTP communication: FAILED"
        echo "Output: $RESULT"
    fi
else
    echo "Skipping GTP test (binary or model missing)"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To test manually:"
echo "  echo 'boardsize 11' | ./katahex gtp -config config.cfg -model $MODEL_FILE"
echo ""
echo "To use in tournament:"
echo "  Import KataHexAgent from agents.Group12.KataHexAgent"
