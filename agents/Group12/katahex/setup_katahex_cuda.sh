#!/bin/bash
# KataHex Setup Script for CSF3 with CUDA Backend
# This script compiles KataHex with GPU support for A100 GPUs
#
# Prerequisites on CSF3:
# - cmake (>= 3.10)
# - gcc/g++ (>= 9.0)
# - CUDA toolkit
# - git
#
# Usage:
#   cd agents/Group12/katahex
#   chmod +x setup_katahex_cuda.sh
#   ./setup_katahex_cuda.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "KataHex CUDA Setup Script for CSF3"
echo "========================================"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Load required modules on CSF3
echo "Loading CSF3 modules..."
module load tools/gcc/12.2.0 2>/dev/null || echo "GCC module not available (might already be loaded)"
module load tools/cmake/3.26.3 2>/dev/null || echo "CMake module not available (might already be loaded)"
module load libs/cuda/12.1.1 2>/dev/null || echo "CUDA module not available (might already be loaded)"

echo ""
echo "Checking prerequisites..."

# Check for required tools
if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found. Try: module load tools/cmake/3.26.3"
    exit 1
fi
echo "✓ cmake: $(cmake --version | head -1)"

if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ not found. Try: module load tools/gcc/12.2.0"
    exit 1
fi
echo "✓ g++: $(g++ --version | head -1)"

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Try: module load libs/cuda/12.1.1"
    exit 1
fi
echo "✓ CUDA: $(nvcc --version | grep release)"

# Step 1: Clone KataHex repository if not exists
if [ ! -d "katahex-src" ]; then
    echo ""
    echo "=== Step 1: Cloning KataHex repository ==="
    git clone --branch Hex2024 --depth 1 https://github.com/selinger/katahex.git katahex-src
else
    echo ""
    echo "=== Step 1: KataHex source already exists ==="
fi

# Step 2: Build KataHex with CUDA backend
echo ""
echo "=== Step 2: Building KataHex with CUDA backend ==="
cd katahex-src

# Create build directory
rm -rf build  # Clean previous builds
mkdir -p build
cd build

# Configure with CUDA backend for A100 GPU
echo "Configuring with cmake (CUDA backend)..."
cmake ../cpp \
    -DUSE_BACKEND=CUDA \
    -DMAX_BOARD_LEN=19 \
    -DCMAKE_BUILD_TYPE=Release \
    -DNO_GIT_REVISION=1 \
    -DCUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_COMPILER=$(which nvcc)

# Build (use multiple cores for speed)
echo "Building..."
make -j$(nproc)

# Copy binary to katahex directory
cp katahex "$SCRIPT_DIR/katahex"
chmod +x "$SCRIPT_DIR/katahex"
echo "Binary copied to: $SCRIPT_DIR/katahex"

cd "$SCRIPT_DIR"

# Step 3: Download pretrained model
echo ""
echo "=== Step 3: Downloading pretrained model ==="

MODEL_FILE="hex27x3.bin.gz"
# URL redirects from KataGo to KataGomo_fork, but curl follows redirects
MODEL_URL="https://github.com/hzyhhzy/KataGo/releases/download/Hex_20240812/hex27x3.bin.gz"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading model from: $MODEL_URL"

    # Use curl with redirect following
    if command -v curl &> /dev/null; then
        curl -L -o "$MODEL_FILE" "$MODEL_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$MODEL_FILE" "$MODEL_URL"
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
    # Check if binary is dynamically linked to CUDA
    if ldd "$SCRIPT_DIR/katahex" | grep -q "libcudart"; then
        echo "  ✓ CUDA-enabled binary detected"
    else
        echo "  ⚠ Warning: Binary may not be CUDA-enabled"
    fi
else
    echo "✗ KataHex binary: MISSING or not executable"
    exit 1
fi

if [ -f "$SCRIPT_DIR/$MODEL_FILE" ]; then
    MODEL_SIZE=$(ls -lh "$SCRIPT_DIR/$MODEL_FILE" | awk '{print $5}')
    echo "✓ Model file: OK ($MODEL_SIZE)"
else
    echo "✗ Model file: MISSING"
    exit 1
fi

if [ -f "$SCRIPT_DIR/config.cfg" ]; then
    echo "✓ Config file: OK"
else
    echo "✗ Config file: MISSING"
    exit 1
fi

# Step 5: Test basic GTP communication
echo ""
echo "=== Step 5: Testing GTP communication ==="

if [ -x "$SCRIPT_DIR/katahex" ] && [ -f "$SCRIPT_DIR/$MODEL_FILE" ]; then
    echo "Running quick test..."

    # Test basic GTP commands
    RESULT=$(timeout 30 bash -c "echo -e 'boardsize 11\nclear_board\nquit' | '$SCRIPT_DIR/katahex' gtp -config '$SCRIPT_DIR/config.cfg' -model '$SCRIPT_DIR/$MODEL_FILE' 2>&1" || echo "TIMEOUT")

    if echo "$RESULT" | grep -q "^="; then
        echo "✓ GTP communication: OK"
    else
        echo "✗ GTP communication: FAILED"
        echo "Output: $RESULT"
        exit 1
    fi
else
    echo "Skipping GTP test (binary or model missing)"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "KataHex is ready with CUDA backend"
echo ""
echo "To test manually:"
echo "  echo 'boardsize 11' | ./katahex gtp -config config.cfg -model $MODEL_FILE"
echo ""
echo "To use in Python:"
echo "  from agents.Group12.KataHexAgent import KataHexAgent"
echo ""
