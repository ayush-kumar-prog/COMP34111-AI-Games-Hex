#!/bin/bash
#
# Start GPU training in Docker container
# This script builds the GPU image and launches training
#

set -e

echo "=============================================================================="
echo "GROUP12 NEURAL HEX AI - DOCKER GPU TRAINING"
echo "=============================================================================="
echo ""

# Step 1: Build GPU Docker image
echo "[1/4] Building GPU Docker image..."
echo "This may take 5-10 minutes on first run..."
echo ""

docker build -f Dockerfile.gpu --build-arg UID=$(id -u) -t hex-gpu . || {
    echo "❌ Docker build failed"
    exit 1
}

echo ""
echo "✅ GPU Docker image built successfully"
echo ""

# Step 2: Verify Docker can access GPU (if available)
echo "[2/4] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_FLAGS="--runtime=nvidia --gpus all"
    DEVICE="cuda"
else
    echo "⚠️  No NVIDIA GPU detected - will use CPU"
    echo "   Training will be much slower but functional"
    GPU_FLAGS=""
    DEVICE="cpu"
fi
echo ""

# Step 3: Create necessary directories
echo "[3/4] Setting up directories..."
mkdir -p models runs data/self_play
echo "✅ Directories ready"
echo ""

# Step 4: Launch training
echo "[4/4] Launching training container..."
echo ""
echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Iterations: 10"
echo "  Games per iteration: 100 (quick mode for testing)"
echo "  Expected duration: 2-4 hours on GPU, 8-12 hours on CPU"
echo ""
echo "Monitor with:"
echo "  docker logs -f hex-training-session"
echo "  tensorboard --logdir=runs"
echo ""
echo "To stop: docker stop hex-training-session"
echo ""
echo "=============================================================================="
echo ""

# Launch container with training
docker run \
    $GPU_FLAGS \
    --cpus=8 \
    --memory=8G \
    -v "$(pwd)":/home/hex \
    -w /home/hex \
    --name hex-training-session \
    --rm \
    hex-gpu \
    python3 train_neural_agent.py \
        --iterations 10 \
        --games 100 \
        --epochs 5 \
        --batch-size 128 \
        --simulations 400 \
        --device $DEVICE \
        --res-blocks 5 \
        --channels 128

echo ""
echo "=============================================================================="
echo "✅ TRAINING COMPLETE"
echo "=============================================================================="
echo "Model saved to: models/hex_model_best.pth"
echo ""
echo "Next steps:"
echo "  1. Test: python3 Hex.py -p1 \"agents.Group12.Group12Agent_neural Group12Agent\""
echo "  2. Compare: Compare with 50-game test results when ready"
echo "=============================================================================="
