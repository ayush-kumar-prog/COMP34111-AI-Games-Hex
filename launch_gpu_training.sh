#!/bin/bash
#
# GPU Training Launcher for Group12 Neural Hex AI
#
# This script launches full GPU training in Docker.
# Expected duration: 12-24 hours
#
# Usage:
#   ./launch_gpu_training.sh [OPTIONS]
#
# Options:
#   --iterations N     Number of training iterations (default: 10)
#   --games N          Games per iteration (default: 1000)
#   --simulations N    MCTS simulations per move (default: 800)
#   --quick            Quick training (100 games, 5 iterations)
#

set -e  # Exit on error

# Default parameters
ITERATIONS=10
GAMES=1000
EPOCHS=10
BATCH_SIZE=256
SIMULATIONS=800
RES_BLOCKS=10
CHANNELS=256

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --games)
            GAMES="$2"
            shift 2
            ;;
        --simulations)
            SIMULATIONS="$2"
            shift 2
            ;;
        --quick)
            ITERATIONS=5
            GAMES=100
            SIMULATIONS=400
            echo "Quick mode enabled: 5 iterations, 100 games, 400 simulations"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================================================="
echo "GROUP12 NEURAL HEX AI - GPU TRAINING LAUNCHER"
echo "=========================================================================="
echo "Configuration:"
echo "  Iterations: $ITERATIONS"
echo "  Games per iteration: $GAMES"
echo "  Epochs per iteration: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  MCTS simulations: $SIMULATIONS"
echo "  ResNet blocks: $RES_BLOCKS"
echo "  Channels: $CHANNELS"
echo ""
echo "Expected duration: 12-24 hours on GPU"
echo "=========================================================================="
echo ""

# Check if GPU Docker image exists
if ! docker images | grep -q hex-gpu; then
    echo "⚠️  GPU Docker image not found. Building now..."
    echo ""
    docker build -f Dockerfile.gpu --build-arg UID=$(id -u) -t hex-gpu .
    echo ""
    echo "✅ GPU Docker image built successfully"
    echo ""
fi

# Check NVIDIA Docker runtime
if ! docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu20.04 nvidia-smi &>/dev/null; then
    echo "❌ ERROR: NVIDIA Docker runtime not available"
    echo ""
    echo "Please ensure:"
    echo "  1. NVIDIA GPU is present"
    echo "  2. NVIDIA drivers are installed"
    echo "  3. nvidia-docker2 is installed"
    echo ""
    echo "Install with:"
    echo "  sudo apt-get install nvidia-docker2"
    echo "  sudo systemctl restart docker"
    echo ""
    exit 1
fi

echo "✅ GPU Docker runtime detected"
echo ""

# Create output directories
mkdir -p models
mkdir -p runs
mkdir -p data/self_play

echo "Starting GPU training container..."
echo ""
echo "To monitor training:"
echo "  1. TensorBoard: tensorboard --logdir=runs"
echo "  2. Watch logs: docker logs -f hex-training"
echo "  3. GPU usage: watch -n 1 nvidia-smi"
echo ""
echo "To stop training: docker stop hex-training"
echo ""
echo "=========================================================================="
echo ""

# Launch training in Docker
docker run \
    --runtime=nvidia \
    --gpus all \
    --cpus=8 \
    --memory=8G \
    -v "$(pwd)":/home/hex \
    -w /home/hex \
    --name hex-training \
    --rm \
    -it \
    hex-gpu \
    python3 train_neural_agent.py \
        --iterations $ITERATIONS \
        --games $GAMES \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --simulations $SIMULATIONS \
        --res-blocks $RES_BLOCKS \
        --channels $CHANNELS \
        --device cuda

echo ""
echo "=========================================================================="
echo "GPU TRAINING COMPLETE"
echo "=========================================================================="
echo "Model saved to: models/hex_model_best.pth"
echo ""
echo "To test the trained agent:"
echo "  python3 Hex.py -p1 \"agents.Group12.Group12Agent_neural Group12Agent\""
echo ""
echo "To compare with other agents:"
echo "  python3 run_comprehensive_tests.py --games 50"
echo "=========================================================================="
