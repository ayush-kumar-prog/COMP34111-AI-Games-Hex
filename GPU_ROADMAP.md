# GPU-Accelerated Hex AI - Implementation Roadmap

**Created**: November 21, 2024
**Goal**: Transform CPU-only agent into championship-level GPU-accelerated neural agent
**Timeline**: 1 week implementation + 12-24 hours training
**Expected Result**: Top 1-3 tournament placement

---

## 🎯 PROJECT VISION

Transform our current CPU-only implementation into an AlphaZero-style neural network agent that leverages available GPU resources (CUDA 12.3, PyTorch 2.5.1, TensorFlow 2.19.0).

### Performance Comparison

| Metric | Current (CPU) | Target (GPU) | Improvement |
|--------|--------------|--------------|-------------|
| MCTS Iterations | 200/move | 50,000/move | 250x |
| Move Selection | Random rollouts | Neural policy | Smarter |
| Evaluation | Hand-crafted heuristics | Learned features | Better |
| Win Rate vs Simple | 20% | 95%+ | 4.75x |
| Tournament Placement | Top 5-10 | Top 1-3 | Championship |

---

## 📋 IMPLEMENTATION PHASES

### Phase 1: Neural Network Architecture (2-4 hours)

**File**: `agents/Group12/neural/hex_network.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HexResBlock(nn.Module):
    """Residual block for hex network"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class HexNeuralNetwork(nn.Module):
    """AlphaZero-style network for Hex"""
    def __init__(self, board_size=11, num_res_blocks=10, num_channels=256):
        super().__init__()
        self.board_size = board_size

        # Input: 11×11×5 tensor
        # Channel 0: Our stones (1 or 0)
        # Channel 1: Opponent stones (1 or 0)
        # Channel 2: Empty cells (1 or 0)
        # Channel 3: Legal moves mask (1 or 0)
        # Channel 4: Edge distances (normalized)

        # Initial convolution
        self.conv_input = nn.Conv2d(5, num_channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            HexResBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy head (move probabilities)
        self.policy_conv = nn.Conv2d(num_channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, board_size * board_size)

        # Value head (position evaluation)
        self.value_conv = nn.Conv2d(num_channels, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Input processing
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value
```

**Checklist**:
- [ ] Create `agents/Group12/neural/` directory
- [ ] Implement `HexNeuralNetwork` class
- [ ] Add board state encoding function
- [ ] Test forward pass with dummy data
- [ ] Verify GPU utilization with `nvidia-smi`

---

### Phase 2: GPU-Accelerated MCTS (4-6 hours)

**File**: `agents/Group12/algorithms/mcts_neural.py`

**Key Features**:
1. **Batch Inference**: Process multiple positions simultaneously
2. **Neural Guidance**: Use network policy for selection/expansion
3. **Virtual Loss**: Enable parallel tree traversal
4. **GPU Rollouts**: Batch simulations on GPU

```python
class NeuralMCTS:
    def __init__(self, network, device='cuda'):
        self.network = network
        self.device = device
        self.network.to(device)
        self.network.eval()

    def search_batch(self, boards, time_limit):
        """Run MCTS on multiple boards simultaneously"""
        # Batch encode boards
        states = torch.stack([self.encode_board(b) for b in boards])
        states = states.to(self.device)

        # Get neural network predictions
        with torch.no_grad():
            policies, values = self.network(states)

        # Use policies to guide MCTS selection
        # Run 10,000-50,000 iterations (feasible with GPU!)
        ...
```

**Checklist**:
- [ ] Implement `NeuralMCTS` class
- [ ] Add batch processing for positions
- [ ] Integrate neural policy into UCB selection
- [ ] Add virtual loss for parallel search
- [ ] Benchmark: target 10,000+ iterations/second

---

### Phase 3: Self-Play Infrastructure (4-6 hours)

**File**: `agents/Group12/training/self_play.py`

**Goal**: Generate training data from self-play games

```python
class SelfPlayWorker:
    def __init__(self, network, num_games=1000):
        self.network = network
        self.mcts = NeuralMCTS(network)
        self.training_data = []

    def generate_game(self):
        """Play one game: Neural MCTS vs Neural MCTS"""
        board = Board(11)
        game_data = []

        while not board.has_ended():
            # Run MCTS guided by neural network
            move, policy_target = self.mcts.search(board)

            # Store training example
            state = self.encode_board(board)
            game_data.append((state, policy_target, None))  # outcome added later

            # Make move
            board.make_move(move)

        # Add game outcome to all positions
        winner = board.get_winner()
        for i, (state, policy, _) in enumerate(game_data):
            value = 1 if winner == self.colour else -1
            game_data[i] = (state, policy, value)

        return game_data

    def generate_dataset(self, num_games=10000):
        """Generate large dataset via self-play"""
        for game_num in range(num_games):
            game_data = self.generate_game()
            self.training_data.extend(game_data)

            if game_num % 100 == 0:
                print(f"Generated {game_num} games, {len(self.training_data)} positions")

        return self.training_data
```

**Checklist**:
- [ ] Implement `SelfPlayWorker` class
- [ ] Add multiprocessing for parallel game generation
- [ ] Create data storage format (HDF5 or PyTorch tensor)
- [ ] Target: 10,000+ games in 4-6 hours on GPU

---

### Phase 4: Training Loop (12-24 hours GPU time)

**File**: `agents/Group12/training/train.py`

```python
class HexTrainer:
    def __init__(self, network, learning_rate=0.001):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        self.device = 'cuda'

    def train_epoch(self, training_data, batch_size=256):
        """Train network on self-play data"""
        self.network.train()

        dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        total_loss = 0
        for states, policy_targets, value_targets in dataloader:
            states = states.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)

            # Forward pass
            policy_pred, value_pred = self.network(states)

            # Compute losses
            policy_loss = F.nll_loss(policy_pred, policy_targets)
            value_loss = F.mse_loss(value_pred, value_targets)
            total_loss_batch = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()

            total_loss += total_loss_batch.item()

        return total_loss / len(dataloader)

    def train(self, num_iterations=10):
        """Full training loop with self-play iteration"""
        for iteration in range(num_iterations):
            print(f"Iteration {iteration+1}/{num_iterations}")

            # Generate self-play data
            print("Generating self-play games...")
            self_play = SelfPlayWorker(self.network)
            training_data = self_play.generate_dataset(num_games=1000)

            # Train on data
            print("Training network...")
            for epoch in range(10):
                loss = self.train_epoch(training_data)
                print(f"  Epoch {epoch+1}, Loss: {loss:.4f}")

            # Save checkpoint
            torch.save(self.network.state_dict(),
                      f'models/hex_network_iter_{iteration}.pth')
```

**Checklist**:
- [ ] Implement training loop with policy + value loss
- [ ] Add learning rate scheduling
- [ ] Implement model checkpointing
- [ ] Monitor training metrics (loss, accuracy)
- [ ] Run for 12-24 hours to convergence

---

### Phase 5: Integration and Testing (2-4 hours)

**File**: `agents/Group12/Group12Agent_neural.py`

```python
class Group12AgentNeural(AgentBase):
    """Neural network + MCTS hybrid agent"""

    def __init__(self, colour: Colour, model_path='models/best_model.pth'):
        super().__init__(colour)

        # Load neural network
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.network = HexNeuralNetwork().to(self.device)
        self.network.load_state_dict(torch.load(model_path))
        self.network.eval()

        # Initialize neural MCTS
        self.mcts = NeuralMCTS(self.network, self.device)

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """Make move using neural network + MCTS"""

        # Handle swap
        if turn == 2 and opp_move:
            if self._should_swap_neural(board, opp_move):
                return Move(-1, -1)

        # Use neural MCTS (50,000 iterations feasible!)
        move = self.mcts.search(board, time_limit=2*10**9)

        return move
```

**Testing Plan**:
```bash
# Test 1: Neural vs Simple (should win 95%+)
python3 test_neural_vs_simple.py --games 50

# Test 2: Neural vs Full (should dominate)
python3 test_neural_vs_full.py --games 50

# Test 3: Neural vs MCTSAgent binary
python3 Hex.py -p1 "agents.Group12.Group12Agent_neural Group12Agent" \
               -p2 "agents.MCTSAgent.MCTSAgent MCTSAgent"
```

**Checklist**:
- [ ] Create `Group12Agent_neural.py`
- [ ] Test GPU inference speed
- [ ] Run 50 games vs Simple agent
- [ ] Run 50 games vs Full agent
- [ ] Verify tournament readiness

---

## 📊 EXPECTED OUTCOMES

### Performance Targets

| Benchmark | Current (CPU) | Target (GPU) | Status |
|-----------|--------------|--------------|--------|
| MCTS iterations/second | 100 | 10,000+ | ⏳ |
| Win vs Simple | 20% | 95%+ | ⏳ |
| Win vs Full | 50% | 99%+ | ⏳ |
| Win vs MCTSAgent | 30%? | 80%+ | ⏳ |
| Tournament placement | Top 5-10 | Top 1-3 | ⏳ |

### Training Metrics

- **Games generated**: 10,000-50,000
- **Training time**: 12-24 hours
- **Model size**: ~50-100 MB
- **Inference time**: <0.1s per position
- **GPU memory**: ~2-4 GB

---

## 🚀 GETTING STARTED

### Step 1: Restore GPU Dockerfile

```dockerfile
FROM nvidia/cuda:12.3.0-runtime-ubuntu20.04

# Install Python 3.11
RUN apt-get update && apt-get install -y python3.11

# Install PyTorch with CUDA
RUN pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install numpy scipy tensorboard
```

### Step 2: Test GPU Access

```bash
# Build with GPU support
docker build --build-arg UID=$UID -t hex-gpu .

# Run with GPU
docker run --runtime=nvidia --gpus all \
           --cpus=8 --memory=8G \
           -v $(pwd):/home/hex \
           --rm -it hex-gpu /bin/bash

# Verify GPU
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Step 3: Create Directory Structure

```bash
mkdir -p agents/Group12/neural
mkdir -p agents/Group12/training
mkdir -p models
mkdir -p data/self_play
```

### Step 4: Begin Implementation

Follow the phases above in order. Each phase builds on the previous one.

---

## 📚 RESOURCES

### Papers
- AlphaGo Zero (Silver et al., 2017)
- MoHex: Deep Hexplayer (Huang et al., 2013)
- Mastering the game of Hex (Browne, 2000)

### Code References
- PyTorch AlphaZero implementation
- Hex-specific MCTS examples
- ResNet architecture patterns

### Tools
- `nvidia-smi`: Monitor GPU utilization
- `tensorboard`: Visualize training
- `torch.profiler`: Profile GPU performance

---

## ⚠️ RISKS AND MITIGATION

### Risk 1: Insufficient Training Time
**Mitigation**: Start with smaller network (5 ResBlocks instead of 10)

### Risk 2: GPU Memory Overflow
**Mitigation**: Reduce batch size, use gradient checkpointing

### Risk 3: Poor Self-Play Quality
**Mitigation**: Bootstrap with heuristic agent, gradually transition to neural

### Risk 4: Tournament Time Limit
**Mitigation**: Optimize inference, use TorchScript compilation

---

## 🎯 SUCCESS CRITERIA

**Minimum Viable Product**:
- [ ] Neural network trains without errors
- [ ] GPU inference works correctly
- [ ] Beats Simple agent 80%+
- [ ] Stays under 3-minute time limit

**Stretch Goals**:
- [ ] Beats Simple agent 95%+
- [ ] 50,000 MCTS iterations per move
- [ ] Top 3 tournament placement
- [ ] Superhuman Hex play

---

**Status**: 🟡 Planning Phase
**Next Action**: Restore GPU Dockerfile and test GPU access
**Estimated Completion**: 1 week from start
