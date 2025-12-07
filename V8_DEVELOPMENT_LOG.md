# V8 Neural Network Development Log

## Project Context
- **Goal**: Train a neural network for Hex that can beat our MCTS+RAVE tournament agent
- **Constraint**: Tournament runs on CPU-only Docker (8 CPUs, 8GB RAM, 5-min game limit)
- **Platform**: CSF3 HPC cluster for training/data generation

---

## V6 Failure Analysis (What Went Wrong)

### The Problem
V6 training produced a model where the **value head learned nothing useful**.

**Root Cause**: All training games were won by RED (first player advantage + deterministic MCTS).

This created training data with values like:
```
Position 1 (RED to move): value = +1 (RED wins)
Position 2 (BLUE to move): value = -1 (BLUE loses)
Position 3 (RED to move): value = +1 (RED wins)
...
```

The value head learned to predict based on **whose turn it is**, not the actual board position. It essentially learned: "if RED to move, predict +1; if BLUE to move, predict -1".

### Evidence
From V5 checkpoint data analysis:
```python
Values min/max/mean: -1.0 1.0 0.0
Unique values: [-1.  1.]
Value distribution:
  -1: 72,506
  0: 0
  +1: 72,594
```
Perfect 50/50 split of +1/-1, alternating every move, because ALL games were won by RED.

---

## V8 Solution: Diverse Outcomes via Random Openings

### Key Insight
To train a useful value head, we need games with **diverse outcomes** (~50/50 Red/Blue wins).

### Implementation
**Random Opening Phase**: Play 3-7 random moves before MCTS takes over.
- This creates varied board states
- Breaks first-player advantage
- Results in ~50% Red wins, ~50% Blue wins

### V8 Data Generator Configuration
```python
BOARD_SIZE = 11
MCTS_ITERATIONS = 200      # Reduced from 400 for speed
RANDOM_OPENING_MIN = 3     # Minimum random opening moves
RANDOM_OPENING_MAX = 7     # Maximum random opening moves
NUM_GAMES = 12000          # Target games
NUM_WORKERS = 12           # Parallel workers
CHECKPOINT_EVERY = 300     # Save progress frequently
```

---

## Timeline of V8 Data Generation Attempts

### Attempt 1: 400 MCTS Iterations (Job 9393456)
- **Submitted**: Dec 5, 2025 ~12:50
- **Configuration**: 400 iterations, 12K games, 12 workers
- **Problem**: Way too slow

**Observations after 30 minutes:**
- 0 games completed
- Workers at 99% CPU (not deadlocked, just slow)
- Each game taking ~60+ seconds
- First batch (300 games) not finished

**Time Estimate Evolution** (this was bad):
- Initial estimate: 5-6 hours
- Revised to: 10-12 hours
- Revised to: 16-17 hours
- Final reality: ~28 hours

**Decision**: Cancel and reduce iterations.

### Attempt 2: 200 MCTS Iterations (Job 9393829)
- **Submitted**: Dec 5, 2025 13:19
- **Configuration**: 200 iterations, 12K games, 12 workers

**Observations after 23 minutes:**
```
Status: "starting"
Games completed: 0
First batch: Processing games 0 to 300...
Workers: All 12 initialized
```

**Why still slow?**
MCTS time isn't linear with iterations. Fixed overhead per game:
- Board copies
- Random rollouts (~50 moves each, regardless of iteration count)
- Win checking with flood fill

200 iterations ≈ 10-15% faster than 400, not 50% faster.

**Current estimate for 12K games**: ~15 hours

---

## Performance Analysis

### MCTS Time Breakdown (per game)
A typical game has ~50 moves. Each move requires:
1. **MCTS search**: N iterations × (select + expand + rollout + backprop)
2. **Rollout**: ~50 random moves to terminal state
3. **Win check**: Flood fill algorithm

With 200 iterations per move:
- 50 moves × 200 iterations = 10,000 MCTS iterations per game
- Each iteration includes a ~50-move random rollout
- Total: ~500,000 board operations per game

**Measured speed**: ~55-60 seconds per game (200 iterations)

### Scaling Options

| Config | Iterations | Games | Est. Time | Quality |
|--------|-----------|-------|-----------|---------|
| Current | 200 | 12,000 | ~15 hours | Good |
| Faster | 100 | 12,000 | ~8-10 hours | Acceptable |
| Fewer games | 200 | 6,000 | ~7-8 hours | Good |
| Minimal | 100 | 6,000 | ~4-5 hours | Acceptable |

---

## Files on CSF3

```
~/scratch/COMP34111-AI-Games-Hex/
├── agents/Group12/training/
│   └── fast_data_gen_v8.py    # V8 data generator
├── submit_v8_datagen.sh        # SLURM job script
├── data/
│   ├── v8_datagen_status.json  # Real-time progress
│   └── expert_games_v8.npz     # Output (when complete)
└── v8_datagen_9393829.out      # Job output log
```

### SLURM Job Configuration
```bash
#SBATCH --job-name=v8_fast_datagen
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
```

---

## Monitoring Commands

```bash
# SSH to CSF3
ssh r36859ak@csf3.itservices.manchester.ac.uk
# Password: Ayushkumar040612!
# Duo: Type 1

# Check job status
squeue -u r36859ak

# Check progress (real-time JSON)
cat data/v8_datagen_status.json

# Check output log
tail -50 v8_datagen_9393829.out

# Cancel job if needed
scancel 9393829
```

---

## Lessons Learned

### 1. Time Estimation
- MCTS has significant fixed overhead per game
- Reducing iterations by 50% ≠ 50% faster
- Always benchmark a small batch before estimating total time

### 2. Value Head Training
- Diverse outcomes are CRITICAL
- All-wins-for-one-color data is useless
- Random openings solve this problem

### 3. Data Generation Architecture
- `pool.map()` waits for ALL items in batch before returning
- First update only appears after slowest worker finishes its batch
- Smaller batches = more frequent progress updates (but more overhead)

---

## Next Steps

1. **Wait for V8 data generation** (~15 hours at current pace)
   - Or cancel and reconfigure for faster completion

2. **Train V8 model** (once data is ready)
   - Policy head: predict MCTS move distribution
   - Value head: predict game outcome from position

3. **Test V8 agent** vs MCTS+RAVE tournament agent

---

## Current Status (Dec 5, 2025 13:42 GMT)

- **Job 9393829**: Running, 23 minutes elapsed
- **Progress**: First batch (0-300) still processing
- **Estimated completion**: ~15 hours from start (~04:19 GMT Dec 6)
- **Decision needed**: Let it run, or cancel and reconfigure?
