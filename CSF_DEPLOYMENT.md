# Group12 Hex AI - CSF3 Deployment Guide

## Quick Start for r36859ak

### Step 1: Request GPU Access (DO THIS FIRST!)
**GPU access is NOT automatic!** Your jobs will wait forever without it.

1. Go to: https://ri.itservices.manchester.ac.uk/csf3/batch/gpu-jobs/
2. Submit a help ticket requesting GPU access
3. Mention: "Free-at-point-of-use V100 access for COMP34111 AI & Games coursework"
4. Wait for confirmation email (usually 1-2 days)

### Step 2: Login to CSF3
```bash
ssh r36859ak@csf3.itservices.manchester.ac.uk
# Enter your university password
```

### Step 3: Setup Project in Scratch
```bash
# Go to scratch (fast, temporary storage - run jobs from here)
cd ~/scratch

# Clone or copy project
# Option A: If you have git access
git clone <your-repo-url> COMP34111-AI-Games-Hex

# Option B: Copy from your local machine (run this locally)
scp -r COMP34111-AI-Games-Hex r36859ak@csf3.itservices.manchester.ac.uk:~/scratch/
```

### Step 4: Submit Training Job
```bash
cd ~/scratch/COMP34111-AI-Games-Hex

# Quick test (10 minutes)
sbatch submit_csf.sh test

# Development training (2-4 hours) - RECOMMENDED TO START
sbatch submit_csf.sh dev

# Standard training (8-12 hours)
sbatch submit_csf.sh standard

# Full training (24-48 hours)
sbatch submit_csf.sh full
```

### Step 5: Monitor Your Job
```bash
# Check job status
squeue

# Monitor GPU usage (replace JOBID with your job number)
gpustat -j JOBID

# Watch output in real-time
tail -f logs/hex_train_*.out

# Cancel a job
scancel JOBID
```

### Step 6: Get Results
```bash
# Results are saved to ~/scratch/hex_results_JOBID/
ls ~/scratch/hex_results_*/

# Copy best model for tournament use
cp ~/scratch/hex_results_*/models/hex_model_best.pth ~/scratch/COMP34111-AI-Games-Hex/models/

# Backup to home directory (important!)
cp -r ~/scratch/hex_results_* ~/
```

---

## CSF3 Key Information

### Storage Areas

| Area | Path | Backed Up | Quota | Use For |
|------|------|-----------|-------|---------|
| Home | `~` or `$HOME` | Yes | ~50GB shared | Jobscripts, code, important results |
| Scratch | `~/scratch` | **NO** | Large | Running jobs, temporary files |
| $TMPDIR | Node-local NVMe | **NO** | 1.6TB | Fast I/O during jobs |

**CRITICAL**: Scratch files unused for 3 months are auto-deleted! Always backup important results to home.

### GPU Resources

| GPU | Partition | Memory | CPUs/GPU | Access | Status |
|-----|-----------|--------|----------|--------|--------|
| V100 | `-p gpuV` | 16GB | Max 8 | - | **DISCONTINUED Oct 2025** |
| A100 | `-p gpuA` | 80GB | Max 12 | Available | **USE THIS** |
| L40S | `-p gpuL` | 48GB | Max 12 | Restricted | Limited access |

**NOTE:** V100 GPUs are no longer available. Use A100 (`-p gpuA -G 1`) instead.

### Time Limits
- Batch jobs: **4 days** maximum
- Interactive jobs: **1 day** maximum

### Common Commands
```bash
# Submit job
sbatch myjobscript.sh

# Check queue
squeue

# Cancel job
scancel JOBID

# GPU status
gpustat -j JOBID

# Check scratch usage
scrusage
```

---

## Training Configurations

| Mode | Iterations | Games/Iter | Est. Time (V100) | Expected Win Rate |
|------|-----------|------------|------------------|-------------------|
| test | 2 | 10 | 10 min | 55-60% |
| dev | 5 | 100 | 2-4 hours | 65-75% |
| standard | 10 | 200 | 8-12 hours | 75-85% |
| full | 20 | 500 | 24-48 hours | 85-90% |
| max | 30 | 1000 | 72+ hours | 90-95% |

---

## Troubleshooting

### Job stuck in queue forever
```bash
# Check if you have GPU access
squeue  # Look at REASON column
# If it says "(Resources)" - normal wait
# If it says "(QOSNotPermitted)" - you need GPU access!
```

### Out of GPU memory
Edit `train_csf.py` to reduce batch size:
```python
'batch_size': 128  # Instead of 256
```

### Job timeout
Use a longer time limit:
```bash
#SBATCH -t 4-0  # 4 days (maximum)
```

### Module not found
```bash
# Check available modules
module avail libs/cuda
module avail apps/binapps/anaconda3

# Load specific versions
module load libs/cuda/12.4.1
```

### Resume interrupted training
```bash
python train_csf.py --mode standard --resume models/hex_model_iter5.pth
```

---

## File Transfer

### Upload to CSF
```bash
# From your local machine
scp -r COMP34111-AI-Games-Hex r36859ak@csf3.itservices.manchester.ac.uk:~/scratch/
```

### Download from CSF
```bash
# From your local machine
scp -r r36859ak@csf3.itservices.manchester.ac.uk:~/scratch/hex_results_* ./
```

### Using VS Code Remote SSH
1. Install "Remote - SSH" extension in VS Code
2. Connect to: `r36859ak@csf3.itservices.manchester.ac.uk`
3. Open folder: `/mnt/iusers01/.../r36859ak/scratch/COMP34111-AI-Games-Hex`

---

## After Training

### Use Trained Model in Tournament
```bash
# Update cmd.txt to use neural agent
echo "agents.Group12.Group12Agent_neural Group12Agent" > agents/Group12/cmd.txt

# Copy trained model
cp models/hex_model_best.pth models/

# Test locally in Docker
docker build -t hex .
docker run --cpus=8 --memory=8G -v $(pwd):/home/hex hex \
    python3 Hex.py -p1 "agents.Group12.Group12Agent_neural Group12Agent" \
                   -p2 "agents.DefaultAgents.NaiveAgent NaiveAgent"
```

---

## Expected Results

### After `dev` training (2-4 hours):
- Model size: ~7 MB
- Network: 6 blocks, 128 channels
- Win rate vs heuristic: 65-75%
- Win rate vs NaiveAgent: 90%+

### After `standard` training (8-12 hours):
- Model size: ~20 MB
- Network: 10 blocks, 256 channels
- Win rate vs heuristic: 75-85%
- Tournament placement: Top 5-10

### After `full` training (24-48 hours):
- Model size: ~40 MB
- Network: 15 blocks, 256 channels
- Win rate vs heuristic: 85-90%
- Tournament placement: Top 3-5

---

## Contact & Help

- CSF Help: https://ri.itservices.manchester.ac.uk/csf3/overview/help/
- Group12 - COMP34111 AI & Games, University of Manchester
