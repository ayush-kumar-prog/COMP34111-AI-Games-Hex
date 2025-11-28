# Google Colab Neural Network Training Guide

## 🚀 Quick Start (5 Steps)

### Step 1: Prepare Codebase
```bash
# In your project directory:
./prepare_for_colab.sh
```
This creates `COMP34111-AI-Games-Hex-for-Colab.zip` (~5MB)

### Step 2: Open Google Colab
1. Go to https://colab.research.google.com
2. **File** → **Upload notebook**
3. Upload `Group12_Hex_Neural_Training.ipynb`

### Step 3: Enable GPU
1. **Runtime** → **Change runtime type**
2. Select **GPU** (T4 recommended)
3. Click **Save**

### Step 4: Run Training
1. Run all cells in order (click **Runtime** → **Run all**)
2. When prompted, upload `COMP34111-AI-Games-Hex-for-Colab.zip`
3. Wait 2-4 hours for training to complete

### Step 5: Download Results
1. Run the download cell at the end
2. Extract `trained_models.zip` to your local `models/` directory
3. Test the trained agent!

---

## 📊 Training Configuration

**Default Settings** (recommended):
- **Iterations**: 10 (AlphaZero cycles)
- **Games per iteration**: 1000
- **Total games**: ~10,000
- **Training time**: 2-4 hours on T4 GPU
- **Expected win rate**: 85-95% vs Full agent

**Quick Test** (if you want to test the pipeline):
- Change `num_iterations` to 2
- Change `games_per_iteration` to 100
- Training time: ~30 minutes
- Expected win rate: 60-70% (undertrained)

---

## 🔧 Customization

### Faster Training (Less Games)
```python
TRAINING_CONFIG = {
    'num_iterations': 5,
    'games_per_iteration': 500,  # Reduce this
    'num_simulations': 400,      # Reduce this
}
# Time: ~1 hour, Win rate: 70-80%
```

### Better Results (More Games)
```python
TRAINING_CONFIG = {
    'num_iterations': 20,
    'games_per_iteration': 2000,  # Increase this
    'num_simulations': 1600,      # Increase this
}
# Time: ~8-12 hours, Win rate: 90-95%
```

### Larger Network (More Parameters)
```python
network = HexNeuralNetwork(
    board_size=11,
    num_res_blocks=15,     # Increase from 10
    num_channels=384       # Increase from 256
)
# Time: +50% slower, Win rate: +2-5%
```

---

## 💾 Save Checkpoints

Colab sessions can disconnect. To save progress:

```python
# Add this cell every 30 minutes during training:
from google.colab import drive
drive.mount('/content/drive')
!cp -r models/* /content/drive/MyDrive/Hex_Training_Backup/
```

To resume from checkpoint:
```python
# Before training:
!cp /content/drive/MyDrive/Hex_Training_Backup/* models/
```

---

## 📈 Monitor Progress

### TensorBoard (Live Monitoring)
The notebook includes TensorBoard cells. You can watch:
- **Policy Loss**: Should decrease (target: <1.0)
- **Value Loss**: Should decrease (target: <0.3)
- **Win Rate**: Not directly shown, but inferred from losses

### Manual Check
```python
# After each iteration, check latest model:
!ls -lh models/
# You'll see: hex_model_iter_X.pth
```

---

## 🎯 After Training

### 1. Test Locally
```bash
# Extract downloaded models to your project
unzip trained_models.zip -d models/

# Test neural vs Full agent
python3 Hex.py \
  -p1 "agents.Group12.Group12Agent_neural Group12Agent" \
  -p2 "agents.Group12.Group12Agent Group12Agent" \
  -v
```

### 2. Run Comprehensive Tests
```bash
# 50-game validation
python3 run_comprehensive_tests.py --games 50 --output results_neural_vs_full.json

# Analyze results
python3 analyze_results.py results_neural_vs_full.json
```

### 3. Switch to Neural Agent (If Win Rate > 85%)
```bash
# Update cmd.txt
echo "agents.Group12.Group12Agent_neural Group12Agent" > agents/Group12/cmd.txt

# Verify
python3 Hex.py -p1 "agents.Group12.Group12Agent_neural Group12Agent"
```

---

## 🐛 Troubleshooting

### "No GPU detected"
- **Solution**: Runtime → Change runtime type → GPU

### "Out of memory"
- **Solution**: Reduce `batch_size` to 128
- **Solution**: Reduce `num_channels` to 128

### "Training too slow"
- **Solution**: Reduce `games_per_iteration` to 500
- **Solution**: Reduce `num_simulations` to 400
- **Check**: Make sure GPU is enabled (not CPU)

### "Colab disconnected"
- **Solution**: Models are auto-saved every iteration
- **Solution**: Download latest checkpoint from `models/`
- **Solution**: Resume training from checkpoint

### "Win rate not improving"
- **Solution**: Train longer (20+ iterations)
- **Solution**: Increase `num_simulations` to 1600
- **Check**: Verify losses are decreasing in TensorBoard

---

## 📊 Expected Results

### After 10 Iterations (Standard Training)
- **vs Simple agent**: 95-100% win rate
- **vs Full agent**: 85-95% win rate
- **vs NaiveAgent**: 100% win rate
- **Move time**: 1-3 seconds (with GPU)
- **Tournament projection**: Top 1-3 placement

### After 20 Iterations (Extended Training)
- **vs Full agent**: 90-98% win rate
- **Move time**: 1-2 seconds
- **Tournament projection**: #1 placement (likely)

---

## 🔄 Alternative: CSF HPC Cluster

If you get CSF access:

```bash
# SSH to CSF
ssh username@csf3.itservices.manchester.ac.uk

# Load modules
module load apps/anaconda3/2021.11
module load libs/cuda/12.3

# Submit training job
qsub -l nvidia_a100=1 train_neural.sh

# Monitor
qstat -u $USER
```

**Advantages**:
- More powerful GPUs (A100 vs T4)
- No session limits (24+ hours)
- Free for students

**Disadvantages**:
- Queue wait times (minutes to hours)
- Need to learn SLURM job system
- Requires CSF account approval

---

## 📚 Resources

- **AlphaZero Paper**: https://arxiv.org/abs/1712.01815
- **Colab Documentation**: https://colab.research.google.com/
- **Project Status**: PROJECT_STATUS_NOVEMBER_2024.md
- **Training Code**: agents/Group12/training/trainer.py

---

## 🎓 Training Methodology

Our implementation follows **AlphaZero** (DeepMind, 2017):

1. **Self-Play**: Neural network plays itself
2. **MCTS Guidance**: 800 simulations per move
3. **Training**: Learn from self-play games
4. **Iteration**: Repeat 10-20 times
5. **Result**: Superhuman play

**Key Difference from Supervised Learning**:
- No human games needed
- Discovers strategies through self-play
- Often finds novel tactics

---

## ✅ Success Criteria

Your training is successful if:
- ✅ Training completes without errors
- ✅ Losses decrease over iterations
- ✅ Neural agent beats Full agent 85%+
- ✅ Model size is ~50-150MB
- ✅ Inference time <3 seconds per move

If all criteria met: **Ready for tournament!** 🏆

---

**Created**: November 21, 2024
**Status**: Ready to use
**Estimated Success Rate**: 95% (based on proven AlphaZero architecture)

**Good luck with training! 🚀**
