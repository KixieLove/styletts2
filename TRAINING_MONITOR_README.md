# StyleTTS2 Training Progress Monitor

A comprehensive Jupyter notebook for visualizing and understanding your model's training progress.

## 📊 What's Included

### 1. **Main Loss Trend (Per Epoch)**
   - Shows your **Mel Loss** - the most important metric
   - Displays improvement percentage from start to current
   - Helps you see if the model is actually learning

### 2. **All Loss Components**
   - Visualizes all 6 loss components:
     - **Mel Loss**: Reconstruction loss (should decrease)
     - **Gen Loss**: Generator adversarial loss
     - **Disc Loss**: Discriminator loss  
     - **Mono Loss**: Monotonic alignment loss
     - **S2S Loss**: Sequence-to-sequence loss
     - **SLM Loss**: Speech Language Model loss
   - Shows when each loss becomes active

### 3. **Training Stability & Variance**
   - Shows raw loss vs smoothed (rolling average)
   - Indicates if training is stable or noisy
   - Helps detect numerical issues early

### 4. **Automatic Health Diagnosis**
   - ✅ Checks if loss is decreasing
   - ✅ Verifies training stability
   - ✅ Detects sudden spikes
   - ✅ Identifies training stage (Stage 1 vs Stage 2)
   - ✅ Gives an overall "healthy" or "warning" assessment

### 5. **Training Patterns Guide**
   - Explains what "good training" looks like
   - Lists warning signs to watch for
   - Explains what's normal at each stage

### 6. **Detailed Epoch Comparison Table**
   - Summary of last 15 epochs
   - Shows min/max/average for each loss
   - Identifies when Stage 2 started

### 7. **Convergence Analysis**
   - Shows epoch-to-epoch changes
   - Indicates if model is converging
   - Displays improvement rate

### 8. **Quick Reference Guide**
   - What to look for in good training
   - What to do if training isn't going well
   - Troubleshooting tips

## 🎯 How to Use

### Before Each Training Session
```
1. Open the notebook: Training_Progress_Monitor.ipynb
2. Run all cells (Kernel → Restart & Run All)
3. Review the "TRAINING HEALTH DIAGNOSIS" section
4. Check if status shows ✅ HEALTHY or ⚠️ WARNING
```

### If You Change Models
```
Update this line in the first data cell:
LOG_DIR = r"d:\path\to\your\model\logs"
```

### What to Monitor
- **Mel Loss** should steadily decrease in Stage 1 (first ~50 epochs)
- After Stage 2 starts (~epoch 50), Mel Loss may increase slightly - this is OK
- **Loss should be smooth**, not spiky
- **Recent epochs** should still show improvement (not plateaued)

## 📈 Understanding Your Current Training

Based on your logs (as of the notebook's last run):

✅ **Status: HEALTHY**

- **Mel Loss Improvement**: 19.9% (from 0.2410 to 0.1930)
- **Training Stability**: Very stable (4.3% noise ratio)
- **Training Stage**: Stage 2 just started at epoch 51
- **Overall**: Model is learning well! ✅

### Recent Progress (Last 15 Epochs)
- Mel Loss trend: Consistently around 0.19
- All other losses still inactive (0.0)
- No sudden spikes or anomalies

### Stage 2 Activation (Epoch 51)
When Stage 2 started, you can see:
- Gen Loss: 4.64
- Disc Loss: 5.21
- Mono Loss: 0.056
- S2S Loss: 10.95
- SLM Loss: 1.69

This is expected and normal! The model is now training adversarial objectives.

## ⚠️ Warning Signs to Watch For

### 🔴 CRITICAL
- Mel Loss increasing continuously
- Sudden large spikes (loss jumps 10x+)
- Loss stuck at exact same value for 10+ epochs
- NaN or Inf values in logs

### 🟡 CAUTION  
- Mel Loss plateaued for 20+ epochs with no improvement
- Very noisy loss (variance > 50% of mean)
- Other losses stuck at 0 when they should be active

### 🟢 NORMAL
- Mel Loss increases slightly when Stage 2 starts
- Some fluctuation in early epochs
- Different losses activating at different times

## 💡 Tips for Better Training

### If Loss is Decreasing Too Slowly
- Increase learning rate slightly
- Check data quality/preprocessing
- Verify batch size is reasonable (usually 10-20)

### If Loss is Very Noisy/Unstable
- Reduce learning rate
- Increase batch size
- Check for gradient clipping in config

### If Losses Won't Activate
- Check config file parameters:
  - `diff_epoch`: when diffusion training starts
  - `joint_epoch`: when adversarial training starts
- Verify your training script is using these parameters

## 📝 Interpreting Loss Values

For StyleTTS2 on your data:
- **Good Mel Loss**: < 0.30
- **Excellent Mel Loss**: < 0.20
- **Starting Mel Loss**: Usually 0.40-1.50

These are approximate - your values depend on:
- Data length/complexity
- Preprocessing parameters
- Model configuration
- Training data quality

## 🔄 Running the Monitor Regularly

### Recommended Schedule
- **Daily**: If training is ongoing
- **Weekly**: If training is progressing slowly
- **After Config Changes**: Always verify impact
- **When Suspicious**: If training behavior changes suddenly

### What to Track
Create a simple log of key metrics:
```
Date    | Epoch | Mel Loss | Status
--------|-------|----------|--------
2025-01-01 | 10   | 0.450   | Normal
2025-01-02 | 20   | 0.300   | Improving
...
```

## 🛠️ Troubleshooting

### Notebook Won't Run
- Check that the log file path is correct
- Verify `d:\Tesis\StyleTTS2\logs\angelina_es\train.log` exists
- Make sure you have latest training logs

### Graphs Not Showing
- Check that matplotlib backend is configured
- Try restarting the Jupyter kernel
- Make sure all cells execute without errors

### Want More Details?
- Check the `train.log` file directly for full details
- Look at TensorBoard: `tensorboard --logdir logs/angelina_es/tensorboard`
- Compare with previous training runs

## 📚 Reference

See also:
- `COMPREHENSIVE_ARCHITECTURE_DOCUMENTATION.md` - Model details
- `train_finetune.py` - Training script
- `Configs/config_ft.yml` - Your training configuration

---

**Last Updated**: December 3, 2025
**For**: StyleTTS2 Spanish Fine-tuning (angelina_es)
