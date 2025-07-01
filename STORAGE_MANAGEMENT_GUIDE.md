# Storage Management Guide

## Overview

The PLC Diagram Processor can consume significant disk space during training and processing. This guide explains where storage is used and how to manage it effectively.

## Current Storage Usage

Based on audit results, your system is using **13.2 GB** total:

- **Environments: 9.9 GB** (largest consumer)
- **Training runs: 2.4 GB** 
- **Models: 707.4 MB**
- **Datasets: 165.1 MB**
- **Processed files: 42.7 MB**
- **Raw files: 7.4 MB**
- **Temp workers: 902 B**

## Storage Locations

### 1. Training Runs (`../plc-data/runs/train/`)
**Size**: 2.4 GB  
**Content**: Each training run creates:
- `best.pt` and `last.pt` model files (~255 MB each)
- Training/validation images and plots
- Logs and metrics

**Problem**: Accumulates over time without cleanup

### 2. Virtual Environments (`environments/`)
**Size**: 9.9 GB  
**Content**: 
- `yolo_env/`: PyTorch + YOLO dependencies (~5 GB)
- `ocr_env/`: PaddleOCR + dependencies (~5 GB)

**Note**: These are necessary for multi-environment operation

### 3. Model Downloads (`../plc-data/models/`)
**Size**: 707.4 MB  
**Content**: Pretrained YOLO models and custom trained models

### 4. Dataset Cache Files
**Location**: Throughout `../plc-data/datasets/`  
**Content**: YOLO-generated `.cache` files for faster loading

### 5. Temporary Worker Files
**Location**: System temp directory  
**Pattern**: `plc_worker_*` folders  
**Content**: JSON communication files between processes

## Storage Management Tools

### 1. Storage Audit Tool

Check current storage usage:
```bash
python cleanup_storage.py --audit
```

### 2. Cleanup Tool

**Dry run** (shows what would be deleted):
```bash
python cleanup_storage.py --cleanup --keep-runs 2
```

**Actual cleanup** (deletes files):
```bash
python cleanup_storage.py --cleanup --keep-runs 2 --force
```

Options:
- `--keep-runs N`: Keep only the latest N training runs (default: 2)
- `--force`: Actually perform deletions (without this, it's a dry run)

### 3. Automatic Cleanup

#### Option A: Environment Variable
```bash
set PLCDP_AUTO_CLEANUP=1
python src/run_pipeline.py --stages training
```

#### Option B: Command Line Flag
```bash
python src/run_pipeline.py --stages training --auto-cleanup
```

#### Option C: Direct in Training Payload
When calling the training worker directly, add:
```json
{
  "action": "train",
  "auto_cleanup": true,
  ...
}
```

## Cleanup Strategies

### Conservative Cleanup (Recommended)
Keep the latest 2-3 training runs:
```bash
python cleanup_storage.py --cleanup --keep-runs 3 --force
```
**Saves**: ~1-2 GB

### Aggressive Cleanup
Keep only the latest training run:
```bash
python cleanup_storage.py --cleanup --keep-runs 1 --force
```
**Saves**: ~2-3 GB

### Cache Cleanup
Remove all YOLO cache files (they regenerate automatically):
```bash
python cleanup_storage.py --cleanup --force
```

## Prevention Strategies

### 1. Enable Auto-Cleanup
Add to your training commands:
```bash
python src/run_pipeline.py --stages training --auto-cleanup --minimal
```

### 2. Use Optimized Training Parameters
The system now automatically optimizes training parameters:
- **Epochs**: Capped at 20 (from 50)
- **Batch size**: Reduced to max 8 (from 16)
- **Patience**: Reduced to 10 (from 20)

### 3. Monitor Storage Regularly
Run periodic audits:
```bash
python cleanup_storage.py --audit
```

### 4. Use Minimal Output Mode
Reduces log file sizes:
```bash
python src/run_pipeline.py --run-all --minimal
```

## Emergency Storage Recovery

If you're running critically low on disk space:

### Immediate Actions (Frees ~3-4 GB)
```bash
# 1. Clean old training runs (keeps only latest)
python cleanup_storage.py --cleanup --keep-runs 1 --force

# 2. Clean cache files
find ../plc-data -name "*.cache" -delete

# 3. Clean temp workers
python cleanup_storage.py --cleanup --force
```

### Extreme Measures (Frees ~10+ GB)
```bash
# Remove one environment (you can recreate it later)
rmdir /s environments\ocr_env

# Or remove both environments (recreate with setup)
rmdir /s environments
```

## Storage Optimization Settings

### Environment Variables
```bash
# Enable auto-cleanup
set PLCDP_AUTO_CLEANUP=1

# Enable minimal output (reduces log sizes)
set PLCDP_MINIMAL_OUTPUT=1

# Reduce worker timeout (prevents hanging processes)
set PLC_WORKER_TIMEOUT=1800
```

### Training Optimizations
The system now automatically:
- Limits epochs to reasonable numbers
- Reduces batch sizes to prevent memory issues
- Disables unnecessary plot generation
- Enables mixed precision training
- Suppresses verbose YOLO output

## Monitoring Commands

### Quick Storage Check
```bash
python cleanup_storage.py --audit
```

### Detailed Training Runs Analysis
```bash
dir /s ..\plc-data\runs\train
```

### Environment Size Check
```bash
dir /s environments
```

## Best Practices

1. **Always use `--minimal` flag** for cleaner output and smaller logs
2. **Enable auto-cleanup** for training runs
3. **Run storage audits** before major training sessions
4. **Keep only 2-3 recent training runs** unless you need specific models
5. **Use optimized training parameters** (automatically applied)
6. **Monitor temp directories** for stuck worker processes

## Troubleshooting

### "Disk space low" errors
1. Run immediate cleanup: `python cleanup_storage.py --cleanup --keep-runs 1 --force`
2. Check for stuck temp workers: `python cleanup_storage.py --cleanup --force`
3. Consider removing one environment temporarily

### Training runs consuming too much space
1. Enable auto-cleanup: `--auto-cleanup` flag
2. Reduce training parameters: `--epochs 10 --batch-size 4`
3. Use minimal output: `--minimal`

### Cache files growing large
1. Delete cache files: `find ../plc-data -name "*.cache" -delete`
2. They will regenerate automatically on next run

## Summary

With the new storage management tools and optimizations:
- **Training is faster** (reduced parameters)
- **Output is cleaner** (minimal mode)
- **Storage is controlled** (auto-cleanup)
- **Monitoring is easy** (audit tool)

Use `python cleanup_storage.py --cleanup --keep-runs 2 --force` to immediately free up ~1 GB of space.
