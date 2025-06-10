# Complete Pipeline Training Steps

## Prerequisites

1. **Ensure you have a downloaded YOLO11 model**:
   ```bash
   # The setup should have already downloaded yolo11m.pt to data/models/pretrained/
   # If not, run:
   python setup/setup.py --download-models
   ```

2. **Verify your dataset is properly configured**:
   ```bash
   # Check that data/dataset/plc_symbols.yaml exists and points to correct paths
   # Verify training/validation data exists in data/dataset/train/ and data/dataset/valid/
   ```

3. **Activate your environment**:
   ```bash
   # Windows
   activate.bat
   
   # Linux/Mac
   source activate.sh
   ```

## Running the Complete Pipeline with Training

### Option 1: Full Pipeline (Recommended)
```bash
# Run complete pipeline with training (10 epochs by default, auto-selects best model)
python src/detection/run_complete_pipeline.py --epochs 10 --parallel

# Use specific model (faster training with nano model)
python src/detection/run_complete_pipeline.py --model yolo11n.pt --epochs 10 --parallel

# Use medium model for better accuracy
python src/detection/run_complete_pipeline.py --model yolo11m.pt --epochs 50 --parallel

# Use large model for best accuracy (slower)
python src/detection/run_complete_pipeline.py --model yolo11l.pt --epochs 50 --parallel
```

### Check Available Models
```bash
# List all available models
python src/detection/run_complete_pipeline.py --list-models

# This will show:
# - Pretrained models (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)
# - Custom/trained models
# - Recommended auto-selected model
# - Usage examples
```

### Option 2: Step-by-Step Execution

#### Step 1: Train the Model
```bash
# Train YOLO11 model
python src/detection/yolo11_train.py --epochs 10

# The trained model will be saved to:
# data/runs/train/plc_symbol_detector_yolo11m_*/weights/best.pt
```

#### Step 2: Run Detection Pipeline
```bash
# Use the trained model for detection (replace with actual path)
python src/detection/run_complete_pipeline.py --skip-training --parallel
```

### Option 3: Advanced Configuration
```bash
# Full pipeline with custom settings
python src/detection/run_complete_pipeline.py \
    --epochs 20 \
    --conf 0.3 \
    --batch-size 16 \
    --workers 6 \
    --parallel \
    --snippet-size 1500 1200 \
    --overlap 500
```

## Pipeline Configuration Options

### Training Parameters
- `--epochs`: Number of training epochs (default: 10)
- `--conf`: Detection confidence threshold (default: 0.25)

### Performance Options
- `--parallel`: Use parallel GPU processing (recommended)
- `--unified`: Use unified parallel pipeline (alternative to --parallel)
- `--batch-size`: GPU batch size (default: 32)
- `--workers`: Number of parallel workers (default: 4)

### Processing Options
- `--snippet-size`: Image snippet size (default: 1500 1200)
- `--overlap`: Snippet overlap (default: 500)
- `--skip-pdf-conversion`: Skip PDF to image conversion if images exist
- `--streaming`: Enable streaming mode for lower memory usage

### Skip Options
- `--skip-training`: Use existing trained model instead of training new one

## Expected Output

The pipeline will create:
```
data/
├── processed/
│   ├── images/           # PDF snippets
│   └── detdiagrams/      # Detection results
│       ├── *_detected.pdf        # PDFs with detection overlays
│       ├── *_detections.json     # Detection coordinates
│       ├── *_coordinates.txt     # Coordinate files
│       ├── *_statistics.json     # Detection statistics
│       └── pipeline_summary.json # Overall summary
└── runs/
    └── train/
        └── plc_symbol_detector_yolo11m_*/  # Training results
            ├── weights/
            │   ├── best.pt       # Best trained model
            │   └── last.pt       # Last checkpoint
            └── results.png       # Training metrics
```

## Performance Expectations

Based on benchmarks:
- **Training**: ~5-30 minutes depending on epochs and dataset size
- **Detection**: ~72 img/s with --parallel flag (excellent performance)
- **Total Pipeline**: Varies by dataset size and training epochs

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `--batch-size` to 16 or 8
2. **No PDFs found**: Ensure PDFs are in `data/raw/pdfs/`
3. **Dataset validation failed**: Check `data/dataset/plc_symbols.yaml` configuration
4. **Model not found**: Run `python setup/setup.py --download-models`

### Performance Tips
1. Use `--parallel` flag for best detection performance
2. Adjust `--batch-size` based on your GPU memory
3. Use `--skip-pdf-conversion` if you've already processed PDFs
4. Monitor GPU usage with `nvidia-smi` during processing

## Next Steps After Pipeline Completion

1. **Review Results**: Check detection accuracy in generated PDFs
2. **Analyze Metrics**: Review `pipeline_summary.json` for performance stats
3. **Fine-tune**: Adjust confidence threshold or retrain with more epochs if needed
4. **Production Use**: Use the trained model for new PDF processing

## Model Reuse

Once trained, you can reuse the model:
```bash
# Find your trained model
ls data/runs/train/plc_symbol_detector_yolo11m_*/weights/best.pt

# Use it directly for detection only
python src/detection/run_complete_pipeline.py --skip-training --parallel
```

The trained model will be automatically detected and used for future runs.
