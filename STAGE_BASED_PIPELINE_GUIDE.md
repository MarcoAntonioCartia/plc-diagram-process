# Complete Stage-Based Pipeline Running Guide

## Prerequisites

1. **Ensure your environment is properly set up**:
   ```bash
   # Activate your environment
   # Windows
   activate.bat
   
   # Linux/Mac
   source activate.sh
   ```

2. **Verify your data structure**:
   ```bash
   # Check that PDFs are in the correct location
   ls data/raw/pdfs/
   
   # Ensure YOLO models are available
   ls data/models/pretrained/
   ```

3. **Check pipeline status**:
   ```bash
   # See all available stages and their current status
   python src/run_pipeline.py --list-stages
   
   # Get detailed pipeline status
   python src/run_pipeline.py --status
   ```

## Running the Complete Pipeline

### Option 1: Full Pipeline (Recommended for First-Time Users)
```bash
# Run all stages in order with default settings
python src/run_pipeline.py --run-all

# Run with verbose output to see detailed progress
python src/run_pipeline.py --run-all --verbose

# Run with multi-environment mode (recommended for production)
python src/run_pipeline.py --run-all --multi-env

# Save execution results for later analysis
python src/run_pipeline.py --run-all --save-results pipeline_results.json
```

### Option 2: Stage-by-Stage Execution (Recommended for Development)

#### Stage 1: Preparation
```bash
# Validate inputs, setup directories, check environment health
python src/run_pipeline.py --stages preparation

# With verbose output to see what's being validated
python src/run_pipeline.py --stages preparation --verbose
```

#### Stage 2: Training/Model Validation
```bash
# Validate YOLO models and setup training environment
python src/run_pipeline.py --stages training

# Force re-validation of models
python src/run_pipeline.py --stages training --force
```

#### Stage 3: Object Detection
```bash
# Run YOLO detection on PDFs
python src/run_pipeline.py --stages detection

# Run detection with multi-environment isolation
python src/run_pipeline.py --stages detection --multi-env
```

#### Stage 4: Text Extraction (OCR)
```bash
# Extract text from detected regions
python src/run_pipeline.py --stages ocr

# Run OCR with multi-environment isolation
python src/run_pipeline.py --stages ocr --multi-env
```

#### Stage 5: Enhancement & Output
```bash
# Create CSV output and enhanced PDFs
python src/run_pipeline.py --stages enhancement

# Force regeneration of outputs
python src/run_pipeline.py --stages enhancement --force
```

### Option 3: Stage Groups
```bash
# Run detection pipeline (training + detection)
python src/run_pipeline.py --stages training detection

# Run text processing pipeline (ocr + enhancement)
python src/run_pipeline.py --stages ocr enhancement

# Run output generation only
python src/run_pipeline.py --stages enhancement --force
```

## Advanced Configuration

### Custom Configuration File
```bash
# Create custom_config.json with your settings
cat > custom_config.json << EOF
{
  "training": {
    "model_name": "yolo11m.pt",
    "force_training": false
  },
  "detection": {
    "confidence_threshold": 0.25,
    "snippet_size": [1500, 1200],
    "overlap": 500
  },
  "ocr": {
    "ocr_confidence": 0.7,
    "ocr_lang": "en"
  },
  "enhancement": {
    "area_grouping": true,
    "alphanumeric_sort": true,
    "enhanced_pdf_version": "short"
  }
}
EOF

# Run pipeline with custom configuration
python src/run_pipeline.py --run-all --config custom_config.json
```

### Environment-Specific Execution
```bash
# Single environment mode (all stages in current Python env)
python src/run_pipeline.py --run-all --single-env

# Multi-environment mode (separate envs for YOLO/OCR)
python src/run_pipeline.py --run-all --multi-env

# Override data root directory
python src/run_pipeline.py --run-all --data-root /path/to/your/data

# Override state directory
python src/run_pipeline.py --run-all --state-dir /path/to/state
```

### Force Re-execution
```bash
# Force re-run all stages (ignore completion status)
python src/run_pipeline.py --run-all --force

# Force re-run specific stages only
python src/run_pipeline.py --stages detection ocr --force

# Re-run without skipping completed stages
python src/run_pipeline.py --run-all --no-skip-completed
```

## Pipeline State Management

### Check Current State
```bash
# Quick status overview
python src/run_pipeline.py --status

# Detailed status with dependencies
python src/run_pipeline.py --status --verbose

# JSON output for scripting
python src/run_pipeline.py --status --json-output
```

### Reset Pipeline State
```bash
# Reset all stages (start completely fresh)
python src/run_pipeline.py --reset

# Reset specific stages only
python src/run_pipeline.py --reset detection ocr enhancement

# Reset and immediately run
python src/run_pipeline.py --reset && python src/run_pipeline.py --run-all
```

### Resume Interrupted Pipeline
```bash
# The pipeline automatically resumes from where it left off
python src/run_pipeline.py --run-all

# Check what stages are completed before resuming
python src/run_pipeline.py --status
python src/run_pipeline.py --run-all
```

## Expected Output Structure

The pipeline will create the following directory structure:
```
data/
├── raw/
│   └── pdfs/                    # Input PDF files
├── processed/
│   ├── images/                  # PDF page images
│   ├── detdiagrams/            # Detection results
│   │   ├── document1/
│   │   │   ├── detections.json         # Detection coordinates
│   │   │   ├── detected_images/        # Images with overlays
│   │   │   └── statistics.json         # Detection statistics
│   │   └── document2/
│   ├── text_extraction/        # OCR results
│   │   ├── document1/
│   │   │   ├── text_extraction.json    # Extracted text regions
│   │   │   └── confidence_stats.json   # OCR confidence metrics
│   │   └── document2/
│   ├── csv_output/             # Final CSV outputs
│   │   ├── combined_text_extraction.csv      # All documents combined
│   │   ├── document1_text_extraction.csv     # Individual document CSVs
│   │   └── document2_text_extraction.csv
│   └── enhanced_pdfs/          # Enhanced PDF outputs
│       ├── document1_enhanced.pdf            # PDFs with overlays
│       └── document2_enhanced.pdf
├── models/
│   ├── pretrained/             # Downloaded YOLO models
│   └── custom/                 # Custom trained models
└── .pipeline_state/            # Pipeline state files
    ├── preparation_state.json
    ├── training_state.json
    ├── detection_state.json
    ├── ocr_state.json
    ├── enhancement_state.json
    └── execution_summary.json
```

## Performance Expectations

Based on the modular architecture:

### Stage Execution Times (Approximate)
- **Preparation**: ~5-10 seconds (validation and setup)
- **Training**: ~10-30 seconds (model validation, no actual training by default)
- **Detection**: ~1-5 minutes per PDF (depends on size and complexity)
- **OCR**: ~30 seconds - 2 minutes per PDF (depends on detected regions)
- **Enhancement**: ~10-30 seconds (CSV generation and PDF enhancement)

### Memory Usage
- **Single Environment**: Uses current Python environment memory
- **Multi Environment**: Isolates heavy dependencies, more memory efficient
- **State Files**: Minimal disk usage (~1-10KB per stage)

## Monitoring and Debugging

### Real-time Monitoring
```bash
# Watch pipeline progress with verbose output
python src/run_pipeline.py --run-all --verbose

# Monitor specific stage execution
python src/run_pipeline.py --stages detection --verbose

# Save detailed logs
python src/run_pipeline.py --run-all --verbose --save-results detailed_log.json
```

### Debug Failed Stages
```bash
# Check which stage failed
python src/run_pipeline.py --status

# Re-run failed stage with verbose output
python src/run_pipeline.py --stages <failed_stage> --force --verbose

# Reset and retry problematic stages
python src/run_pipeline.py --reset <failed_stage>
python src/run_pipeline.py --stages <failed_stage> --verbose
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# The pipeline automatically handles missing dependencies in CI/test environments
# If you see import errors, check your environment activation:
python -c "import torch; print('PyTorch OK')"
python -c "import paddleocr; print('PaddleOCR OK')"

# Use multi-environment mode to isolate dependencies:
python src/run_pipeline.py --run-all --multi-env
```

#### 2. CUDA/GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Use multi-environment mode to avoid CUDA conflicts
python src/run_pipeline.py --run-all --multi-env

# Force single environment if multi-env fails
python src/run_pipeline.py --run-all --single-env
```

#### 3. Stage Dependencies
```bash
# Check stage dependency status
python src/run_pipeline.py --status --verbose

# Reset problematic dependency chain
python src/run_pipeline.py --reset preparation training detection
python src/run_pipeline.py --stages preparation training detection
```

#### 4. State File Corruption
```bash
# Reset all state files
python src/run_pipeline.py --reset

# Reset specific corrupted stage
python src/run_pipeline.py --reset <stage_name>

# Check state directory
ls -la .pipeline_state/
```

#### 5. No PDFs Found
```bash
# Verify PDF location
ls data/raw/pdfs/

# Override data root if PDFs are elsewhere
python src/run_pipeline.py --run-all --data-root /path/to/your/data
```

### Performance Optimization

#### For Large Datasets
```bash
# Use multi-environment mode for better resource management
python src/run_pipeline.py --run-all --multi-env

# Process in stages to monitor progress
python src/run_pipeline.py --stages preparation training
python src/run_pipeline.py --stages detection
python src/run_pipeline.py --stages ocr enhancement
```

#### For Development/Testing
```bash
# Run only preparation to validate setup quickly
python src/run_pipeline.py --stages preparation

# Test specific stages without dependencies
python src/run_pipeline.py --stages detection --force

# Use quiet mode for automated scripts
python src/run_pipeline.py --run-all --quiet --json-output
```

## Integration with Existing Workflows

### Scripting and Automation
```bash
#!/bin/bash
# automated_pipeline.sh

# Check if pipeline is ready
if python src/run_pipeline.py --status --json-output | jq -r '.overall_progress' | grep -q "1.0"; then
    echo "Pipeline already completed"
    exit 0
fi

# Run pipeline with error handling
if python src/run_pipeline.py --run-all --save-results results.json; then
    echo "Pipeline completed successfully"
    # Process results.json as needed
else
    echo "Pipeline failed, checking status..."
    python src/run_pipeline.py --status --verbose
    exit 1
fi
```

### CI/CD Integration
```yaml
# .github/workflows/pipeline.yml
- name: Run PLC Pipeline
  run: |
    python src/run_pipeline.py --run-all --json-output > pipeline_results.json
    
- name: Check Pipeline Results
  run: |
    if jq -r '.success' pipeline_results.json | grep -q "true"; then
      echo "Pipeline succeeded"
    else
      echo "Pipeline failed"
      exit 1
    fi
```

## Next Steps After Pipeline Completion

1. **Review Results**:
   ```bash
   # Check CSV outputs
   ls data/processed/csv_output/
   
   # Review enhanced PDFs
   ls data/processed/enhanced_pdfs/
   
   # Analyze execution summary
   cat .pipeline_state/execution_summary.json | jq
   ```

2. **Validate Output Quality**:
   ```bash
   # Check detection statistics
   find data/processed/detdiagrams -name "statistics.json" -exec cat {} \;
   
   # Review OCR confidence metrics
   find data/processed/text_extraction -name "confidence_stats.json" -exec cat {} \;
   ```

3. **Iterate and Improve**:
   ```bash
   # Adjust configuration and re-run specific stages
   python src/run_pipeline.py --stages enhancement --force --config improved_config.json
   
   # Re-run with different confidence thresholds
   python src/run_pipeline.py --stages detection ocr enhancement --force
   ```

## Model Management

### Using Different Models
```bash
# The pipeline automatically detects available models
python src/run_pipeline.py --list-stages

# To use a specific model, create a config file:
echo '{"training": {"model_name": "yolo11s.pt"}}' > model_config.json
python src/run_pipeline.py --run-all --config model_config.json
```

### Training Integration
The stage-based pipeline integrates with the existing training workflow:
```bash
# Train a model using the existing training pipeline
python src/detection/run_complete_pipeline.py --epochs 10

# Then use the trained model in the stage-based pipeline
python src/run_pipeline.py --run-all
```

The new stage-based pipeline provides a robust, maintainable, and user-friendly interface for processing PLC diagrams while maintaining full compatibility with existing workflows and data structures.
