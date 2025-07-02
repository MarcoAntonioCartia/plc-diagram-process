# Detection Stage Documentation

## Overview

The Detection Stage is a critical component of the PLC Diagram Processor pipeline responsible for identifying and localizing PLC symbols within processed diagram images using YOLO11 object detection models. This stage transforms PDF diagrams into annotated outputs with precise coordinate mapping and symbol classification.

## Architecture

### Core Components

The detection system follows a modular architecture with clear separation of concerns:

```
src/detection/
├── Core Pipeline Components
│   ├── detect_pipeline.py          # Main detection orchestrator
│   ├── yolo11_infer.py             # YOLO11 inference engine
│   ├── yolo11_train.py             # Model training functionality
│   └── coordinate_transform.py     # Coordinate mapping utilities
├── Integration Components
│   ├── detection_manager.py        # High-level detection coordinator
│   ├── detect_pipeline_subprocess.py # Subprocess execution wrapper
│   └── yolo_compatibility.py       # YOLO version compatibility layer
├── Reconstruction Components
│   └── reconstruct_with_detections.py # PDF reconstruction with overlays
├── Performance Components
│   ├── benchmark_detection.py      # Performance benchmarking
│   ├── benchmark_all_pipelines.py  # Comprehensive benchmarks
│   └── profile_gpu_pipeline.py     # GPU profiling utilities
├── Validation Components
│   ├── validate_pipeline_structure.py # Pipeline validation
│   └── lightweight_pipeline_runner.py # Lightweight execution
└── Legacy Components
    ├── detect_pipeline_gpu_optimized.py # GPU-optimized implementation
    ├── detect_pipeline_parallel.py      # Parallel processing version
    └── unified_parallel_pipeline.py     # Unified parallel approach
```

### Design Principles

1. **Modular Architecture**: Each component has a single responsibility
2. **Environment Isolation**: Heavy dependencies are isolated in subprocess execution
3. **Configuration Management**: Centralized configuration through the config system
4. **Smart Model Loading**: Automatic detection and loading of best available models
5. **Coordinate Precision**: Accurate transformation from snippet to global coordinates
6. **Performance Optimization**: Multiple execution strategies for different use cases

## File-by-File Documentation

### Core Pipeline Components

#### `detect_pipeline.py`
**Purpose**: Main orchestrator for the complete detection pipeline
**Functionality**:
- Coordinates PDF-to-image conversion, detection, and reconstruction
- Manages the complete workflow: PDF → Snippets → Detection → Coordinate Mapping → Reconstructed PDF
- Provides both standalone and integrated execution modes
- Handles path resolution for both relative and absolute inputs

**Key Classes**:
- `PLCDetectionPipeline`: Main pipeline orchestrator

**Usage**:
```bash
python src/detection/detect_pipeline.py --diagrams "path/to/pdfs" --conf 0.5
```

**Integration**: Used by the stage-based pipeline and as a standalone tool

#### `yolo11_infer.py`
**Purpose**: YOLO11 inference engine with smart model management
**Functionality**:
- Automatic detection and loading of best available custom models
- Fallback to pretrained models when custom models unavailable
- Single image and batch processing capabilities
- Confidence-based filtering and result formatting
- Path resolution for cross-platform compatibility

**Key Functions**:
- `load_model()`: Smart model loading with auto-detection
- `predict_image()`: Single image inference
- `predict_folder()`: Batch folder processing

**Model Loading Logic**:
1. Search for custom trained models in `models/custom/`
2. Select most recent model based on modification time
3. Load model metadata for validation
4. Fallback to pretrained models if no custom models found

**Usage**:
```bash
python src/detection/yolo11_infer.py --input "image.png" --conf 0.25 --save-images
```

#### `yolo11_train.py`
**Purpose**: YOLO11 model training with configuration management
**Functionality**:
- Automated training pipeline with smart device detection
- Dataset validation and structure verification
- Model metadata generation and storage
- Custom model deployment to `models/custom/` directory
- Error handling for common training issues (CSV corruption, etc.)

**Key Functions**:
- `train_yolo11()`: Main training function
- `validate_dataset()`: Dataset structure validation
- `get_best_device()`: Automatic device selection

**Training Process**:
1. Validate dataset structure and configuration
2. Load pretrained model as starting point
3. Execute training with specified parameters
4. Copy best model to custom models directory
5. Generate and save model metadata

**Usage**:
```bash
python src/detection/yolo11_train.py --model yolo11m.pt --epochs 10 --batch 16
```

#### `coordinate_transform.py`
**Purpose**: Coordinate transformation utilities for snippet-to-global mapping
**Functionality**:
- Transforms detection coordinates from snippet-relative to global PDF coordinates
- Validates coordinate bounds and consistency
- Generates human-readable coordinate mappings
- Provides detection statistics and analysis

**Key Functions**:
- `transform_detections_to_global()`: Main coordinate transformation
- `transform_single_detection()`: Individual detection transformation
- `validate_coordinates()`: Coordinate validation
- `get_detection_statistics()`: Statistical analysis

**Coordinate System**:
- **Snippet Coordinates**: Relative to individual image snippet (0,0 at snippet top-left)
- **Global Coordinates**: Relative to full PDF page (0,0 at page top-left)
- **Transformation**: Global = Snippet_Global_Offset + Snippet_Relative

### Integration Components

#### `detection_manager.py`
**Purpose**: High-level detection coordinator without heavy dependencies
**Functionality**:
- Provides abstraction layer for detection operations
- Manages subprocess execution for environment isolation
- Validates setup and configuration before execution
- Coordinates training and inference operations

**Key Classes**:
- `DetectionManager`: Main coordination class

**Design Philosophy**: Never imports ultralytics, torch, or heavy dependencies directly to maintain lightweight operation in the main process.

#### `detect_pipeline_subprocess.py`
**Purpose**: Subprocess execution wrapper for isolated detection
**Functionality**:
- Executes detection pipeline in isolated subprocess
- Handles input/output serialization via JSON
- Provides error isolation and recovery
- Enables heavy dependency usage without main process contamination

**Execution Flow**:
1. Receive input parameters via JSON file
2. Import heavy dependencies (ultralytics, torch)
3. Execute detection pipeline
4. Serialize results to JSON output
5. Return success/failure status

#### `yolo_compatibility.py`
**Purpose**: YOLO version compatibility layer
**Functionality**:
- Provides placeholder classes for unknown YOLO layers
- Maintains model compatibility across YOLO versions
- Handles channel dimension preservation
- Enables loading of models trained with different YOLO versions

**Key Classes**:
- `RobustPlaceholder`: Base placeholder for unknown layers
- Various YOLO layer placeholders (C3k2, RepC3, etc.)

**Compatibility Strategy**:
1. Parse constructor arguments to determine channel dimensions
2. Use identity mapping when input/output channels match
3. Insert 1×1 convolution for channel adaptation when needed
4. Register placeholders only when native classes unavailable

### Reconstruction Components

#### `reconstruct_with_detections.py`
**Purpose**: PDF reconstruction with detection overlays
**Functionality**:
- Reconstructs full PDF pages from image snippets
- Overlays detection bounding boxes and labels
- Generates annotated PDFs with detection results
- Creates coordinate mapping files and statistics
- Provides comparison with original PDFs

**Key Functions**:
- `reconstruct_pdf_with_detections()`: Main reconstruction orchestrator
- `reconstruct_page_with_detections()`: Single page reconstruction
- `overlay_detections()`: Detection visualization
- `create_pdf_from_images()`: PDF generation from images

**Output Files**:
- `*_detected.pdf`: Annotated PDF with detection overlays
- `*_detections.json`: Machine-readable detection results
- `*_coordinates.txt`: Human-readable coordinate mapping
- `*_statistics.json`: Detection statistics and analysis

### Performance Components

#### `benchmark_detection.py`
**Purpose**: Performance benchmarking for detection operations
**Functionality**:
- Compares sequential vs parallel detection performance
- Tests different batch sizes and worker configurations
- Measures throughput and processing time
- Identifies optimal configuration parameters

**Benchmarking Metrics**:
- Processing time (seconds)
- Throughput (images/second)
- Speedup ratios
- Resource utilization

#### `benchmark_all_pipelines.py`
**Purpose**: Comprehensive pipeline performance comparison
**Functionality**:
- Benchmarks multiple pipeline implementations
- Compares different optimization strategies
- Provides performance recommendations
- Generates detailed performance reports

#### `profile_gpu_pipeline.py`
**Purpose**: GPU-specific performance profiling
**Functionality**:
- Profiles GPU memory usage and utilization
- Identifies bottlenecks in GPU processing
- Optimizes batch sizes for GPU efficiency
- Monitors CUDA operations and synchronization

### Validation Components

#### `validate_pipeline_structure.py`
**Purpose**: Pipeline structure and configuration validation
**Functionality**:
- Validates pipeline component availability
- Checks configuration consistency
- Verifies model and dataset accessibility
- Ensures proper environment setup

#### `lightweight_pipeline_runner.py`
**Purpose**: Minimal overhead pipeline execution
**Functionality**:
- Provides lightweight execution for testing
- Minimal dependency requirements
- Fast startup and execution
- Suitable for CI/CD environments

### Legacy Components

#### `detect_pipeline_gpu_optimized.py`
**Purpose**: GPU-optimized detection implementation (Legacy)
**Functionality**:
- Advanced GPU batch processing with tensor optimization
- Custom dataset and dataloader implementation
- Automatic mixed precision (AMP) support
- GPU memory management and synchronization
- Parallel PDF processing and reconstruction

**Optimization Features**:
- Pre-loaded tensor batching
- GPU warmup for consistent performance
- Memory pinning for faster data transfer
- Asynchronous data loading
- CUDA synchronization management

**Status**: Legacy implementation maintained for reference and specialized use cases

#### `detect_pipeline_parallel.py`
**Purpose**: CPU parallel processing implementation (Legacy)
**Functionality**:
- Multi-process CPU parallelization
- Thread-based I/O operations
- Load balancing across CPU cores
- Memory-efficient processing

**Status**: Legacy implementation superseded by GPU-optimized version

#### `unified_parallel_pipeline.py`
**Purpose**: Unified parallel processing approach (Legacy)
**Functionality**:
- Combines CPU and GPU parallelization strategies
- Adaptive resource allocation
- Hybrid processing pipelines

**Status**: Legacy implementation used for research and development

## Integration with Pipeline System

### Stage-Based Integration

The detection stage integrates with the main pipeline through:

1. **Stage Manager**: `src/pipeline/stages/detection_stage.py`
2. **Detection Worker**: `src/workers/detection_worker.py`
3. **Multi-Environment Manager**: `src/utils/multi_env_manager.py`

### Dependency Management

The detection stage uses smart dependency checking:

1. **Model-Based Dependencies**: Checks for available trained models instead of rigid stage dependencies
2. **Environment Isolation**: Heavy dependencies (PyTorch, Ultralytics) isolated in subprocess execution
3. **Graceful Fallbacks**: Automatic fallback to pretrained models when custom models unavailable

### Configuration Integration

Detection configuration is managed through:

1. **Central Configuration**: `src/config.py` provides unified configuration management
2. **Model Path Resolution**: Automatic resolution of model paths across different environments
3. **Data Path Management**: Consistent data directory structure and access

## Model Management

### Model Types

1. **Pretrained Models**: Base YOLO11 models (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
2. **Custom Models**: Fine-tuned models for PLC symbol detection
3. **Model Metadata**: JSON files containing training information and metrics

### Model Storage Structure

```
models/
├── pretrained/
│   ├── yolo11m.pt
│   └── ...
└── custom/
    ├── plc_symbol_detector_best.pt
    ├── plc_symbol_detector_best.json
    └── ...
```

### Model Selection Logic

1. **Auto-Detection**: Automatically selects best available custom model
2. **Metadata Validation**: Verifies model metadata and training metrics
3. **Fallback Strategy**: Falls back to pretrained models when necessary
4. **Version Compatibility**: Handles models trained with different YOLO versions

## Usage Examples

### Standalone Detection

```bash
# Process single PDF with custom confidence
python src/detection/detect_pipeline.py --diagrams "path/to/pdfs" --conf 0.5

# Process with specific model
python src/detection/detect_pipeline.py --diagrams "path/to/pdfs" --model "custom_model.pt"

# Skip PDF conversion (use existing images)
python src/detection/detect_pipeline.py --diagrams "path/to/pdfs" --skip-pdf-conversion
```

### Pipeline Integration

```bash
# Run detection stage only
python src/run_pipeline.py --stages detection --force

# Run with custom parameters
python src/run_pipeline.py --stages detection --detection-conf 0.5 --force
```

### Training New Models

```bash
# Train with default parameters
python src/detection/yolo11_train.py --model yolo11m.pt --epochs 10

# Train with custom configuration
python src/detection/yolo11_train.py --model yolo11l.pt --epochs 50 --batch 32 --device 0
```

### Performance Benchmarking

```bash
# Benchmark detection performance
python src/detection/benchmark_detection.py --model "best.pt" --images "path/to/images"

# GPU optimization benchmarking
python src/detection/legacy/detect_pipeline_gpu_optimized.py --benchmark --diagrams "path/to/pdfs"
```

## Configuration Parameters

### Detection Parameters

- **confidence_threshold**: Detection confidence threshold (0.0-1.0, default: 0.25)
- **model_path**: Path to YOLO model (None for auto-detection)
- **snippet_size**: Image snippet dimensions (default: [1500, 1200])
- **overlap**: Snippet overlap in pixels (default: 500)

### Performance Parameters

- **batch_size**: GPU batch size for inference (default: 32)
- **num_workers**: Number of parallel workers (default: 4)
- **device**: Processing device ('auto', 'cpu', '0', '1', etc.)
- **use_amp**: Enable automatic mixed precision (default: True)

### Output Parameters

- **save_images**: Save annotated images (default: False)
- **output_dir**: Custom output directory (default: auto-generated)
- **exist_ok**: Allow overwriting existing outputs (default: True)

## Error Handling and Troubleshooting

### Common Issues

1. **Model Not Found**
   - **Cause**: No trained models available
   - **Solution**: Run training or copy models to `models/custom/`
   - **Command**: `python src/detection/yolo11_train.py`

2. **Path Resolution Errors**
   - **Cause**: Incorrect relative path handling
   - **Solution**: Use absolute paths or verify working directory
   - **Fix**: Path resolution logic handles both relative and absolute paths

3. **GPU Memory Issues**
   - **Cause**: Batch size too large for available GPU memory
   - **Solution**: Reduce batch size or use CPU processing
   - **Parameter**: `--batch-size 16` or `--device cpu`

4. **Environment Dependencies**
   - **Cause**: Missing PyTorch or Ultralytics installation
   - **Solution**: Install detection environment dependencies
   - **Command**: `pip install -r requirements-detection.txt`

### Debugging Tools

1. **Validation Script**: `validate_pipeline_structure.py`
2. **Lightweight Runner**: `lightweight_pipeline_runner.py`
3. **Configuration Checker**: Built into detection manager
4. **Verbose Logging**: Enable with `--verbose` flag

## Performance Optimization

### GPU Optimization

1. **Batch Processing**: Use appropriate batch sizes for GPU memory
2. **Mixed Precision**: Enable AMP for faster inference
3. **Memory Management**: Proper CUDA synchronization and cleanup
4. **Tensor Optimization**: Pre-loaded tensors and memory pinning

### CPU Optimization

1. **Parallel Processing**: Multi-process and multi-thread execution
2. **I/O Optimization**: Asynchronous file operations
3. **Memory Efficiency**: Streaming processing for large datasets
4. **Load Balancing**: Dynamic work distribution

### Configuration Recommendations

- **Small Datasets (<100 images)**: Sequential processing, batch_size=1
- **Medium Datasets (100-1000 images)**: GPU processing, batch_size=16-32
- **Large Datasets (>1000 images)**: GPU optimization, batch_size=32-64
- **Limited GPU Memory**: Reduce batch_size or use CPU processing
- **High-Performance Requirements**: Use legacy GPU-optimized pipeline

## Development Guidelines

### Adding New Features

1. **Maintain Modularity**: Keep components focused and independent
2. **Environment Isolation**: Avoid importing heavy dependencies in main process
3. **Configuration Integration**: Use centralized configuration management
4. **Error Handling**: Implement comprehensive error handling and recovery
5. **Documentation**: Update this README with new functionality

### Testing Procedures

1. **Unit Testing**: Test individual components in isolation
2. **Integration Testing**: Test pipeline integration and data flow
3. **Performance Testing**: Benchmark new implementations
4. **Validation Testing**: Verify output accuracy and consistency

### Code Standards

1. **Type Hints**: Use type hints for function parameters and returns
2. **Documentation**: Comprehensive docstrings for all functions and classes
3. **Error Messages**: Clear, actionable error messages
4. **Logging**: Appropriate logging levels and messages
5. **Configuration**: Externalize configuration parameters

## Future Enhancements

### Planned Improvements

1. **Model Ensemble**: Support for multiple model ensemble detection
2. **Real-time Processing**: Streaming detection for real-time applications
3. **Cloud Integration**: Support for cloud-based model serving
4. **Advanced Metrics**: Enhanced detection quality metrics and validation
5. **Auto-tuning**: Automatic parameter optimization based on dataset characteristics

### Research Directions

1. **Advanced Architectures**: Integration of newer YOLO versions and architectures
2. **Domain Adaptation**: Improved transfer learning for PLC-specific domains
3. **Multi-scale Detection**: Enhanced detection of symbols at different scales
4. **Uncertainty Quantification**: Confidence estimation and uncertainty modeling

## Conclusion

The Detection Stage provides a comprehensive, production-ready solution for PLC symbol detection with emphasis on modularity, performance, and maintainability. The architecture supports both standalone and integrated usage while maintaining flexibility for future enhancements and optimizations.

For additional support or questions, refer to the main project documentation or contact the dev team.
