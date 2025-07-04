# Utils Module Documentation

## Overview

The Utils module provides shared utilities and helper functions used across the PLC Diagram Processor pipeline. This module includes GPU management, multi-environment coordination, PDF creation utilities, progress display, and various system management tools.

## Architecture

### Core Components

```
src/utils/
├── Environment Management
│   ├── multi_env_manager.py           # Multi-environment coordination
│   ├── gpu_manager.py                 # GPU resource management
│   └── runtime_flags.py               # Runtime configuration flags
├── Data Management
│   ├── dataset_manager.py             # Dataset management utilities
│   ├── model_manager.py               # Model management utilities
│   ├── network_drive_manager.py       # Network storage management
│   └── onedrive_manager.py            # OneDrive integration (legacy)
├── PDF Creation
│   ├── basic_pdf_creator.py           # Basic PDF generation
│   ├── enhanced_pdf_creator.py        # Enhanced PDF with annotations
│   ├── detection_pdf_creator.py       # Detection result PDFs
│   ├── detection_text_extraction_pdf_creator.py  # Combined detection/text PDFs
│   └── pdf_enhancer.py                # PDF enhancement utilities
├── System Utilities
│   ├── progress_display.py            # Progress visualization
│   ├── gpu_sanity_checker.py          # GPU diagnostics
│   ├── cleanup_storage.py             # Storage management utility
│   ├── fix_config_path.py             # Configuration path fixer
│   └── pdf_annotator.py               # Native PDF annotation system
└── Documentation
    └── gpu_conflict_guide.md           # GPU troubleshooting guide
```

## File-by-File Documentation

### Environment Management

#### `multi_env_manager.py`
**Purpose**: Coordination and management of multiple isolated Python environments
**Functionality**:
- Manages execution of pipeline stages in isolated environments
- Handles communication between main process and worker environments
- Provides subprocess coordination for heavy dependency isolation
- Includes error handling and recovery for multi-environment operations

**Key Classes**:
- `MultiEnvironmentManager`: Main coordination class

**Key Functions**:
- `run_training_pipeline()`: Execute training in isolated yolo_env
- `run_detection_pipeline()`: Execute detection in isolated yolo_env
- `run_ocr_pipeline()`: Execute OCR in isolated ocr_env
- `check_environment_health()`: Validate environment status

**Environment Isolation Strategy**:
- **yolo_env**: PyTorch, Ultralytics, CUDA dependencies
- **ocr_env**: PaddleOCR, text processing dependencies
- **core_env**: Lightweight dependencies for coordination

**Usage**:
```python
from src.utils.multi_env_manager import MultiEnvironmentManager

manager = MultiEnvironmentManager(project_root)
result = manager.run_training_pipeline(training_config)
```

#### `gpu_manager.py`
**Purpose**: GPU resource management and optimization
**Functionality**:
- Detects available GPU resources and capabilities
- Manages GPU memory allocation and monitoring
- Provides device selection and optimization recommendations
- Handles GPU conflict resolution and troubleshooting

**Key Functions**:
- `detect_gpu_setup()`: Comprehensive GPU detection
- `get_optimal_device()`: Device selection for workloads
- `monitor_gpu_usage()`: Real-time GPU monitoring
- `resolve_gpu_conflicts()`: Conflict resolution utilities

**GPU Management Features**:
- **Device Detection**: Automatic detection of CUDA, ROCm, and CPU capabilities
- **Memory Monitoring**: Real-time GPU memory usage tracking
- **Conflict Resolution**: Handles multiple process GPU access
- **Performance Optimization**: Optimal device selection for workloads

#### `runtime_flags.py`
**Purpose**: Runtime configuration and feature flags
**Functionality**:
- Manages runtime configuration flags and settings
- Provides feature toggles for experimental functionality
- Handles environment variable processing
- Includes configuration validation and defaults

**Configuration Categories**:
- **Performance Flags**: GPU usage, parallel processing, memory limits
- **Debug Flags**: Verbose output, debug mode, profiling
- **Feature Flags**: Experimental features, compatibility modes

### Data Management

#### `dataset_manager.py`
**Purpose**: Dataset management and organization utilities
**Functionality**:
- Manages dataset downloads and organization
- Handles dataset version control and activation
- Provides dataset validation and integrity checking
- Includes symlink management for efficient storage

**Key Functions**:
- `download_dataset()`: Dataset download from storage backends
- `activate_dataset()`: Dataset activation and linking
- `validate_dataset()`: Dataset structure validation
- `list_available_datasets()`: Available dataset discovery

#### `model_manager.py`
**Purpose**: Model management and deployment utilities
**Functionality**:
- Manages YOLO model downloads and storage
- Handles model version control and metadata
- Provides model validation and integrity checking
- Includes custom model deployment and organization

**Key Functions**:
- `download_model()`: Model download from repositories
- `deploy_custom_model()`: Custom model deployment
- `validate_model()`: Model integrity validation
- `list_available_models()`: Available model discovery

#### `network_drive_manager.py`
**Purpose**: Network storage backend management
**Functionality**:
- Manages network drive access and mounting
- Handles file transfer and synchronization
- Provides network storage validation and testing
- Includes error handling for network connectivity issues

**Key Features**:
- **Network Detection**: Automatic network drive detection
- **Path Translation**: Windows/Linux path compatibility
- **Transfer Optimization**: Efficient file transfer mechanisms
- **Error Recovery**: Network connectivity error handling

#### `onedrive_manager.py`
**Purpose**: OneDrive integration for cloud storage (legacy)
**Functionality**:
- Provides OneDrive API integration for cloud storage
- Handles authentication and file access
- Includes download and upload capabilities
- Legacy component maintained for compatibility

### PDF Creation

#### `basic_pdf_creator.py`
**Purpose**: Basic PDF generation utilities
**Functionality**:
- Creates simple PDF documents from images
- Handles basic page layout and formatting
- Provides batch PDF creation capabilities
- Includes basic metadata and annotation support

**Key Functions**:
- `create_pdf_from_images()`: Image-to-PDF conversion
- `merge_pdfs()`: PDF merging utilities
- `add_basic_annotations()`: Simple annotation support

#### `enhanced_pdf_creator.py`
**Purpose**: Enhanced PDF generation with advanced features
**Functionality**:
- Creates feature-rich PDF documents with annotations
- Handles complex layouts and formatting
- Provides advanced annotation and overlay capabilities
- Includes quality optimization and compression

**Key Features**:
- **Advanced Layouts**: Complex page layouts and formatting
- **Rich Annotations**: Text, shapes, and overlay annotations
- **Quality Control**: Image quality optimization and compression
- **Metadata Management**: Comprehensive PDF metadata handling

#### `detection_pdf_creator.py`
**Purpose**: PDF creation for detection results visualization
**Functionality**:
- Creates PDFs with detection result overlays
- Handles bounding box visualization and labeling
- Provides confidence score display and formatting
- Includes batch processing for multiple detection results

**Key Functions**:
- `create_detection_pdf()`: Detection result PDF generation
- `overlay_detections()`: Detection visualization overlays
- `format_detection_labels()`: Label formatting and positioning

#### `detection_text_extraction_pdf_creator.py`
**Purpose**: Combined detection and text extraction PDF creation
**Functionality**:
- Creates PDFs combining detection and OCR results
- Handles visualization of both detection boxes and extracted text
- Provides comprehensive result presentation
- Includes comparison and analysis visualizations

#### `pdf_enhancer.py`
**Purpose**: PDF enhancement and optimization utilities
**Functionality**:
- Enhances existing PDF documents with additional content
- Handles PDF optimization and compression
- Provides quality improvement and formatting utilities
- Includes batch enhancement processing

### System Utilities

#### `progress_display.py`
**Purpose**: Progress visualization and monitoring utilities
**Functionality**:
- Provides real-time progress bars and status displays
- Handles multi-stage progress tracking
- Includes time estimation and performance metrics
- Supports both console and programmatic progress reporting

**Key Classes**:
- `ProgressDisplay`: Main progress tracking class
- `StageProgress`: Stage-specific progress tracking

**Features**:
- **Real-time Updates**: Live progress bar updates
- **Time Estimation**: ETA calculation and display
- **Multi-stage Tracking**: Progress across multiple pipeline stages
- **Performance Metrics**: Throughput and timing information

#### `gpu_sanity_checker.py`
**Purpose**: GPU diagnostics and troubleshooting utilities
**Functionality**:
- Performs comprehensive GPU system diagnostics
- Validates GPU driver and library installations
- Provides troubleshooting recommendations
- Includes performance benchmarking and testing

**Diagnostic Categories**:
- **Hardware Detection**: GPU hardware identification
- **Driver Validation**: Driver version and compatibility checking
- **Library Testing**: CUDA/PyTorch library validation
- **Performance Testing**: Basic performance benchmarking

#### `cleanup_storage.py`
**Purpose**: Storage management and cleanup utilities
**Functionality**:
- Audits current storage usage across project directories
- Cleans up old training runs and temporary files
- Manages cache files and temporary worker directories
- Provides storage optimization recommendations

**Key Functions**:
- `audit_storage()`: Comprehensive storage usage analysis
- `cleanup_training_runs()`: Remove old training runs keeping latest N
- `cleanup_cache_files()`: Clean YOLO cache files
- `cleanup_temp_workers()`: Remove temporary worker directories

**Storage Management Features**:
- **Usage Analysis**: Detailed storage usage breakdown by directory
- **Automated Cleanup**: Configurable cleanup of old files and runs
- **Safety Features**: Dry-run mode and confirmation prompts
- **Space Recovery**: Identifies and removes large temporary files

#### `fix_config_path.py`
**Purpose**: Configuration path management and migration utilities
**Functionality**:
- Updates configuration paths for version migrations
- Validates configuration file integrity
- Handles path translation between different environments
- Provides configuration backup and recovery

**Key Functions**:
- `update_config_paths()`: Update paths in configuration files
- `validate_config()`: Validate configuration file structure
- `backup_config()`: Create configuration backups
- `migrate_version()`: Handle version-specific migrations

**Configuration Management Features**:
- **Path Migration**: Automatic path updates for version changes
- **Validation**: Configuration file structure validation
- **Backup Management**: Automatic configuration backups
- **Cross-platform**: Windows/Linux path compatibility

#### `pdf_annotator.py`
**Purpose**: Native PDF annotation system for ML detection results
**Functionality**:
- Creates native PDF annotations from detection results
- Handles coordinate transformation for proper annotation placement
- Provides professional annotation styling and formatting
- Supports both detection boxes and text annotations

**Key Classes**:
- `PDFAnnotator`: Main annotation creation class

**Key Functions**:
- `create_annotated_pdf()`: Generate annotated PDF from detection results
- `add_detection_annotations()`: Add YOLO detection boxes as annotations
- `add_text_annotations()`: Add OCR text as PDF annotations
- `transform_coordinates()`: Convert detection coordinates to PDF space

**Annotation Features**:
- **Native Annotations**: True PDF annotations visible in all viewers
- **Coordinate Accuracy**: Precise coordinate transformation for rotated PDFs
- **Professional Styling**: Configurable colors, opacity, and styling
- **Metadata Support**: Rich annotation metadata and tooltips

## Integration with Pipeline System

### Multi-Environment Integration

The utils module provides the foundation for multi-environment pipeline execution:

1. **Environment Coordination**: `multi_env_manager.py` coordinates between environments
2. **Resource Management**: GPU and system resource management
3. **Data Management**: Dataset and model management across environments
4. **Progress Tracking**: Unified progress tracking across all stages

### Configuration Integration

Utils components integrate with the centralized configuration system:

1. **Runtime Configuration**: Runtime flags and environment variables
2. **Resource Configuration**: GPU and system resource settings
3. **Storage Configuration**: Data storage and management settings

## Usage Examples

### Multi-Environment Management

```python
from src.utils.multi_env_manager import MultiEnvironmentManager

# Initialize manager
manager = MultiEnvironmentManager(project_root="/path/to/project")

# Run training in isolated environment
training_config = {
    "model_path": "yolo11m.pt",
    "epochs": 50,
    "batch_size": 8
}
result = manager.run_training_pipeline(training_config)

# Run detection in isolated environment
detection_config = {
    "input_folder": "path/to/images",
    "confidence": 0.5
}
result = manager.run_detection_pipeline(detection_config)
```

### GPU Management

```python
from src.utils.gpu_manager import detect_gpu_setup, get_optimal_device

# Detect GPU capabilities
gpu_info = detect_gpu_setup()
print(f"CUDA available: {gpu_info['cuda_available']}")
print(f"GPU count: {gpu_info['gpu_count']}")

# Get optimal device for workload
device = get_optimal_device(workload_type="training")
print(f"Recommended device: {device}")
```

### Progress Tracking

```python
from src.utils.progress_display import create_stage_progress

# Create progress tracker
progress = create_stage_progress("training")
progress.start_stage("Starting model training...")

# Update progress
progress.update_progress("Processing epoch 1/50...")
progress.complete_file("Model Training", "Training completed successfully")
```

### PDF Creation

```python
from src.utils.detection_pdf_creator import create_detection_pdf

# Create detection result PDF
create_detection_pdf(
    image_folder="path/to/images",
    detection_results="detections.json",
    output_pdf="detection_results.pdf"
)
```

### Dataset Management

```python
from src.utils.dataset_manager import download_dataset, activate_dataset

# Download dataset
download_dataset(
    dataset_name="plc_symbols_v11_latest",
    storage_backend="network_drive"
)

# Activate dataset
activate_dataset("plc_symbols_v11_latest")
```

### Storage Management

```python
from src.utils.cleanup_storage import audit_storage, cleanup_training_runs

# Audit current storage usage
storage_info = audit_storage()
print(f"Total storage used: {storage_info['total_size']}")

# Clean up old training runs (dry run)
freed_space = cleanup_training_runs(keep_latest=2, dry_run=True)
print(f"Would free: {freed_space} bytes")

# Actually perform cleanup
freed_space = cleanup_training_runs(keep_latest=2, dry_run=False)
```

### Configuration Management

```python
from src.utils.fix_config_path import update_config_paths

# Update configuration paths for version migration
update_config_paths(old_version="0.3", new_version="0.4")

# Validate configuration after update
validate_config()
```

### PDF Annotation

```python
from src.utils.pdf_annotator import PDFAnnotator

# Create annotated PDF with detection results
annotator = PDFAnnotator(detection_confidence_threshold=0.8)
annotated_pdf = annotator.create_annotated_pdf(
    detection_file="detections.json",
    text_extraction_file="text_extraction.json",
    pdf_file="input.pdf",
    output_file="annotated_output.pdf"
)
```

## Configuration

### Multi-Environment Settings

- **PLCDP_MULTI_ENV**: Enable multi-environment mode ('1' or '0')
- **PLCDP_PROJECT_ROOT**: Project root directory path
- **PLCDP_ENV_TIMEOUT**: Environment operation timeout (seconds)

### GPU Management Settings

- **PLCDP_GPU_DEVICE**: Preferred GPU device ('auto', 'cpu', '0', '1', etc.)
- **PLCDP_GPU_MEMORY_LIMIT**: GPU memory limit (MB)
- **PLCDP_FORCE_CPU**: Force CPU-only processing ('1' or '0')

### Progress Display Settings

- **PLCDP_VERBOSE**: Enable verbose progress output ('1' or '0')
- **PLCDP_QUIET**: Disable progress display ('1' or '0')
- **PLCDP_PROGRESS_STYLE**: Progress bar style ('bar', 'spinner', 'minimal')

## Error Handling and Troubleshooting

### Common Issues

1. **Environment Communication Failures**
   - **Cause**: Inter-process communication issues
   - **Solution**: Check environment health and restart if needed
   - **Tool**: `multi_env_manager.check_environment_health()`

2. **GPU Detection Issues**
   - **Cause**: Driver or library compatibility problems
   - **Solution**: Run GPU diagnostics and follow recommendations
   - **Tool**: `gpu_sanity_checker.py`

3. **Dataset Access Issues**
   - **Cause**: Network connectivity or permission problems
   - **Solution**: Validate network storage access and permissions
   - **Tool**: `network_drive_manager.py`

4. **Progress Display Issues**
   - **Cause**: Terminal compatibility or output redirection
   - **Solution**: Use alternative progress styles or disable
   - **Parameter**: Set `PLCDP_PROGRESS_STYLE=minimal`

### Debugging Tools

1. **Environment Health Check**: Validate multi-environment setup
2. **GPU Diagnostics**: Comprehensive GPU system validation
3. **Network Testing**: Network storage connectivity testing
4. **Progress Testing**: Progress display functionality testing

## Performance Optimization

### Multi-Environment Performance

- **Process Isolation**: Efficient process isolation and communication
- **Resource Management**: Optimal resource allocation across environments
- **Error Recovery**: Fast error recovery and environment restart

### GPU Performance

- **Device Selection**: Optimal GPU device selection for workloads
- **Memory Management**: Efficient GPU memory allocation and monitoring
- **Conflict Resolution**: Minimal overhead GPU conflict resolution

## Development Guidelines

### Adding New Features

1. **Environment Compatibility**: Ensure compatibility with multi-environment setup
2. **Resource Management**: Consider GPU and system resource usage
3. **Error Handling**: Implement robust error handling for system operations
4. **Configuration**: Use configurable parameters for utility settings

### Code Standards

1. **Documentation**: Comprehensive docstrings for utility functions
2. **Error Messages**: Clear, actionable error messages for system issues
3. **Logging**: Appropriate logging for system operations
4. **Performance**: Consider efficiency for frequently used utilities

## Future Enhancements

### Planned Improvements

1. **Advanced GPU Management**: Enhanced GPU scheduling and load balancing
2. **Cloud Integration**: Support for cloud-based resource management
3. **Performance Monitoring**: Advanced performance monitoring and optimization
4. **Automated Troubleshooting**: Automated issue detection and resolution

### Research Directions

1. **Distributed Processing**: Support for distributed multi-machine processing
2. **Resource Optimization**: Advanced resource allocation and optimization
3. **Intelligent Caching**: Smart caching strategies for improved performance
4. **Predictive Management**: Predictive resource management and scaling

## Conclusion

The Utils module provides essential shared utilities for the PLC Diagram Processor pipeline. The system handles multi-environment coordination, GPU management, data management, and various system utilities with focus on reliability and performance.

The modular architecture supports both standalone utility usage and integrated pipeline operations, while comprehensive error handling and diagnostics help maintain system reliability. This documentation covers the current implementation and provides guidance for usage and troubleshooting.
