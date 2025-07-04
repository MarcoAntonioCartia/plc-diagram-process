# Workers Module Documentation

## Overview

The Workers module provides isolated execution workers for pipeline stages that require heavy dependencies. These workers run in separate Python environments to avoid dependency conflicts and provide clean isolation between different processing stages.

## Architecture

### Core Components

```
src/workers/
├── Core Workers
│   ├── training_worker.py        # YOLO training execution in yolo_env
│   ├── detection_worker.py       # Detection processing in yolo_env
│   └── ocr_worker.py             # OCR processing in ocr_env
└── Coordination
    └── __init__.py               # Worker module initialization
```

## File-by-File Documentation

### Core Workers

#### `training_worker.py`
**Purpose**: Isolated YOLO training execution worker for multi-environment mode
**Functionality**:
- Executes YOLO training in isolated `yolo_env` subprocess
- Handles input/output serialization via JSON files
- Provides error isolation and recovery for training operations
- Implements performance optimizations and cleanup procedures

**Execution Flow**:
1. **Input Processing**: Receives training parameters via JSON input file
2. **Environment Setup**: Configures isolated training environment
3. **Dependency Import**: Imports heavy dependencies (PyTorch, Ultralytics)
4. **Training Execution**: Runs `yolo11_train.py` as subprocess
5. **Result Processing**: Extracts training results and metrics
6. **Output Serialization**: Returns results via JSON output file
7. **Cleanup**: Optional automatic cleanup of old training runs

**Key Features**:
- **Direct Output**: No output capture or processing for clean performance
- **Worker Configuration**: Uses optimized worker count (8 workers)
- **Memory Management**: Removes problematic environment variables
- **Batch Size Limiting**: Prevents GPU memory issues with large batch sizes
- **Auto-Cleanup**: Optional cleanup of old training runs

**Input Format**:
```json
{
  "model_path": "path/to/pretrained/model.pt",
  "data_yaml_path": "path/to/data.yaml",
  "epochs": 50,
  "batch_size": 8,
  "patience": 20,
  "project_name": "plc_symbol_detector",
  "verbose": false,
  "auto_cleanup": false
}
```

**Output Format**:
```json
{
  "status": "success",
  "results": {
    "save_dir": "path/to/training/results",
    "epochs_completed": 50,
    "best_model_path": "path/to/best.pt",
    "metrics": {
      "precision": 0.85,
      "recall": 0.82,
      "mAP50": 0.88,
      "mAP50-95": 0.75
    }
  }
}
```

**Usage**: Invoked automatically by `MultiEnvironmentManager` during training

#### `detection_worker.py`
**Purpose**: Isolated detection processing worker for multi-environment mode
**Functionality**:
- Executes YOLO detection in isolated `yolo_env` subprocess
- Handles batch processing of image folders
- Provides error isolation for detection operations
- Manages detection result formatting and output

**Execution Flow**:
1. **Input Processing**: Receives detection parameters via JSON input file
2. **Environment Setup**: Configures isolated detection environment
3. **Model Loading**: Loads YOLO model in isolated environment
4. **Detection Processing**: Processes images or image folders
5. **Result Formatting**: Formats detection results for output
6. **Output Serialization**: Returns results via JSON output file

**Key Features**:
- **Batch Processing**: Efficient processing of multiple images
- **Model Caching**: Reuses loaded models for multiple detections
- **Result Formatting**: Standardized detection result format
- **Error Isolation**: Isolated error handling for detection failures

**Input Format**:
```json
{
  "pdf_path": "path/to/input.pdf",
  "output_dir": "path/to/output",
  "confidence_threshold": 0.25,
  "snippet_size": [1500, 1200],
  "overlap": 500
}
```

**Output Format**:
```json
{
  "status": "success",
  "results": {
    "detection_count": 42,
    "output_files": [
      "path/to/detection_results.json",
      "path/to/annotated_images.pdf"
    ],
    "processing_time": 125.3
  }
}
```

#### `ocr_worker.py`
**Purpose**: Isolated OCR processing worker for multi-environment mode
**Functionality**:
- Executes PaddleOCR text extraction in isolated `ocr_env` subprocess
- Handles detection result processing for text extraction
- Provides error isolation for OCR operations
- Manages text extraction result formatting and output

**Execution Flow**:
1. **Input Processing**: Receives OCR parameters via JSON input file
2. **Environment Setup**: Configures isolated OCR environment
3. **OCR Initialization**: Initializes PaddleOCR in isolated environment
4. **Text Extraction**: Processes detection results for text extraction
5. **Result Formatting**: Formats text extraction results
6. **Output Serialization**: Returns results via JSON output file

**Key Features**:
- **Detection Integration**: Processes detection results for text extraction
- **OCR Optimization**: Optimized PaddleOCR configuration for PLC diagrams
- **Coordinate Handling**: Proper coordinate system management
- **Error Isolation**: Isolated error handling for OCR failures

**Input Format**:
```json
{
  "detection_folder": "path/to/detections",
  "pdf_folder": "path/to/pdfs",
  "output_folder": "path/to/output",
  "confidence_threshold": 0.7,
  "language": "en"
}
```

**Output Format**:
```json
{
  "status": "success",
  "results": {
    "text_extraction_count": 28,
    "output_files": [
      "path/to/text_extraction.json",
      "path/to/text_overlay.pdf"
    ],
    "processing_time": 89.7
  }
}
```

## Integration with Pipeline System

### Multi-Environment Coordination

Workers integrate with the pipeline through the `MultiEnvironmentManager`:

1. **Worker Invocation**: Manager invokes workers via subprocess execution
2. **Parameter Passing**: JSON-based parameter serialization
3. **Result Collection**: JSON-based result deserialization
4. **Error Handling**: Isolated error handling and recovery

### Communication Protocol

```
Main Process → JSON Input → Worker Process → JSON Output → Main Process
     ↓              ↓            ↓              ↓            ↓
Coordinator → temp_input.json → Worker → temp_output.json → Result
```

### Environment Isolation

- **yolo_env**: Used by `training_worker.py` and `detection_worker.py`
- **ocr_env**: Used by `ocr_worker.py`
- **core_env**: Used by main coordination process

## Worker Execution Model

### Subprocess Execution

Workers are executed as separate Python processes:

```python
# Example worker invocation
subprocess.run([
    python_executable,
    worker_script_path,
    "--input", input_json_path,
    "--output", output_json_path
], timeout=timeout_seconds)
```

### Error Handling Strategy

1. **Input Validation**: Validate input parameters before processing
2. **Dependency Checking**: Verify required dependencies are available
3. **Processing Isolation**: Isolate processing errors from coordination
4. **Result Validation**: Validate output before returning results
5. **Cleanup**: Ensure proper cleanup on both success and failure

### Performance Considerations

- **Process Overhead**: Subprocess creation and communication overhead
- **Memory Isolation**: Each worker has isolated memory space
- **Resource Management**: Workers manage their own GPU/CPU resources
- **Timeout Protection**: Workers have execution timeout protection

## Usage Examples

### Training Worker Usage

```python
# Via MultiEnvironmentManager
from src.utils.multi_env_manager import MultiEnvironmentManager

manager = MultiEnvironmentManager(project_root)
training_config = {
    "model_path": "yolo11m.pt",
    "epochs": 50,
    "batch_size": 8
}
result = manager.run_training_pipeline(training_config)
```

### Detection Worker Usage

```python
# Via MultiEnvironmentManager
detection_config = {
    "pdf_path": "input.pdf",
    "output_dir": "output/",
    "confidence_threshold": 0.5
}
result = manager.run_detection_pipeline(detection_config)
```

### OCR Worker Usage

```python
# Via MultiEnvironmentManager
ocr_config = {
    "detection_folder": "detections/",
    "pdf_folder": "pdfs/",
    "output_folder": "output/"
}
result = manager.run_ocr_pipeline(ocr_config)
```

## Configuration

### Worker Environment Variables

- **PLCDP_MULTI_ENV**: Enable multi-environment mode ('1' or '0')
- **PLCDP_VERBOSE**: Enable verbose worker output ('1' or '0')
- **PLCDP_AUTO_CLEANUP**: Enable automatic cleanup ('1' or '0')
- **PLCDP_WORKER_TIMEOUT**: Worker execution timeout (default: 3600 seconds)

### Performance Parameters

- **workers**: Number of dataloader workers (default: 8)
- **timeout**: Worker execution timeout in seconds
- **memory_limit**: Memory limit per worker process
- **batch_size**: Processing batch size for workers

### Error Handling Parameters

- **max_retries**: Maximum retry attempts for failed workers
- **retry_delay**: Delay between retry attempts
- **error_threshold**: Error threshold for worker failure

## Error Handling and Troubleshooting

### Common Issues

1. **Worker Timeout**
   - **Cause**: Worker execution exceeding timeout limit
   - **Solution**: Increase timeout or optimize processing
   - **Parameter**: Increase `PLCDP_WORKER_TIMEOUT`

2. **Environment Dependencies Missing**
   - **Cause**: Required packages not installed in worker environment
   - **Solution**: Install missing dependencies in appropriate environment
   - **Command**: Activate environment and install packages

3. **JSON Serialization Errors**
   - **Cause**: Non-serializable objects in input/output data
   - **Solution**: Ensure all data is JSON-serializable
   - **Tool**: Validate input/output data structure

4. **GPU Memory Issues in Workers**
   - **Cause**: Multiple workers competing for GPU resources
   - **Solution**: Reduce batch size or use CPU processing
   - **Parameter**: Adjust batch size in worker configuration

### Debugging Tools

1. **Worker Logs**: Check worker output for detailed error information
2. **Environment Testing**: Test worker environments independently
3. **JSON Validation**: Validate input/output JSON structure
4. **Resource Monitoring**: Monitor worker resource usage

## Performance Optimization

### Worker Performance

- **Process Reuse**: Minimize subprocess creation overhead
- **Memory Management**: Efficient memory usage in isolated processes
- **Resource Allocation**: Optimal resource allocation per worker
- **Batch Processing**: Efficient batch processing within workers

### Communication Optimization

- **JSON Optimization**: Efficient JSON serialization/deserialization
- **File I/O**: Optimized temporary file handling
- **Error Propagation**: Efficient error information propagation

## Development Guidelines

### Adding New Workers

1. **Environment Isolation**: Ensure proper environment isolation
2. **JSON Interface**: Use standardized JSON input/output interface
3. **Error Handling**: Implement comprehensive error handling
4. **Resource Management**: Proper resource allocation and cleanup
5. **Documentation**: Document worker interface and usage

### Worker Interface Standards

1. **Command Line Interface**: Standardized `--input` and `--output` parameters
2. **JSON Schema**: Well-defined input/output JSON schemas
3. **Error Reporting**: Consistent error reporting format
4. **Exit Codes**: Proper exit code handling for success/failure

### Code Standards

1. **Documentation**: Comprehensive docstrings for worker functions
2. **Error Messages**: Clear, actionable error messages
3. **Logging**: Appropriate logging for worker operations
4. **Testing**: Unit tests for worker functionality

## Future Enhancements

### Planned Improvements

1. **Worker Pooling**: Persistent worker pools for improved performance
2. **Advanced Communication**: More efficient inter-process communication
3. **Resource Scheduling**: Advanced resource scheduling and allocation
4. **Monitoring**: Enhanced worker monitoring and diagnostics

### Research Directions

1. **Distributed Workers**: Support for distributed worker execution
2. **Dynamic Scaling**: Dynamic worker scaling based on workload
3. **Intelligent Scheduling**: Intelligent worker scheduling and load balancing
4. **Performance Optimization**: Advanced performance optimization techniques

## Conclusion

The Workers module provides essential isolated execution capabilities for the PLC Diagram Processor pipeline. The system handles heavy dependency isolation, inter-process communication, and error recovery with focus on reliability and performance.

The modular worker architecture enables clean separation of concerns while maintaining efficient communication between pipeline stages. This documentation covers the current implementation and provides guidance for usage, troubleshooting, and development.
