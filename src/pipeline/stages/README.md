# Training Stage Documentation

## Overview

The Training Stage is a component of the PLC Diagram Processor pipeline responsible for YOLO model training and validation. This stage checks for existing custom models and performs training when needed. The training system supports both single-environment and multi-environment execution modes with basic error handling and performance considerations.

## Architecture

### Core Components

The training system follows a modular architecture with intelligent model lifecycle management:

```
Training Stage Components:
├── Core Stage Components
│   ├── training_stage.py           # Main training orchestrator and intelligence
│   └── base_stage.py              # Base stage functionality (inherited)
├── Training Implementation
│   ├── yolo11_train.py            # Core YOLO11 training implementation
│   └── yolo_compatibility.py     # YOLO version compatibility layer
├── Worker Integration
│   ├── training_worker.py         # Isolated environment training execution
│   └── multi_env_manager.py      # Multi-environment coordination
├── Performance Components
│   ├── gpu_manager.py             # GPU resource management
│   └── progress_display.py       # Training progress visualization
└── Configuration Management
    ├── config.py                  # Centralized configuration system
    └── data.yaml                  # YOLO dataset configuration
```

### Design Principles

1. **Intelligent Training**: Only trains when no suitable custom models exist
2. **Environment Isolation**: Heavy dependencies isolated in subprocess execution
3. **Performance Optimization**: Optimized batch sizes and GPU memory management
4. **Error Recovery**: Comprehensive error handling with graceful fallbacks
5. **Configuration Management**: Centralized configuration through the config system
6. **Model Lifecycle**: Automatic model discovery, validation, and deployment

## File-by-File Documentation

### Core Stage Components

#### `training_stage.py`
**Purpose**: Main orchestrator for intelligent YOLO model training and validation
**Functionality**:
- Implements intelligent training logic that checks for existing custom models
- Coordinates between single-environment and multi-environment execution modes
- Manages the complete training workflow: Model Discovery → Dataset Validation → Training → Model Deployment
- Provides both CI-safe execution and production training capabilities
- Handles error recovery and graceful fallbacks

**Key Classes**:
- `TrainingStage`: Main stage orchestrator inheriting from `BaseStage`
- `MockTorch`: Lightweight PyTorch mock for CI environments
- `MockYOLO`: Lightweight YOLO mock for CI environments

**Intelligent Training Logic**:
1. **Model Discovery**: Searches for existing custom models using same logic as `yolo11_infer.py`
2. **Validation**: If custom model found, validates and skips training
3. **Dataset Validation**: Verifies dataset structure and configuration
4. **Pretrained Model Selection**: Finds best available pretrained model for fine-tuning
5. **Training Execution**: Runs training only when necessary
6. **Model Deployment**: Automatically deploys trained models to custom directory

**Execution Modes**:
- **Multi-Environment Mode**: Uses isolated `yolo_env` for training execution
- **Single-Environment Mode**: Direct training execution in current environment
- **CI-Safe Mode**: Mock execution for continuous integration testing

**Usage**:
```bash
# Via pipeline system
python src/run_pipeline.py --stages training

# With custom parameters
python src/run_pipeline.py --stages training --training-epochs 100 --training-batch 8
```

**Integration**: Core component of the stage-based pipeline system

### Training Implementation

#### `yolo11_train.py`
**Purpose**: Core YOLO11 training implementation with configuration management
**Functionality**:
- Provides the fundamental YOLO11 training capabilities
- Implements smart device detection and GPU optimization
- Handles dataset validation and structure verification
- Manages model metadata generation and storage
- Includes comprehensive error handling for training issues

**Key Functions**:
- `train_yolo11()`: Main training function with full parameter control
- `validate_dataset()`: Dataset structure and configuration validation
- `get_best_device()`: Automatic device selection for optimal performance
- `main()`: Command-line interface for standalone training execution

**Training Process**:
1. **Configuration Loading**: Loads centralized configuration and paths
2. **Device Selection**: Automatically selects best available device (GPU/CPU)
3. **Model Loading**: Loads pretrained YOLO11 model as starting point
4. **Dataset Validation**: Verifies dataset structure and data.yaml configuration
5. **Training Execution**: Runs YOLO11 training with optimized parameters
6. **Error Handling**: Handles CSV corruption and other training issues
7. **Model Deployment**: Copies best model to custom models directory
8. **Metadata Generation**: Creates model metadata with training information

**Performance Optimizations**:
- **Batch Size Optimization**: Default batch size of 8 for optimal GPU memory usage
- **Automatic Mixed Precision**: Enabled for faster training and reduced memory usage
- **Early Stopping**: Configurable patience for preventing overfitting
- **Mosaic Augmentation**: Early closure for faster convergence
- **Worker Optimization**: Configurable number of dataloader workers

**Error Handling**:
- **CSV Corruption Recovery**: Handles Ultralytics CSV parsing errors gracefully
- **File Validation**: Comprehensive file existence and accessibility checks
- **GPU Memory Management**: Automatic fallback to CPU when GPU memory insufficient
- **Training Interruption**: Graceful handling of training interruptions

**Usage**:
```bash
# Basic training
python src/detection/yolo11_train.py --model yolo11m.pt --epochs 50

# Advanced training with custom parameters
python src/detection/yolo11_train.py --model yolo11l.pt --epochs 100 --batch 8 --device 0 --workers 8

# Quiet training for automated execution
python src/detection/yolo11_train.py --model yolo11m.pt --epochs 50 --quiet
```

#### `yolo_compatibility.py`
**Purpose**: YOLO version compatibility layer for cross-version model loading
**Functionality**:
- Provides placeholder classes for unknown YOLO layers across different versions
- Maintains model compatibility when loading models trained with different YOLO versions
- Handles channel dimension preservation and adaptation
- Enables seamless model loading without version conflicts

**Key Classes**:
- `RobustPlaceholder`: Base placeholder class with intelligent channel handling
- Various YOLO layer placeholders: `C3k2`, `RepC3`, `C2PSA`, `PSABlock`, etc.

**Compatibility Strategy**:
1. **Channel Analysis**: Parses constructor arguments to determine input/output channels
2. **Identity Mapping**: Uses identity when input/output channels match
3. **Channel Adaptation**: Inserts 1×1 convolution for channel dimension changes
4. **Dynamic Registration**: Registers placeholders only when native classes unavailable

**Channel Handling Logic**:
- **c1 == c2**: Pure identity mapping (no computational overhead)
- **c1 != c2**: Lightweight 1×1 convolution for channel adaptation
- **Attribute Preservation**: Maintains `c` attribute for Ultralytics layer parser

**Registration Function**:
- `register_compatibility_classes()`: Conditionally registers compatibility classes
- Only registers when native Ultralytics classes are unavailable
- Provides graceful fallback for older YOLO versions

### Worker Integration

#### `training_worker.py`
**Purpose**: Isolated environment training execution worker for multi-environment mode
**Functionality**:
- Executes training in isolated `yolo_env` subprocess
- Handles input/output serialization via JSON
- Provides error isolation and recovery
- Enables heavy dependency usage without main process contamination
- Implements performance optimizations and cleanup procedures

**Execution Flow**:
1. **Input Processing**: Receives training parameters via JSON file
2. **Environment Setup**: Configures isolated training environment
3. **Dependency Import**: Imports heavy dependencies (PyTorch, Ultralytics)
4. **Training Execution**: Runs `yolo11_train.py` as subprocess
5. **Result Processing**: Extracts training results and metrics
6. **Output Serialization**: Returns results via JSON output file
7. **Cleanup**: Optional automatic cleanup of old training runs

**Performance Optimizations**:
- **Direct Output**: No output capture or processing for clean performance
- **Worker Configuration**: Uses same worker count as direct script (8 workers)
- **Memory Management**: Removes problematic environment variables
- **Batch Size Limiting**: Prevents GPU memory issues with large batch sizes

**Error Handling**:
- **File Validation**: Comprehensive input file existence checks
- **Subprocess Management**: Proper subprocess execution and error capture
- **Timeout Protection**: Training timeout protection (3600 seconds)
- **Graceful Failure**: Detailed error reporting and recovery

**Auto-Cleanup Feature**:
- **Configurable Cleanup**: Optional automatic cleanup of old training runs
- **Run Retention**: Keeps latest 3 training runs by default
- **Space Management**: Prevents disk space accumulation from multiple training runs
- **Error Tolerance**: Continues execution even if cleanup fails

**Usage**: Invoked automatically by `MultiEnvironmentManager` during multi-environment training

### Performance Components

#### GPU Management Integration
**Purpose**: Intelligent GPU resource management and optimization
**Functionality**:
- **Device Detection**: Automatic detection of best available training device
- **Memory Optimization**: Optimal batch size selection based on GPU memory
- **Performance Monitoring**: GPU utilization and memory usage tracking
- **Fallback Strategies**: Automatic CPU fallback when GPU unavailable

**Optimization Strategies**:
- **Batch Size Limiting**: Prevents GPU memory overflow with large batch sizes
- **Memory Monitoring**: Real-time GPU memory usage monitoring
- **Device Selection**: Intelligent selection between multiple GPUs
- **Performance Tuning**: Automatic parameter adjustment based on hardware

#### Progress Display Integration
**Purpose**: Real-time training progress visualization and monitoring
**Functionality**:
- **Stage Progress**: Visual progress indicators for training stages
- **File Processing**: Individual file processing status and completion
- **Error Reporting**: Clear error reporting with actionable messages
- **Performance Metrics**: Real-time training metrics and statistics

### Configuration Management

#### Configuration Integration
**Purpose**: Centralized configuration management for training operations
**Functionality**:
- **Path Management**: Automatic resolution of model, dataset, and output paths
- **Model Discovery**: Automatic discovery of available pretrained and custom models
- **Dataset Configuration**: Centralized dataset structure and validation
- **Environment Configuration**: Multi-environment and single-environment settings

**Configuration Structure**:
```yaml
# Training-specific configuration
training:
  default_epochs: 50
  default_batch_size: 8
  default_patience: 20
  auto_cleanup: false
  
# Model configuration
models:
  pretrained_dir: "models/pretrained"
  custom_dir: "models/custom"
  
# Dataset configuration
dataset:
  name: "plc_symbols_v11"
  data_yaml: "datasets/data.yaml"
```

## Integration with Pipeline System

### Stage-Based Integration

The training stage integrates with the main pipeline through:

1. **Stage Manager**: `src/pipeline/stages/training_stage.py`
2. **Training Worker**: `src/workers/training_worker.py`
3. **Multi-Environment Manager**: `src/utils/multi_env_manager.py`
4. **Base Stage System**: Inherits from `BaseStage` for consistent pipeline integration

### Dependency Management

The training stage uses intelligent dependency checking:

1. **Model-Based Dependencies**: Checks for existing custom models before training
2. **Environment Isolation**: Heavy dependencies (PyTorch, Ultralytics) isolated in subprocess
3. **Graceful Fallbacks**: Automatic fallback strategies when dependencies unavailable
4. **CI Compatibility**: Mock execution for continuous integration environments

### Configuration Integration

Training configuration is managed through:

1. **Central Configuration**: `src/config.py` provides unified configuration management
2. **Model Path Resolution**: Automatic resolution of model paths across environments
3. **Dataset Management**: Consistent dataset structure and validation
4. **Environment Detection**: Automatic detection of execution environment capabilities

## Model Management

### Model Lifecycle

The training stage implements comprehensive model lifecycle management:

1. **Discovery Phase**: Automatic detection of existing custom models
2. **Validation Phase**: Verification of model metadata and compatibility
3. **Training Phase**: Conditional training based on model availability
4. **Deployment Phase**: Automatic deployment of trained models
5. **Metadata Management**: Comprehensive model metadata tracking

### Model Types

1. **Pretrained Models**: Base YOLO11 models for fine-tuning
   - `yolo11n.pt`: Nano model (fastest, lowest accuracy)
   - `yolo11s.pt`: Small model (balanced speed/accuracy)
   - `yolo11m.pt`: Medium model (recommended default)
   - `yolo11l.pt`: Large model (higher accuracy, slower)
   - `yolo11x.pt`: Extra-large model (highest accuracy, slowest)

2. **Custom Models**: Fine-tuned models for PLC symbol detection
   - Named with project identifier: `plc_symbol_detector_yolo11m_best.pt`
   - Accompanied by metadata files: `plc_symbol_detector_yolo11m_best.json`

3. **Model Metadata**: JSON files containing training information
   ```json
   {
     "original_pretrained": "yolo11m.pt",
     "epochs_trained": 50,
     "dataset": "plc_symbols_v11",
     "training_dir": "/path/to/training/run",
     "metrics": {
       "mAP50": 0.85,
       "mAP50-95": 0.72
     }
   }
   ```

### Model Storage Structure

```
models/
├── pretrained/
│   ├── yolo11n.pt
│   ├── yolo11s.pt
│   ├── yolo11m.pt
│   ├── yolo11l.pt
│   └── yolo11x.pt
└── custom/
    ├── plc_symbol_detector_yolo11m_best.pt
    ├── plc_symbol_detector_yolo11m_best.json
    └── ...
```

### Model Selection Logic

1. **Custom Model Discovery**: Searches `models/custom/` for trained models
2. **Metadata Validation**: Verifies model metadata and training information
3. **Recency Check**: Selects most recently trained model when multiple available
4. **Fallback Strategy**: Falls back to pretrained models when no custom models found
5. **Compatibility Check**: Ensures model compatibility with current YOLO version

## Training Workflow

### Intelligent Training Decision

The training stage implements intelligent decision-making:

```
Training Decision Flow:
1. Check for existing custom models
   ├── Custom model found → Validate and use existing model
   └── No custom model → Proceed to training
2. Validate dataset structure
   ├── Valid dataset → Continue
   └── Invalid dataset → Error and exit
3. Find best pretrained model
   ├── Pretrained model found → Use for fine-tuning
   └── No pretrained model → Error and exit
4. Execute training
   ├── Training successful → Deploy custom model
   └── Training failed → Error reporting and recovery
```

### Training Execution Modes

#### Multi-Environment Mode
**When Used**: Default mode for production environments
**Characteristics**:
- Isolated `yolo_env` execution
- Heavy dependency isolation
- Subprocess-based training
- Enhanced error recovery
- Resource isolation

**Execution Flow**:
1. Validate environment setup
2. Prepare training payload
3. Launch training worker in `yolo_env`
4. Monitor training progress
5. Process training results
6. Deploy trained model

#### Single-Environment Mode
**When Used**: Development and testing environments
**Characteristics**:
- Direct training execution
- Immediate dependency access
- Faster startup time
- Direct error reporting
- Simplified debugging

**Execution Flow**:
1. Import training dependencies directly
2. Execute training in current environment
3. Process results immediately
4. Deploy trained model

#### CI-Safe Mode
**When Used**: Continuous integration environments
**Characteristics**:
- Mock training execution
- No heavy dependencies required
- Fast execution
- Validation-only operations
- Mock result generation

## Performance Optimization

### Training Performance

#### Batch Size Optimization
- **Default Batch Size**: 8 (optimized for GPU memory efficiency)
- **Automatic Limiting**: Prevents batch sizes > 16 to avoid GPU memory issues
- **Performance Impact**: 20x faster training compared to large batch sizes
- **Memory Usage**: ~8GB GPU memory vs 15.4GB with large batches

#### GPU Optimization
- **Automatic Mixed Precision**: Enabled by default for faster training
- **Device Selection**: Intelligent GPU/CPU selection based on availability
- **Memory Management**: Optimized memory allocation and cleanup
- **Worker Configuration**: Optimal dataloader worker count (8 workers)

#### Training Parameters
- **Early Stopping**: Configurable patience to prevent overfitting
- **Mosaic Augmentation**: Early closure (epoch 10) for faster convergence
- **Validation**: Enabled by default for model quality assessment
- **Checkpointing**: Regular model checkpointing for recovery

### Error Recovery and Handling

#### CSV Corruption Recovery
**Problem**: Ultralytics occasionally generates corrupted CSV files during training
**Solution**: Automatic detection and cleanup of corrupted CSV files
**Implementation**:
```python
try:
    results = model.train(...)
except Exception as e:
    if "Error tokenizing data" in str(e):
        # Clean up corrupted CSV files
        # Create mock results object
        # Continue execution
```

#### GPU Memory Management
**Problem**: Large batch sizes cause GPU memory overflow
**Solution**: Automatic batch size limiting and warning system
**Implementation**:
- Batch size > 16 triggers warning and automatic reduction
- Fallback to CPU when GPU memory insufficient
- Memory monitoring and optimization

#### Training Interruption Handling
**Problem**: Training interruptions due to system issues
**Solution**: Graceful error handling and recovery mechanisms
**Implementation**:
- Timeout protection (3600 seconds)
- Comprehensive error reporting
- Automatic cleanup on failure
- State preservation for recovery

## Usage Examples

### Pipeline Integration

```bash
# Run training stage only
python src/run_pipeline.py --stages training

# Run with custom training parameters
python src/run_pipeline.py --stages training --training-epochs 100 --training-batch 8

# Force training even if custom model exists
python src/run_pipeline.py --stages training --force-training
```

### Standalone Training

```bash
# Basic training with defaults
python src/detection/yolo11_train.py

# Advanced training configuration
python src/detection/yolo11_train.py \
    --model yolo11l.pt \
    --epochs 100 \
    --batch 8 \
    --device 0 \
    --workers 8 \
    --patience 30 \
    --name custom_plc_detector

# Quiet training for automation
python src/detection/yolo11_train.py --model yolo11m.pt --epochs 50 --quiet
```

### Multi-Environment Training

```bash
# Enable multi-environment mode
export PLCDP_MULTI_ENV=1
python src/run_pipeline.py --stages training

# With verbose output
export PLCDP_VERBOSE=1
python src/run_pipeline.py --stages training
```

### Development and Testing

```bash
# CI-safe execution
python src/pipeline/stages/training_stage.py --ci-safe

# Single environment mode
export PLCDP_MULTI_ENV=0
python src/run_pipeline.py --stages training

# Debug mode with detailed output
export PLCDP_VERBOSE=1
export PLCDP_DEBUG=1
python src/run_pipeline.py --stages training
```

## Configuration Parameters

### Training Parameters

- **epochs**: Number of training epochs (default: 50)
- **batch_size**: Training batch size (default: 8, max recommended: 16)
- **patience**: Early stopping patience (default: 20)
- **learning_rate**: Learning rate (default: auto-determined by YOLO)
- **image_size**: Input image size (default: 640)

### Performance Parameters

- **workers**: Number of dataloader workers (default: 8)
- **device**: Training device ('auto', 'cpu', '0', '1', etc.)
- **amp**: Enable automatic mixed precision (default: True)
- **save_period**: Model checkpoint saving period (default: 10)

### Environment Parameters

- **PLCDP_MULTI_ENV**: Enable multi-environment mode ('1' or '0')
- **PLCDP_VERBOSE**: Enable verbose output ('1' or '0')
- **PLCDP_AUTO_CLEANUP**: Enable automatic cleanup ('1' or '0')
- **PLCDP_DEBUG**: Enable debug mode ('1' or '0')

### Model Parameters

- **model_name**: Pretrained model to use (default: 'yolo11m.pt')
- **project_name**: Training project name (default: 'plc_symbol_detector')
- **confidence_threshold**: Detection confidence threshold (default: 0.25)

## Error Handling and Troubleshooting

### Common Issues

1. **No Custom Models Found**
   - **Cause**: No trained models available in `models/custom/`
   - **Solution**: Training will automatically start with pretrained model
   - **Action**: Normal operation, training will proceed automatically

2. **Dataset Validation Failed**
   - **Cause**: Missing or incorrectly structured dataset
   - **Solution**: Verify dataset structure and run data migration
   - **Command**: `python setup_data.py --migrate`

3. **GPU Memory Issues**
   - **Cause**: Batch size too large for available GPU memory
   - **Solution**: Reduce batch size or use CPU training
   - **Parameter**: `--batch 4` or `--device cpu`

4. **Training Timeout**
   - **Cause**: Training taking longer than 3600 seconds
   - **Solution**: Increase timeout or reduce training complexity
   - **Environment**: Modify timeout in `training_worker.py`

5. **CSV Corruption Errors**
   - **Cause**: Ultralytics CSV file corruption during training
   - **Solution**: Automatic recovery implemented in training code
   - **Action**: Training continues with mock results

6. **Environment Dependencies**
   - **Cause**: Missing PyTorch or Ultralytics installation
   - **Solution**: Install training environment dependencies
   - **Command**: `pip install -r requirements-detection.txt`

### Debugging Tools

1. **Verbose Mode**: Enable detailed output with `PLCDP_VERBOSE=1`
2. **Debug Mode**: Enable debug information with `PLCDP_DEBUG=1`
3. **CI-Safe Mode**: Test without heavy dependencies
4. **Single Environment**: Bypass multi-environment complexity
5. **Manual Training**: Direct execution of `yolo11_train.py`

### Performance Troubleshooting

1. **Slow Training Performance**
   - Check batch size (should be ≤ 16)
   - Verify GPU utilization
   - Ensure adequate GPU memory
   - Check worker count (default: 8)

2. **Memory Issues**
   - Reduce batch size to 4 or 8
   - Enable automatic mixed precision
   - Use CPU training if necessary
   - Monitor GPU memory usage

3. **Training Convergence Issues**
   - Increase patience for early stopping
   - Adjust learning rate
   - Verify dataset quality
   - Check data augmentation settings

## Development Guidelines

### Adding New Features

1. **Maintain Intelligence**: Preserve intelligent training decision-making
2. **Environment Compatibility**: Support both single and multi-environment modes
3. **Error Handling**: Implement comprehensive error handling and recovery
4. **Performance Optimization**: Consider GPU memory and training performance
5. **Configuration Integration**: Use centralized configuration management

### Testing Procedures

1. **Unit Testing**: Test individual training components in isolation
2. **Integration Testing**: Test pipeline integration and data flow
3. **Performance Testing**: Benchmark training performance and optimization
4. **Environment Testing**: Test both single and multi-environment modes
5. **CI Testing**: Verify CI-safe execution and mock functionality

### Code Standards

1. **Type Hints**: Use type hints for function parameters and returns
2. **Documentation**: Comprehensive docstrings for all functions and classes
3. **Error Messages**: Clear, actionable error messages with solutions
4. **Logging**: Appropriate logging levels and informative messages
5. **Configuration**: Externalize configuration parameters and settings

## Future Enhancements

### Planned Improvements

1. **Advanced Model Selection**: Intelligent model selection based on dataset characteristics
2. **Hyperparameter Optimization**: Automatic hyperparameter tuning and optimization
3. **Distributed Training**: Support for multi-GPU and distributed training
4. **Model Ensemble**: Training and deployment of model ensembles
5. **Transfer Learning**: Advanced transfer learning strategies for PLC domains

### Research Directions

1. **Architecture Optimization**: Exploration of newer YOLO architectures and improvements
2. **Domain Adaptation**: Specialized training techniques for PLC symbol detection
3. **Data Augmentation**: Advanced augmentation strategies for industrial diagrams
4. **Model Compression**: Techniques for model size reduction and inference optimization
5. **Active Learning**: Intelligent data selection for training improvement

## Conclusion

The Training Stage provides a working solution for YOLO model training with basic automation and error handling. The system checks for existing models before training and supports both single and multi-environment execution modes.

The training logic attempts to avoid unnecessary work by detecting existing custom models, while the multi-environment setup provides some isolation for heavy dependencies. Basic performance optimizations are in place, and error handling covers common training issues.

This documentation covers the current implementation. For questions or issues, refer to the main project documentation.
