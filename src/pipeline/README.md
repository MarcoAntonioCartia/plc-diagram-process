# Pipeline Module Documentation

## Overview

The Pipeline module provides the core stage-based execution framework for the PLC Diagram Processor. This module orchestrates the execution of different processing stages in a coordinated manner, handling dependencies, error recovery, and progress tracking.

## Architecture

### Core Components

```
src/pipeline/
├── Core Framework
│   ├── base_stage.py           # Base stage class and common functionality
│   └── stage_manager.py        # Stage execution coordination and management
├── Stage Implementations
│   └── stages/                 # Individual stage implementations
│       ├── preparation_stage.py    # Data preparation and validation
│       ├── training_stage.py       # Model training and validation
│       ├── detection_stage.py      # Object detection processing
│       ├── ocr_stage.py            # Text extraction processing
│       └── enhancement_stage.py    # PDF enhancement and output
└── Coordination
    └── __init__.py             # Pipeline module initialization
```

## File-by-File Documentation

### Core Framework

#### `base_stage.py`
**Purpose**: Base class providing common functionality for all pipeline stages
**Functionality**:
- Defines standard stage interface and lifecycle methods
- Provides dependency checking and validation
- Handles stage configuration and parameter management
- Includes error handling and recovery mechanisms

**Key Classes**:
- `BaseStage`: Abstract base class for all pipeline stages

**Key Methods**:
- `execute()`: Main stage execution method (abstract)
- `validate_inputs()`: Input validation and dependency checking
- `check_dependencies()`: Dependency stage validation
- `get_stage_config()`: Stage-specific configuration retrieval

**Stage Lifecycle**:
1. **Initialization**: Stage setup and configuration loading
2. **Validation**: Input validation and dependency checking
3. **Execution**: Main stage processing logic
4. **Cleanup**: Resource cleanup and finalization

**Usage**:
```python
from src.pipeline.base_stage import BaseStage

class CustomStage(BaseStage):
    def __init__(self, name="custom", description="Custom processing stage"):
        super().__init__(name, description, required_env="core", dependencies=[])
    
    def execute(self):
        # Stage implementation
        return {"status": "success"}
```

#### `stage_manager.py`
**Purpose**: Coordination and management of stage execution
**Functionality**:
- Manages stage execution order and dependencies
- Handles inter-stage communication and data flow
- Provides progress tracking and error recovery
- Coordinates multi-environment stage execution

**Key Classes**:
- `StageManager`: Main stage coordination class

**Key Functions**:
- `execute_stages()`: Execute specified stages in dependency order
- `validate_stage_dependencies()`: Validate stage dependency graph
- `handle_stage_failure()`: Stage failure recovery and error handling
- `track_stage_progress()`: Progress tracking and reporting

**Execution Flow**:
1. **Stage Discovery**: Identify available stages and dependencies
2. **Dependency Resolution**: Resolve stage execution order
3. **Stage Execution**: Execute stages in dependency order
4. **Progress Tracking**: Track and report execution progress
5. **Error Handling**: Handle stage failures and recovery

## Stage Implementations

The `stages/` subdirectory contains individual stage implementations. Each stage inherits from `BaseStage` and implements specific processing functionality:

### Available Stages

1. **Preparation Stage** (`preparation_stage.py`)
   - **Purpose**: Data preparation and validation
   - **Dependencies**: None (entry point)
   - **Environment**: Core environment
   - **Functionality**: Dataset validation, directory setup, configuration validation

2. **Training Stage** (`training_stage.py`)
   - **Purpose**: YOLO model training and validation
   - **Dependencies**: Preparation stage
   - **Environment**: yolo_env (multi-environment mode)
   - **Functionality**: Model training, validation, deployment

3. **Detection Stage** (`detection_stage.py`)
   - **Purpose**: Object detection processing
   - **Dependencies**: Training stage (for custom models)
   - **Environment**: yolo_env (multi-environment mode)
   - **Functionality**: PDF processing, object detection, result generation

4. **OCR Stage** (`ocr_stage.py`)
   - **Purpose**: Text extraction from detection results
   - **Dependencies**: Detection stage
   - **Environment**: ocr_env (multi-environment mode)
   - **Functionality**: Text extraction, coordinate calibration, result formatting

5. **Enhancement Stage** (`enhancement_stage.py`)
   - **Purpose**: PDF enhancement and final output generation
   - **Dependencies**: OCR stage
   - **Environment**: Core environment
   - **Functionality**: PDF enhancement, result compilation, output formatting

### Stage Dependencies

```
Preparation → Training → Detection → OCR → Enhancement
     ↓           ↓          ↓        ↓         ↓
   Setup    Model Train  Detection  Text    Final
   Data     & Deploy    Processing Extract  Output
```

## Integration with Pipeline System

### Main Pipeline Integration

The pipeline module integrates with the main execution system through:

1. **Main Runner**: `src/run_pipeline.py` uses the stage manager
2. **Configuration System**: Centralized configuration management
3. **Multi-Environment Manager**: Coordination with isolated environments
4. **Progress Display**: Unified progress tracking and reporting

### Execution Modes

#### Sequential Execution
- **Default Mode**: Stages execute in dependency order
- **Error Handling**: Failure in one stage stops subsequent stages
- **Progress Tracking**: Linear progress through stage sequence

#### Selective Execution
- **Stage Selection**: Execute only specified stages
- **Dependency Validation**: Ensure required dependencies are met
- **Force Execution**: Override dependency checks when needed

#### Multi-Environment Execution
- **Environment Isolation**: Heavy stages run in isolated environments
- **Resource Management**: Optimal resource allocation per environment
- **Error Recovery**: Environment-specific error handling and recovery

## Usage Examples

### Basic Stage Execution

```python
from src.pipeline.stage_manager import StageManager

# Initialize stage manager
manager = StageManager()

# Execute all stages
result = manager.execute_stages(["preparation", "training", "detection"])

# Execute specific stages
result = manager.execute_stages(["detection"], force=True)
```

### Custom Stage Implementation

```python
from src.pipeline.base_stage import BaseStage

class CustomProcessingStage(BaseStage):
    def __init__(self):
        super().__init__(
            name="custom_processing",
            description="Custom processing stage",
            required_env="core",
            dependencies=["preparation"]
        )
    
    def execute(self):
        # Implement custom processing logic
        print("Executing custom processing...")
        
        # Return stage results
        return {
            "status": "success",
            "files_processed": 42,
            "processing_time": 125.3
        }
    
    def validate_inputs(self):
        # Implement input validation
        return True
```

### Stage Manager Configuration

```python
# Configure stage manager
manager = StageManager(
    config_path="config.yaml",
    multi_env_enabled=True,
    progress_tracking=True
)

# Set stage-specific configuration
manager.set_stage_config("training", {
    "epochs": 100,
    "batch_size": 8,
    "patience": 30
})

# Execute with configuration
result = manager.execute_stages(["training"])
```

## Configuration

### Pipeline Configuration

- **stage_execution_order**: Custom stage execution order
- **multi_environment_enabled**: Enable multi-environment execution
- **progress_tracking_enabled**: Enable progress tracking
- **error_recovery_enabled**: Enable automatic error recovery

### Stage Configuration

Each stage can have specific configuration parameters:

```yaml
stages:
  preparation:
    validate_dataset: true
    setup_directories: true
  
  training:
    epochs: 50
    batch_size: 8
    patience: 20
  
  detection:
    confidence_threshold: 0.25
    snippet_size: [1500, 1200]
  
  ocr:
    confidence_threshold: 0.7
    language: "en"
```

### Environment Configuration

- **PLCDP_MULTI_ENV**: Enable multi-environment mode ('1' or '0')
- **PLCDP_STAGE_TIMEOUT**: Stage execution timeout (seconds)
- **PLCDP_PROGRESS_ENABLED**: Enable progress tracking ('1' or '0')

## Error Handling and Troubleshooting

### Common Issues

1. **Stage Dependency Failures**
   - **Cause**: Required dependency stage not completed successfully
   - **Solution**: Execute dependency stages first or use force mode
   - **Parameter**: Use `force=True` to override dependency checks

2. **Environment Setup Issues**
   - **Cause**: Multi-environment setup problems
   - **Solution**: Validate environment configuration and dependencies
   - **Tool**: Check environment health with multi-environment manager

3. **Stage Timeout Issues**
   - **Cause**: Stage execution exceeding timeout limits
   - **Solution**: Increase timeout or optimize stage processing
   - **Parameter**: Increase `PLCDP_STAGE_TIMEOUT`

4. **Configuration Issues**
   - **Cause**: Invalid or missing stage configuration
   - **Solution**: Validate configuration files and parameters
   - **Tool**: Use configuration validation utilities

### Debugging Tools

1. **Stage Validation**: Validate individual stage functionality
2. **Dependency Checking**: Verify stage dependency graph
3. **Progress Monitoring**: Monitor stage execution progress
4. **Error Logging**: Detailed error logging and reporting

## Performance Optimization

### Stage Performance

- **Parallel Execution**: Execute independent stages in parallel
- **Resource Management**: Optimal resource allocation per stage
- **Caching**: Cache stage results for repeated executions
- **Optimization**: Stage-specific performance optimizations

### Pipeline Performance

- **Execution Planning**: Optimal stage execution planning
- **Resource Scheduling**: Efficient resource scheduling across stages
- **Progress Optimization**: Efficient progress tracking and reporting

## Development Guidelines

### Adding New Stages

1. **Inherit from BaseStage**: Use the base stage class for consistency
2. **Define Dependencies**: Clearly specify stage dependencies
3. **Implement Interface**: Implement required abstract methods
4. **Error Handling**: Include comprehensive error handling
5. **Documentation**: Document stage functionality and usage

### Stage Interface Standards

1. **Execution Method**: Implement `execute()` method with proper return format
2. **Validation Method**: Implement `validate_inputs()` for input validation
3. **Configuration**: Support stage-specific configuration parameters
4. **Error Reporting**: Consistent error reporting and logging

### Code Standards

1. **Documentation**: Comprehensive docstrings for stage methods
2. **Error Messages**: Clear, actionable error messages
3. **Logging**: Appropriate logging for stage operations
4. **Testing**: Unit tests for stage functionality

## Future Enhancements

### Planned Improvements

1. **Parallel Stage Execution**: Support for parallel execution of independent stages
2. **Advanced Dependency Management**: More sophisticated dependency resolution
3. **Dynamic Stage Loading**: Dynamic loading of stage implementations
4. **Performance Monitoring**: Advanced performance monitoring and optimization

### Research Directions

1. **Distributed Pipeline**: Support for distributed pipeline execution
2. **Intelligent Scheduling**: AI-based stage scheduling and optimization
3. **Adaptive Execution**: Adaptive execution based on system resources
4. **Pipeline Optimization**: Advanced pipeline optimization techniques

## Conclusion

The Pipeline module provides a working stage-based execution framework for the PLC Diagram Processor. The system handles stage coordination, dependency management, and execution flow with support for multi-environment operation.

The modular architecture enables easy addition of new stages while maintaining consistent execution patterns and error handling. This documentation covers the current implementation and provides guidance for usage, development, and troubleshooting.
