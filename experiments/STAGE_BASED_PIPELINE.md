# Stage-Based Pipeline Architecture

This document describes the new stage-based pipeline architecture that replaces the monolithic `run_pipeline.py` script with a modular, maintainable, and CI-friendly system.

## Overview

The stage-based pipeline breaks down the PLC diagram processing workflow into discrete, manageable stages that can be executed independently or as part of a complete pipeline. Each stage has clear inputs, outputs, and dependencies.

## Architecture Components

### 1. Core Components

- **`src/pipeline/base_stage.py`** - Abstract base class for all pipeline stages
- **`src/pipeline/stage_manager.py`** - Orchestrates stage execution and manages state
- **`src/run_pipeline.py`** - New command-line interface for the pipeline

### 2. Pipeline Stages

Located in `src/pipeline/stages/`:

1. **Preparation Stage** (`preparation_stage.py`)
   - Environment: `core`
   - Dependencies: None
   - Purpose: Validate inputs, setup directories, check environment health

2. **Training Stage** (`training_stage.py`)
   - Environment: `yolo_env`
   - Dependencies: `preparation`
   - Purpose: Train or validate YOLO models

3. **Detection Stage** (`detection_stage.py`)
   - Environment: `yolo_env`
   - Dependencies: `training`
   - Purpose: Run YOLO object detection on PDFs

4. **OCR Stage** (`ocr_stage.py`)
   - Environment: `ocr_env`
   - Dependencies: `detection`
   - Purpose: Extract text from detected regions

5. **Enhancement Stage** (`enhancement_stage.py`)
   - Environment: `core`
   - Dependencies: `ocr`
   - Purpose: Create CSV output and enhanced PDFs

### 3. Output Processing

Located in `src/output/`:

- **`csv_formatter.py`** - Formats text extraction results into CSV with area-based grouping
- **`area_grouper.py`** - Groups text regions by spatial areas and symbol associations

## Key Features

### 1. CI Safety

- **Automatic CI Detection**: Detects CI environments (`CI=true`, `GITHUB_ACTIONS=true`, `PYTEST_CURRENT_TEST`)
- **Mock Dependencies**: Uses lightweight mocks for heavy dependencies in CI
- **Lazy Imports**: Heavy dependencies are only imported when needed
- **Graceful Degradation**: Falls back to mock implementations when dependencies are missing

### 2. State Management

- **Persistent State**: Each stage saves its completion state to JSON files
- **Dependency Tracking**: Stages validate that their dependencies have completed
- **Resume Capability**: Skip completed stages and resume from where you left off
- **Force Re-run**: Option to force re-execution of specific stages

### 3. Multi-Environment Support

- **Environment Isolation**: Supports separate environments for YOLO and OCR to avoid CUDA conflicts
- **Single Environment**: Can also run in a single environment for simpler setups
- **Worker Process**: Uses subprocess workers for heavy operations

### 4. Flexible Execution

- **Individual Stages**: Run specific stages independently
- **Stage Groups**: Run subsets of stages
- **Complete Pipeline**: Run all stages in dependency order
- **Configuration Override**: Support for custom configurations

## Usage Examples

### Command Line Interface

```bash
# Show pipeline status
python src/run_pipeline.py --status

# List all available stages
python src/run_pipeline.py --list-stages

# Run complete pipeline
python src/run_pipeline.py --run-all

# Run specific stages
python src/run_pipeline.py --stages preparation training detection

# Force re-run specific stages
python src/run_pipeline.py --stages detection ocr --force

# Reset pipeline state
python src/run_pipeline.py --reset

# Reset specific stages
python src/run_pipeline.py --reset preparation training

# Run with multi-environment mode
python src/run_pipeline.py --run-all --multi-env

# Save execution results
python src/run_pipeline.py --run-all --save-results results.json

# JSON output for automation
python src/run_pipeline.py --status --json-output
```

### Programmatic Usage

```python
from src.pipeline.stage_manager import StageManager

# Initialize stage manager
manager = StageManager()

# Get pipeline status
status = manager.get_pipeline_status()
print(f"Progress: {status['overall_progress']:.1%}")

# Run specific stage
result = manager.run_single_stage('preparation')
if result.success:
    print("Preparation completed successfully")

# Run complete pipeline
summary = manager.run_stages()
if summary['success']:
    print(f"Pipeline completed in {summary['total_duration']:.2f}s")
```

## Stage Development

### Creating a New Stage

1. **Inherit from BaseStage**:
```python
from ..base_stage import BaseStage

class MyStage(BaseStage):
    def __init__(self, name="my_stage", description="My custom stage", 
                 required_env="core", dependencies=None):
        super().__init__(name, description, required_env, dependencies or [])
```

2. **Implement Required Methods**:
```python
def execute(self) -> Dict[str, Any]:
    """Main execution logic"""
    # Your stage logic here
    return {'status': 'success', 'data': 'result'}

def execute_ci_safe(self) -> Dict[str, Any]:
    """CI-safe execution without heavy dependencies"""
    return {'status': 'ci_mock', 'mock_mode': True}

def validate_inputs(self) -> bool:
    """Validate stage inputs"""
    # Check dependencies, input files, etc.
    return True
```

3. **Register the Stage**:
Add your stage to the `StageManager._register_default_stages()` method.

### Stage Best Practices

1. **CI Compatibility**: Always provide a `execute_ci_safe()` method
2. **Error Handling**: Use try-catch blocks and return meaningful error messages
3. **State Management**: Use `self.save_state()` and `self.load_state()` for persistence
4. **Dependency Validation**: Check that required inputs exist in `validate_inputs()`
5. **Progress Reporting**: Update progress for long-running operations
6. **Lazy Imports**: Import heavy dependencies only when needed

## Configuration

### Stage Configuration

Each stage can be configured through the stage manager:

```python
config = {
    'preparation': {
        'validate_pdfs': True
    },
    'training': {
        'model_name': 'yolo11m.pt',
        'force_training': False
    },
    'detection': {
        'confidence_threshold': 0.25,
        'snippet_size': [1500, 1200]
    },
    'ocr': {
        'ocr_confidence': 0.7,
        'ocr_lang': 'en'
    },
    'enhancement': {
        'area_grouping': True,
        'alphanumeric_sort': True
    }
}

manager.run_stages(config=config)
```

### Environment Variables

- `PLCDP_MULTI_ENV=1` - Enable multi-environment mode
- `PLCDP_VERBOSE=1` - Enable verbose logging
- `PLCDP_QUIET=1` - Suppress non-essential output
- `CI=true` - Detected automatically, enables CI-safe mode

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_pipeline_stages.py

# Run with verbose output
python -m pytest tests/ -v

# Run in CI mode
CI=true python -m pytest tests/
```

### Test Structure

- **`tests/test_pipeline_stages.py`** - Main test suite for pipeline stages
- **CI Compatibility** - All tests work in CI environments with mock dependencies
- **Integration Tests** - Test stage dependencies and execution order
- **Unit Tests** - Test individual stage functionality

## Migration from Old Pipeline

### Key Changes

1. **Modular Architecture**: Monolithic script split into discrete stages
2. **State Management**: Pipeline state is now persistent and resumable
3. **CI Safety**: No more import errors in CI environments
4. **Better Error Handling**: Clearer error messages and recovery options
5. **Flexible Execution**: Can run individual stages or custom combinations

### Migration Steps

1. **Update Scripts**: Replace calls to old `run_pipeline.py` with new stage-based commands
2. **Configuration**: Update configuration files to use new stage-based format
3. **Environment Setup**: Consider using multi-environment mode for better isolation
4. **Testing**: Update tests to use new stage-based architecture

### Backward Compatibility

The new pipeline maintains compatibility with existing data structures and output formats. Existing processed data can be used with the new pipeline without modification.

## Troubleshooting

### Common Issues

1. **Import Errors in CI**:
   - Solution: Ensure `CI=true` environment variable is set
   - The pipeline automatically uses mock implementations in CI

2. **Stage Dependencies**:
   - Solution: Check that prerequisite stages have completed successfully
   - Use `--reset` to clear problematic stage state

3. **Multi-Environment Issues**:
   - Solution: Ensure environments are properly set up using `MultiEnvironmentManager`
   - Check that worker scripts exist in `src/workers/`

4. **State File Corruption**:
   - Solution: Use `--reset` to clear corrupted state files
   - State files are stored in `.pipeline_state/` directory

### Debug Mode

Enable verbose output for debugging:

```bash
python src/run_pipeline.py --run-all --verbose
```

Or set environment variable:

```bash
export PLCDP_VERBOSE=1
python src/run_pipeline.py --run-all
```

## Future Enhancements

### Planned Features

1. **Web Interface**: Browser-based pipeline management and monitoring
2. **Parallel Execution**: Run independent stages in parallel
3. **Cloud Integration**: Support for cloud-based processing
4. **Advanced Scheduling**: Cron-like scheduling for automated processing
5. **Plugin System**: Easy integration of custom stages

### Extension Points

The architecture is designed for extensibility:

- **Custom Stages**: Easy to add new processing stages
- **Output Formats**: Pluggable output formatters
- **Environment Managers**: Support for different execution environments
- **State Backends**: Alternative storage for pipeline state

## Contributing

When contributing to the stage-based pipeline:

1. **Follow Stage Patterns**: Use existing stages as templates
2. **Maintain CI Compatibility**: Always provide CI-safe implementations
3. **Add Tests**: Include tests for new stages and functionality
4. **Update Documentation**: Keep this document updated with changes
5. **Consider Dependencies**: Minimize heavy dependencies and use lazy imports

## Support

For issues with the stage-based pipeline:

1. Check the troubleshooting section above
2. Run with `--verbose` for detailed logging
3. Check CI logs for import or dependency issues
4. Review stage state files in `.pipeline_state/`
5. Use `--reset` to clear problematic state

The stage-based architecture provides a robust, maintainable, and extensible foundation for the PLC diagram processing pipeline while maintaining full compatibility with existing workflows and data.
