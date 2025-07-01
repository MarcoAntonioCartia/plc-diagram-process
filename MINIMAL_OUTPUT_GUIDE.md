# Minimal Output Mode Guide

This guide explains the new minimal output mode for the PLC Diagram Processor pipeline, which provides a clean, single-line continuous update display.

## Overview

The minimal output mode transforms the verbose pipeline output into a streamlined, single-line display that shows only the current operation. This is ideal for:

- Production environments where clean output is preferred
- Automated scripts and CI/CD pipelines
- Users who want to see progress without verbose details
- Testing and development where you want to focus on results

## How to Enable

### Command Line Options

```bash
# Using --minimal flag
python src/run_pipeline.py --run-all --minimal

# Using --quiet flag (same effect)
python src/run_pipeline.py --run-all --quiet

# For specific stages
python src/run_pipeline.py --stages training detection --minimal
```

### Environment Variable

```bash
# Set environment variable
export PLCDP_MINIMAL_OUTPUT=1
python src/run_pipeline.py --run-all

# Or inline
PLCDP_MINIMAL_OUTPUT=1 python src/run_pipeline.py --run-all
```

## Output Comparison

### Normal Mode (Default)
```
==================================================
Stage TRAINING: Train or validate YOLO models
Environment: yolo_env
CI Mode: False
==================================================

X Starting training...
  → Processing: model validation
    Checking for existing custom models...
    Validating dataset structure...
    Starting model training...
  ✓ Completed: model validation - training successful

V Stage training completed successfully in 45.2s

==================================================
Stage DETECTION: Run YOLO object detection
Environment: yolo_env
CI Mode: False
==================================================
...
```

### Minimal Mode
```
training: Checking for existing models...
training: Validating dataset structure...
training: Starting model training...
training ✓
detection: Processing file1.pdf...
detection: Processing file2.pdf...
detection ✓
```

## Features

### Single-Line Updates
- Each stage shows only its current operation
- Previous lines are cleared and replaced
- Only major milestones are preserved

### Stage Completion Markers
- `✓` indicates successful completion
- `✗` indicates failure with brief error message
- Completed stages remain visible

### Reduced Verbosity
- No stage headers or separators
- No environment information
- No timing details (unless errors occur)
- Minimal pipeline startup/completion messages

## Implementation Details

### Progress Display Classes

The minimal mode is implemented through:

1. **ProgressDisplay**: Detects minimal mode and switches to single-line updates
2. **StageProgressDisplay**: Adapts stage progress to minimal format
3. **BaseStage**: Suppresses verbose stage headers and completion messages
4. **StageManager**: Reduces pipeline execution logging

### Environment Detection

The system checks for minimal mode using:
- `PLCDP_MINIMAL_OUTPUT=1` environment variable
- `PLCDP_QUIET=1` environment variable (legacy compatibility)
- Command line flags `--minimal` or `--quiet`

### Backward Compatibility

- Default behavior remains unchanged (verbose mode)
- All existing functionality preserved
- Can switch between modes without code changes

## Testing

### Test Script
```bash
# Run the test script to see both modes
python test_minimal_output.py
```

### Manual Testing
```bash
# Test minimal mode
python src/run_pipeline.py --stages preparation --minimal

# Test normal mode
python src/run_pipeline.py --stages preparation --verbose
```

## Use Cases

### Production Deployment
```bash
# Clean output for production logs
python src/run_pipeline.py --run-all --minimal > production.log
```

### CI/CD Integration
```bash
# Minimal output for automated builds
PLCDP_MINIMAL_OUTPUT=1 python src/run_pipeline.py --run-all
```

### Development Testing
```bash
# Quick testing with minimal noise
python src/run_pipeline.py --stages training detection --minimal
```

### Monitoring Scripts
```bash
# Clean output for monitoring systems
python src/run_pipeline.py --run-all --minimal --json-output
```

## Configuration

### Stage-Specific Behavior

Each stage adapts its output in minimal mode:

- **Preparation**: Shows validation steps
- **Training**: Shows model checking and training progress
- **Detection**: Shows file processing progress
- **OCR**: Shows text extraction progress
- **Enhancement**: Shows output generation progress

### Error Handling

In minimal mode, errors are still displayed clearly:
```
training: Model validation failed ✗ No pretrained models found
```

### Progress Indicators

Minimal mode uses simple progress indicators:
- `...` for ongoing operations
- `✓` for successful completion
- `✗` for failures

## Best Practices

1. **Use minimal mode for production** to reduce log noise
2. **Use verbose mode for debugging** to see detailed information
3. **Combine with --json-output** for structured logging
4. **Set environment variables** for consistent behavior across scripts
5. **Test both modes** to ensure functionality works correctly

## Troubleshooting

### If minimal mode doesn't work:
1. Check that `PLCDP_MINIMAL_OUTPUT=1` is set
2. Verify the command line flag is correct (`--minimal` or `--quiet`)
3. Ensure you're using the updated pipeline runner

### If output is still verbose:
1. Check for conflicting `--verbose` flag
2. Verify environment variables are set correctly
3. Check that the progress display classes are imported correctly

### If errors are not visible:
1. Errors are always shown, even in minimal mode
2. Check the exit code for pipeline failures
3. Use `--json-output` for structured error information
