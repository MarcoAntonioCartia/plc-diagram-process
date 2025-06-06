# PLC Diagram Processor Tests

This directory contains all test scripts for the PLC Diagram Processor project.

## Test Files

### `run_all_tests.py`
Runs all tests in sequence and provides a summary report.

**Usage:**
```bash
python tests/run_all_tests.py
```

**What it does:**
- Executes all test scripts in order
- Captures output from each test
- Provides a summary of passed/failed tests
- Returns appropriate exit code for CI/CD integration

### 1. `test_network_drive.py`
Tests the network drive functionality for dataset storage and retrieval.

**Usage:**
```bash
python tests/test_network_drive.py
```

**What it tests:**
- Network drive access and connectivity
- Dataset listing from network storage
- Dataset manager initialization
- Configuration loading

### 2. `test_wsl_poppler.py`
Tests WSL (Windows Subsystem for Linux) integration for poppler utilities.

**Usage:**
```bash
python tests/test_wsl_poppler.py
```

**What it tests:**
- WSL availability on Windows
- Poppler-utils installation in WSL
- Wrapper script functionality for Windows/WSL integration

### 3. `test_pipeline.py`
Tests the complete PLC detection pipeline setup and configuration.

**Usage:**
```bash
python tests/test_pipeline.py
```

**What it tests:**
- Folder structure validation
- Model file availability
- Configuration file validity
- Pipeline script existence
- Shows usage examples for the complete pipeline

### 4. `validate_setup.py`
Comprehensive validation of the entire project setup.

**Usage:**
```bash
python tests/validate_setup.py
```

**What it tests:**
- Python module imports (PyTorch, YOLO, etc.)
- Directory structure
- Configuration files
- Manager class initialization
- YOLO functionality
- Management script availability

## Running All Tests

### Option 1: Using the test runner script
```bash
# From project root
python tests/run_all_tests.py
```

### Option 2: Running tests individually
```bash
# From project root
python tests/validate_setup.py
python tests/test_network_drive.py
python tests/test_wsl_poppler.py
python tests/test_pipeline.py
```

## Test Requirements

- **validate_setup.py**: Requires virtual environment activated with all dependencies installed
- **test_network_drive.py**: Requires network drive access and proper configuration
- **test_wsl_poppler.py**: Windows only, requires WSL installed
- **test_pipeline.py**: Requires project structure and configuration files

## Adding New Tests

When adding new test files:
1. Place them in this `tests/` directory
2. Update import paths to use `project_root = Path(__file__).resolve().parent.parent`
3. Add documentation to this README
4. Follow the naming convention: `test_<feature>.py`
