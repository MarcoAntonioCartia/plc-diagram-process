# PLC Diagram Processor - Setup & Data Management

This directory contains all setup-related files and utilities for the PLC Diagram Processor project.

## Files in this Directory

### Core Setup
- **`setup.py`** - Main setup script for complete project installation
- **`config/download_config.yaml`** - Configuration for data and model downloads

### Management Scripts
- **`manage_datasets.py`** - Dataset download and management utility
- **`manage_models.py`** - YOLO model download and management utility
- **`validate_setup.py`** - Setup validation and testing script

## Quick Start

### 1. Initial Setup
```bash
# Run from project root
python setup/setup.py
```

This will:
- Install system dependencies
- Create virtual environment
- Install Python dependencies
- Set up data directory structure
- Optionally download datasets and models

### 2. Validate Setup
```bash
python setup/validate_setup.py
```

### 3. Manage Data
```bash
# Interactive dataset management
python setup/manage_datasets.py --interactive

# Interactive model management
python setup/manage_models.py --interactive
```

## Configuration

### Download Configuration (`config/download_config.yaml`)

```yaml
# OneDrive Dataset Configuration
onedrive:
  base_url: "https://your-sharepoint-url"
  dataset:
    architecture: "YOLO"
    version: "v11"
    dataset_pattern: "plc_diagrams_yolov11_setA_*.zip"
    auto_select_latest: true
    keep_old_versions: true

# YOLO Model Configuration
models:
  default_model: "yolo11m.pt"
  available_models:
    yolo11: ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
    yolo10: ["yolo10n.pt", "yolo10s.pt", "yolo10m.pt", "yolo10l.pt", "yolo10x.pt"]
  download_multiple: true
  verify_downloads: true

# Setup Behavior
setup:
  auto_download_dataset: false      # Set true for automatic download
  auto_download_models: false       # Set true for automatic download
  prompt_for_downloads: true        # Ask user during setup
  show_empty_folder_warnings: true  # Warn if folders remain empty
```

## Data Management System

### Dataset Management

The system implements a sophisticated dataset management approach:

1. **Download**: Datasets are downloaded to `../plc-data/datasets/downloaded/`
2. **Activation**: Active dataset is linked to `../plc-data/datasets/train/`, `valid/`, `test/`
3. **Method**: Uses symlinks with automatic fallback to copying (Windows compatibility)
4. **Configuration**: Automatically updates `../plc-data/datasets/plc_symbols.yaml`

#### Dataset Commands
```bash
# List available datasets from OneDrive
python setup/manage_datasets.py --list-available

# Download latest dataset
python setup/manage_datasets.py --download-latest

# Download specific dataset
python setup/manage_datasets.py --download dataset_name

# Activate downloaded dataset
python setup/manage_datasets.py --activate dataset_name

# Show current status
python setup/manage_datasets.py --status

# Interactive mode
python setup/manage_datasets.py --interactive
```

### Model Management

YOLO model downloads and management:

1. **Download**: Models are downloaded to `../plc-data/models/pretrained/`
2. **Verification**: Models are verified for integrity after download
3. **Selection**: Interactive selection of multiple models
4. **Cleanup**: Tools for managing disk space

#### Model Commands
```bash
# List available models
python setup/manage_models.py --list-available

# Download specific model
python setup/manage_models.py --download yolo11m.pt

# Download multiple models
python setup/manage_models.py --download-multiple yolo11m.pt yolo11l.pt

# Interactive mode
python setup/manage_models.py --interactive

# Get model information
python setup/manage_models.py --info yolo11m.pt

# Cleanup old models
python setup/manage_models.py --cleanup
```

## Directory Structure Created

```
../plc-data/                           # Data root (sibling to project)
├── datasets/
│   ├── downloaded/                    # Raw downloaded datasets
│   │   ├── plc_diagrams_yolov11_setA_20241205/
│   │   └── plc_diagrams_yolov11_setA_20241201/
│   ├── train -> downloaded/setA_20241205/train/     # Symlink (or copy)
│   ├── valid -> downloaded/setA_20241205/valid/     # Symlink (or copy)
│   ├── test -> downloaded/setA_20241205/test/       # Symlink (or copy)
│   └── plc_symbols.yaml              # Auto-updated configuration
├── models/
│   ├── pretrained/                    # Downloaded YOLO models
│   │   ├── yolo11m.pt
│   │   └── yolo11n.pt
│   └── custom/                        # Trained models (from runs/)
├── processed/                         # Processed data
├── raw/                              # Raw input data
└── runs/                             # Training/inference outputs
```

## Key Features

### 1. Symlink + Fallback Strategy
- **Primary**: Uses symlinks for efficient dataset activation (no disk duplication)
- **Fallback**: Automatically copies files if symlinks fail (Windows compatibility)
- **Seamless**: User doesn't need to know which method is used

### 2. Configuration-Driven
- **Flexible**: All behavior controlled via YAML configuration
- **Documented**: Clear comments explaining each option
- **Extensible**: Easy to add new options

### 3. Interactive Management
- **User-Friendly**: Interactive CLI interfaces for all operations
- **Comprehensive**: Both command-line and interactive modes
- **Informative**: Detailed status and progress information

### 4. Robust Error Handling
- **Graceful**: Fallback mechanisms for common failures
- **Clear**: Informative error messages and warnings
- **Validated**: Comprehensive validation at multiple levels

### 5. Existing Workflow Preservation
- **Compatible**: No changes required to existing training scripts
- **Preserved**: Maintains existing directory structure
- **Integrated**: Seamless integration with current workflow

## Usage Examples

### Complete Setup Workflow
```bash
# 1. Initial setup
python setup/setup.py

# 2. Validate everything works
python setup/validate_setup.py

# 3. Configure OneDrive URL (edit config/download_config.yaml)
# Update the base_url with your SharePoint link

# 4. Download datasets
python setup/manage_datasets.py --interactive

# 5. Download models
python setup/manage_models.py --interactive

# 6. Start training (existing workflow)
python src/detection/yolo11_train.py --epochs 10
```

### Daily Usage
```bash
# Check current dataset status
python setup/manage_datasets.py --status

# Switch to different dataset
python setup/manage_datasets.py --activate other_dataset

# Download new models
python setup/manage_models.py --download yolo11x.pt

# Validate setup after changes
python setup/validate_setup.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Make sure virtual environment is activated
   - Run `python setup/validate_setup.py` to check dependencies

2. **Dataset Download Fails**
   - Check OneDrive URL in `config/download_config.yaml`
   - Verify network connectivity
   - Check SharePoint permissions

3. **Symlinks Don't Work**
   - System automatically falls back to copying
   - No action needed, but uses more disk space

4. **Model Download Timeouts**
   - Large models (PyTorch, etc.) can take 30-60 minutes
   - Be patient, progress is shown
   - Check network stability

### Getting Help

1. **Validation**: Run `python setup/validate_setup.py` for comprehensive testing
2. **Status**: Use `--status` flags to check current state
3. **Interactive**: Use `--interactive` modes for guided operations
4. **Logs**: Check console output for detailed error messages

## Advanced Configuration

### Custom OneDrive Structure
If your OneDrive has a different structure, update the pattern in `config/download_config.yaml`:

```yaml
onedrive:
  dataset:
    dataset_pattern: "your_custom_pattern_*.zip"
```

### Automatic Downloads
For automated setups, enable automatic downloads:

```yaml
setup:
  auto_download_dataset: true
  auto_download_models: true
  prompt_for_downloads: false
```

### Model Selection
Customize available models:

```yaml
models:
  available_models:
    yolo11: ["yolo11n.pt", "yolo11m.pt"]  # Only nano and medium
    custom: ["your_custom_model.pt"]
```
