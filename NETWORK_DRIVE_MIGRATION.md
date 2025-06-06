# Network Drive Migration Summary

This document summarizes the changes made to migrate from OneDrive to network drive storage for dataset management.

## Overview

The PLC Diagram Processor project has been updated to use a network drive as the primary storage backend for datasets, while keeping OneDrive support as a legacy option for potential future cloud storage implementation.

## Key Changes

### 1. New Network Drive Manager
- **File**: `src/utils/network_drive_manager.py`
- **Purpose**: Handles dataset retrieval from network storage locations
- **Features**:
  - Lists available datasets in the network path
  - Parses filenames to extract version info (YYYYMMDDVV format)
  - Copies datasets from network to local storage
  - Verifies YOLO dataset structure
  - Provides cleanup functionality for old downloads

### 2. Configuration Updates
- **File**: `setup/config/download_config.yaml`
- **Changes**:
  - Added `storage_backend` parameter (default: "network_drive")
  - Added `network_drive` section with configuration
  - Kept `onedrive` section as legacy (marked as such)
  - Network path configured as: `S:\99_Automation\Datasets plc-diagram-processor`

### 3. OneDrive Manager (Legacy)
- **File**: `src/utils/onedrive_manager.py`
- **Status**: Preserved but marked as LEGACY
- **Note**: Added header comment explaining it's for future cloud storage

### 4. Updated Scripts
The following scripts were updated to support both storage backends:
- `setup/setup.py` - Main setup script
- `setup/manage_datasets.py` - Dataset management utility

### 5. Documentation Updates
- **setup/README.md**: Updated with network drive configuration and examples
- Added switching instructions between storage backends

## Network Drive Structure

Expected folder structure on the network drive:
```
S:\99_Automation\Datasets plc-diagram-processor\
└── YOLO\
    └── v11\
        └── plc_symbols_v11_YYYYMMDDVV.zip
```

Where YYYYMMDDVV represents:
- YYYYMMDD: Date (e.g., 20250605)
- VV: Version number (e.g., 01)

## Usage

### Testing Network Drive Access
```bash
# Test network drive connectivity and configuration
python tests/test_network_drive.py
```

### Managing Datasets
```bash
# Interactive dataset management
python setup/manage_datasets.py --interactive

# List available datasets
python setup/manage_datasets.py --list-available

# Download latest dataset
python setup/manage_datasets.py --download-latest

# Activate a dataset
python setup/manage_datasets.py --activate plc_symbols_v11_2025060501
```

## Configuration

### Using Network Drive (Default)
```yaml
storage_backend: "network_drive"
network_drive:
  base_path: "S:\\99_Automation\\Datasets plc-diagram-processor"
  dataset:
    architecture: "YOLO"
    version: "v11"
    dataset_pattern: "plc_symbols_v11_*.zip"
    auto_select_latest: true
```

### Switching to OneDrive (Legacy)
```yaml
storage_backend: "onedrive"
onedrive:
  base_url: "https://your-sharepoint-url"
  dataset:
    architecture: "YOLO"
    version: "v11"
    dataset_pattern: "plc_symbols_v11_*.zip"
    auto_select_latest: true
```

## Benefits of Network Drive Approach

1. **Direct Access**: No authentication or web scraping required
2. **Faster Downloads**: Local network speeds vs internet downloads
3. **Reliability**: No dependency on SharePoint/OneDrive availability
4. **Simplicity**: Straightforward file operations
5. **Control**: Full control over dataset organization and access

## Backward Compatibility

The OneDrive functionality is preserved as legacy code. To revert to OneDrive:
1. Change `storage_backend` to "onedrive" in config
2. Update OneDrive URL in configuration
3. All scripts will automatically use OneDrive manager

## Troubleshooting

### Network Drive Not Accessible
- Verify drive is mapped/mounted
- Check path in configuration
- Ensure read permissions

### No Datasets Found
- Verify folder structure matches expected pattern
- Check filename format matches pattern
- Ensure files have .zip extension

### Download Fails
- Check available disk space
- Verify network connectivity
- Ensure write permissions in local directories

## Future Enhancements

Potential improvements for the network drive approach:
1. Add progress bars for large file copies
2. Implement incremental/differential updates
3. Add dataset versioning metadata
4. Support for multiple network locations
5. Automatic cleanup policies
