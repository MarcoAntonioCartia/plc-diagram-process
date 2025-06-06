#!/usr/bin/env python3
"""
Test script for Network Drive Manager
Tests the network drive dataset retrieval functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / 'src'))

try:
    import yaml
    from utils.network_drive_manager import NetworkDriveManager
    from utils.dataset_manager import DatasetManager
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you have activated the virtual environment and installed dependencies")
    sys.exit(1)


def load_config():
    """Load download configuration"""
    config_path = project_root / 'setup' / 'config' / 'download_config.yaml'
    
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None


def test_network_drive_access():
    """Test network drive access and dataset listing"""
    print("=== Testing Network Drive Manager ===")
    print()
    
    # Load configuration
    config = load_config()
    if not config:
        return False
    
    # Check storage backend
    storage_backend = config.get('storage_backend', 'network_drive')
    print(f"Storage backend configured: {storage_backend}")
    
    if storage_backend != 'network_drive':
        print("Warning: Storage backend is not set to 'network_drive'")
        print("Update config/download_config.yaml to use network_drive")
        return False
    
    # Initialize network drive manager
    try:
        network_manager = NetworkDriveManager(config)
        print("\nNetwork Drive Manager initialized successfully")
    except Exception as e:
        print(f"Error initializing Network Drive Manager: {e}")
        return False
    
    # Test network access
    print("\n1. Testing network drive access...")
    if network_manager.check_network_access():
        print("   ✓ Network drive is accessible")
    else:
        print("   ✗ Network drive is NOT accessible")
        print("   Please check:")
        print("   - Network drive is mapped/mounted")
        print("   - Path is correct in config")
        print("   - You have read permissions")
        return False
    
    # List available datasets
    print("\n2. Listing available datasets...")
    datasets = network_manager.list_available_datasets()
    
    if datasets:
        print(f"   ✓ Found {len(datasets)} datasets:")
        for i, dataset in enumerate(datasets, 1):
            print(f"      {i}. {dataset['name']}")
            print(f"         Date: {dataset['date']}")
            print(f"         Version: {dataset['version']}")
            print(f"         Size: {dataset['size_mb']:.1f} MB")
    else:
        print("   ✗ No datasets found")
        print("   Please check:")
        print("   - Dataset files exist in the network path")
        print("   - Files match the pattern in config")
        return False
    
    # Test dataset manager
    print("\n3. Testing Dataset Manager...")
    try:
        dataset_manager = DatasetManager(config)
        print("   ✓ Dataset Manager initialized successfully")
        
        # List downloaded datasets
        downloaded = dataset_manager.list_downloaded_datasets()
        if downloaded:
            print(f"   Found {len(downloaded)} downloaded datasets:")
            for dataset in downloaded:
                print(f"      - {dataset}")
        else:
            print("   No datasets downloaded yet")
        
        # Check active dataset
        active = dataset_manager.get_active_dataset()
        if active:
            print(f"   Active dataset: {active}")
        else:
            print("   No active dataset")
            
    except Exception as e:
        print(f"   ✗ Error with Dataset Manager: {e}")
        return False
    
    print("\n=== All tests passed! ===")
    print("\nYou can now use:")
    print("  python setup/manage_datasets.py --interactive")
    print("to download and manage datasets from the network drive.")
    
    return True


def main():
    """Main test function"""
    print("PLC Diagram Processor - Network Drive Test")
    print("=" * 50)
    
    success = test_network_drive_access()
    
    if not success:
        print("\n⚠ Some tests failed. Please check the issues above.")
        sys.exit(1)
    else:
        print("\n✓ Network drive setup is working correctly!")
        sys.exit(0)


if __name__ == "__main__":
    main()
