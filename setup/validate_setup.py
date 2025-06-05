#!/usr/bin/env python3
"""
Setup Validation Script for PLC Diagram Processor
Validates that the complete setup is working correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

def test_imports():
    """Test that all required modules can be imported"""
    print("=== Testing Module Imports ===")
    
    required_modules = [
        ('yaml', 'PyYAML'),
        ('torch', 'PyTorch'),
        ('ultralytics', 'Ultralytics YOLO'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('requests', 'Requests'),
    ]
    
    failed_imports = []
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"  SUCCESS: {display_name}")
        except ImportError as e:
            print(f"  FAILED: {display_name} - {e}")
            failed_imports.append(display_name)
    
    # Test project modules
    project_modules = [
        ('utils.dataset_manager', 'Dataset Manager'),
        ('utils.model_manager', 'Model Manager'),
        ('utils.onedrive_manager', 'OneDrive Manager'),
    ]
    
    for module_name, display_name in project_modules:
        try:
            __import__(module_name)
            print(f"  SUCCESS: {display_name}")
        except ImportError as e:
            print(f"  FAILED: {display_name} - {e}")
            failed_imports.append(display_name)
    
    if failed_imports:
        print(f"\nFailed imports: {', '.join(failed_imports)}")
        return False
    else:
        print("\nAll modules imported successfully!")
        return True


def test_directory_structure():
    """Test that the directory structure is correct"""
    print("\n=== Testing Directory Structure ===")
    
    data_root = project_root.parent / 'plc-data'
    
    required_dirs = [
        'datasets',
        'models/pretrained',
        'models/custom',
        'raw/pdfs',
        'processed/images',
        'runs/detect'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        full_path = data_root / dir_path
        if full_path.exists():
            print(f"  SUCCESS: {dir_path}")
        else:
            print(f"  MISSING: {dir_path}")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\nMissing directories: {', '.join(missing_dirs)}")
        return False
    else:
        print("\nAll directories exist!")
        return True


def test_configuration():
    """Test that configuration files exist and are valid"""
    print("\n=== Testing Configuration ===")
    
    config_files = [
        'setup/config/download_config.yaml',
        'plc-data/datasets/plc_symbols.yaml'
    ]
    
    missing_configs = []
    
    for config_file in config_files:
        if config_file.startswith('plc-data/'):
            config_path = project_root.parent / config_file
        else:
            config_path = project_root / config_file
        
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    yaml.safe_load(f)
                print(f"  SUCCESS: {config_file}")
            except Exception as e:
                print(f"  INVALID: {config_file} - {e}")
                missing_configs.append(config_file)
        else:
            print(f"  MISSING: {config_file}")
            missing_configs.append(config_file)
    
    if missing_configs:
        print(f"\nMissing/invalid configs: {', '.join(missing_configs)}")
        return False
    else:
        print("\nAll configurations valid!")
        return True


def test_managers():
    """Test that the manager classes can be instantiated"""
    print("\n=== Testing Manager Classes ===")
    
    try:
        import yaml
        from utils.dataset_manager import DatasetManager
        from utils.model_manager import ModelManager
        from utils.onedrive_manager import OneDriveManager
        
        # Load config
        config_path = project_root / 'setup' / 'config' / 'download_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test Dataset Manager
        try:
            dataset_manager = DatasetManager(config)
            print("  SUCCESS: Dataset Manager")
        except Exception as e:
            print(f"  FAILED: Dataset Manager - {e}")
            return False
        
        # Test Model Manager
        try:
            model_manager = ModelManager(config)
            print("  SUCCESS: Model Manager")
        except Exception as e:
            print(f"  FAILED: Model Manager - {e}")
            return False
        
        # Test OneDrive Manager
        try:
            onedrive_manager = OneDriveManager(config)
            print("  SUCCESS: OneDrive Manager")
        except Exception as e:
            print(f"  FAILED: OneDrive Manager - {e}")
            return False
        
        print("\nAll managers initialized successfully!")
        return True
        
    except Exception as e:
        print(f"  FAILED: Manager testing - {e}")
        return False


def test_yolo_functionality():
    """Test basic YOLO functionality"""
    print("\n=== Testing YOLO Functionality ===")
    
    try:
        from ultralytics import YOLO
        
        # Try to create a YOLO model instance
        model = YOLO('yolo11n.pt')  # This will download if not present
        print("  SUCCESS: YOLO model creation")
        
        # Test basic model info
        print(f"  Model type: {type(model)}")
        print(f"  Model task: {getattr(model, 'task', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"  FAILED: YOLO functionality - {e}")
        return False


def test_scripts():
    """Test that management scripts are executable"""
    print("\n=== Testing Management Scripts ===")
    
    scripts = [
        'setup/manage_datasets.py',
        'setup/manage_models.py',
        'setup/validate_setup.py'
    ]
    
    missing_scripts = []
    
    for script in scripts:
        script_path = project_root / script
        if script_path.exists():
            # Check if script is executable (on Unix systems)
            if os.name != 'nt':  # Not Windows
                if os.access(script_path, os.X_OK):
                    print(f"  SUCCESS: {script} (executable)")
                else:
                    print(f"  WARNING: {script} (not executable)")
            else:
                print(f"  SUCCESS: {script}")
        else:
            print(f"  MISSING: {script}")
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"\nMissing scripts: {', '.join(missing_scripts)}")
        return False
    else:
        print("\nAll scripts present!")
        return True


def show_system_info():
    """Show system information"""
    print("\n=== System Information ===")
    
    import platform
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Project root: {project_root}")
    print(f"Data root: {project_root.parent / 'plc-data'}")
    
    # Show virtual environment info
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"Virtual environment: {sys.prefix}")
    else:
        print("Virtual environment: Not detected")


def main():
    """Run all validation tests"""
    print("PLC Diagram Processor Setup Validation")
    print("=" * 50)
    
    show_system_info()
    
    tests = [
        ("Module Imports", test_imports),
        ("Directory Structure", test_directory_structure),
        ("Configuration", test_configuration),
        ("Manager Classes", test_managers),
        ("YOLO Functionality", test_yolo_functionality),
        ("Management Scripts", test_scripts),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSETUP VALIDATION SUCCESSFUL!")
        print("Your PLC Diagram Processor is ready to use.")
        print("\nNext steps:")
        print("1. Download datasets: python setup/manage_datasets.py --interactive")
        print("2. Download models: python setup/manage_models.py --interactive")
        print("3. Run training: python src/detection/yolo11_train.py")
        return True
    else:
        print(f"\nSETUP VALIDATION FAILED!")
        print(f"{total - passed} tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure you've run: python setup.py")
        print("2. Activate the virtual environment")
        print("3. Check that all dependencies are installed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
