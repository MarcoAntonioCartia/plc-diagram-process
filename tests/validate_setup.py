#!/usr/bin/env python3
"""
Setup Validation Script for PLC Diagram Processor
Validates that the complete setup is working correctly
Supports both single-environment and multi-environment setups
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

def detect_setup_type():
    """Detect whether this is a single-env or multi-env setup"""
    environments_dir = project_root / "environments"
    single_env_dirs = [
        project_root / "plcdp",
        project_root / "venv", 
        project_root / ".venv"
    ]
    
    # Check for multi-environment setup
    if environments_dir.exists():
        detection_env = environments_dir / "detection_env"
        ocr_env = environments_dir / "ocr_env"
        if detection_env.exists() and ocr_env.exists():
            return "multi-env", {
                "detection_env": detection_env,
                "ocr_env": ocr_env
            }
    
    # Check for single environment setup
    for env_dir in single_env_dirs:
        if env_dir.exists():
            return "single-env", {"env": env_dir}
    
    return "unknown", {}

def find_python_executable(env_path):
    """Find Python executable in an environment directory"""
    possible_paths = [
        env_path / "bin" / "python",           # Linux/Mac
        env_path / "Scripts" / "python.exe",   # Windows
        env_path / "Scripts" / "python",       # Windows alternative
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    return None

def run_in_environment(python_path, code, timeout=30):
    """Run code in a specific environment"""
    try:
        result = subprocess.run([
            str(python_path), "-c", code
        ], capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def test_imports():
    """Test that all required modules can be imported"""
    print("=== Testing Module Imports ===")
    
    setup_type, env_info = detect_setup_type()
    print(f"Detected setup type: {setup_type}")
    
    if setup_type == "multi-env":
        return test_multi_env_imports(env_info)
    elif setup_type == "single-env":
        return test_single_env_imports(env_info)
    else:
        print("  ERROR: No virtual environment detected")
        print("  Please run setup first: python setup/setup.py")
        return False

def test_single_env_imports(env_info):
    """Test imports in single environment setup"""
    print("Testing single environment setup...")
    
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
        ('utils.network_drive_manager', 'Network Drive Manager'),
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

def test_multi_env_imports(env_info):
    """Test imports in multi-environment setup"""
    print("Testing multi-environment setup...")
    
    # Test detection environment
    detection_python = find_python_executable(env_info["detection_env"])
    if not detection_python:
        print("  FAILED: Detection environment Python not found")
        return False
    
    print(f"  Detection environment: {detection_python}")
    
    detection_modules = [
        ("import torch; print(f'PyTorch {torch.__version__}')", "PyTorch"),
        ("import ultralytics; print(f'Ultralytics {ultralytics.__version__}')", "Ultralytics"),
        ("import cv2; print(f'OpenCV {cv2.__version__}')", "OpenCV"),
        ("import numpy; print(f'NumPy {numpy.__version__}')", "NumPy"),
    ]
    
    detection_failed = []
    for code, name in detection_modules:
        success, output, error = run_in_environment(detection_python, code)
        if success:
            print(f"    SUCCESS: {name} - {output}")
        else:
            print(f"    FAILED: {name} - {error}")
            detection_failed.append(name)
    
    # Test OCR environment
    ocr_python = find_python_executable(env_info["ocr_env"])
    if not ocr_python:
        print("  FAILED: OCR environment Python not found")
        return False
    
    print(f"  OCR environment: {ocr_python}")
    
    ocr_modules = [
        ("import paddle; print(f'PaddlePaddle {paddle.__version__}')", "PaddlePaddle"),
        ("import paddleocr; print('PaddleOCR imported successfully')", "PaddleOCR"),
        ("import PIL; print(f'Pillow {PIL.__version__}')", "Pillow"),
        ("import numpy; print(f'NumPy {numpy.__version__}')", "NumPy"),
    ]
    
    ocr_failed = []
    for code, name in ocr_modules:
        success, output, error = run_in_environment(ocr_python, code, timeout=60)  # PaddleOCR can be slow
        if success:
            print(f"    SUCCESS: {name} - {output}")
        else:
            print(f"    FAILED: {name} - {error}")
            ocr_failed.append(name)
    
    # Test core modules (should be available in current environment)
    core_modules = [
        ('yaml', 'PyYAML'),
        ('pandas', 'Pandas'),
        ('requests', 'Requests'),
    ]
    
    core_failed = []
    for module_name, display_name in core_modules:
        try:
            __import__(module_name)
            print(f"  SUCCESS: {display_name} (core)")
        except ImportError as e:
            print(f"  FAILED: {display_name} (core) - {e}")
            core_failed.append(display_name)
    
    # Test project modules
    project_modules = [
        ('utils.dataset_manager', 'Dataset Manager'),
        ('utils.model_manager', 'Model Manager'),
        ('utils.onedrive_manager', 'OneDrive Manager'),
        ('utils.network_drive_manager', 'Network Drive Manager'),
    ]
    
    project_failed = []
    for module_name, display_name in project_modules:
        try:
            __import__(module_name)
            print(f"  SUCCESS: {display_name}")
        except ImportError as e:
            print(f"  FAILED: {display_name} - {e}")
            project_failed.append(display_name)
    
    all_failed = detection_failed + ocr_failed + core_failed + project_failed
    
    if all_failed:
        print(f"\nFailed imports: {', '.join(all_failed)}")
        return False
    else:
        print("\nAll multi-environment imports successful!")
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
    
    setup_type, env_info = detect_setup_type()
    
    if setup_type == "multi-env":
        # Test YOLO in detection environment
        detection_python = find_python_executable(env_info["detection_env"])
        if not detection_python:
            print("  FAILED: Detection environment not found")
            return False
        
        yolo_test_code = """
try:
    from ultralytics import YOLO
    # Test basic YOLO import first
    print('SUCCESS: Ultralytics YOLO imported successfully')
    
    # Try to create a model (this will download if needed)
    try:
        model = YOLO('yolo11n.pt')
        print(f'SUCCESS: YOLO model created - Type: {type(model).__name__}')
        print(f'Model task: {getattr(model, "task", "Unknown")}')
    except FileNotFoundError:
        print('INFO: YOLO model file not found - will be downloaded when needed')
        print('SUCCESS: YOLO functionality available (model download required)')
    except Exception as model_e:
        print(f'WARNING: YOLO model creation failed: {model_e}')
        print('SUCCESS: YOLO import working (model issues can be resolved later)')
        
except ImportError as e:
    print(f'FAILED: YOLO import failed: {e}')
    raise
except Exception as e:
    print(f'FAILED: Unexpected error: {e}')
    raise
        """
        
        success, output, error = run_in_environment(detection_python, yolo_test_code, timeout=120)
        if success:
            print(f"  {output}")
            return True
        else:
            print(f"  FAILED: YOLO functionality - {error}")
            return False
    
    elif setup_type == "single-env":
        # Test YOLO in current environment
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
    
    else:
        print("  FAILED: No environment detected for YOLO testing")
        return False


def test_scripts():
    """Test that management scripts are executable"""
    print("\n=== Testing Management Scripts ===")
    
    scripts = [
        'setup/manage_datasets.py',
        'setup/manage_models.py',
        'tests/validate_setup.py'
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
        
        # Show setup-specific next steps
        setup_type, env_info = detect_setup_type()
        print(f"\nSetup type: {setup_type}")
        
        if setup_type == "multi-env":
            print("\nNext steps for multi-environment setup:")
            print("1. Download datasets: python setup/manage_datasets.py --interactive")
            print("2. Download models: python setup/manage_models.py --interactive")
            print("3. Test detection pipeline: python tests/test_multi_env_imports.py")
            print("4. Run detection training (in detection_env):")
            detection_python = find_python_executable(env_info["detection_env"])
            if detection_python:
                print(f"   {detection_python} src/detection/yolo11_train.py")
            print("5. Run OCR processing (in ocr_env):")
            ocr_python = find_python_executable(env_info["ocr_env"])
            if ocr_python:
                print(f"   {ocr_python} src/ocr/run_text_extraction.py")
        elif setup_type == "single-env":
            print("\nNext steps for single-environment setup:")
            print("1. Activate environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
            print("2. Download datasets: python setup/manage_datasets.py --interactive")
            print("3. Download models: python setup/manage_models.py --interactive")
            print("4. Run training: python src/detection/yolo11_train.py")
        else:
            print("\nNext steps:")
            print("1. Run setup: python setup/setup.py")
            print("2. Re-run validation: python tests/validate_setup.py")
        
        return True
    else:
        print(f"\nSETUP VALIDATION FAILED!")
        print(f"{total - passed} tests failed. Please check the errors above.")
        
        setup_type, env_info = detect_setup_type()
        print(f"\nDetected setup type: {setup_type}")
        
        print("\nTroubleshooting:")
        if setup_type == "unknown":
            print("1. Run setup first: python setup/setup.py")
            print("2. For multi-environment: python setup/setup.py --multi-env")
        elif setup_type == "multi-env":
            print("1. Check if environments were created properly:")
            print(f"   Detection env: {env_info.get('detection_env', 'Not found')}")
            print(f"   OCR env: {env_info.get('ocr_env', 'Not found')}")
            print("2. Re-run setup if needed: python setup/setup.py --multi-env")
            print("3. Test individual environments: python tests/test_multi_env_imports.py")
        elif setup_type == "single-env":
            print("1. Activate the virtual environment")
            print("2. Install missing dependencies: pip install -r requirements.txt")
            print("3. Check that all dependencies are installed")
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
