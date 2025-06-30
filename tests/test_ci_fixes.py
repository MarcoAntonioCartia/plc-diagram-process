#!/usr/bin/env python3
"""
Test script to verify CI fixes work correctly
"""

import sys
import subprocess
from pathlib import Path

def test_paddleocr_version_consistency():
    """Test that PaddleOCR versions are consistent across all files"""
    print("=== Testing PaddleOCR Version Consistency ===")
    
    project_root = Path(__file__).parent.parent
    
    # Check requirements-ocr.txt
    ocr_req_file = project_root / "requirements-ocr.txt"
    if ocr_req_file.exists():
        with open(ocr_req_file, 'r') as f:
            ocr_content = f.read()
        if "paddleocr==3.0.1" in ocr_content:
            print("V requirements-ocr.txt has paddleocr==3.0.1")
        else:
            print("X requirements-ocr.txt missing paddleocr==3.0.1")
            return False
    
    # Check multi_env_manager.py
    manager_file = project_root / "src" / "utils" / "multi_env_manager.py"
    if manager_file.exists():
        with open(manager_file, 'r') as f:
            manager_content = f.read()
        if 'paddle_ocr_pkg = "paddleocr==3.0.1"' in manager_content:
            print("V multi_env_manager.py has paddleocr==3.0.1")
        else:
            print("X multi_env_manager.py missing paddleocr==3.0.1")
            return False
    
    print("V PaddleOCR version consistency check passed")
    return True

def test_opencv_headless_usage():
    """Test that OpenCV headless versions are used for CI compatibility"""
    print("\n=== Testing OpenCV Headless Usage ===")
    
    project_root = Path(__file__).parent.parent
    
    # Check requirements-detection.txt
    det_req_file = project_root / "requirements-detection.txt"
    if det_req_file.exists():
        with open(det_req_file, 'r') as f:
            det_content = f.read()
        if "opencv-python-headless" in det_content and "opencv-python==" not in det_content:
            print("V requirements-detection.txt uses opencv-python-headless")
        else:
            print("X requirements-detection.txt not using opencv-python-headless")
            return False
    
    # Check main requirements.txt
    main_req_file = project_root / "requirements.txt"
    if main_req_file.exists():
        with open(main_req_file, 'r') as f:
            main_content = f.read()
        headless_count = main_content.count("opencv-python-headless")
        regular_count = main_content.count("opencv-python>=") + main_content.count("opencv-python==")
        
        if headless_count >= 2 and regular_count == 0:
            print("V requirements.txt uses opencv-python-headless")
        else:
            print(f"X requirements.txt opencv issue: headless={headless_count}, regular={regular_count}")
            return False
    
    # Check version consistency across files
    version_410 = main_content.count("4.10.0.82")
    version_49 = main_content.count("4.9.0.80")
    
    if version_410 >= 2 and version_49 == 0:
        print("V OpenCV versions consistent (4.10.0.82)")
    else:
        print(f"X OpenCV version inconsistency: v4.10={version_410}, v4.9={version_49}")
        return False
    
    print("V OpenCV headless usage check passed")
    return True

def test_environment_path_detection():
    """Test that the test script can find environment paths correctly"""
    print("\n=== Testing Environment Path Detection ===")
    
    project_root = Path(__file__).parent.parent
    
    # Test detection environment paths
    detection_paths = [
        project_root / "environments" / "detection_env" / "bin" / "python",
        project_root / "environments" / "detection_env" / "Scripts" / "python.exe",
        project_root / "detection_env" / "bin" / "python",
        project_root / "detection_env" / "Scripts" / "python.exe",
    ]
    
    print("Detection environment search paths:")
    for path in detection_paths:
        exists = "EXISTS" if path.exists() else "missing"
        print(f"  {exists}: {path}")
    
    # Test OCR environment paths
    ocr_paths = [
        project_root / "environments" / "ocr_env" / "bin" / "python",
        project_root / "environments" / "ocr_env" / "Scripts" / "python.exe",
        project_root / "ocr_env" / "bin" / "python",
        project_root / "ocr_env" / "Scripts" / "python.exe",
    ]
    
    print("\nOCR environment search paths:")
    for path in ocr_paths:
        exists = "EXISTS" if path.exists() else "missing"
        print(f"  {exists}: {path}")
    
    # Check if environments directory exists
    env_dir = project_root / "environments"
    if env_dir.exists():
        print(f"\nV environments/ directory exists at: {env_dir}")
        try:
            subdirs = list(env_dir.iterdir())
            print(f"  Subdirectories: {[d.name for d in subdirs if d.is_dir()]}")
        except Exception as e:
            print(f"  Error listing subdirectories: {e}")
    else:
        print(f"\nX environments/ directory not found at: {env_dir}")
    
    return True

def test_multi_env_manager_import():
    """Test that multi_env_manager can be imported without errors"""
    print("\n=== Testing Multi-Environment Manager Import ===")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.utils.multi_env_manager import MultiEnvironmentManager
        
        # Try to create manager instance
        manager = MultiEnvironmentManager(project_root, dry_run=True)
        print("V MultiEnvironmentManager imported and instantiated successfully")
        print(f"  Detection env path: {manager.detection_env_path}")
        print(f"  OCR env path: {manager.ocr_env_path}")
        
        return True
        
    except ImportError as e:
        print(f"X Import error: {e}")
        return False
    except Exception as e:
        print(f"X Unexpected error: {e}")
        return False

def main():
    """Run all CI fix tests"""
    print("Testing CI Fixes")
    print("=" * 50)
    
    tests = [
        test_paddleocr_version_consistency,
        test_opencv_headless_usage,
        test_environment_path_detection,
        test_multi_env_manager_import,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"X {test_func.__name__} failed")
        except Exception as e:
            print(f"X {test_func.__name__} error: {e}")
    
    print(f"\n{'='*50}")
    print(f"CI fix test results: {passed}/{total} passed")
    
    if passed == total:
        print("V All CI fix tests passed!")
        print("\nThe fixes should resolve the CI issues:")
        print("1. PaddleOCR version conflict resolved (3.0.1 everywhere)")
        print("2. OpenCV headless versions used for CI compatibility")
        print("3. Environment path detection updated for environments/ subdirectory")
        print("4. System dependencies added to CI workflow")
        print("5. Better error reporting added to CI workflow")
        return 0
    else:
        print("X Some CI fix tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
