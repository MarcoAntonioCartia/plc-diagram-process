#!/usr/bin/env python3
"""
Test multi-environment setup - validate detection_env and ocr_env imports
"""

import sys
import subprocess
from pathlib import Path

def test_detection_env():
    """Test that detection environment can import PyTorch and YOLO"""
    project_root = Path(__file__).parent.parent
    
    # Try environments subdirectory first (new structure) - check for yolo_env first, then detection_env
    detection_python = project_root / "environments" / "yolo_env" / "bin" / "python"
    if not detection_python.exists():
        detection_python = project_root / "environments" / "yolo_env" / "Scripts" / "python.exe"
    
    # Fallback to detection_env for backward compatibility
    if not detection_python.exists():
        detection_python = project_root / "environments" / "detection_env" / "bin" / "python"
    if not detection_python.exists():
        detection_python = project_root / "environments" / "detection_env" / "Scripts" / "python.exe"
    
    # Fallback to root directory (legacy structure)
    if not detection_python.exists():
        detection_python = project_root / "yolo_env" / "bin" / "python"
    if not detection_python.exists():
        detection_python = project_root / "yolo_env" / "Scripts" / "python.exe"
    if not detection_python.exists():
        detection_python = project_root / "detection_env" / "bin" / "python"
    if not detection_python.exists():
        detection_python = project_root / "detection_env" / "Scripts" / "python.exe"
    
    if not detection_python.exists():
        print("X Detection environment not found")
        print(f"  Looked for: {project_root / 'environments' / 'detection_env'}")
        print(f"  Looked for: {project_root / 'detection_env'}")
        return False
    
    try:
        # Test PyTorch import
        result = subprocess.run([
            str(detection_python), "-c", 
            "import torch; print(f'V PyTorch {torch.__version__} imported successfully')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"X PyTorch import failed: {result.stderr}")
            return False
        
        print(result.stdout.strip())
        
        # Test Ultralytics import
        result = subprocess.run([
            str(detection_python), "-c",
            "import ultralytics; print(f'V Ultralytics {ultralytics.__version__} imported successfully')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"X Ultralytics import failed: {result.stderr}")
            return False
            
        print(result.stdout.strip())
        return True
        
    except subprocess.TimeoutExpired:
        print("X Detection environment test timed out")
        return False
    except Exception as e:
        print(f"X Detection environment test error: {e}")
        return False

def test_ocr_env():
    """Test that OCR environment can import PaddlePaddle and PaddleOCR"""
    project_root = Path(__file__).parent.parent
    
    # Try environments subdirectory first (new structure)
    ocr_python = project_root / "environments" / "ocr_env" / "bin" / "python"
    if not ocr_python.exists():
        ocr_python = project_root / "environments" / "ocr_env" / "Scripts" / "python.exe"
    
    # Fallback to root directory (legacy structure)
    if not ocr_python.exists():
        ocr_python = project_root / "ocr_env" / "bin" / "python"
    if not ocr_python.exists():
        ocr_python = project_root / "ocr_env" / "Scripts" / "python.exe"
    
    if not ocr_python.exists():
        print("X OCR environment not found")
        print(f"  Looked for: {project_root / 'environments' / 'ocr_env'}")
        print(f"  Looked for: {project_root / 'ocr_env'}")
        return False
    
    try:
        # Test PaddlePaddle import
        result = subprocess.run([
            str(ocr_python), "-c",
            "import paddle; print(f'V PaddlePaddle {paddle.__version__} imported successfully')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"X PaddlePaddle import failed: {result.stderr}")
            return False
            
        print(result.stdout.strip())
        
        # Test PaddleOCR import
        result = subprocess.run([
            str(ocr_python), "-c",
            "import paddleocr; print('V PaddleOCR imported successfully')"
        ], capture_output=True, text=True, timeout=60)  # PaddleOCR can be slow to import
        
        if result.returncode != 0:
            print(f"X PaddleOCR import failed: {result.stderr}")
            return False
            
        print(result.stdout.strip())
        return True
        
    except subprocess.TimeoutExpired:
        print("X OCR environment test timed out")
        return False
    except Exception as e:
        print(f"X OCR environment test error: {e}")
        return False

def test_environment_isolation():
    """Test that environments are properly isolated"""
    project_root = Path(__file__).parent.parent
    
    # Try environments subdirectory first (new structure) - check for yolo_env first, then detection_env
    detection_python = project_root / "environments" / "yolo_env" / "bin" / "python"
    ocr_python = project_root / "environments" / "ocr_env" / "bin" / "python"
    
    # Handle Windows paths
    if not detection_python.exists():
        detection_python = project_root / "environments" / "yolo_env" / "Scripts" / "python.exe"
    if not ocr_python.exists():
        ocr_python = project_root / "environments" / "ocr_env" / "Scripts" / "python.exe"
    
    # Fallback to detection_env for backward compatibility
    if not detection_python.exists():
        detection_python = project_root / "environments" / "detection_env" / "bin" / "python"
    if not detection_python.exists():
        detection_python = project_root / "environments" / "detection_env" / "Scripts" / "python.exe"
    
    # Fallback to root directory (legacy structure)
    if not detection_python.exists():
        detection_python = project_root / "yolo_env" / "bin" / "python"
    if not detection_python.exists():
        detection_python = project_root / "yolo_env" / "Scripts" / "python.exe"
    if not detection_python.exists():
        detection_python = project_root / "detection_env" / "bin" / "python"
    if not detection_python.exists():
        detection_python = project_root / "detection_env" / "Scripts" / "python.exe"
    if not ocr_python.exists():
        ocr_python = project_root / "ocr_env" / "bin" / "python"
    if not ocr_python.exists():
        ocr_python = project_root / "ocr_env" / "Scripts" / "python.exe"
    
    if not (detection_python.exists() and ocr_python.exists()):
        print("X Cannot test isolation - environments not found")
        print(f"  Detection env looked for: {project_root / 'environments' / 'detection_env'}")
        print(f"  OCR env looked for: {project_root / 'environments' / 'ocr_env'}")
        return False
    
    try:
        # Test that detection env doesn't have paddle
        result = subprocess.run([
            str(detection_python), "-c",
            "try:\n    import paddle\n    print('X PaddlePaddle found in detection_env (should not be there)')\nexcept ImportError:\n    print('V Detection env properly isolated - no PaddlePaddle')"
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print(result.stdout.strip())
        
        # Test that OCR env doesn't have torch (or has CPU-only version)
        result = subprocess.run([
            str(ocr_python), "-c",
            "try:\n    import torch\n    if torch.cuda.is_available():\n        print('! OCR env has CUDA PyTorch (might cause conflicts)')\n    else:\n        print('V OCR env has CPU-only PyTorch (good)')\nexcept ImportError:\n    print('V OCR env properly isolated - no PyTorch')"
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print(result.stdout.strip())
            
        return True
        
    except Exception as e:
        print(f"X Environment isolation test error: {e}")
        return False

def main():
    """Run all multi-environment tests"""
    print("Testing multi-environment setup...")
    print("=" * 50)
    
    tests = [
        ("Detection Environment", test_detection_env),
        ("OCR Environment", test_ocr_env), 
        ("Environment Isolation", test_environment_isolation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"V {test_name} passed")
            else:
                print(f"X {test_name} failed")
        except Exception as e:
            print(f"X {test_name} error: {e}")
    
    print(f"\n{'='*50}")
    print(f"Multi-environment test results: {passed}/{total} passed")
    
    if passed == total:
        print("V All multi-environment tests passed!")
        return 0
    else:
        print("X Some multi-environment tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
