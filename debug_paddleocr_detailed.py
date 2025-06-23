"""
Detailed PaddleOCR debugging script
Tests various initialization methods and provides detailed error information
"""

import sys
import os
import traceback

print("=== PaddleOCR Detailed Debug ===")
print(f"Python: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Current Directory: {os.getcwd()}")
print()

# Test 1: Check if paddle can be imported
print("Test 1: Import paddle")
try:
    import paddle
    print(f"✓ Paddle imported successfully, version: {paddle.__version__}")
    
    # Check CUDA availability
    print(f"  CUDA available: {paddle.is_compiled_with_cuda()}")
    if paddle.is_compiled_with_cuda():
        print(f"  GPU count: {paddle.device.cuda.device_count()}")
except Exception as e:
    print(f"✗ Failed to import paddle: {e}")
    traceback.print_exc()
print()

# Test 2: Check PaddleOCR import
print("Test 2: Import PaddleOCR")
try:
    from paddleocr import PaddleOCR
    print("✓ PaddleOCR imported successfully")
except Exception as e:
    print(f"✗ Failed to import PaddleOCR: {e}")
    traceback.print_exc()
print()

# Test 3: Try different PaddleOCR initialization methods
print("Test 3: PaddleOCR Initialization Methods")

# Method 1: Minimal initialization
print("\nMethod 1: Minimal (no parameters)")
try:
    ocr = PaddleOCR()
    print("✓ Minimal PaddleOCR initialized successfully!")
    del ocr
except Exception as e:
    print(f"✗ Failed: {e}")
    print(f"  Error type: {type(e).__name__}")

# Method 2: English only
print("\nMethod 2: English language only")
try:
    ocr = PaddleOCR(lang='en')
    print("✓ English PaddleOCR initialized successfully!")
    del ocr
except Exception as e:
    print(f"✗ Failed: {e}")
    print(f"  Error type: {type(e).__name__}")

# Method 3: Force CPU mode
print("\nMethod 3: Force CPU mode")
try:
    # Set environment variable to force CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    ocr = PaddleOCR(use_gpu=False)
    print("✓ CPU-only PaddleOCR initialized successfully!")
    del ocr
except Exception as e:
    print(f"✗ Failed: {e}")
    print(f"  Error type: {type(e).__name__}")

# Method 4: With device parameter
print("\nMethod 4: With device='cpu' parameter")
try:
    ocr = PaddleOCR(device='cpu')
    print("✓ Device='cpu' PaddleOCR initialized successfully!")
    del ocr
except Exception as e:
    print(f"✗ Failed: {e}")
    print(f"  Error type: {type(e).__name__}")

# Method 5: Check available parameters
print("\nMethod 5: Check PaddleOCR parameters")
try:
    import inspect
    sig = inspect.signature(PaddleOCR.__init__)
    print("Available PaddleOCR parameters:")
    for param_name, param in sig.parameters.items():
        if param_name != 'self':
            default = param.default if param.default != inspect.Parameter.empty else 'no default'
            print(f"  - {param_name}: {default}")
except Exception as e:
    print(f"✗ Failed to inspect PaddleOCR: {e}")

# Test 4: Check DLL dependencies
print("\n\nTest 4: Check DLL Dependencies")
if sys.platform == 'win32':
    import pathlib
    venv_path = pathlib.Path(sys.executable).parent.parent
    dll_paths = [
        venv_path / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        venv_path / "Lib" / "site-packages" / "nvidia" / "cuda_nvrtc" / "bin",
        venv_path / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
        venv_path / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
    ]
    
    for path in dll_paths:
        if path.exists():
            dll_files = list(path.glob("*.dll"))
            print(f"✓ {path.name}: {len(dll_files)} DLL files")
            # Check for specific cudnn file
            cudnn_files = [f for f in dll_files if "cudnn" in f.name.lower()]
            if cudnn_files:
                print(f"  Found cuDNN files: {[f.name for f in cudnn_files]}")
        else:
            print(f"✗ {path.name}: Directory not found")

# Test 5: Environment variables
print("\n\nTest 5: Environment Variables")
env_vars = ['PATH', 'CUDA_PATH', 'CUDA_HOME', 'CUDNN_PATH', 'CUDA_VISIBLE_DEVICES']
for var in env_vars:
    value = os.environ.get(var, 'Not set')
    if var == 'PATH' and value != 'Not set':
        # Just show if CUDA/nvidia paths are in PATH
        has_cuda = 'cuda' in value.lower() or 'nvidia' in value.lower()
        print(f"{var}: {'Contains CUDA/NVIDIA paths' if has_cuda else 'No CUDA/NVIDIA paths found'}")
    else:
        print(f"{var}: {value}")

print("\n=== Debug Complete ===")
