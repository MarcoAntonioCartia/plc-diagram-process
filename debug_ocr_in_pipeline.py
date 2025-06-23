"""
Debug OCR initialization within the pipeline context
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

print("=== OCR Pipeline Debug ===")
print(f"Python: {sys.executable}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Test direct import and initialization
print("\nTest 1: Direct PaddleOCR initialization")
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR()
    print("✓ Direct initialization successful")
    del ocr
except Exception as e:
    print(f"✗ Direct initialization failed: {e}")

# Test pipeline initialization
print("\nTest 2: Pipeline TextExtractionPipeline initialization")
try:
    from ocr.text_extraction_pipeline import TextExtractionPipeline
    
    # Try with default settings
    print("  Trying default initialization...")
    pipeline = TextExtractionPipeline()
    print(f"  OCR available: {pipeline.ocr_available}")
    print(f"  OCR object: {pipeline.ocr}")
    
except Exception as e:
    print(f"✗ Pipeline initialization failed: {e}")
    import traceback
    traceback.print_exc()

# Test GPU/CPU settings
print("\nTest 3: GPU/CPU Environment")
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

# Check if we're forcing CPU mode
if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
    print("WARNING: CUDA_VISIBLE_DEVICES is set to -1, forcing CPU mode!")
    print("To enable GPU, unset this variable or set it to '0'")
