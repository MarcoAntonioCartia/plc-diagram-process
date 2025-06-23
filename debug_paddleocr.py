import sys
from paddleocr import PaddleOCR
import torch

def get_device():
    """Determines the appropriate compute device."""
    try:
        # We need torch to be imported to check for CUDA
        if torch.cuda.is_available():
            print("CUDA is available. Setting device to 'gpu'.")
            return 'gpu'
        else:
            print("CUDA not available. Setting device to 'cpu'.")
            return 'cpu'
    except Exception as e:
        print(f"Could not determine CUDA availability: {e}. Defaulting to 'cpu'.")
        return 'cpu'

def run_tests():
    """Run a sequence of OCR initialization tests."""
    print("--- Starting PaddleOCR Initialization Test ---")
    device = get_device()
    use_gpu_flag = True if device == 'gpu' else False
    
    try:
        print("\n[Attempt 1] Initializing PP-OCRv5...")
        ocr_v5 = PaddleOCR(ocr_version="PP-OCRv5", lang='en', use_angle_cls=True, device=device)
        print("✅ SUCCESS: PP-OCRv5 initialized.")
        return 0
    except Exception:
        print("❌ FAILURE: PP-OCRv5 failed.")
        import traceback
        traceback.print_exc()

    try:
        print("\n[Attempt 2] Initializing PP-OCRv4...")
        ocr_v4 = PaddleOCR(ocr_version="PP-OCRv4", lang='en', use_angle_cls=True, device=device)
        print("✅ SUCCESS: PP-OCRv4 initialized.")
        return 0
    except Exception:
        print("❌ FAILURE: PP-OCRv4 failed.")
        import traceback
        traceback.print_exc()

    try:
        print("\n[Attempt 3] Initializing Default OCR...")
        ocr_default = PaddleOCR(lang='en', use_gpu=use_gpu_flag, use_angle_cls=True)
        print("✅ SUCCESS: Default OCR initialized.")
        return 0
    except Exception:
        print("❌ FAILURE: Default OCR failed.")
        import traceback
        traceback.print_exc()

    print("\n--- ❌ All OCR initialization attempts failed. ---")
    return 1

if __name__ == "__main__":
    sys.exit(run_tests()) 