from paddleocr import PaddleOCR
import sys

def test_paddleocr_versions():
    print("Testing PaddleOCR versions...")
    
    # Test different versions
    versions_to_test = ["PP-OCRv4", "PP-OCRv5", "PP-OCRv3"]
    
    for version in versions_to_test:
        try:
            print(f"\nTesting {version}...")
            ocr = PaddleOCR(ocr_version=version, lang="en", use_angle_cls=True)
            print(f"✓ {version} is available and working!")
            
            # Test a simple OCR call
            import numpy as np
            test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255  # White image
            result = ocr.ocr(test_img, cls=True)
            print(f"  OCR test successful: {len(result)} results")
            
        except Exception as e:
            print(f"✗ {version} failed: {e}")
    
    # Also test without specifying version
    try:
        print(f"\nTesting default version...")
        ocr = PaddleOCR(lang="en", use_angle_cls=True)
        print(f"✓ Default version is available!")
    except Exception as e:
        print(f"✗ Default version failed: {e}")

if __name__ == "__main__":
    test_paddleocr_versions()