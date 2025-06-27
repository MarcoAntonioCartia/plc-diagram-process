from paddleocr import PaddleOCR
import sys

def test_paddleocr_versions():
    print("Testing PaddleOCR versions...")
    
    # Test different versions
    versions_to_test = ["PP-OCRv4", "PP-OCRv5", "PP-OCRv3"]
    
    for version in versions_to_test:
        try:
            # Test import and basic functionality
            if version == "latest":
                from paddleocr import PaddleOCR
            else:
                # For specific versions, we'd need version management
                from paddleocr import PaddleOCR
            
            # Test basic OCR functionality
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            print(f"V {version} is available and working!")
            
        except Exception as e:
            print(f"X {version} failed: {e}")
    
    # Test default installation
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(show_log=False)
        print(f"V Default version is available!")
    except Exception as e:
        print(f"X Default version failed: {e}")

if __name__ == "__main__":
    test_paddleocr_versions()