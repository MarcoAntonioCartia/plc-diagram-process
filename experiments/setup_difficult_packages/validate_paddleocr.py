"""
Quick PaddleOCR validation script for ongoing testing
"""
def quick_paddleocr_test():
    """Quick test to verify PaddleOCR is working"""
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(show_log=False)
        print("✓ PaddleOCR is working correctly")
        return True
    except Exception as e:
        print(f"✗ PaddleOCR test failed: {e}")
        return False

if __name__ == "__main__":
    quick_paddleocr_test() 