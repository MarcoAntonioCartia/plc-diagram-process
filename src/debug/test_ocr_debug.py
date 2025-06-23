"""
Debug OCR issue by testing on a single ROI
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
from pathlib import Path

def test_ocr_on_roi():
    """Test OCR on the generated ROI to debug the issue"""
    
    # Initialize OCR
    print("Initializing OCR...")
    try:
        ocr = PaddleOCR(ocr_version="PP-OCRv5", lang="en", use_angle_cls=True)
        print("✓ OCR initialized successfully!")
    except Exception as e:
        print(f"OCR initialization failed: {e}")
        return
    
    # Load the ROI image that was generated
    roi_path = "debug_rois/page1_classxxxx_prob0.98_x1596_y403_w307_h127.png"
    
    if not Path(roi_path).exists():
        print(f"ROI file not found: {roi_path}")
        return
    
    print(f"Loading ROI: {roi_path}")
    roi = cv2.imread(roi_path)
    
    if roi is None:
        print("Failed to load ROI image")
        return
    
    print(f"ROI shape: {roi.shape}")
    
    # Test OCR
    print("Running OCR...")
    try:
        ocr_results = ocr.ocr(roi)
        print(f"OCR results type: {type(ocr_results)}")
        print(f"OCR results length: {len(ocr_results) if ocr_results else 0}")
        
        if ocr_results:
            print(f"First result type: {type(ocr_results[0])}")
            
            # Check if it's an OCRResult object
            ocr_result = ocr_results[0]
            if hasattr(ocr_result, '__dict__'):
                print("OCRResult attributes:")
                for attr in dir(ocr_result):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(ocr_result, attr)
                            if not callable(value):
                                print(f"  {attr}: {type(value)} = {value}")
                        except:
                            print(f"  {attr}: <error accessing>")
            
            # Also check if it has the expected attributes
            print(f"Has rec_texts: {hasattr(ocr_result, 'rec_texts')}")
            print(f"Has rec_scores: {hasattr(ocr_result, 'rec_scores')}")
            print(f"Has rec_polys: {hasattr(ocr_result, 'rec_polys')}")
        
    except Exception as e:
        print(f"OCR failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ocr_on_roi()
