#!/usr/bin/env python3
"""
Simple PaddleOCR 3.x Test
Test the actual PaddleOCR 3.x API and check GPU usage
"""

import os
import cv2
import numpy as np
from pathlib import Path

def test_paddleocr_gpu():
    """Test PaddleOCR 3.x GPU usage and API"""
    print("Testing PaddleOCR 3.x...")
    print("=" * 50)
    
    # Check if we're in the right environment
    try:
        import paddle
        print(f"Paddle version: {paddle.__version__}")
        print(f"CUDA available: {paddle.device.is_compiled_with_cuda()}")
        if paddle.device.is_compiled_with_cuda():
            gpu_count = paddle.device.cuda.device_count()
            print(f"GPU count: {gpu_count}")
        print()
    except Exception as e:
        print(f"Error checking Paddle: {e}")
    
    # Test PaddleOCR initialization
    try:
        from paddleocr import PaddleOCR
        print("Initializing PaddleOCR 3.x...")
        
        # Test different initialization approaches
        ocr = PaddleOCR(lang='en')
        print("✓ PaddleOCR initialized successfully")
        print(f"OCR type: {type(ocr)}")
        print()
        
        # Create a simple test image with text
        test_image = create_test_image()
        
        # Test the predict method
        print("Testing OCR prediction...")
        results = ocr.predict(test_image)
        print(f"Results type: {type(results)}")
        print(f"Results length: {len(results) if results else 0}")
        
        if results and len(results) > 0:
            result = results[0]
            print(f"First result type: {type(result)}")
            print(f"Result attributes: {dir(result)}")
            
            # Try to access result data
            if hasattr(result, 'res'):
                print(f"result.res type: {type(result.res)}")
                print(f"result.res keys: {result.res.keys() if hasattr(result.res, 'keys') else 'No keys'}")
                
                if 'rec_texts' in result.res:
                    texts = result.res['rec_texts']
                    scores = result.res['rec_scores']
                    print(f"Found {len(texts)} text regions:")
                    for text, score in zip(texts, scores):
                        print(f"  '{text}' (confidence: {score:.3f})")
            else:
                print("No 'res' attribute found in result")
                
        return True
        
    except Exception as e:
        print(f"Error testing PaddleOCR: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_image():
    """Create a simple test image with text"""
    # Create a white image
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    # Add some text using OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Test Text 123', (50, 100), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'PaddleOCR', (50, 150), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    return img

def test_paddlex_vs_paddleocr():
    """Test the difference between PaddleX and PaddleOCR APIs"""
    print("\nTesting PaddleX vs PaddleOCR...")
    print("=" * 50)
    
    try:
        # Test PaddleX pipeline (which might be what's actually being used)
        import paddlex
        print(f"PaddleX version: {paddlex.__version__}")
        
        # Try the OCR pipeline from PaddleX
        from paddlex import create_pipeline
        ocr_pipeline = create_pipeline(pipeline="OCR")
        print(f"PaddleX OCR pipeline type: {type(ocr_pipeline)}")
        
        test_image = create_test_image()
        results = ocr_pipeline.predict(test_image)
        print(f"PaddleX results type: {type(results)}")
        
        if results:
            print(f"PaddleX results: {results}")
        
    except Exception as e:
        print(f"Error testing PaddleX: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    success = test_paddleocr_gpu()
    test_paddlex_vs_paddleocr()
    
    if success:
        print("\n✓ PaddleOCR 3.x test completed")
    else:
        print("\n✗ PaddleOCR 3.x test failed") 