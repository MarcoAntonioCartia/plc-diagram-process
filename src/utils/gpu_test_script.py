#!/usr/bin/env python3
"""
Test script specifically for PP-OCRv5 model with PaddleOCR
"""

import os
import time
import paddle
from paddleocr import PaddleOCR
import cv2
import numpy as np

# Your specific image path
IMAGE_PATH = r"D:\MarMe\github\0.3\plc-diagram-processor\debug_rois\page1_classTag-ID_prob0.87_x2235_y1291_w175_h102.png"

def setup_gpu():
    """Setup GPU for PaddleOCR"""
    print("=== GPU Setup ===")
    
    if paddle.device.is_compiled_with_cuda():
        gpu_count = paddle.device.cuda.device_count()
        print(f"✅ Available GPUs: {gpu_count}")
        
        if gpu_count > 0:
            gpu_name = paddle.device.cuda.get_device_name(0)
            print(f"  GPU 0: {gpu_name}")
            
            # Set GPU device
            paddle.device.set_device('gpu:0')
            print("✅ GPU device set to gpu:0")
            return True
        else:
            print("❌ No GPUs available")
            return False
    else:
        print("❌ PaddlePaddle not compiled with CUDA")
        return False

def check_image_file():
    """Check if the image file exists and display info"""
    print(f"\n=== Image File Check ===")
    print(f"Image path: {IMAGE_PATH}")
    
    if os.path.exists(IMAGE_PATH):
        print("✅ Image file exists")
        
        # Try to read with different extensions
        possible_extensions = ['', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for ext in possible_extensions:
            full_path = IMAGE_PATH + ext
            if os.path.exists(full_path):
                print(f"✅ Found file: {full_path}")
                
                # Load and display image info
                image = cv2.imread(full_path)
                if image is not None:
                    height, width, channels = image.shape
                    print(f"  Image dimensions: {width}x{height}x{channels}")
                    print(f"  File size: {os.path.getsize(full_path)} bytes")
                    return full_path
                else:
                    print(f"  Could not load image: {full_path}")
        
        print("❌ Could not find valid image file with common extensions")
        return None
    else:
        print("❌ Image file not found")
        return None

def test_pp_ocrv5():
    """Test PP-OCRv5 model specifically"""
    print(f"\n=== PP-OCRv5 Model Test ===")
    
    # Find the image file
    image_path = check_image_file()
    if not image_path:
        return False
    
    try:
        # Initialize PaddleOCR with PP-OCRv5 models
        print("Initializing PP-OCRv5 models...")
        
        # For newest PaddleOCR versions, use default initialization (PP-OCRv5)
        ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='en'
            # Note: model name parameters removed - defaults use PP-OCRv5
        )
        
        print("✅ PP-OCRv5 models loaded successfully")
        
        # Load and preprocess image
        print(f"Processing image: {os.path.basename(image_path)}")
        image = cv2.imread(image_path)
        
        if image is None:
            print("❌ Could not load image")
            return False
        
        # Display image info
        height, width = image.shape[:2]
        print(f"Image size: {width}x{height}")
        
        # Run OCR inference
        print("Running PP-OCRv5 inference...")
        start_time = time.time()
        
        result = ocr.predict(image_path)
        
        inference_time = time.time() - start_time
        print(f"✅ Inference completed in {inference_time:.3f}s")
        
        # Parse and display results
        print("\n=== OCR Results ===")
        
        if isinstance(result, dict):
            if 'rec_texts' in result and 'dt_polys' in result:
                texts = result['rec_texts']
                boxes = result['dt_polys']
                
                print(f"Found {len(texts)} text regions:")
                
                for i, (text_item, box) in enumerate(zip(texts, boxes)):
                    if isinstance(text_item, dict):
                        text = text_item.get('text', '')
                        confidence = text_item.get('score', 1.0)
                    else:
                        text = str(text_item)
                        confidence = 1.0
                    
                    # Calculate box center for reference
                    box_array = np.array(box)
                    center_x = int(box_array[:, 0].mean())
                    center_y = int(box_array[:, 1].mean())
                    
                    print(f"  {i+1}. Text: '{text}'")
                    print(f"      Confidence: {confidence:.3f}")
                    print(f"      Position: ({center_x}, {center_y})")
                    print(f"      Box: {box}")
                    print()
                
                if len(texts) == 0:
                    print("  No text detected")
                
            else:
                print(f"  Unexpected result format - keys: {list(result.keys())}")
                print(f"  Raw result: {result}")
        else:
            print(f"  Unexpected result type: {type(result)}")
            print(f"  Raw result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ PP-OCRv5 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_preprocessing():
    """Test with image preprocessing for better OCR results"""
    print(f"\n=== Test with Image Preprocessing ===")
    
    image_path = check_image_file()
    if not image_path:
        return False
    
    try:
        # Load image
        image = cv2.imread(image_path)
        
        # Apply preprocessing techniques
        print("Applying image preprocessing...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply different preprocessing techniques
        preprocessed_images = {
            'original': image,
            'grayscale': gray,
            'threshold': cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            'adaptive_threshold': cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            'gaussian_blur': cv2.GaussianBlur(gray, (3, 3), 0),
            'morphology': cv2.morphologyEx(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        }
        
        # Initialize OCR
        ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='en'
        )
        
        # Test each preprocessing method
        best_result = None
        best_confidence = 0
        best_method = 'original'
        
        for method_name, processed_image in preprocessed_images.items():
            print(f"\nTesting with {method_name} preprocessing...")
            
            try:
                start_time = time.time()
                result = ocr.predict(processed_image)
                processing_time = time.time() - start_time
                
                if isinstance(result, dict) and 'rec_texts' in result:
                    texts = result['rec_texts']
                    if texts:
                        # Calculate average confidence
                        confidences = []
                        all_text = []
                        
                        for text_item in texts:
                            if isinstance(text_item, dict):
                                confidence = text_item.get('score', 1.0)
                                text = text_item.get('text', '')
                            else:
                                confidence = 1.0
                                text = str(text_item)
                            
                            confidences.append(confidence)
                            all_text.append(text)
                        
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        print(f"  Time: {processing_time:.3f}s")
                        print(f"  Texts found: {len(texts)}")
                        print(f"  Average confidence: {avg_confidence:.3f}")
                        print(f"  Extracted text: {' | '.join(all_text)}")
                        
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_result = result
                            best_method = method_name
                    else:
                        print(f"  No text detected")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        print(f"\n✅ Best result with '{best_method}' preprocessing (confidence: {best_confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=== PP-OCRv5 PaddleOCR Test ===")
    print(f"Testing with image: {os.path.basename(IMAGE_PATH)}")
    
    # Setup GPU
    gpu_available = setup_gpu()
    
    if not gpu_available:
        print("⚠️  Continuing with CPU...")
    
    # Test basic PP-OCRv5
    basic_success = test_pp_ocrv5()
    
    # Test with preprocessing
    preprocess_success = test_with_preprocessing()
    
    print("\n=== Test Summary ===")
    print(f"Basic PP-OCRv5 test: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Preprocessing test: {'✅ PASS' if preprocess_success else '❌ FAIL'}")
    print(f"GPU usage: {'✅ ENABLED' if gpu_available else '❌ DISABLED'}")
    
    if basic_success:
        print(f"\n🚀 PP-OCRv5 working with your image!")
        print("For your pipeline, use:")
        print("import paddle; paddle.device.set_device('gpu:0')")
        print("from paddleocr import PaddleOCR")
        print("ocr = PaddleOCR(use_textline_orientation=True, lang='en')  # Uses PP-OCRv5 by default")
        print("result = ocr.predict(image_path)")
    
    return basic_success

if __name__ == "__main__":
    main()