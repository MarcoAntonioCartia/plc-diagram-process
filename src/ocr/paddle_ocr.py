# https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/quick_start.html
"""
Enhanced PaddleOCR utilities for PLC diagram text extraction
Provides both simple and advanced OCR functionality
"""

from paddleocr import PaddleOCR
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

class PLCOCRProcessor:
    """Enhanced OCR processor specifically designed for PLC diagrams"""
    
    def __init__(self, lang="en", confidence_threshold=0.7):
        """
        Initialize the OCR processor
        
        Args:
            lang: Language for OCR (default: English)
            confidence_threshold: Minimum confidence for text detection
        """
        self.ocr = PaddleOCR(ocr_version="PP-OCRv4", lang=lang, use_angle_cls=True, show_log=False)
        self.confidence_threshold = confidence_threshold
    
    def extract_text(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from image file (legacy function for compatibility)
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detected text regions with metadata
        """
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        return self.extract_text_from_image(img)
    
    def extract_text_from_image(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text from OpenCV image
        
        Args:
            img: OpenCV image (numpy array)
            
        Returns:
            List of detected text regions with metadata
        """
        try:
            result = self.ocr.ocr(img, cls=True)
            texts = []
            
            if result and result[0]:
                for line in result[0]:
                    if line:
                        box, (txt, score) = line
                        
                        if score >= self.confidence_threshold and txt.strip():
                            # Convert box coordinates to standard format
                            box_array = np.array(box)
                            min_x = float(np.min(box_array[:, 0]))
                            min_y = float(np.min(box_array[:, 1]))
                            max_x = float(np.max(box_array[:, 0]))
                            max_y = float(np.max(box_array[:, 1]))
                            
                            texts.append({
                                "text": txt.strip(),
                                "confidence": float(score),
                                "bbox": [min_x, min_y, max_x, max_y],
                                "original_box": box,
                                "source": "paddle_ocr"
                            })
            
            return texts
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return []
    
    def extract_text_from_region(self, img: np.ndarray, bbox: Tuple[int, int, int, int], 
                               expand_ratio: float = 0.1) -> List[Dict[str, Any]]:
        """
        Extract text from a specific region of an image
        
        Args:
            img: OpenCV image
            bbox: Bounding box as (x1, y1, x2, y2)
            expand_ratio: How much to expand the region (0.1 = 10% expansion)
            
        Returns:
            List of detected text regions
        """
        x1, y1, x2, y2 = bbox
        
        # Expand the region
        width, height = x2 - x1, y2 - y1
        expand_x = int(width * expand_ratio)
        expand_y = int(height * expand_ratio)
        
        # Calculate expanded coordinates
        exp_x1 = max(0, x1 - expand_x)
        exp_y1 = max(0, y1 - expand_y)
        exp_x2 = min(img.shape[1], x2 + expand_x)
        exp_y2 = min(img.shape[0], y2 + expand_y)
        
        # Extract region
        roi = img[exp_y1:exp_y2, exp_x1:exp_x2]
        
        if roi.size == 0:
            return []
        
        # Run OCR on region
        texts = self.extract_text_from_image(roi)
        
        # Adjust coordinates back to original image space
        for text in texts:
            bbox_roi = text["bbox"]
            text["bbox"] = [
                bbox_roi[0] + exp_x1,
                bbox_roi[1] + exp_y1,
                bbox_roi[2] + exp_x1,
                bbox_roi[3] + exp_y1
            ]
            text["region_bbox"] = [exp_x1, exp_y1, exp_x2, exp_y2]
        
        return texts
    
    def preprocess_image_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy
        
        Args:
            img: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply adaptive thresholding to improve text contrast
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Optional: Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        return processed

# Legacy function for backward compatibility
def extract_text(image_path):
    """Legacy function - use PLCOCRProcessor for new code"""
    processor = PLCOCRProcessor()
    return processor.extract_text(image_path)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    
    from src.config import get_config
    
    # Initialize processor
    processor = PLCOCRProcessor(confidence_threshold=0.6)
    
    # Try to find test images
    config = get_config()
    test_images_dir = Path(config.config['data_root']) / "datasets" / "test" / "images"
    
    if test_images_dir.exists():
        # Get first few images from test set
        image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        
        if image_files:
            print(f"Found {len(image_files)} test images")
            
            # Process first 3 images as examples
            for i, test_image in enumerate(image_files[:3]):
                print(f"\nProcessing test image {i+1}: {test_image.name}")
                texts = processor.extract_text(str(test_image))
                
                if texts:
                    print(f"Found {len(texts)} text regions:")
                    for j, text in enumerate(texts[:5]):  # Show first 5 results
                        print(f"  {j+1}. '{text['text']}' (confidence: {text['confidence']:.3f})")
                    if len(texts) > 5:
                        print(f"  ... and {len(texts) - 5} more")
                else:
                    print("  No text detected")
        else:
            print("No test images found")
    else:
        print(f"Test images directory not found: {test_images_dir}")
        print("Available directories:")
        datasets_dir = Path(config.config['data_root']) / "datasets"
        if datasets_dir.exists():
            for item in datasets_dir.iterdir():
                if item.is_dir():
                    print(f"  {item}")
