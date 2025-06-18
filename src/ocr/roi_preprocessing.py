"""
ROI Preprocessing Module
Enhances ROI images for better OCR performance
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

class ROIPreprocessor:
    """
    Preprocesses ROI images to improve OCR accuracy
    """
    
    def __init__(self):
        self.debug_mode = False
    
    def set_debug_mode(self, debug: bool):
        """Enable/disable debug mode for saving intermediate images"""
        self.debug_mode = debug
    
    def preprocess_roi(self, roi_image: np.ndarray, 
                      symbol_class: str = "unknown",
                      debug_path: Optional[Path] = None) -> np.ndarray:
        """
        Apply preprocessing pipeline to ROI image
        
        Args:
            roi_image: Input ROI image (BGR format)
            symbol_class: Class of the detected symbol for class-specific processing
            debug_path: Path to save debug images (if debug_mode is True)
            
        Returns:
            Preprocessed image optimized for OCR
        """
        if roi_image is None or roi_image.size == 0:
            return roi_image
        
        # Convert to grayscale if needed
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image.copy()
        
        # Save original for debugging
        if self.debug_mode and debug_path:
            cv2.imwrite(str(debug_path.with_suffix('.0_original.png')), roi_image)
            cv2.imwrite(str(debug_path.with_suffix('.1_gray.png')), gray)
        
        # Apply preprocessing pipeline
        processed = self._apply_preprocessing_pipeline(gray, symbol_class)
        
        # Save processed for debugging
        if self.debug_mode and debug_path:
            cv2.imwrite(str(debug_path.with_suffix('.9_final.png')), processed)
        
        return processed
    
    def _apply_preprocessing_pipeline(self, gray_image: np.ndarray, 
                                    symbol_class: str) -> np.ndarray:
        """
        Apply the complete preprocessing pipeline
        
        Args:
            gray_image: Grayscale input image
            symbol_class: Symbol class for adaptive processing
            
        Returns:
            Preprocessed image
        """
        # Step 1: Noise reduction
        denoised = self._denoise_image(gray_image)
        
        # Step 2: Contrast enhancement
        enhanced = self._enhance_contrast(denoised)
        
        # Step 3: Resize if too small
        resized = self._resize_if_needed(enhanced)
        
        # Step 4: Binarization (convert to black/white)
        binary = self._binarize_image(resized, symbol_class)
        
        # Step 5: Morphological operations
        cleaned = self._morphological_cleanup(binary)
        
        # Step 6: Add padding
        padded = self._add_padding(cleaned)
        
        return padded
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction"""
        # Use bilateral filter to reduce noise while preserving edges
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _resize_if_needed(self, image: np.ndarray, min_height: int = 32) -> np.ndarray:
        """Resize image if it's too small for OCR"""
        h, w = image.shape[:2]
        
        if h < min_height:
            # Calculate scale factor to reach minimum height
            scale_factor = min_height / h
            new_width = int(w * scale_factor)
            new_height = min_height
            
            # Use INTER_CUBIC for upscaling
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            return resized
        
        return image
    
    def _binarize_image(self, image: np.ndarray, symbol_class: str) -> np.ndarray:
        """Convert image to binary (black/white)"""
        # Try adaptive thresholding first
        binary_adaptive = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # For certain symbol classes, try Otsu's method as well
        if symbol_class in ['Tag-ID', 'xxxx']:  # Text-heavy symbols
            _, binary_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Choose the method that produces more reasonable text regions
            # (heuristic: adaptive usually works better for mixed lighting)
            return binary_adaptive
        
        return binary_adaptive
    
    def _morphological_cleanup(self, binary_image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up the binary image"""
        # Remove small noise
        kernel_small = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_small)
        
        # Close small gaps in text
        kernel_close = np.ones((1, 3), np.uint8)  # Horizontal kernel for text
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        
        return cleaned
    
    def _add_padding(self, image: np.ndarray, padding: int = 10) -> np.ndarray:
        """Add padding around the image to avoid edge detection issues"""
        return cv2.copyMakeBorder(
            image, padding, padding, padding, padding, 
            cv2.BORDER_CONSTANT, value=255  # White padding
        )
    
    def preprocess_for_symbol_class(self, roi_image: np.ndarray, 
                                  symbol_class: str) -> np.ndarray:
        """
        Apply class-specific preprocessing
        
        Args:
            roi_image: Input ROI image
            symbol_class: Symbol class name
            
        Returns:
            Preprocessed image optimized for the specific symbol class
        """
        # Base preprocessing
        processed = self.preprocess_roi(roi_image, symbol_class, None)
        
        # Class-specific adjustments
        if symbol_class == 'Tag-ID':
            # Tag-IDs often have small text, enhance for readability
            processed = self._enhance_for_small_text(processed)
        elif symbol_class in ['C0082', 'X8164', 'X8022', 'X8117']:
            # Component symbols may have alphanumeric codes
            processed = self._enhance_for_alphanumeric(processed)
        elif symbol_class == 'xxxx':
            # Unknown symbols, use general enhancement
            processed = self._enhance_general(processed)
        
        return processed
    
    def _enhance_for_small_text(self, image: np.ndarray) -> np.ndarray:
        """Enhance image specifically for small text recognition"""
        # Upscale more aggressively for small text
        h, w = image.shape[:2]
        if h < 48:  # Increase threshold for small text
            scale_factor = 48 / h
            new_width = int(w * scale_factor)
            new_height = 48
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        return sharpened
    
    def _enhance_for_alphanumeric(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for alphanumeric character recognition"""
        # Apply slight dilation to make characters more solid
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(image, kernel, iterations=1)
        
        return dilated
    
    def _enhance_general(self, image: np.ndarray) -> np.ndarray:
        """General enhancement for unknown symbol types"""
        # Just return the standard processed image
        return image

def create_preprocessed_roi_folder(base_folder: Path, 
                                 detection_data: dict,
                                 pdf_file: Path) -> Path:
    """
    Create a folder structure for preprocessed ROIs
    
    Args:
        base_folder: Base folder for saving ROIs
        detection_data: Detection data
        pdf_file: Source PDF file
        
    Returns:
        Path to the preprocessed ROI folder
    """
    # Create folder name based on PDF
    pdf_name = pdf_file.stem
    preprocessed_folder = base_folder / f"{pdf_name}_preprocessed_rois"
    preprocessed_folder.mkdir(parents=True, exist_ok=True)
    
    return preprocessed_folder
