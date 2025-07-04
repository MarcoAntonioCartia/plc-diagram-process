"""
Hybrid Text Extraction Pipeline for PLC Diagrams
Combines PaddleOCR and PyMuPDF for optimal text extraction from detected symbols
"""

import json
import cv2
import fitz  # PyMuPDF
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from paddleocr import PaddleOCR
# Ensure correct CUDA DLL order when Paddle is about to be used
from src.utils.gpu_manager import GPUManager

# Import our new modules
from .detection_deduplication import deduplicate_detections, analyze_detection_overlaps
from .roi_preprocessing import ROIPreprocessor, create_preprocessed_roi_folder

@dataclass
class TextRegion:
    """Represents a detected text region with metadata"""
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    source: str  # 'ocr' or 'pdf'
    page: int
    associated_symbol: Optional[Dict] = None

@dataclass
class PLCTextPattern:
    """Represents a PLC-specific text pattern"""
    pattern: str
    regex: str
    priority: int
    description: str

class TextExtractionPipeline:
    """
    Hybrid text extraction pipeline that combines:
    1. PyMuPDF for direct PDF text extraction (fast, accurate for digital text)
    2. PaddleOCR for image-based text extraction (handles scanned/image text)
    3. Smart fusion and PLC pattern recognition
    """
    
    def _get_device(self, device: Optional[str]) -> str:
        """
        PaddleOCR 3.x doesn't use device parameters anymore.
        Device selection is handled automatically by PaddleOCR.
        This method is kept for compatibility but doesn't affect initialization.
        """
        if device:
            print(f"Note: Device parameter '{device}' specified but PaddleOCR 3.x handles device selection automatically")
        
        print("PaddleOCR 3.x will automatically select the best available device (GPU/CPU)")
        return "auto"

    def __init__(self, confidence_threshold: float = 0.5, ocr_lang: str = "en", 
                 enable_nms: bool = True, nms_iou_threshold: float = 0.5,
                 enable_roi_preprocessing: bool = False,
                 perform_deduplication: bool = True, deduplication_iou_threshold: float = 0.5,
                 bbox_padding: float = 0, duplicate_iou_threshold: float = 0.7,
                 device: Optional[str] = None):
        """
        Initialize the text extraction pipeline with PaddleOCR 3.x
        
        Args:
            confidence_threshold: Minimum confidence for OCR results (lowered to 0.5)
            ocr_lang: Language for OCR (default: English)
            enable_nms: Whether to apply Non-Maximum Suppression to remove overlapping detections
            nms_iou_threshold: IoU threshold for NMS
            enable_roi_preprocessing: Whether to apply ROI preprocessing for better OCR
            perform_deduplication: Whether to perform detection deduplication
            deduplication_iou_threshold: IoU threshold for detection deduplication
            bbox_padding: Padding in pixels to add around YOLO bounding boxes (default: 0)
            duplicate_iou_threshold: IoU threshold for duplicate detection filtering (default: 0.7)
            device: Device preference (for logging only, PaddleOCR 3.x auto-selects)
        """
        self.confidence_threshold = confidence_threshold
        self.ocr_lang = ocr_lang
        self.enable_nms = enable_nms
        self.nms_iou_threshold = nms_iou_threshold
        self.enable_roi_preprocessing = enable_roi_preprocessing
        self.perform_deduplication = perform_deduplication
        self.deduplication_iou_threshold = deduplication_iou_threshold
        self.bbox_padding = bbox_padding
        self.duplicate_iou_threshold = duplicate_iou_threshold
        
        # Log device preference but don't use it for initialization
        self.device = self._get_device(device)
        
        # ------------------------------------------------------------------
        # Prepare the process for Paddle usage (set env vars, clear Torch caches
        # if it was imported earlier, etc.). This mitigates cuDNN / cuBLAS DLL
        # conflicts when Torch and Paddle live in the same interpreter.
        # ------------------------------------------------------------------
        try:
            GPUManager.global_instance().use_paddle()
        except Exception as _gpu_exc:  # pragma: no cover – never fatal
            print(f"[GPUManager] Warning: could not switch to paddle mode: {_gpu_exc}")
        
        # Initialize ROI preprocessor
        if self.enable_roi_preprocessing:
            self.roi_preprocessor = ROIPreprocessor()
            self.roi_preprocessor.set_debug_mode(True)  # Enable debug mode for now
        
        # Initialize PaddleOCR 3.x with the new simplified API
        self.ocr = None
        self.ocr_available = False
        
        # PaddleOCR 3.x initialization - much simpler than 2.x
        try:
            print(f"Initializing PaddleOCR 3.x with language: {ocr_lang}")
            
            # PaddleOCR 3.x simple initialization
            # Device selection is automatic - no parameters needed
            self.ocr = PaddleOCR(lang=ocr_lang)
            self.ocr_available = True
            print("[OK] PaddleOCR 3.x initialized successfully!")
            print("Device selection is handled automatically by PaddleOCR 3.x")
            
        except Exception as e:
            print(f"[ERROR] PaddleOCR 3.x initialization failed: {e}")
            print("Setting OCR to None - text extraction will use PDF text only.")
            self.ocr = None
            self.ocr_available = False
        
        # Define PLC text patterns (ordered by priority)
        self.plc_patterns = [
            PLCTextPattern("input", r"I\d+\.\d+", 10, "Input addresses (I0.1, I1.2, etc.)"),
            PLCTextPattern("output", r"Q\d+\.\d+", 10, "Output addresses (Q0.1, Q2.3, etc.)"),
            PLCTextPattern("memory", r"M\d+\.\d+", 9, "Memory addresses (M0.1, M1.2, etc.)"),
            PLCTextPattern("timer", r"T\d+", 8, "Timer addresses (T1, T2, etc.)"),
            PLCTextPattern("counter", r"C\d+", 8, "Counter addresses (C1, C2, etc.)"),
            PLCTextPattern("function_block", r"FB\d+", 7, "Function blocks (FB1, FB2, etc.)"),
            PLCTextPattern("data_block", r"DB\d+", 7, "Data blocks (DB1, DB2, etc.)"),
            PLCTextPattern("analog_input", r"AI\d+", 6, "Analog inputs (AI1, AI2, etc.)"),
            PLCTextPattern("analog_output", r"AO\d+", 6, "Analog outputs (AO1, AO2, etc.)"),
            PLCTextPattern("variable", r"[A-Z][A-Z0-9_]{2,}", 5, "Variable names"),
            PLCTextPattern("numeric", r"\d+\.?\d*", 3, "Numeric values"),
            PLCTextPattern("label", r"[A-Za-z][A-Za-z0-9_]{1,}", 2, "General labels"),
        ]
        
        # Compile regex patterns
        self.compiled_patterns = {
            pattern.pattern: re.compile(pattern.regex, re.IGNORECASE)
            for pattern in self.plc_patterns
        }
    
    def extract_text_from_detection_results(self, detection_file: Path, pdf_file: Path, 
                                          output_dir: Path) -> Dict[str, Any]:
        """
        Extract text from a PDF using detection results to guide the process
        
        Args:
            detection_file: Path to detection JSON file
            pdf_file: Path to original PDF file
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing extracted text results
        """
        print(f"Processing text extraction for: {pdf_file.name}")
        
        # Load detection results
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        # Apply duplicate detection filtering for Tag-ID class
        original_count = sum(len(page['detections']) for page in detection_data['pages'])
        print(f"Original detections: {original_count}")
        
        # Filter duplicates for Tag-ID class specifically
        detection_data = self._filter_duplicate_detections(detection_data)
        filtered_count = sum(len(page['detections']) for page in detection_data['pages'])
        print(f"After duplicate filtering: {filtered_count} (removed {original_count - filtered_count} duplicates)")
        
        # Apply Non-Maximum Suppression to remove overlapping detections
        if self.enable_nms:
            detection_data = deduplicate_detections(
                detection_data, 
                iou_threshold=self.nms_iou_threshold,
                class_specific=True
            )
            nms_count = sum(len(page['detections']) for page in detection_data['pages'])
            print(f"After NMS: {nms_count}")
        
        # Extract text using both methods
        pdf_texts = self._extract_pdf_text_near_detections(pdf_file, detection_data)
        ocr_texts = self._extract_ocr_text_from_regions(detection_data, pdf_file)
        
        # Combine and associate texts with symbols
        print(f"Combining texts: {len(pdf_texts)} PDF + {len(ocr_texts)} OCR = {len(pdf_texts) + len(ocr_texts)} total")
        combined_results = self._combine_and_associate_texts(
            pdf_texts, ocr_texts, detection_data
        )
        print(f"After deduplication: {len(combined_results)} combined text regions")
        
        # Apply PLC pattern recognition and filtering
        filtered_results = self._apply_plc_pattern_filtering(combined_results)
        print(f"After PLC pattern filtering: {len(filtered_results)} final text regions")
        
        # Generate output
        output_data = {
            "source_pdf": str(pdf_file),
            "detection_file": str(detection_file),
            "extraction_method": "hybrid",
            "total_text_regions": len(filtered_results),
            "text_regions": filtered_results,
            "plc_patterns_found": self._analyze_plc_patterns(filtered_results),
            "statistics": self._generate_extraction_statistics(filtered_results)
        }
        
        # Save results
        output_file = output_dir / f"{pdf_file.stem}_text_extraction.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Text extraction completed. Found {len(filtered_results)} text regions.")
        print(f"Results saved to: {output_file}")
        
        return output_data
    
    def _extract_pdf_text_near_detections(self, pdf_file: Path, detection_data: Dict) -> List[TextRegion]:
        """Extract text from PDF only near detected symbol regions"""
        text_regions = []
        
        try:
            doc = fitz.open(str(pdf_file))
            
            # Create detection regions for filtering
            detection_regions = []
            for page_data in detection_data["pages"]:
                page_number = page_data.get("page", page_data.get("page_num", 1))
                page_num = page_number - 1
                
                for detection in page_data["detections"]:
                    bbox = detection.get("bbox_global", detection.get("global_bbox", None))
                    if isinstance(bbox, dict):
                        bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        continue
                    
                    try:
                        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                        # Expand detection region by 100% to capture nearby text
                        width, height = x2 - x1, y2 - y1
                        expand_x, expand_y = width * 1.0, height * 1.0
                        
                        expanded_bbox = (
                            max(0, x1 - expand_x),
                            max(0, y1 - expand_y),
                            x2 + expand_x,
                            y2 + expand_y
                        )
                        detection_regions.append((page_num, expanded_bbox))
                    except (ValueError, TypeError):
                        continue
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text with detailed information
                text_dict = page.get_text("dict")
                
                for block in text_dict["blocks"]:
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text and len(text) > 0:
                                bbox = span["bbox"]  # x0, y0, x1, y1
                                text_center_x = (bbox[0] + bbox[2]) / 2
                                text_center_y = (bbox[1] + bbox[3]) / 2
                                
                                # Check if text is near any detection region
                                is_near_detection = False
                                for det_page, det_bbox in detection_regions:
                                    if det_page == page_num:
                                        if (det_bbox[0] <= text_center_x <= det_bbox[2] and
                                            det_bbox[1] <= text_center_y <= det_bbox[3]):
                                            is_near_detection = True
                                            break
                                
                                if is_near_detection:
                                    text_region = TextRegion(
                                        text=text,
                                        confidence=1.0,  # PDF text is always high confidence
                                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                                        source="pdf",
                                        page=page_num + 1
                                    )
                                    text_regions.append(text_region)
            
            doc.close()
            print(f"PDF text extraction: Found {len(text_regions)} text regions near detections")
            
        except Exception as e:
            print(f"Warning: Could not extract PDF text: {e}")
        
        return text_regions
    
    def _extract_ocr_text_from_regions(self, detection_data: Dict, pdf_file: Path) -> List[TextRegion]:
        """Extract text using OCR from detected symbol regions"""
        text_regions = []
        
        # Safety check: If OCR is not available, return empty list
        if not self.ocr_available or self.ocr is None:
            print("Warning: OCR not available - skipping OCR text extraction")
            return text_regions
        
        try:
            # Convert PDF pages to images for OCR
            doc = fitz.open(str(pdf_file))
            
            for page_data in detection_data["pages"]:
                # Handle both "page" and "page_num" keys for compatibility
                page_number = page_data.get("page", page_data.get("page_num", 1))
                page_num = page_number - 1
                
                # Get page image at 2x zoom for better OCR
                page = doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to OpenCV format
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                print(f"Page {page_num + 1} image dimensions: {img.shape} (H x W x C)")
                
                # Get original PDF dimensions for coordinate scaling
                original_width = page_data.get("original_width", img.shape[1] // 2)  # Divide by 2 because of 2x zoom
                original_height = page_data.get("original_height", img.shape[0] // 2)
                current_width = img.shape[1] // 2  # Actual PDF width (before 2x zoom)
                current_height = img.shape[0] // 2  # Actual PDF height (before 2x zoom)
                
                # Calculate scaling factors
                scale_x = current_width / original_width
                scale_y = current_height / original_height
                
                print(f"Coordinate scaling: Original ({original_width}x{original_height}) -> Current ({current_width}x{current_height})")
                print(f"Scale factors: X={scale_x:.3f}, Y={scale_y:.3f}")
                
                # Process each detection region
                for detection in page_data["detections"]:
                    # Handle both "bbox_global" and "global_bbox" keys for compatibility
                    # Detection data uses "bbox_global"
                    bbox = detection.get("bbox_global", detection.get("global_bbox", None))
                    if isinstance(bbox, dict):
                        bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        print(f"Warning: Invalid bbox format in detection: {bbox}")
                        continue
                    
                    # Convert coordinates to float to handle string values
                    try:
                        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Invalid coordinate values in bbox {bbox}: {e}")
                        continue
                    
                    # Scale coordinates from original PDF size to current PDF size, then for 2x zoom
                    x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
                    x1, y1, x2, y2 = x1 * 2, y1 * 2, x2 * 2, y2 * 2
                    
                    # Apply configurable padding to capture associated text
                    roi_x1 = max(0, int(x1 - self.bbox_padding))
                    roi_y1 = max(0, int(y1 - self.bbox_padding))
                    roi_x2 = min(img.shape[1], int(x2 + self.bbox_padding))
                    roi_y2 = min(img.shape[0], int(y2 + self.bbox_padding))
                    
                    # Validate ROI coordinates
                    if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
                        print(f"Warning: Invalid ROI coordinates: ({roi_x1}, {roi_y1}, {roi_x2}, {roi_y2})")
                        continue
                    
                    # Extract ROI
                    roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
                    
                    # Validate ROI is not empty
                    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                        print(f"Warning: Empty ROI extracted: shape {roi.shape}")
                        continue

                    # Validate ROI is not empty and has valid dimensions
                    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                        print(f"Warning: Empty ROI extracted: shape {roi.shape}")
                        continue
                    
                    if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                        # Apply ROI preprocessing if enabled
                        if self.enable_roi_preprocessing:
                            det_class = detection.get('class_name', 'unknown')
                            processed_roi = self.roi_preprocessor.preprocess_for_symbol_class(
                                roi, det_class
                            )
                        else:
                            processed_roi = roi
                        
                        # Run OCR on processed ROI using PaddleOCR 3.x API
                        try:
                            # PaddleOCR 3.x returns results in predict() method format
                            ocr_results = self.ocr.predict(processed_roi)
                        except Exception as e:
                            print(f"Warning: OCR failed on ROI: {e}")
                            continue
                        
                        if ocr_results and len(ocr_results) > 0:
                            try:
                                # PaddleOCR 3.x returns a list of OCRResult objects
                                ocr_result = ocr_results[0]
                                
                                # Extract data from PaddleOCR 3.x result format
                                # The result is a dictionary-like object with direct key access
                                if 'rec_texts' in ocr_result and 'rec_scores' in ocr_result and 'rec_polys' in ocr_result:
                                    texts = ocr_result['rec_texts']
                                    scores = ocr_result['rec_scores']
                                    polys = ocr_result['rec_polys']
                                    
                                    print(f"OCR found {len(texts)} text regions in ROI")
                                    
                                    # Process extracted texts
                                    if texts and scores and polys:
                                        for text, confidence, bbox_roi in zip(texts, scores, polys):
                                            if confidence >= self.confidence_threshold and text.strip():
                                                try:
                                                    # Convert ROI coordinates back to page coordinates
                                                    roi_bbox = np.array(bbox_roi)
                                                    roi_bbox[:, 0] += roi_x1  # Add ROI offset X
                                                    roi_bbox[:, 1] += roi_y1  # Add ROI offset Y
                                                    roi_bbox = roi_bbox / 2   # Scale back from 2x zoom
                                                    
                                                    # Get bounding box in page coordinates
                                                    min_x = float(np.min(roi_bbox[:, 0]))
                                                    min_y = float(np.min(roi_bbox[:, 1]))
                                                    max_x = float(np.max(roi_bbox[:, 0]))
                                                    max_y = float(np.max(roi_bbox[:, 1]))
                                                    
                                                    # Transform to global coordinates using associated symbol's snippet position
                                                    global_bbox = self._transform_to_global_coordinates(
                                                        (min_x, min_y, max_x, max_y), 
                                                        detection, 
                                                        original_width, 
                                                        original_height,
                                                        current_width,
                                                        current_height
                                                    )
                                                    
                                                    text_region = TextRegion(
                                                        text=text.strip(),
                                                        confidence=float(confidence),
                                                        bbox=global_bbox,
                                                        source="ocr",
                                                        page=page_num + 1,
                                                        associated_symbol=detection
                                                    )
                                                    text_regions.append(text_region)
                                                    print(f"OCR found text: '{text.strip()}' (confidence: {confidence:.3f}) at global coords: {global_bbox}")
                                                except Exception as e:
                                                    print(f"Warning: Error processing OCR text '{text}': {e}")
                                                    continue
                                    else:
                                        print(f"Warning: Empty OCR data in result")
                                else:
                                    print(f"Warning: Missing expected keys in OCR result. Available keys: {list(ocr_result.keys()) if hasattr(ocr_result, 'keys') else 'No keys'}")
                                    
                            except Exception as e:
                                print(f"Warning: Error processing OCR results: {e}")
                                import traceback
                                traceback.print_exc()
                                continue
            
            doc.close()
            print(f"OCR text extraction: Found {len(text_regions)} text regions from {len(detection_data.get('pages', []))} pages")
            
        except Exception as e:
            print(f"Warning: OCR extraction failed: {e}")
        
        return text_regions
    
    def _transform_to_global_coordinates(self, page_bbox: Tuple[float, float, float, float], 
                                       detection: Dict, original_width: float, original_height: float,
                                       current_width: float, current_height: float) -> Tuple[float, float, float, float]:
        """Transform page-level coordinates to global coordinates using snippet position"""
        try:
            # First try to get offset from bbox comparison (most accurate)
            bbox_snippet = detection.get("bbox_snippet", {})
            bbox_global = detection.get("bbox_global", {})
            
            if bbox_snippet and bbox_global:
                # Calculate offset from snippet to global coordinates
                if isinstance(bbox_snippet, dict) and isinstance(bbox_global, dict):
                    offset_x = bbox_global.get("x1", 0) - bbox_snippet.get("x1", 0)
                    offset_y = bbox_global.get("y1", 0) - bbox_snippet.get("y1", 0)
                else:
                    # Handle list format
                    offset_x = bbox_global[0] - bbox_snippet[0] if len(bbox_global) >= 4 and len(bbox_snippet) >= 4 else 0
                    offset_y = bbox_global[1] - bbox_snippet[1] if len(bbox_global) >= 4 and len(bbox_snippet) >= 4 else 0
                
                print(f"Using bbox offset: ({offset_x}, {offset_y})")
            else:
                # Fallback: use snippet position
                snippet_pos = detection.get("snippet_position", {})
                if not snippet_pos:
                    print(f"Warning: Cannot transform coordinates - no snippet position or bbox info available")
                    return page_bbox
                
                row = snippet_pos.get("row", 0)
                col = snippet_pos.get("col", 0)
                
                # For edge snippets, we need to look up the actual coordinates
                # Default calculation for non-edge snippets
                snippet_width = 1500
                snippet_height = 1200
                overlap = 500
                step_w = snippet_width - overlap  # 1000
                step_h = snippet_height - overlap  # 700
                
                # Calculate default offset
                offset_x = col * step_w
                offset_y = row * step_h
                
                # Calculate grid dimensions to detect edge snippets
                cols = max(1, (original_width - overlap) // step_w)
                rows = max(1, (original_height - overlap) // step_h)
                
                if original_width > cols * step_w:
                    cols += 1
                if original_height > rows * step_h:
                    rows += 1
                
                # Adjust for edge snippets
                # Right edge adjustment (last column)
                if col == cols - 1 and col > 0:  # Last column
                    # Ensure snippet doesn't exceed image width
                    offset_x = max(col * step_w, original_width - snippet_width)
                
                # Bottom edge adjustment (last row)
                if row == rows - 1 and row > 0:  # Last row
                    # Ensure snippet doesn't exceed image height
                    offset_y = max(row * step_h, original_height - snippet_height)
                
                print(f"Snippet position: row={row}, col={col}, adjusted offset=({offset_x}, {offset_y})")
            
            # Apply transformation
            x1, y1, x2, y2 = page_bbox
            
            # Scale from current page size back to original size if needed
            if current_width != original_width or current_height != original_height:
                scale_x = original_width / current_width
                scale_y = original_height / current_height
                x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
            
            # Add snippet offset to get global coordinates
            global_x1 = x1 + offset_x
            global_y1 = y1 + offset_y
            global_x2 = x2 + offset_x
            global_y2 = y2 + offset_y
            
            print(f"Coordinate transform: page({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) + offset({offset_x},{offset_y}) = global({global_x1:.1f},{global_y1:.1f},{global_x2:.1f},{global_y2:.1f})")
            
            return (global_x1, global_y1, global_x2, global_y2)
            
        except Exception as e:
            print(f"Warning: Error transforming coordinates: {e}")
            return page_bbox
    
    def _combine_and_associate_texts(self, pdf_texts: List[TextRegion], 
                                   ocr_texts: List[TextRegion], 
                                   detection_data: Dict) -> List[TextRegion]:
        """Combine PDF and OCR texts, removing duplicates and associating with symbols"""
        combined_texts = []
        
        # Start with PDF texts (higher priority)
        for pdf_text in pdf_texts:
            combined_texts.append(pdf_text)
        
        # Add OCR texts that don't overlap significantly with PDF texts
        for ocr_text in ocr_texts:
            is_duplicate = False
            
            for pdf_text in pdf_texts:
                if (pdf_text.page == ocr_text.page and 
                    self._texts_overlap(pdf_text, ocr_text) and
                    self._texts_similar(pdf_text.text, ocr_text.text)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                combined_texts.append(ocr_text)
        
        # Associate texts with nearby symbols
        for text_region in combined_texts:
            if not text_region.associated_symbol:
                text_region.associated_symbol = self._find_nearest_symbol(
                    text_region, detection_data
                )
        
        return combined_texts
    
    def _texts_overlap(self, text1: TextRegion, text2: TextRegion, threshold: float = 0.5) -> bool:
        """Check if two text regions overlap significantly"""
        x1_1, y1_1, x2_1, y2_1 = text1.bbox
        x1_2, y1_2, x2_2, y2_2 = text2.bbox
        
        # Calculate intersection
        x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        intersection = x_overlap * y_overlap
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate overlap ratio
        if area1 > 0 and area2 > 0:
            overlap_ratio = intersection / min(area1, area2)
            return overlap_ratio > threshold
        
        return False
    
    def _texts_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar (simple string similarity)"""
        text1_clean = re.sub(r'[^\w]', '', text1.lower())
        text2_clean = re.sub(r'[^\w]', '', text2.lower())
        
        if not text1_clean or not text2_clean:
            return False
        
        # Simple similarity check
        if text1_clean == text2_clean:
            return True
        
        # Check if one is contained in the other
        if text1_clean in text2_clean or text2_clean in text1_clean:
            return len(text1_clean) / len(text2_clean) > threshold or len(text2_clean) / len(text1_clean) > threshold
        
        return False
    
    def _find_nearest_symbol(self, text_region: TextRegion, detection_data: Dict) -> Optional[Dict]:
        """Find the nearest detected symbol to a text region"""
        min_distance = float('inf')
        nearest_symbol = None
        
        # Get text center
        tx1, ty1, tx2, ty2 = text_region.bbox
        text_center_x = (tx1 + tx2) / 2
        text_center_y = (ty1 + ty2) / 2
        
        for page_data in detection_data["pages"]:
            # Handle both "page" and "page_num" keys for compatibility
            # Detection data uses "page_num", text data uses "page"
            page_number = page_data.get("page_num", page_data.get("page", 0))
            
            # Ensure both are integers for comparison
            try:
                page_number = int(page_number)
                text_page = int(text_region.page)
            except (ValueError, TypeError):
                continue
                
            if page_number == text_page:
                for detection in page_data["detections"]:
                    # Handle both "bbox_global" and "global_bbox" keys for compatibility
                    # Detection data uses "bbox_global"
                    bbox = detection.get("bbox_global", detection.get("global_bbox", None))
                    if isinstance(bbox, dict):
                        bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        continue
                    
                    # Convert coordinates to float to handle string values
                    try:
                        sx1, sy1, sx2, sy2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    except (ValueError, TypeError):
                        continue

                    symbol_center_x = (sx1 + sx2) / 2
                    symbol_center_y = (sy1 + sy2) / 2
                    
                    # Calculate distance
                    distance = np.sqrt(
                        (text_center_x - symbol_center_x) ** 2 + 
                        (text_center_y - symbol_center_y) ** 2
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_symbol = detection
        
        # Only associate if reasonably close (within 300 pixels for large diagrams)
        if min_distance < 300:
            return nearest_symbol
        
        return None
    
    def _apply_plc_pattern_filtering(self, text_regions: List[TextRegion]) -> List[Dict[str, Any]]:
        """Apply PLC pattern recognition and filtering"""
        filtered_results = []
        
        for text_region in text_regions:
            text = text_region.text.strip()
            
            # Skip very short or empty texts
            if len(text) < 1:
                continue
            
            # Check against PLC patterns
            matched_patterns = []
            for pattern in self.plc_patterns:
                if self.compiled_patterns[pattern.pattern].search(text):
                    matched_patterns.append({
                        "pattern": pattern.pattern,
                        "priority": pattern.priority,
                        "description": pattern.description
                    })
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(text, matched_patterns)
            
            # Create result entry
            result = {
                "text": text,
                "confidence": text_region.confidence,
                "bbox": text_region.bbox,
                "source": text_region.source,
                "page": text_region.page,
                "matched_patterns": matched_patterns,
                "relevance_score": relevance_score,
                "associated_symbol": text_region.associated_symbol
            }
            
            filtered_results.append(result)
        
        # Sort by relevance score (highest first)
        filtered_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return filtered_results
    
    def _calculate_relevance_score(self, text: str, matched_patterns: List[Dict]) -> float:
        """Calculate relevance score for a text based on PLC patterns"""
        base_score = 1.0
        
        # Add pattern bonuses
        pattern_bonus = sum(pattern["priority"] for pattern in matched_patterns)
        
        # Length bonus (prefer reasonable length texts)
        length_bonus = min(2.0, len(text) / 5.0)
        
        # Alphanumeric bonus (PLC texts often contain numbers)
        alphanumeric_bonus = 1.0 if re.search(r'\d', text) else 0.0
        
        total_score = base_score + pattern_bonus + length_bonus + alphanumeric_bonus
        
        return round(total_score, 2)
    
    def _analyze_plc_patterns(self, text_regions: List[Dict]) -> Dict[str, int]:
        """Analyze which PLC patterns were found"""
        pattern_counts = {}
        
        for text_region in text_regions:
            for pattern in text_region["matched_patterns"]:
                pattern_name = pattern["pattern"]
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
        
        return pattern_counts
    
    def _generate_extraction_statistics(self, text_regions: List[Dict]) -> Dict[str, Any]:
        """Generate statistics about the extraction process"""
        if not text_regions:
            return {"total_texts": 0}
        
        pdf_count = sum(1 for tr in text_regions if tr["source"] == "pdf")
        ocr_count = sum(1 for tr in text_regions if tr["source"] == "ocr")
        
        avg_confidence = sum(tr["confidence"] for tr in text_regions) / len(text_regions)
        avg_relevance = sum(tr["relevance_score"] for tr in text_regions) / len(text_regions)
        
        associated_count = sum(1 for tr in text_regions if tr["associated_symbol"] is not None)
        
        return {
            "total_texts": len(text_regions),
            "pdf_extracted": pdf_count,
            "ocr_extracted": ocr_count,
            "average_confidence": round(avg_confidence, 3),
            "average_relevance_score": round(avg_relevance, 2),
            "texts_with_associated_symbols": associated_count,
            "association_rate": round(associated_count / len(text_regions) * 100, 1)
        }

    def process_detection_folder(self, detection_folder: Path, pdf_folder: Path, 
                               output_folder: Path) -> Dict[str, Any]:
        """
        Process all detection files in a folder
        
        Args:
            detection_folder: Folder containing detection JSON files
            pdf_folder: Folder containing original PDF files
            output_folder: Folder to save text extraction results
            
        Returns:
            Summary of processing results
        """
        output_folder.mkdir(parents=True, exist_ok=True)
        
        detection_files = list(detection_folder.glob("*_detections.json"))
        
        if not detection_files:
            print(f"No detection files found in {detection_folder}")
            return {"processed_files": 0, "results": []}
        
        results = []
        
        for detection_file in detection_files:
            # Find corresponding PDF file
            pdf_name = detection_file.name.replace("_detections.json", ".pdf")
            pdf_file = pdf_folder / pdf_name
            
            if not pdf_file.exists():
                print(f"Warning: PDF file not found: {pdf_file}")
                continue
            
            try:
                result = self.extract_text_from_detection_results(
                    detection_file, pdf_file, output_folder
                )
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {detection_file}: {e}")
                continue
        
        # Generate summary
        summary = {
            "processed_files": len(results),
            "total_text_regions": sum(r["total_text_regions"] for r in results),
            "results": results
        }
        
        # Save summary
        summary_file = output_folder / "text_extraction_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nText extraction completed for {len(results)} files")
        print(f"Total text regions extracted: {summary['total_text_regions']}")
        print(f"Summary saved to: {summary_file}")
        
        return summary
    
    def _filter_duplicate_detections(self, detection_data: Dict) -> Dict:
        """Filter duplicate detections using IoU-based approach"""
        filtered_data = {
            "pages": [],
            "metadata": detection_data.get("metadata", {})
        }
        
        for page_data in detection_data["pages"]:
            filtered_page = {
                "page": page_data.get("page", page_data.get("page_num", 1)),
                "page_num": page_data.get("page_num", page_data.get("page", 1)),
                "original_width": page_data.get("original_width"),
                "original_height": page_data.get("original_height"),
                "detections": []
            }
            
            detections = page_data.get("detections", [])
            if not detections:
                filtered_page["detections"] = []
                filtered_data["pages"].append(filtered_page)
                continue
            
            # Group detections by class for class-specific filtering
            class_groups = {}
            for detection in detections:
                class_name = detection.get("class_name", "unknown")
                if class_name not in class_groups:
                    class_groups[class_name] = []
                class_groups[class_name].append(detection)
            
            # Filter duplicates within each class
            for class_name, class_detections in class_groups.items():
                if class_name == "Tag-ID":
                    # Apply duplicate filtering for Tag-ID class
                    filtered_detections = self._filter_class_duplicates(class_detections)
                    print(f"  Tag-ID class: {len(class_detections)} -> {len(filtered_detections)} (removed {len(class_detections) - len(filtered_detections)} duplicates)")
                else:
                    # Keep all detections for other classes
                    filtered_detections = class_detections
                
                filtered_page["detections"].extend(filtered_detections)
            
            filtered_data["pages"].append(filtered_page)
        
        return filtered_data
    
    def _filter_class_duplicates(self, detections: List[Dict]) -> List[Dict]:
        """Filter duplicate detections within a single class using IoU"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
        
        filtered = []
        
        for detection in sorted_detections:
            bbox = detection.get("bbox_global", detection.get("global_bbox", None))
            if isinstance(bbox, dict):
                bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            
            try:
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            except (ValueError, TypeError):
                continue
            
            # Check if this detection overlaps significantly with any already filtered detection
            is_duplicate = False
            for existing in filtered:
                existing_bbox = existing.get("bbox_global", existing.get("global_bbox", None))
                if isinstance(existing_bbox, dict):
                    existing_bbox = [existing_bbox["x1"], existing_bbox["y1"], existing_bbox["x2"], existing_bbox["y2"]]
                if not (isinstance(existing_bbox, list) and len(existing_bbox) == 4):
                    continue
                
                try:
                    ex1, ey1, ex2, ey2 = float(existing_bbox[0]), float(existing_bbox[1]), float(existing_bbox[2]), float(existing_bbox[3])
                except (ValueError, TypeError):
                    continue
                
                # Calculate IoU
                iou = self._calculate_iou((x1, y1, x2, y2), (ex1, ey1, ex2, ey2))
                
                if iou > self.duplicate_iou_threshold:
                    is_duplicate = True
                    print(f"    Duplicate detected: IoU={iou:.3f} > {self.duplicate_iou_threshold}")
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
