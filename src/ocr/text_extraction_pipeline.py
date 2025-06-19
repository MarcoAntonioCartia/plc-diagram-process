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
    
    def __init__(self, confidence_threshold: float = 0.5, ocr_lang: str = "en", 
                 enable_nms: bool = True, nms_iou_threshold: float = 0.5,
                 enable_roi_preprocessing: bool = False):
        """
        Initialize the text extraction pipeline
        
        Args:
            confidence_threshold: Minimum confidence for OCR results (lowered to 0.5)
            ocr_lang: Language for OCR (default: English)
            enable_nms: Whether to apply Non-Maximum Suppression to remove overlapping detections
            nms_iou_threshold: IoU threshold for NMS
            enable_roi_preprocessing: Whether to apply ROI preprocessing for better OCR
        """
        self.confidence_threshold = confidence_threshold
        self.ocr_lang = ocr_lang
        self.enable_nms = enable_nms
        self.nms_iou_threshold = nms_iou_threshold
        self.enable_roi_preprocessing = enable_roi_preprocessing
        
        # Initialize ROI preprocessor
        if self.enable_roi_preprocessing:
            self.roi_preprocessor = ROIPreprocessor()
            self.roi_preprocessor.set_debug_mode(True)  # Enable debug mode for now
        
        # Initialize PaddleOCR with version compatibility
        try:
            # Try PP-OCRv5 first (newest)
            print("Trying PP-OCRv5...")
            self.ocr = PaddleOCR(ocr_version="PP-OCRv5", lang=ocr_lang, use_angle_cls=True)
            print("✓ PP-OCRv5 initialized successfully!")
        except Exception as e:
            try:
                # Fallback to PP-OCRv4
                print("PP-OCRv5 not available, trying PP-OCRv4...")
                self.ocr = PaddleOCR(ocr_version="PP-OCRv4", lang=ocr_lang, use_angle_cls=True)
                print("✓ PP-OCRv4 initialized successfully!")
            except Exception as e2:
                # Fallback to default version
                print("PP-OCRv4 not available, using default version...")
                self.ocr = PaddleOCR(lang=ocr_lang)
                print("✓ Default PaddleOCR version initialized!")     

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
        
        # Apply Non-Maximum Suppression to remove overlapping detections
        if self.enable_nms:
            print(f"Original detections: {sum(len(page['detections']) for page in detection_data['pages'])}")
            detection_data = deduplicate_detections(
                detection_data, 
                iou_threshold=self.nms_iou_threshold,
                class_specific=True
            )
            print(f"After NMS: {sum(len(page['detections']) for page in detection_data['pages'])}")
        
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
                    
                    # Expand region by 50% to capture associated text
                    width, height = x2 - x1, y2 - y1
                    expand_x, expand_y = width * 0.5, height * 0.5
                    
                    roi_x1 = max(0, int(x1 - expand_x))
                    roi_y1 = max(0, int(y1 - expand_y))
                    roi_x2 = min(img.shape[1], int(x2 + expand_x))
                    roi_y2 = min(img.shape[0], int(y2 + expand_y))
                    
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

                    # --- DEBUG: Save ROI and print info ---
                    import os
                    debug_dir = "debug_rois"
                    os.makedirs(debug_dir, exist_ok=True)
                    det_conf = detection.get('confidence', 0)
                    det_class = detection.get('class_name', 'unknown')
                    roi_filename = f"{debug_dir}/page{page_num+1}_class{det_class}_prob{det_conf:.2f}_x{roi_x1}_y{roi_y1}_w{roi_x2-roi_x1}_h{roi_y2-roi_y1}.png"
                    
                    # Only save if ROI is valid
                    if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                        try:
                            cv2.imwrite(roi_filename, roi)
                            print(f"Saved ROI: {roi_filename}, shape: {roi.shape}, bbox: {bbox}, expanded: ({roi_x1}, {roi_y1}, {roi_x2}, {roi_y2}), confidence: {det_conf}")
                        except Exception as e:
                            print(f"Warning: Failed to save ROI {roi_filename}: {e}")
                    else:
                        print(f"Skipped saving invalid ROI: shape {roi.shape}")
                    # --- END DEBUG ---
                    
                    if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                        # Apply ROI preprocessing if enabled
                        if self.enable_roi_preprocessing:
                            debug_path = Path(roi_filename) if self.roi_preprocessor.debug_mode else None
                            processed_roi = self.roi_preprocessor.preprocess_for_symbol_class(
                                roi, det_class
                            )
                        else:
                            processed_roi = roi
                        
                        # Run OCR on processed ROI
                        try:
                            ocr_results = self.ocr.ocr(processed_roi)
                        except Exception as e:
                            print(f"Warning: OCR failed on ROI: {e}")
                            continue
                        
                        if ocr_results and len(ocr_results) > 0:
                            try:
                                ocr_result = ocr_results[0]
                                
                                # Handle new PaddleOCR format (OCRResult object)
                                texts, scores, polys = None, None, None
                                
                                if hasattr(ocr_result, 'json') and 'res' in ocr_result.json:
                                    # New format: OCRResult object with nested data
                                    res_data = ocr_result.json['res']
                                    if 'rec_texts' in res_data and 'rec_scores' in res_data and 'rec_polys' in res_data:
                                        texts = res_data['rec_texts']
                                        scores = res_data['rec_scores']
                                        polys = res_data['rec_polys']
                                elif hasattr(ocr_result, 'rec_texts') and hasattr(ocr_result, 'rec_scores') and hasattr(ocr_result, 'rec_polys'):
                                    # Direct attributes (fallback)
                                    texts = ocr_result.rec_texts
                                    scores = ocr_result.rec_scores
                                    polys = ocr_result.rec_polys
                                
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
                                                
                                                # Get bounding box
                                                min_x = float(np.min(roi_bbox[:, 0]))
                                                min_y = float(np.min(roi_bbox[:, 1]))
                                                max_x = float(np.max(roi_bbox[:, 0]))
                                                max_y = float(np.max(roi_bbox[:, 1]))
                                                
                                                text_region = TextRegion(
                                                    text=text.strip(),
                                                    confidence=float(confidence),
                                                    bbox=(min_x, min_y, max_x, max_y),
                                                    source="ocr",
                                                    page=page_num + 1,
                                                    associated_symbol=detection
                                                )
                                                text_regions.append(text_region)
                                                print(f"OCR found text: '{text.strip()}' (confidence: {confidence:.3f})")
                                            except Exception as e:
                                                print(f"Warning: Error processing OCR text '{text}': {e}")
                                                continue
                                
                                # Handle old PaddleOCR format (list of lists)
                                elif isinstance(ocr_result, list):
                                    for line in ocr_result:
                                        if not isinstance(line, (list, tuple)):
                                            continue  # skip metadata/config keys
                                        
                                        try:
                                            if len(line) == 2 and isinstance(line[1], (list, tuple)) and len(line[1]) == 2:
                                                bbox_roi, (text, confidence) = line
                                            elif len(line) == 3:
                                                bbox_roi, text, confidence = line
                                            else:
                                                print(f"Warning: Unexpected OCR result format: {line}")
                                                continue
                                        except (ValueError, IndexError, TypeError) as e:
                                            print(f"Warning: Error parsing OCR line {line}: {e}")
                                            continue

                                        if confidence >= self.confidence_threshold and text.strip():
                                            try:
                                                # Convert ROI coordinates back to page coordinates
                                                roi_bbox = np.array(bbox_roi)
                                                roi_bbox[:, 0] += roi_x1  # Add ROI offset X
                                                roi_bbox[:, 1] += roi_y1  # Add ROI offset Y
                                                roi_bbox = roi_bbox / 2   # Scale back from 2x zoom
                                                
                                                # Get bounding box
                                                min_x = float(np.min(roi_bbox[:, 0]))
                                                min_y = float(np.min(roi_bbox[:, 1]))
                                                max_x = float(np.max(roi_bbox[:, 0]))
                                                max_y = float(np.max(roi_bbox[:, 1]))
                                                
                                                text_region = TextRegion(
                                                    text=text.strip(),
                                                    confidence=float(confidence),
                                                    bbox=(min_x, min_y, max_x, max_y),
                                                    source="ocr",
                                                    page=page_num + 1,
                                                    associated_symbol=detection
                                                )
                                                text_regions.append(text_region)
                                                print(f"OCR found text: '{text.strip()}' (confidence: {confidence:.3f})")
                                            except Exception as e:
                                                print(f"Warning: Error processing OCR text '{text}': {e}")
                                                continue
                                else:
                                    print(f"Warning: Unexpected OCR result type: {type(ocr_result)}")
                                    
                            except Exception as e:
                                print(f"Warning: Error processing OCR results: {e}")
                                continue
            
            doc.close()
            print(f"OCR text extraction: Found {len(text_regions)} text regions from {len(detection_data.get('pages', []))} pages")
            
        except Exception as e:
            print(f"Warning: OCR extraction failed: {e}")
        
        return text_regions
    
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
