"""
Enhanced Text Extraction Pipeline for PLC Diagrams
Integrates detection preprocessing and dual verification paths
"""

import json
import cv2
import fitz  # PyMuPDF
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import our modules
from .text_extraction_pipeline import TextExtractionPipeline, TextRegion, PLCTextPattern
from .detection_preprocessor import DetectionPreprocessor
from .paddle_ocr import PLCOCRProcessor

@dataclass
class VerificationResult:
    """Results from dual verification paths"""
    pdf_extraction: Dict[str, Any]
    ocr_extraction: Dict[str, Any]
    cross_validation: Dict[str, Any]
    consensus_confidence: float
    discrepancies: List[Dict[str, Any]]

class EnhancedTextExtractionPipeline(TextExtractionPipeline):
    """
    Enhanced pipeline with preprocessing and dual verification
    """
    
    def __init__(self, confidence_threshold: float = 0.7, ocr_lang: str = "en",
                 preprocess_detections: bool = True, iou_threshold: float = 0.5):
        """
        Initialize enhanced pipeline
        
        Args:
            confidence_threshold: Minimum confidence for OCR results
            ocr_lang: Language for OCR
            preprocess_detections: Whether to preprocess detection results
            iou_threshold: IoU threshold for detection preprocessing
        """
        super().__init__(confidence_threshold, ocr_lang)
        
        # Initialize detection preprocessor
        self.preprocess_detections = preprocess_detections
        if preprocess_detections:
            self.detection_preprocessor = DetectionPreprocessor(
                iou_threshold=iou_threshold,
                confidence_threshold=0.25  # Lower threshold for preprocessing
            )
        
        # Initialize enhanced OCR processor
        self.enhanced_ocr = PLCOCRProcessor(
            lang=ocr_lang,
            confidence_threshold=confidence_threshold
        )
        
        # Verification settings
        self.verification_enabled = True
        self.consensus_threshold = 0.8
    
    def extract_text_with_verification(self, detection_file: Path, pdf_file: Path, 
                                     output_dir: Path) -> VerificationResult:
        """
        Extract text using dual verification paths
        
        Args:
            detection_file: Path to detection JSON file
            pdf_file: Path to original PDF file
            output_dir: Directory to save results
            
        Returns:
            VerificationResult with both extraction paths and cross-validation
        """
        print(f"Processing with dual verification: {pdf_file.name}")
        
        # Step 1: Preprocess detection results if enabled
        processed_detection_file = detection_file
        if self.preprocess_detections:
            print("Preprocessing detection results...")
            processed_detection_file = self._preprocess_detections(detection_file, output_dir)
        
        # Step 2: Run both extraction paths
        print("Running dual extraction paths...")
        
        # Path 1: PDF-first approach (fast, accurate for digital text)
        pdf_result = self._extract_via_pdf_path(processed_detection_file, pdf_file, output_dir)
        
        # Path 2: OCR-first approach (robust for scanned/image text)
        ocr_result = self._extract_via_ocr_path(processed_detection_file, pdf_file, output_dir)
        
        # Step 3: Cross-validate results
        print("Cross-validating extraction results...")
        cross_validation = self._cross_validate_extractions(pdf_result, ocr_result)
        
        # Step 4: Generate consensus result
        verification_result = VerificationResult(
            pdf_extraction=pdf_result,
            ocr_extraction=ocr_result,
            cross_validation=cross_validation,
            consensus_confidence=cross_validation.get("consensus_confidence", 0.0),
            discrepancies=cross_validation.get("discrepancies", [])
        )
        
        # Step 5: Save comprehensive results
        self._save_verification_results(verification_result, output_dir, pdf_file.stem)
        
        return verification_result
    
    def _preprocess_detections(self, detection_file: Path, output_dir: Path) -> Path:
        """Preprocess detection results to remove overlaps"""
        output_file = output_dir / f"{detection_file.stem}_processed.json"
        
        if output_file.exists():
            print(f"Using existing preprocessed detections: {output_file}")
            return output_file
        
        self.detection_preprocessor.preprocess_detection_file(detection_file, output_file)
        return output_file
    
    def _extract_via_pdf_path(self, detection_file: Path, pdf_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Extraction Path 1: PDF-first approach
        Prioritizes direct PDF text extraction, uses OCR for regions without text
        """
        print("  Path 1: PDF-first extraction...")
        
        # Load detection results
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        # Extract text directly from PDF
        pdf_texts = self._extract_pdf_text(pdf_file)
        
        # Identify regions without PDF text coverage
        uncovered_regions = self._find_uncovered_detection_regions(pdf_texts, detection_data)
        
        # Use OCR only for uncovered regions
        ocr_texts = []
        if uncovered_regions:
            print(f"    OCR processing {len(uncovered_regions)} uncovered regions...")
            ocr_texts = self._extract_ocr_from_specific_regions(uncovered_regions, pdf_file)
        
        # Combine results
        all_texts = pdf_texts + ocr_texts
        filtered_results = self._apply_plc_pattern_filtering(all_texts)
        
        return {
            "method": "pdf_first",
            "pdf_text_count": len(pdf_texts),
            "ocr_text_count": len(ocr_texts),
            "total_text_regions": len(filtered_results),
            "text_regions": filtered_results,
            "uncovered_regions": len(uncovered_regions)
        }
    
    def _extract_via_ocr_path(self, detection_file: Path, pdf_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Extraction Path 2: OCR-first approach
        Uses OCR for all detected regions, validates with PDF text where available
        """
        print("  Path 2: OCR-first extraction...")
        
        # Load detection results
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        # Use OCR for all detection regions
        ocr_texts = self._extract_ocr_text_from_regions(detection_data, pdf_file)
        
        # Get PDF text for validation
        pdf_texts = self._extract_pdf_text(pdf_file)
        
        # Validate OCR results against PDF text
        validated_ocr_texts = self._validate_ocr_with_pdf(ocr_texts, pdf_texts)
        
        # Apply PLC pattern filtering
        filtered_results = self._apply_plc_pattern_filtering(validated_ocr_texts)
        
        return {
            "method": "ocr_first",
            "ocr_text_count": len(ocr_texts),
            "pdf_text_count": len(pdf_texts),
            "validated_ocr_count": len(validated_ocr_texts),
            "total_text_regions": len(filtered_results),
            "text_regions": filtered_results
        }
    
    def _find_uncovered_detection_regions(self, pdf_texts: List[TextRegion], 
                                        detection_data: Dict) -> List[Dict]:
        """Find detection regions not covered by PDF text"""
        uncovered_regions = []
        
        for page_data in detection_data.get("pages", []):
            page_num = page_data["page"]
            page_pdf_texts = [t for t in pdf_texts if t.page == page_num]
            
            for detection in page_data.get("detections", []):
                detection_bbox = detection.get("global_bbox", detection.get("bbox_global", {}))
                if isinstance(detection_bbox, dict):
                    det_bbox = (detection_bbox["x1"], detection_bbox["y1"], 
                              detection_bbox["x2"], detection_bbox["y2"])
                else:
                    det_bbox = tuple(detection_bbox)
                
                # Check if any PDF text overlaps with this detection
                has_pdf_coverage = False
                for pdf_text in page_pdf_texts:
                    if self._bboxes_overlap(det_bbox, pdf_text.bbox, threshold=0.3):
                        has_pdf_coverage = True
                        break
                
                if not has_pdf_coverage:
                    uncovered_regions.append({
                        "detection": detection,
                        "page": page_num,
                        "bbox": det_bbox
                    })
        
        return uncovered_regions
    
    def _extract_ocr_from_specific_regions(self, regions: List[Dict], pdf_file: Path) -> List[TextRegion]:
        """Extract OCR text from specific regions only"""
        text_regions = []
        
        try:
            doc = fitz.open(str(pdf_file))
            
            # Group regions by page for efficient processing
            regions_by_page = {}
            for region in regions:
                page_num = region["page"]
                if page_num not in regions_by_page:
                    regions_by_page[page_num] = []
                regions_by_page[page_num].append(region)
            
            # Process each page
            for page_num, page_regions in regions_by_page.items():
                page = doc[page_num - 1]  # Convert to 0-based indexing
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to OpenCV format
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Extract text from each region
                for region in page_regions:
                    bbox = region["bbox"]
                    # Scale bbox for 2x zoom
                    scaled_bbox = tuple(coord * 2 for coord in bbox)
                    
                    ocr_results = self.enhanced_ocr.extract_text_from_region(
                        img, scaled_bbox, expand_ratio=0.2
                    )
                    
                    # Convert results to TextRegion objects
                    for ocr_result in ocr_results:
                        # Scale coordinates back
                        orig_bbox = tuple(coord / 2 for coord in ocr_result["bbox"])
                        
                        text_region = TextRegion(
                            text=ocr_result["text"],
                            confidence=ocr_result["confidence"],
                            bbox=orig_bbox,
                            source="ocr_targeted",
                            page=page_num,
                            associated_symbol=region["detection"]
                        )
                        text_regions.append(text_region)
            
            doc.close()
            
        except Exception as e:
            print(f"Error in targeted OCR extraction: {e}")
        
        return text_regions
    
    def _validate_ocr_with_pdf(self, ocr_texts: List[TextRegion], 
                             pdf_texts: List[TextRegion]) -> List[TextRegion]:
        """Validate OCR results against PDF text"""
        validated_texts = []
        
        for ocr_text in ocr_texts:
            # Find overlapping PDF texts
            overlapping_pdf_texts = [
                pdf_text for pdf_text in pdf_texts
                if (pdf_text.page == ocr_text.page and 
                    self._texts_overlap(pdf_text, ocr_text, threshold=0.3))
            ]
            
            if overlapping_pdf_texts:
                # Use PDF text if available and similar
                best_pdf_match = max(overlapping_pdf_texts, 
                                   key=lambda x: self._text_similarity_score(x.text, ocr_text.text))
                
                if self._texts_similar(best_pdf_match.text, ocr_text.text, threshold=0.7):
                    # Use PDF text (higher confidence)
                    validated_text = TextRegion(
                        text=best_pdf_match.text,
                        confidence=1.0,
                        bbox=ocr_text.bbox,  # Use OCR bbox (more precise for regions)
                        source="pdf_validated",
                        page=ocr_text.page,
                        associated_symbol=ocr_text.associated_symbol
                    )
                    validated_texts.append(validated_text)
                else:
                    # Keep OCR text but mark as unvalidated
                    ocr_text.source = "ocr_unvalidated"
                    validated_texts.append(ocr_text)
            else:
                # No PDF overlap, keep OCR text
                validated_texts.append(ocr_text)
        
        return validated_texts
    
    def _cross_validate_extractions(self, pdf_result: Dict, ocr_result: Dict) -> Dict[str, Any]:
        """Cross-validate results from both extraction paths"""
        pdf_texts = pdf_result["text_regions"]
        ocr_texts = ocr_result["text_regions"]
        
        # Find matching texts between paths
        matches = []
        discrepancies = []
        consensus_texts = []
        
        # Compare texts from both methods
        for pdf_text in pdf_texts:
            best_match = None
            best_similarity = 0
            
            for ocr_text in ocr_texts:
                if (pdf_text["page"] == ocr_text["page"] and 
                    self._bboxes_overlap(tuple(pdf_text["bbox"]), tuple(ocr_text["bbox"]), 0.3)):
                    
                    similarity = self._text_similarity_score(pdf_text["text"], ocr_text["text"])
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = ocr_text
            
            if best_match and best_similarity > 0.8:
                # High confidence match
                consensus_text = pdf_text.copy()
                consensus_text["consensus_confidence"] = best_similarity
                consensus_text["verified_by"] = "both_methods"
                consensus_texts.append(consensus_text)
                matches.append({
                    "pdf_text": pdf_text,
                    "ocr_text": best_match,
                    "similarity": best_similarity
                })
            elif best_match and best_similarity > 0.5:
                # Partial match - record discrepancy
                discrepancies.append({
                    "type": "text_mismatch",
                    "pdf_text": pdf_text["text"],
                    "ocr_text": best_match["text"],
                    "similarity": best_similarity,
                    "bbox": pdf_text["bbox"],
                    "page": pdf_text["page"]
                })
        
        # Add unmatched texts from both methods
        for pdf_text in pdf_texts:
            if not any(m["pdf_text"] == pdf_text for m in matches):
                consensus_text = pdf_text.copy()
                consensus_text["consensus_confidence"] = 0.7
                consensus_text["verified_by"] = "pdf_only"
                consensus_texts.append(consensus_text)
        
        for ocr_text in ocr_texts:
            if not any(m["ocr_text"] == ocr_text for m in matches):
                consensus_text = ocr_text.copy()
                consensus_text["consensus_confidence"] = 0.6
                consensus_text["verified_by"] = "ocr_only"
                consensus_texts.append(consensus_text)
        
        # Calculate overall consensus confidence
        if consensus_texts:
            avg_confidence = sum(t["consensus_confidence"] for t in consensus_texts) / len(consensus_texts)
        else:
            avg_confidence = 0.0
        
        return {
            "matches": matches,
            "discrepancies": discrepancies,
            "consensus_texts": consensus_texts,
            "consensus_confidence": avg_confidence,
            "match_rate": len(matches) / max(len(pdf_texts), len(ocr_texts)) if pdf_texts or ocr_texts else 0
        }
    
    def _text_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Clean texts
        clean1 = re.sub(r'[^\w]', '', text1.lower())
        clean2 = re.sub(r'[^\w]', '', text2.lower())
        
        if clean1 == clean2:
            return 1.0
        
        # Simple Levenshtein-like similarity
        if not clean1 or not clean2:
            return 0.0
        
        # Jaccard similarity for character n-grams
        ngrams1 = set(clean1[i:i+2] for i in range(len(clean1)-1))
        ngrams2 = set(clean2[i:i+2] for i in range(len(clean2)-1))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union
    
    def _bboxes_overlap(self, bbox1: Tuple, bbox2: Tuple, threshold: float = 0.5) -> bool:
        """Check if two bounding boxes overlap"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        intersection = x_overlap * y_overlap
        
        # Calculate areas
        area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
        area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
        
        if area1 == 0 or area2 == 0:
            return False
        
        # Calculate overlap ratio
        overlap_ratio = intersection / min(area1, area2)
        return overlap_ratio > threshold
    
    def _save_verification_results(self, result: VerificationResult, output_dir: Path, filename_stem: str):
        """Save comprehensive verification results"""
        output_file = output_dir / f"{filename_stem}_verification_results.json"
        
        output_data = {
            "filename": filename_stem,
            "extraction_methods": {
                "pdf_first": result.pdf_extraction,
                "ocr_first": result.ocr_extraction
            },
            "cross_validation": result.cross_validation,
            "consensus_confidence": result.consensus_confidence,
            "discrepancies": result.discrepancies,
            "summary": {
                "total_consensus_texts": len(result.cross_validation.get("consensus_texts", [])),
                "match_rate": result.cross_validation.get("match_rate", 0),
                "discrepancy_count": len(result.discrepancies),
                "recommended_method": self._recommend_extraction_method(result)
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Verification results saved to: {output_file}")
    
    def _recommend_extraction_method(self, result: VerificationResult) -> str:
        """Recommend the best extraction method based on results"""
        pdf_count = result.pdf_extraction["total_text_regions"]
        ocr_count = result.ocr_extraction["total_text_regions"]
        match_rate = result.cross_validation.get("match_rate", 0)
        consensus_confidence = result.consensus_confidence
        
        if match_rate > 0.8 and consensus_confidence > 0.8:
            return "both_methods_reliable"
        elif pdf_count > ocr_count * 1.2 and match_rate > 0.6:
            return "pdf_first_recommended"
        elif ocr_count > pdf_count * 1.2:
            return "ocr_first_recommended"
        else:
            return "manual_review_recommended"

def main():
    """CLI interface for enhanced text extraction"""
    import sys
    import argparse
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    
    from src.config import get_config
    
    parser = argparse.ArgumentParser(description='Enhanced Text Extraction with Dual Verification')
    parser.add_argument('--detection-file', '-d', type=str, required=True,
                       help='Detection JSON file to process')
    parser.add_argument('--pdf-file', '-p', type=str, required=True,
                       help='Original PDF file')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--confidence', '-c', type=float, default=0.7,
                       help='OCR confidence threshold (default: 0.7)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for detection preprocessing (default: 0.5)')
    parser.add_argument('--no-preprocess', action='store_true',
                       help='Skip detection preprocessing')
    
    args = parser.parse_args()
    
    # Initialize enhanced pipeline
    pipeline = EnhancedTextExtractionPipeline(
        confidence_threshold=args.confidence,
        preprocess_detections=not args.no_preprocess,
        iou_threshold=args.iou_threshold
    )
    
    # Run extraction with verification
    detection_file = Path(args.detection_file)
    pdf_file = Path(args.pdf_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = pipeline.extract_text_with_verification(detection_file, pdf_file, output_dir)
        
        print(f"\nExtraction completed successfully!")
        print(f"Consensus confidence: {result.consensus_confidence:.3f}")
        print(f"Discrepancies found: {len(result.discrepancies)}")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 