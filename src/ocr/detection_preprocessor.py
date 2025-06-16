"""
Detection Results Preprocessor for Text Extraction
Removes overlapping and redundant detection boxes before text extraction
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class DetectionBox:
    """Represents a detection box with metadata"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    source_snippet: str
    page: int
    snippet_position: Dict[str, int]  # row, col
    
    def area(self) -> float:
        """Calculate bounding box area"""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)
    
    def iou(self, other: 'DetectionBox') -> float:
        """Calculate Intersection over Union with another detection box"""
        if self.page != other.page:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = self.bbox
        x1_2, y1_2, x2_2, y2_2 = other.bbox
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0
        
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        union = self.area() + other.area() - intersection
        
        return intersection / union if union > 0 else 0.0

class DetectionPreprocessor:
    """
    Preprocesses detection results to remove overlapping and redundant detections
    """
    
    def __init__(self, iou_threshold: float = 0.5, confidence_threshold: float = 0.25):
        """
        Initialize the preprocessor
        
        Args:
            iou_threshold: IoU threshold for considering detections as overlapping
            confidence_threshold: Minimum confidence to keep detections
        """
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
    
    def preprocess_detection_file(self, detection_file: Path, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Preprocess a detection JSON file to remove overlapping detections
        
        Args:
            detection_file: Path to detection JSON file
            output_file: Optional output file path (default: same as input with _processed suffix)
            
        Returns:
            Processed detection data
        """
        # Load detection results
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        # Process each page
        processed_data = {
            "source_file": str(detection_file),
            "preprocessing_settings": {
                "iou_threshold": self.iou_threshold,
                "confidence_threshold": self.confidence_threshold
            },
            "original_pdf": detection_data.get("original_pdf", ""),
            "pages": []
        }
        
        total_original = 0
        total_processed = 0
        
        for page_data in detection_data.get("pages", []):
            original_detections = page_data.get("detections", [])
            total_original += len(original_detections)
            
            # Convert to DetectionBox objects
            detection_boxes = self._convert_to_detection_boxes(original_detections, page_data["page"])
            
            # Apply NMS and filtering
            filtered_boxes = self._apply_nms_and_filtering(detection_boxes)
            total_processed += len(filtered_boxes)
            
            # Convert back to original format
            processed_detections = self._convert_from_detection_boxes(filtered_boxes)
            
            processed_page = {
                "page": page_data["page"],
                "original_width": page_data.get("original_width", 0),
                "original_height": page_data.get("original_height", 0),
                "original_detection_count": len(original_detections),
                "processed_detection_count": len(processed_detections),
                "detections": processed_detections
            }
            
            processed_data["pages"].append(processed_page)
        
        processed_data["summary"] = {
            "total_original_detections": total_original,
            "total_processed_detections": total_processed,
            "reduction_percentage": ((total_original - total_processed) / total_original * 100) if total_original > 0 else 0
        }
        
        # Save processed results
        if output_file is None:
            output_file = detection_file.parent / f"{detection_file.stem}_processed.json"
        
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"Preprocessing completed:")
        print(f"  Original detections: {total_original}")
        print(f"  Processed detections: {total_processed}")
        print(f"  Reduction: {processed_data['summary']['reduction_percentage']:.1f}%")
        print(f"  Saved to: {output_file}")
        
        return processed_data
    
    def _convert_to_detection_boxes(self, detections: List[Dict], page_num: int) -> List[DetectionBox]:
        """Convert detection dictionaries to DetectionBox objects"""
        boxes = []
        
        for detection in detections:
            # Handle different bbox formats
            if "global_bbox" in detection:
                bbox = detection["global_bbox"]
            elif "bbox_global" in detection:
                bbox_dict = detection["bbox_global"]
                bbox = [bbox_dict["x1"], bbox_dict["y1"], bbox_dict["x2"], bbox_dict["y2"]]
            elif "bbox" in detection:
                bbox = detection["bbox"]
                if isinstance(bbox, dict):
                    bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
            else:
                continue  # Skip if no bbox found
            
            # Ensure bbox is a list/tuple of 4 numbers
            if len(bbox) != 4:
                continue
            
            box = DetectionBox(
                class_id=detection.get("class_id", 0),
                class_name=detection.get("class_name", "unknown"),
                confidence=detection.get("confidence", 0.0),
                bbox=tuple(bbox),
                source_snippet=detection.get("snippet_source", "unknown"),
                page=page_num,
                snippet_position=detection.get("snippet_position", {"row": 0, "col": 0})
            )
            
            boxes.append(box)
        
        return boxes
    
    def _apply_nms_and_filtering(self, detection_boxes: List[DetectionBox]) -> List[DetectionBox]:
        """Apply Non-Maximum Suppression and confidence filtering"""
        # Filter by confidence
        filtered_boxes = [box for box in detection_boxes if box.confidence >= self.confidence_threshold]
        
        if not filtered_boxes:
            return []
        
        # Group by class for class-specific NMS
        class_groups = {}
        for box in filtered_boxes:
            class_name = box.class_name
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(box)
        
        # Apply NMS to each class group
        final_boxes = []
        for class_name, boxes in class_groups.items():
            nms_boxes = self._apply_nms_single_class(boxes)
            final_boxes.extend(nms_boxes)
        
        return final_boxes
    
    def _apply_nms_single_class(self, boxes: List[DetectionBox]) -> List[DetectionBox]:
        """Apply NMS to boxes of a single class"""
        if not boxes:
            return []
        
        # Sort by confidence (descending)
        boxes = sorted(boxes, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while boxes:
            # Take the box with highest confidence
            current = boxes.pop(0)
            keep.append(current)
            
            # Remove boxes with high IoU overlap
            remaining = []
            for box in boxes:
                if current.iou(box) < self.iou_threshold:
                    remaining.append(box)
                else:
                    # Log suppressed detection for debugging
                    print(f"    Suppressed overlapping detection: {box.class_name} "
                          f"(conf: {box.confidence:.3f}, IoU: {current.iou(box):.3f})")
            
            boxes = remaining
        
        return keep
    
    def _convert_from_detection_boxes(self, boxes: List[DetectionBox]) -> List[Dict]:
        """Convert DetectionBox objects back to dictionary format"""
        detections = []
        
        for box in boxes:
            detection = {
                "class_id": box.class_id,
                "class_name": box.class_name,
                "confidence": box.confidence,
                "global_bbox": list(box.bbox),
                "snippet_source": box.source_snippet,
                "snippet_position": box.snippet_position,
                "bbox_global": {
                    "x1": box.bbox[0],
                    "y1": box.bbox[1],
                    "x2": box.bbox[2],
                    "y2": box.bbox[3]
                }
            }
            detections.append(detection)
        
        return detections

if __name__ == "__main__":
    import sys
    import argparse
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    
    parser = argparse.ArgumentParser(description='Preprocess Detection Results for Text Extraction')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input detection file or folder')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file or folder (default: same as input with _processed suffix)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for NMS (default: 0.5)')
    parser.add_argument('--confidence-threshold', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    preprocessor = DetectionPreprocessor(
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold
    )
    
    if input_path.is_file():
        # Process single file
        preprocessor.preprocess_detection_file(input_path, output_path)
    elif input_path.is_dir():
        # Process folder
        detection_files = list(input_path.glob("*_detections.json"))
        if not detection_files:
            print(f"No detection files found in {input_path}")
            exit(1)
        
        for detection_file in detection_files:
            output_file = output_path / f"{detection_file.stem}_processed.json" if output_path else None
            preprocessor.preprocess_detection_file(detection_file, output_file)
    else:
        print(f"Error: Input path not found: {input_path}")
        exit(1) 