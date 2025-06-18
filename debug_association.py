#!/usr/bin/env python3
"""
Debug script to test text-symbol association
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

def debug_association():
    """Debug the text-symbol association issue"""
    
    # Load detection data
    detection_file = Path("../plc-data/processed/detdiagrams2/1150_detections.json")
    
    if not detection_file.exists():
        print(f"Detection file not found: {detection_file}")
        return
    
    with open(detection_file, 'r') as f:
        detection_data = json.load(f)
    
    print("=== DETECTION DATA ANALYSIS ===")
    print(f"Detection file: {detection_file}")
    print(f"Number of pages: {len(detection_data.get('pages', []))}")
    
    for i, page_data in enumerate(detection_data.get("pages", [])):
        page_num = page_data.get("page", page_data.get("page_num", "unknown"))
        detections = page_data.get("detections", [])
        print(f"Page {i+1}: page_num={page_num}, detections={len(detections)}")
        
        if detections:
            # Show first few detections
            for j, detection in enumerate(detections[:3]):
                bbox = detection.get("global_bbox", detection.get("bbox_global", "missing"))
                class_name = detection.get("class_name", "unknown")
                confidence = detection.get("confidence", "unknown")
                print(f"  Detection {j+1}: {class_name} (conf={confidence}) bbox={bbox}")
    
    # Load text extraction results
    text_file = Path("../plc-data/processed/textextracted/1150_text_extraction.json")
    
    if not text_file.exists():
        print(f"Text extraction file not found: {text_file}")
        return
    
    with open(text_file, 'r') as f:
        text_data = json.load(f)
    
    print("\n=== TEXT EXTRACTION ANALYSIS ===")
    print(f"Text file: {text_file}")
    print(f"Total text regions: {text_data.get('total_text_regions', 0)}")
    print(f"Association rate: {text_data.get('statistics', {}).get('association_rate', 0)}%")
    
    # Show first few text regions
    text_regions = text_data.get("text_regions", [])
    print(f"\nFirst 5 text regions:")
    for i, region in enumerate(text_regions[:5]):
        text = region.get("text", "")
        page = region.get("page", "unknown")
        bbox = region.get("bbox", [])
        associated = region.get("associated_symbol") is not None
        print(f"  {i+1}. '{text}' (page={page}) bbox={bbox} associated={associated}")
    
    print("\n=== COORDINATE COMPARISON ===")
    # Compare coordinate ranges
    if detection_data.get("pages") and text_regions:
        det_page = detection_data["pages"][0]
        det_detections = det_page.get("detections", [])
        
        if det_detections:
            det_bbox = det_detections[0].get("global_bbox", det_detections[0].get("bbox_global", []))
            if isinstance(det_bbox, dict):
                det_bbox = [det_bbox["x1"], det_bbox["y1"], det_bbox["x2"], det_bbox["y2"]]
            print(f"Sample detection bbox: {det_bbox}")
        
        if text_regions:
            text_bbox = text_regions[0].get("bbox", [])
            print(f"Sample text bbox: {text_bbox}")
            
            # Check if coordinates are in similar ranges
            if det_bbox and text_bbox and len(det_bbox) == 4 and len(text_bbox) == 4:
                det_range = max(det_bbox) - min(det_bbox)
                text_range = max(text_bbox) - min(text_bbox)
                print(f"Detection coordinate range: {det_range:.1f}")
                print(f"Text coordinate range: {text_range:.1f}")
                
                if abs(det_range - text_range) > 1000:
                    print("WARNING: Coordinate ranges are very different - possible scaling issue!")

if __name__ == "__main__":
    debug_association()
