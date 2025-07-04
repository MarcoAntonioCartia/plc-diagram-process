#!/usr/bin/env python3
"""
Coordinate Validation Debug Script
Analyzes where our transformed coordinates land compared to actual PDF text positions
"""

import json
import fitz  # PyMuPDF
from pathlib import Path

def debug_coordinate_validation():
    """Debug coordinate transformation by comparing with actual PDF text positions"""
    
    print("=" * 60)
    print("COORDINATE VALIDATION DEBUG")
    print("=" * 60)
    
    # Load files
    pdf_file = Path("pdf_debug/1150.pdf")
    detection_file = Path("pdf_debug/1150_detections.json")
    
    if not pdf_file.exists() or not detection_file.exists():
        print("❌ Required files not found")
        return
    
    # Open PDF and get text positions
    doc = fitz.open(str(pdf_file))
    page = doc[0]
    
    print(f"PDF Info:")
    print(f"  Dimensions: {page.rect.width} x {page.rect.height}")
    print(f"  Rotation: {page.rotation}°")
    print(f"  MediaBox: {page.mediabox}")
    
    # Extract text with positions
    print(f"\nExtracting text positions from PDF...")
    text_dict = page.get_text("dict")
    
    # Find some key text elements for reference
    reference_texts = []
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text and len(text) >= 4:  # Look for substantial text
                        bbox = span["bbox"]
                        reference_texts.append({
                            "text": text,
                            "bbox": bbox,
                            "x1": bbox[0], "y1": bbox[1], 
                            "x2": bbox[2], "y2": bbox[3]
                        })
    
    # Sort by position and show first 10
    reference_texts.sort(key=lambda x: (x["y1"], x["x1"]))
    print(f"\nFirst 10 text elements in PDF:")
    for i, ref in enumerate(reference_texts[:10]):
        print(f"  {i+1:2d}. '{ref['text'][:20]}' at ({ref['x1']:.0f},{ref['y1']:.0f})-({ref['x2']:.0f},{ref['y2']:.0f})")
    
    # Load detection data
    with open(detection_file, 'r') as f:
        detection_data = json.load(f)
    
    # Analyze first few detections
    print(f"\nAnalyzing detection coordinate transformations:")
    
    # Original image dimensions
    original_width = 9362
    original_height = 6623
    grid_rows = 6
    grid_cols = 4
    
    detections = []
    for page_data in detection_data.get("pages", []):
        if page_data.get("page", 1) == 1:
            detections = page_data.get("detections", [])
            break
    
    print(f"Found {len(detections)} detections")
    
    # Test coordinate transformation for first few detections
    for i, detection in enumerate(detections[:5]):
        snippet_bbox = detection.get("bbox_snippet", {})
        snippet_position = detection.get("snippet_position", {})
        class_name = detection.get("class_name", "unknown")
        confidence = detection.get("confidence", 0.0)
        
        if not snippet_bbox or not snippet_position:
            continue
            
        # Get coordinates
        sx1 = snippet_bbox.get("x1", 0)
        sy1 = snippet_bbox.get("y1", 0)
        sx2 = snippet_bbox.get("x2", 0)
        sy2 = snippet_bbox.get("y2", 0)
        
        row = snippet_position.get("row", 0)
        col = snippet_position.get("col", 0)
        
        print(f"\nDetection {i+1}: {class_name} ({confidence:.1%})")
        print(f"  Grid position: r{row}c{col}")
        print(f"  Snippet coords: ({sx1:.0f},{sy1:.0f})-({sx2:.0f},{sy2:.0f})")
        
        # Transform to original image coordinates
        grid_col = col - 2
        if grid_col < 0 or grid_col >= grid_cols or row < 0 or row >= grid_rows:
            print(f"  ❌ Invalid grid position")
            continue
            
        cell_width = original_width / grid_cols
        cell_height = original_height / grid_rows
        
        cell_base_x = grid_col * cell_width
        cell_base_y = row * cell_height
        
        orig_x1 = cell_base_x + sx1
        orig_y1 = cell_base_y + sy1
        orig_x2 = cell_base_x + sx2
        orig_y2 = cell_base_y + sy2
        
        print(f"  Original coords: ({orig_x1:.0f},{orig_y1:.0f})-({orig_x2:.0f},{orig_y2:.0f})")
        
        # Transform to PDF coordinates (with rotation)
        if page.rotation == 90:
            rot_x1 = orig_y1
            rot_y1 = original_width - orig_x2
            rot_x2 = orig_y2
            rot_y2 = original_width - orig_x1
            
            if rot_x1 > rot_x2:
                rot_x1, rot_x2 = rot_x2, rot_x1
            if rot_y1 > rot_y2:
                rot_y1, rot_y2 = rot_y2, rot_y1
            
            scale_x = page.rect.width / original_height
            scale_y = page.rect.height / original_width
            
            pdf_x1 = rot_x1 * scale_x
            pdf_y1 = rot_y1 * scale_y
            pdf_x2 = rot_x2 * scale_x
            pdf_y2 = rot_y2 * scale_y
            
            print(f"  Rotated coords: ({rot_x1:.0f},{rot_y1:.0f})-({rot_x2:.0f},{rot_y2:.0f})")
            print(f"  PDF coords: ({pdf_x1:.0f},{pdf_y1:.0f})-({pdf_x2:.0f},{pdf_y2:.0f})")
            
            # Check if coordinates are within PDF bounds
            within_bounds = (0 <= pdf_x1 <= page.rect.width and 
                           0 <= pdf_y1 <= page.rect.height and
                           0 <= pdf_x2 <= page.rect.width and 
                           0 <= pdf_y2 <= page.rect.height)
            
            print(f"  Within PDF bounds: {'✓' if within_bounds else '❌'}")
            
            # Find nearest text elements
            center_x = (pdf_x1 + pdf_x2) / 2
            center_y = (pdf_y1 + pdf_y2) / 2
            
            nearest_texts = []
            for ref in reference_texts:
                ref_center_x = (ref["x1"] + ref["x2"]) / 2
                ref_center_y = (ref["y1"] + ref["y2"]) / 2
                distance = ((center_x - ref_center_x)**2 + (center_y - ref_center_y)**2)**0.5
                nearest_texts.append((distance, ref))
            
            nearest_texts.sort()
            print(f"  Nearest text elements:")
            for j, (dist, ref) in enumerate(nearest_texts[:3]):
                print(f"    {j+1}. '{ref['text'][:15]}' (distance: {dist:.0f})")
    
    # Test with a known text element
    print(f"\n" + "─" * 40)
    print("REVERSE LOOKUP TEST")
    print("─" * 40)
    
    # Take a known text element and see if we can find a detection near it
    if reference_texts:
        test_text = reference_texts[0]  # First text element
        print(f"Testing with text: '{test_text['text']}'")
        print(f"Text position: ({test_text['x1']:.0f},{test_text['y1']:.0f})-({test_text['x2']:.0f},{test_text['y2']:.0f})")
        
        # Find detections near this text
        text_center_x = (test_text["x1"] + test_text["x2"]) / 2
        text_center_y = (test_text["y1"] + test_text["y2"]) / 2
        
        print(f"Looking for detections near ({text_center_x:.0f},{text_center_y:.0f})...")
        
        # Check all detections
        close_detections = []
        for i, detection in enumerate(detections):
            snippet_bbox = detection.get("bbox_snippet", {})
            snippet_position = detection.get("snippet_position", {})
            
            if not snippet_bbox or not snippet_position:
                continue
            
            # Transform this detection
            sx1 = snippet_bbox.get("x1", 0)
            sy1 = snippet_bbox.get("y1", 0)
            sx2 = snippet_bbox.get("x2", 0)
            sy2 = snippet_bbox.get("y2", 0)
            
            row = snippet_position.get("row", 0)
            col = snippet_position.get("col", 0)
            grid_col = col - 2
            
            if grid_col < 0 or grid_col >= grid_cols or row < 0 or row >= grid_rows:
                continue
            
            cell_width = original_width / grid_cols
            cell_height = original_height / grid_rows
            cell_base_x = grid_col * cell_width
            cell_base_y = row * cell_height
            
            orig_x1 = cell_base_x + sx1
            orig_y1 = cell_base_y + sy1
            orig_x2 = cell_base_x + sx2
            orig_y2 = cell_base_y + sy2
            
            if page.rotation == 90:
                rot_x1 = orig_y1
                rot_y1 = original_width - orig_x2
                rot_x2 = orig_y2
                rot_y2 = original_width - orig_x1
                
                if rot_x1 > rot_x2:
                    rot_x1, rot_x2 = rot_x2, rot_x1
                if rot_y1 > rot_y2:
                    rot_y1, rot_y2 = rot_y2, rot_y1
                
                scale_x = page.rect.width / original_height
                scale_y = page.rect.height / original_width
                
                pdf_x1 = rot_x1 * scale_x
                pdf_y1 = rot_y1 * scale_y
                pdf_x2 = rot_x2 * scale_x
                pdf_y2 = rot_y2 * scale_y
                
                det_center_x = (pdf_x1 + pdf_x2) / 2
                det_center_y = (pdf_y1 + pdf_y2) / 2
                
                distance = ((text_center_x - det_center_x)**2 + (text_center_y - det_center_y)**2)**0.5
                
                if distance < 100:  # Within 100 pixels
                    close_detections.append((distance, detection, (pdf_x1, pdf_y1, pdf_x2, pdf_y2)))
        
        close_detections.sort()
        print(f"Found {len(close_detections)} detections within 100 pixels:")
        for j, (dist, det, coords) in enumerate(close_detections[:3]):
            class_name = det.get("class_name", "unknown")
            confidence = det.get("confidence", 0.0)
            print(f"  {j+1}. {class_name} ({confidence:.1%}) at ({coords[0]:.0f},{coords[1]:.0f})-({coords[2]:.0f},{coords[3]:.0f}), distance: {dist:.0f}")
    
    doc.close()
    
    print(f"\n" + "=" * 60)
    print("COORDINATE VALIDATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    debug_coordinate_validation()
