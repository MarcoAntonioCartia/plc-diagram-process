#!/usr/bin/env python3
"""
Test script to create highly visible PDF annotations
"""

import fitz  # PyMuPDF
from pathlib import Path

def test_visible_annotations():
    """Create a test PDF with large, visible annotations"""
    
    print("Creating test PDF with visible annotations...")
    
    # Open original PDF
    pdf_file = Path("pdf_debug/1150.pdf")
    doc = fitz.open(str(pdf_file))
    page = doc[0]
    
    print(f"PDF dimensions: {page.rect.width} x {page.rect.height}")
    
    # Add some large, obvious test annotations
    test_annotations = [
        # Top-left corner
        {"rect": (50, 50, 200, 150), "color": (1, 0, 0), "label": "TEST RED"},
        # Top-right corner  
        {"rect": (page.rect.width-200, 50, page.rect.width-50, 150), "color": (0, 1, 0), "label": "TEST GREEN"},
        # Bottom-left corner
        {"rect": (50, page.rect.height-150, 200, page.rect.height-50), "color": (0, 0, 1), "label": "TEST BLUE"},
        # Center
        {"rect": (page.rect.width/2-100, page.rect.height/2-50, page.rect.width/2+100, page.rect.height/2+50), "color": (1, 0, 1), "label": "TEST CENTER"},
    ]
    
    # Add the test annotations
    for i, ann_data in enumerate(test_annotations):
        rect = fitz.Rect(ann_data["rect"])
        color = ann_data["color"]
        label = ann_data["label"]
        
        # Add rectangle annotation
        annot = page.add_rect_annot(rect)
        annot.set_colors(stroke=color, fill=color)
        annot.set_border(width=3)
        annot.set_opacity(0.3)  # Semi-transparent
        annot.set_info(content=f"{label}\nTest annotation {i+1}", title="Visibility Test")
        annot.update()
        
        print(f"Added {label} annotation at {rect}")
    
    # Also add some text annotations
    text_points = [
        (100, 100, "Text annotation 1"),
        (page.rect.width-100, 100, "Text annotation 2"),
        (100, page.rect.height-100, "Text annotation 3"),
        (page.rect.width/2, page.rect.height/2, "Text annotation CENTER"),
    ]
    
    for x, y, text in text_points:
        point = fitz.Point(x, y)
        annot = page.add_text_annot(point, text)
        annot.set_info(title="Text Test", content=f"Test text: {text}")
        annot.set_colors(stroke=(1, 0.5, 0))  # Orange
        annot.update()
        
        print(f"Added text annotation '{text}' at ({x}, {y})")
    
    # Save test PDF
    output_file = Path("pdf_debug/1150_visibility_test.pdf")
    doc.save(str(output_file))
    doc.close()
    
    print(f"✓ Test PDF saved: {output_file}")
    print(f"✓ File size: {output_file.stat().st_size:,} bytes")
    
    # Now test with our actual detection coordinates but larger boxes
    print("\nTesting with actual detection coordinates (enlarged)...")
    
    # Reload and test with real coordinates
    doc = fitz.open(str(pdf_file))
    page = doc[0]
    
    # Use some coordinates from our debug output
    test_detections = [
        {"coords": (517, 2016, 562, 2047), "label": "Tag-ID (90.8%)", "color": (0, 0, 1)},
        {"coords": (517, 1674, 562, 1706), "label": "Tag-ID (92.9%)", "color": (0, 0, 1)},
        {"coords": (508, 1544, 552, 1575), "label": "Tag-ID (91.8%)", "color": (0, 0, 1)},
        {"coords": (572, 1548, 610, 1570), "label": "C0082 (82.2%)", "color": (0, 0.8, 0)},
    ]
    
    for i, det in enumerate(test_detections):
        x1, y1, x2, y2 = det["coords"]
        
        # Enlarge the box by 20 pixels in each direction
        enlarged_rect = fitz.Rect(x1-20, y1-20, x2+20, y2+20)
        
        # Ensure it's within bounds
        enlarged_rect = enlarged_rect & page.rect
        
        annot = page.add_rect_annot(enlarged_rect)
        annot.set_colors(stroke=det["color"], fill=det["color"])
        annot.set_border(width=2)
        annot.set_opacity(0.4)
        annot.set_info(content=det["label"], title="Detection Test")
        annot.update()
        
        print(f"Added enlarged detection box: {det['label']} at {enlarged_rect}")
    
    # Save enlarged detection test
    output_file2 = Path("pdf_debug/1150_detection_test.pdf")
    doc.save(str(output_file2))
    doc.close()
    
    print(f"✓ Detection test PDF saved: {output_file2}")
    print(f"✓ File size: {output_file2.stat().st_size:,} bytes")
    
    print("\nIf you can see the test annotations but not the detection boxes,")
    print("then the issue is that our detection boxes are too small or positioned incorrectly.")

if __name__ == "__main__":
    test_visible_annotations()
