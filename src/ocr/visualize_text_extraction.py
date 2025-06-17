"""
Visualize Text Extraction Results with YOLO Detection Boxes
Shows extracted text overlaid on the original PDF with detection boxes
"""

import json
import cv2
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any

def visualize_text_extraction(text_extraction_file: Path, pdf_file: Path, 
                            output_file: Path = None, show_confidence: bool = True):
    """
    Create a visualization of text extraction results with YOLO detection boxes
    
    Args:
        text_extraction_file: Path to text extraction JSON file
        pdf_file: Path to original PDF file
        output_file: Output image file (optional)
        show_confidence: Whether to show confidence scores
    """
    print(f"Visualizing text extraction results...")
    
    # Load text extraction results
    with open(text_extraction_file, 'r') as f:
        text_data = json.load(f)
    
    # Load PDF and convert to image
    doc = fitz.open(str(pdf_file))
    page = doc[0]  # First page
    
    # Convert page to image
    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better visualization
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    
    # Convert to OpenCV format
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Scale factor for coordinates
    scale_factor = 2.0
    
    # Draw detection boxes and associated text
    for text_region in text_data["text_regions"]:
        # Get text information
        text = text_region["text"]
        confidence = text_region["confidence"]
        source = text_region["source"]
        bbox = text_region["bbox"]
        associated_symbol = text_region.get("associated_symbol")
        
        # Scale coordinates for 2x zoom
        x1, y1, x2, y2 = [int(coord * scale_factor) for coord in bbox]
        
        # Choose color based on source
        if source == "pdf":
            color = (0, 255, 0)  # Green for PDF text
        else:
            color = (255, 0, 0)  # Red for OCR text
        
        # Draw text bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw text label
        label = f"{text}"
        if show_confidence:
            label += f" ({confidence:.2f})"
        
        # Calculate text position
        text_x = x1
        text_y = y1 - 10 if y1 > 20 else y2 + 20
        
        # Draw text background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (text_x, text_y - text_height - 5), 
                     (text_x + text_width, text_y + 5), (255, 255, 255), -1)
        
        # Draw text
        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw associated symbol box if available
        if associated_symbol and "global_bbox" in associated_symbol:
            symbol_bbox = associated_symbol["global_bbox"]
            if isinstance(symbol_bbox, list) and len(symbol_bbox) == 4:
                sx1, sy1, sx2, sy2 = [int(coord * scale_factor) for coord in symbol_bbox]
                
                # Draw symbol box in blue
                cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (255, 0, 255), 3)
                
                # Draw connection line
                text_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                symbol_center = ((sx1 + sx2) // 2, (sy1 + sy2) // 2)
                cv2.line(img, text_center, symbol_center, (0, 255, 255), 2)
    
    # Add legend
    legend_y = 30
    cv2.putText(img, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # PDF text (green)
    cv2.rectangle(img, (10, legend_y + 10), (30, legend_y + 30), (0, 255, 0), -1)
    cv2.putText(img, "PDF Text", (35, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # OCR text (red)
    cv2.rectangle(img, (10, legend_y + 40), (30, legend_y + 60), (255, 0, 0), -1)
    cv2.putText(img, "OCR Text", (35, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Symbol boxes (magenta)
    cv2.rectangle(img, (10, legend_y + 70), (30, legend_y + 90), (255, 0, 255), -1)
    cv2.putText(img, "YOLO Detections", (35, legend_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add statistics
    stats_text = f"Total Text Regions: {text_data['total_text_regions']}"
    cv2.putText(img, stats_text, (10, img.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    pdf_count = sum(1 for tr in text_data["text_regions"] if tr["source"] == "pdf")
    ocr_count = sum(1 for tr in text_data["text_regions"] if tr["source"] == "ocr")
    stats_text2 = f"PDF: {pdf_count}, OCR: {ocr_count}"
    cv2.putText(img, stats_text2, (10, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save or display image
    if output_file:
        cv2.imwrite(str(output_file), img)
        print(f"Visualization saved to: {output_file}")
    else:
        # Display image
        cv2.imshow("Text Extraction Results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    doc.close()
    
    return img

def main():
    parser = argparse.ArgumentParser(description='Visualize Text Extraction Results')
    parser.add_argument('--text-file', '-t', type=str, required=True,
                       help='Text extraction JSON file')
    parser.add_argument('--pdf-file', '-p', type=str, required=True,
                       help='Original PDF file')
    parser.add_argument('--output', '-o', type=str,
                       help='Output image file (optional, will display if not provided)')
    parser.add_argument('--no-confidence', action='store_true',
                       help='Hide confidence scores')
    
    args = parser.parse_args()
    
    text_file = Path(args.text_file)
    pdf_file = Path(args.pdf_file)
    output_file = Path(args.output) if args.output else None
    
    if not text_file.exists():
        print(f"Error: Text extraction file not found: {text_file}")
        return 1
    
    if not pdf_file.exists():
        print(f"Error: PDF file not found: {pdf_file}")
        return 1
    
    try:
        visualize_text_extraction(
            text_file, pdf_file, output_file, 
            show_confidence=not args.no_confidence
        )
        return 0
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())