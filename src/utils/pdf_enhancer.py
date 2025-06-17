"""
PDF Enhancer Utility
Overlays YOLO detection boxes and OCR text results on original PDFs
"""

import json
import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import argparse

class PDFEnhancer:
    """
    Enhances PDFs with detection boxes and text extraction results
    """
    
    def __init__(self, font_size: float = 8, line_width: float = 1.0):
        """
        Initialize PDF enhancer
        
        Args:
            font_size: Font size for text annotations
            line_width: Line width for boxes and lines
        """
        self.font_size = font_size
        self.line_width = line_width
        
        # Color definitions (RGB format for PyMuPDF)
        self.colors = {
            "yolo_detection": (1, 0, 1),      # Magenta
            "pdf_text": (0, 1, 0),            # Green
            "ocr_text": (1, 0, 0),            # Red
            "connection": (1, 1, 0),          # Yellow
            "background": (1, 1, 1),          # White
            "text_color": (0, 0, 0)           # Black
        }
    
    def enhance_pdf_with_detections(self, 
                                  detection_file: Path,
                                  pdf_file: Path, 
                                  output_file: Optional[Path] = None) -> Path:
        """
        Enhance PDF with YOLO detection boxes only
        
        Args:
            detection_file: Path to detection JSON file
            pdf_file: Path to original PDF file
            output_file: Output PDF file (optional)
            
        Returns:
            Path to enhanced PDF file
        """
        print(f"Enhancing PDF with detections: {pdf_file.name}")
        
        # Load detection data
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        # Open PDF
        doc = fitz.open(str(pdf_file))
        
        # Process each page
        for page_data in detection_data.get("pages", []):
            page_num = page_data.get("page", page_data.get("page_num", 1)) - 1
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            
            # Draw detection boxes
            for detection in page_data.get("detections", []):
                self._draw_detection_box(page, detection)
        
        # Save enhanced PDF
        if output_file is None:
            output_file = pdf_file.parent / f"{pdf_file.stem}_with_detections.pdf"
        
        doc.save(str(output_file))
        doc.close()
        
        print(f"Enhanced PDF saved to: {output_file}")
        return output_file
    
    def enhance_pdf_with_text_extraction(self,
                                       text_extraction_file: Path,
                                       pdf_file: Path,
                                       output_file: Optional[Path] = None) -> Path:
        """
        Enhance PDF with text extraction results
        
        Args:
            text_extraction_file: Path to text extraction JSON file
            pdf_file: Path to original PDF file
            output_file: Output PDF file (optional)
            
        Returns:
            Path to enhanced PDF file
        """
        print(f"Enhancing PDF with text extraction: {pdf_file.name}")
        
        # Load text extraction data
        with open(text_extraction_file, 'r') as f:
            text_data = json.load(f)
        
        # Open PDF
        doc = fitz.open(str(pdf_file))
        
        # Process each page
        for page_data in text_data.get("pages", []):
            page_num = page_data.get("page", 1) - 1
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            
            # Group text regions by page
            page_text_regions = [tr for tr in text_data["text_regions"] if tr["page"] == page_num + 1]
            
            # Draw text regions and connections
            for text_region in page_text_regions:
                self._draw_text_region(page, text_region)
        
        # Add legend and statistics
        self._add_legend_and_stats(doc[0], text_data)
        
        # Save enhanced PDF
        if output_file is None:
            output_file = pdf_file.parent / f"{pdf_file.stem}_with_text_extraction.pdf"
        
        doc.save(str(output_file))
        doc.close()
        
        print(f"Enhanced PDF saved to: {output_file}")
        return output_file
    
    def enhance_pdf_complete(self,
                           detection_file: Path,
                           text_extraction_file: Path,
                           pdf_file: Path,
                           output_file: Optional[Path] = None) -> Path:
        """
        Enhance PDF with both detections and text extraction
        
        Args:
            detection_file: Path to detection JSON file
            text_extraction_file: Path to text extraction JSON file
            pdf_file: Path to original PDF file
            output_file: Output PDF file (optional)
            
        Returns:
            Path to enhanced PDF file
        """
        print(f"Creating complete enhanced PDF: {pdf_file.name}")
        
        # Load both datasets
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        with open(text_extraction_file, 'r') as f:
            text_data = json.load(f)
        
        # Open PDF
        doc = fitz.open(str(pdf_file))
        
        # Process each page
        for page_data in detection_data.get("pages", []):
            page_num = page_data.get("page", page_data.get("page_num", 1)) - 1
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            
            # Draw detection boxes first (background)
            for detection in page_data.get("detections", []):
                self._draw_detection_box(page, detection)
            
            # Draw text regions (foreground)
            page_text_regions = [tr for tr in text_data["text_regions"] if tr["page"] == page_num + 1]
            for text_region in page_text_regions:
                self._draw_text_region(page, text_region)
        
        # Add comprehensive legend and statistics
        self._add_complete_legend_and_stats(doc[0], detection_data, text_data)
        
        # Save enhanced PDF
        if output_file is None:
            output_file = pdf_file.parent / f"{pdf_file.stem}_complete_enhanced.pdf"
        
        doc.save(str(output_file))
        doc.close()
        
        print(f"Complete enhanced PDF saved to: {output_file}")
        return output_file
    
    def _draw_detection_box(self, page: fitz.Page, detection: Dict):
        """Draw a YOLO detection box on the page"""
        bbox = detection.get("global_bbox", detection.get("bbox_global", []))
        if not bbox or len(bbox) != 4:
            return
        
        x1, y1, x2, y2 = bbox
        
        # Create rectangle
        rect = fitz.Rect(x1, y1, x2, y2)
        
        # Draw box
        page.draw_rect(rect, color=self.colors["yolo_detection"], 
                      width=self.line_width, fill=None)
        
        # Add label
        class_name = detection.get("class_name", "unknown")
        confidence = detection.get("confidence", 0.0)
        label = f"{class_name} ({confidence:.2f})"
        
        # Position label above box
        text_point = fitz.Point(x1, y1 - 5)
        page.insert_text(text_point, label, fontsize=self.font_size, 
                        color=self.colors["text_color"])
    
    def _draw_text_region(self, page: fitz.Page, text_region: Dict):
        """Draw a text region on the page"""
        bbox = text_region["bbox"]
        text = text_region["text"]
        confidence = text_region["confidence"]
        source = text_region["source"]
        associated_symbol = text_region.get("associated_symbol")
        
        x1, y1, x2, y2 = bbox
        
        # Choose color based on source
        color = self.colors["pdf_text"] if source == "pdf" else self.colors["ocr_text"]
        
        # Draw text box
        rect = fitz.Rect(x1, y1, x2, y2)
        page.draw_rect(rect, color=color, width=self.line_width, fill=None)
        
        # Add text label
        label = f"{text} ({confidence:.2f})"
        text_point = fitz.Point(x1, y1 - 5)
        page.insert_text(text_point, label, fontsize=self.font_size, 
                        color=self.colors["text_color"])
        
        # Draw connection to associated symbol if available
        if associated_symbol and "global_bbox" in associated_symbol:
            symbol_bbox = associated_symbol["global_bbox"]
            if isinstance(symbol_bbox, list) and len(symbol_bbox) == 4:
                sx1, sy1, sx2, sy2 = symbol_bbox
                
                # Calculate centers
                text_center = fitz.Point((x1 + x2) / 2, (y1 + y2) / 2)
                symbol_center = fitz.Point((sx1 + sx2) / 2, (sy1 + sy2) / 2)
                
                # Draw connection line
                page.draw_line(text_center, symbol_center, 
                              color=self.colors["connection"], width=1)
    
    def _add_legend_and_stats(self, page: fitz.Page, text_data: Dict):
        """Add legend and statistics to the page"""
        # Add legend
        legend_y = 30
        legend_x = 10
        
        # PDF text (green)
        page.draw_rect(fitz.Rect(legend_x, legend_y, legend_x + 20, legend_y + 20), 
                      color=self.colors["pdf_text"], fill=self.colors["pdf_text"])
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 15), 
                        "PDF Text", fontsize=self.font_size, color=self.colors["text_color"])
        
        # OCR text (red)
        page.draw_rect(fitz.Rect(legend_x, legend_y + 30, legend_x + 20, legend_y + 50), 
                      color=self.colors["ocr_text"], fill=self.colors["ocr_text"])
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 45), 
                        "OCR Text", fontsize=self.font_size, color=self.colors["text_color"])
        
        # Statistics
        total_texts = text_data["total_text_regions"]
        pdf_count = sum(1 for tr in text_data["text_regions"] if tr["source"] == "pdf")
        ocr_count = sum(1 for tr in text_data["text_regions"] if tr["source"] == "ocr")
        
        stats_text = f"Total Text Regions: {total_texts} (PDF: {pdf_count}, OCR: {ocr_count})"
        page.insert_text(fitz.Point(legend_x, legend_y + 80), 
                        stats_text, fontsize=self.font_size, color=self.colors["text_color"])
    
    def _add_complete_legend_and_stats(self, page: fitz.Page, detection_data: Dict, text_data: Dict):
        """Add comprehensive legend and statistics"""
        legend_y = 30
        legend_x = 10
        
        # YOLO detections (magenta)
        page.draw_rect(fitz.Rect(legend_x, legend_y, legend_x + 20, legend_y + 20), 
                      color=self.colors["yolo_detection"], fill=self.colors["yolo_detection"])
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 15), 
                        "YOLO Detections", fontsize=self.font_size, color=self.colors["text_color"])
        
        # PDF text (green)
        page.draw_rect(fitz.Rect(legend_x, legend_y + 30, legend_x + 20, legend_y + 50), 
                      color=self.colors["pdf_text"], fill=self.colors["pdf_text"])
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 45), 
                        "PDF Text", fontsize=self.font_size, color=self.colors["text_color"])
        
        # OCR text (red)
        page.draw_rect(fitz.Rect(legend_x, legend_y + 60, legend_x + 20, legend_y + 80), 
                      color=self.colors["ocr_text"], fill=self.colors["ocr_text"])
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 75), 
                        "OCR Text", fontsize=self.font_size, color=self.colors["text_color"])
        
        # Connections (yellow)
        page.draw_line(fitz.Point(legend_x, legend_y + 95), fitz.Point(legend_x + 20, legend_y + 95), 
                      color=self.colors["connection"], width=2)
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 100), 
                        "Text-Symbol Connections", fontsize=self.font_size, color=self.colors["text_color"])
        
        # Statistics
        total_detections = sum(len(p.get("detections", [])) for p in detection_data.get("pages", []))
        total_texts = text_data["total_text_regions"]
        pdf_count = sum(1 for tr in text_data["text_regions"] if tr["source"] == "pdf")
        ocr_count = sum(1 for tr in text_data["text_regions"] if tr["source"] == "ocr")
        
        stats_text = f"YOLO Detections: {total_detections} | Text Regions: {total_texts} (PDF: {pdf_count}, OCR: {ocr_count})"
        page.insert_text(fitz.Point(legend_x, legend_y + 130), 
                        stats_text, fontsize=self.font_size, color=self.colors["text_color"])

def main():
    """CLI interface for PDF enhancement"""
    parser = argparse.ArgumentParser(description='Enhance PDFs with detection and text extraction results')
    parser.add_argument('--detection-file', '-d', type=str,
                       help='Detection JSON file')
    parser.add_argument('--text-file', '-t', type=str,
                       help='Text extraction JSON file')
    parser.add_argument('--pdf-file', '-p', type=str, required=True,
                       help='Original PDF file')
    parser.add_argument('--output', '-o', type=str,
                       help='Output PDF file')
    parser.add_argument('--mode', '-m', type=str, choices=['detections', 'text', 'complete'], 
                       default='complete', help='Enhancement mode')
    
    args = parser.parse_args()
    
    enhancer = PDFEnhancer()
    
    try:
        if args.mode == 'detections' and args.detection_file:
            enhancer.enhance_pdf_with_detections(
                Path(args.detection_file), Path(args.pdf_file), 
                Path(args.output) if args.output else None
            )
        elif args.mode == 'text' and args.text_file:
            enhancer.enhance_pdf_with_text_extraction(
                Path(args.text_file), Path(args.pdf_file), 
                Path(args.output) if args.output else None
            )
        elif args.mode == 'complete' and args.detection_file and args.text_file:
            enhancer.enhance_pdf_complete(
                Path(args.detection_file), Path(args.text_file), Path(args.pdf_file), 
                Path(args.output) if args.output else None
            )
        else:
            print("Error: Missing required files for selected mode")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"PDF enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())