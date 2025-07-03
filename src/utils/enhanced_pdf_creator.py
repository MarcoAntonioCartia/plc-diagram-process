"""
Enhanced PDF Creator
Creates multi-page enhanced PDFs with proper coordinate mapping
Supports both long (4-page) and short (1-page) versions
"""

import json
import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import argparse

class EnhancedPDFCreator:
    """
    Creates enhanced PDFs with detection boxes and text extraction results
    Supports long version (4 pages) and short version (1 page)
    """
    
    def __init__(self, font_size: float = 10, line_width: float = 1.5, 
                 detection_confidence_threshold: float = 0.8,
                 text_confidence_threshold: float = 0.5):
        """
        Initialize Enhanced PDF Creator
        
        Args:
            font_size: Font size for text annotations
            line_width: Line width for boxes and lines
            detection_confidence_threshold: Minimum confidence for showing detection boxes
            text_confidence_threshold: Minimum confidence for showing text regions
        """
        self.font_size = font_size
        self.line_width = line_width
        self.detection_confidence_threshold = detection_confidence_threshold
        self.text_confidence_threshold = text_confidence_threshold
        
        # Color definitions (RGB format for PyMuPDF)
        self.colors = {
            "detection_box": (0, 0, 1),         # Blue for detection boxes
            "text_box": (0, 0.8, 0),            # Green for text boxes
            "text_label": (0, 0.6, 0),          # Darker green for text labels
            "connection": (0.7, 0.7, 0.7),      # Light gray for connections
            "background": (1, 1, 1),            # White
            "text_color": (0, 0, 0),            # Black
            "confidence_high": (0, 0.8, 0),     # Green for high confidence
            "confidence_medium": (1, 0.6, 0),   # Orange for medium confidence
            "confidence_low": (1, 0, 0),        # Red for low confidence
            "page_title": (0.2, 0.2, 0.2)      # Dark gray for page titles
        }
    
    def create_enhanced_pdf(self,
                           detection_file: Path,
                           text_extraction_file: Path,
                           pdf_file: Path,
                           output_file: Path,
                           version: str = 'long') -> Path:
        """
        Create enhanced PDF with detection boxes and text extraction
        
        Args:
            detection_file: Path to detection JSON file
            text_extraction_file: Path to text extraction JSON file
            pdf_file: Path to original PDF file
            output_file: Output PDF file path
            version: 'long' (4 pages) or 'short' (1 page)
            
        Returns:
            Path to enhanced PDF file
        """
        print(f"Creating {version} version enhanced PDF: {pdf_file.name}")
        
        # Load data
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        with open(text_extraction_file, 'r') as f:
            text_data = json.load(f)
        
        # Open original PDF
        original_doc = fitz.open(str(pdf_file))
        
        if version == 'long':
            enhanced_doc = self._create_long_version(original_doc, detection_data, text_data)
        else:
            enhanced_doc = self._create_short_version(original_doc, detection_data, text_data)
        
        # Save enhanced PDF
        enhanced_doc.save(str(output_file))
        enhanced_doc.close()
        original_doc.close()
        
        print(f"Enhanced PDF saved to: {output_file}")
        return output_file
    
    def _create_long_version(self, original_doc: fitz.Document, 
                           detection_data: Dict, text_data: Dict) -> fitz.Document:
        """Create 4-page long version"""
        enhanced_doc = fitz.open()
        
        for page_num in range(len(original_doc)):
            original_page = original_doc[page_num]
            
            # Page 1: Original PDF (clean)
            page1 = enhanced_doc.new_page(width=original_page.rect.width, 
                                        height=original_page.rect.height)
            page1.show_pdf_page(original_page.rect, original_doc, page_num)
            self._add_page_title(page1, "Page 1: Original PDF", 1)
            
            # Page 2: PDF + YOLO Detections
            page2 = enhanced_doc.new_page(width=original_page.rect.width, 
                                        height=original_page.rect.height)
            page2.show_pdf_page(original_page.rect, original_doc, page_num)
            self._add_page_title(page2, "Page 2: YOLO Detections", 2)
            self._draw_detections_only(page2, detection_data, page_num + 1)
            self._add_detection_legend(page2, detection_data, page_num + 1)
            
            # Page 3: PDF + OCR Text
            page3 = enhanced_doc.new_page(width=original_page.rect.width, 
                                        height=original_page.rect.height)
            page3.show_pdf_page(original_page.rect, original_doc, page_num)
            self._add_page_title(page3, "Page 3: OCR Text Extraction", 3)
            self._draw_text_regions_only(page3, text_data, page_num + 1, original_page.rect)
            self._add_text_legend(page3, text_data, page_num + 1)
            
            # Page 4: PDF + Both (Combined)
            page4 = enhanced_doc.new_page(width=original_page.rect.width, 
                                        height=original_page.rect.height)
            page4.show_pdf_page(original_page.rect, original_doc, page_num)
            self._add_page_title(page4, "Page 4: Combined View", 4)
            self._draw_detections_only(page4, detection_data, page_num + 1)
            self._draw_text_regions_only(page4, text_data, page_num + 1, original_page.rect)
            self._add_combined_legend(page4, detection_data, text_data, page_num + 1)
        
        return enhanced_doc
    
    def _create_short_version(self, original_doc: fitz.Document, 
                            detection_data: Dict, text_data: Dict) -> fitz.Document:
        """Create 1-page short version (same as page 4 of long version)"""
        enhanced_doc = fitz.open()
        
        for page_num in range(len(original_doc)):
            original_page = original_doc[page_num]
            
            # Create new page with proper orientation handling
            page = self._create_page_with_orientation(enhanced_doc, original_page)
            
            # Show original PDF content with proper rotation
            self._show_pdf_page_with_rotation(page, original_doc, page_num, original_page)
            
            self._add_page_title(page, "Enhanced PLC Diagram", 1)
            self._draw_detections_only(page, detection_data, page_num + 1, original_page)
            self._draw_text_regions_only(page, text_data, page_num + 1, original_page)
            self._add_combined_legend(page, detection_data, text_data, page_num + 1)
        
        return enhanced_doc
    
    def _create_page_with_orientation(self, enhanced_doc: fitz.Document, original_page: fitz.Page) -> fitz.Page:
        """Create a new page with proper orientation handling"""
        # Use the original page's MediaBox dimensions (before rotation)
        mediabox = original_page.mediabox
        rotation = original_page.rotation
        
        print(f"Creating page with MediaBox: {mediabox}, Rotation: {rotation}°")
        
        # Create page with MediaBox dimensions
        page = enhanced_doc.new_page(width=mediabox.width, height=mediabox.height)
        
        # Apply the same rotation as the original
        if rotation != 0:
            page.set_rotation(rotation)
            print(f"Applied {rotation}° rotation to new page")
        
        return page
    
    def _show_pdf_page_with_rotation(self, page: fitz.Page, original_doc: fitz.Document, 
                                   page_num: int, original_page: fitz.Page):
        """Show PDF page content with proper rotation handling"""
        # Get the original page's properties
        rotation = original_page.rotation
        mediabox = original_page.mediabox
        
        print(f"Showing PDF page {page_num + 1} with rotation {rotation}°")
        
        # Create the appropriate rectangle for showing the PDF content
        if rotation == 0:
            # No rotation needed
            show_rect = fitz.Rect(0, 0, mediabox.width, mediabox.height)
        elif rotation == 90:
            # 90° rotation: width and height are swapped in the display
            show_rect = fitz.Rect(0, 0, mediabox.width, mediabox.height)
        elif rotation == 180:
            # 180° rotation
            show_rect = fitz.Rect(0, 0, mediabox.width, mediabox.height)
        elif rotation == 270:
            # 270° rotation: width and height are swapped in the display
            show_rect = fitz.Rect(0, 0, mediabox.width, mediabox.height)
        else:
            # Default case
            show_rect = fitz.Rect(0, 0, mediabox.width, mediabox.height)
        
        # Show the PDF page content
        page.show_pdf_page(show_rect, original_doc, page_num)
    
    def _add_page_title(self, page: fitz.Page, title: str, page_num: int):
        """Add title to page"""
        title_rect = fitz.Rect(10, 10, 400, 30)
        page.draw_rect(title_rect, color=self.colors["background"], fill=self.colors["background"])
        page.insert_text(fitz.Point(15, 25), title, fontsize=self.font_size + 2, 
                        color=self.colors["page_title"])
    
    def _draw_detections_only(self, page: fitz.Page, detection_data: Dict, target_page: int, original_page: fitz.Page = None):
        """Draw only YOLO detection boxes"""
        for page_data in detection_data.get("pages", []):
            page_num = page_data.get("page", page_data.get("page_num", 1))
            if page_num != target_page:
                continue
                
            for detection in page_data.get("detections", []):
                confidence = detection.get("confidence", 0.0)
                
                # Only draw if confidence meets threshold
                if confidence < self.detection_confidence_threshold:
                    continue
                
                # Get global bbox
                bbox = detection.get("bbox_global", detection.get("global_bbox", None))
                if isinstance(bbox, dict):
                    bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                
                x1, y1, x2, y2 = bbox
                
                # Create rectangle
                rect = fitz.Rect(x1, y1, x2, y2)
                
                # Choose color based on confidence level
                if confidence >= 0.95:
                    box_color = self.colors["confidence_high"]
                elif confidence >= 0.85:
                    box_color = self.colors["confidence_medium"]
                else:
                    box_color = self.colors["detection_box"]
                
                # Draw box
                page.draw_rect(rect, color=box_color, width=self.line_width, fill=None)
                
                # Add label
                class_name = detection.get("class_name", "unknown")
                label = f"{class_name} {confidence:.0%}"
                
                # Position label above box
                text_point = fitz.Point(x1, y1 - 8)
                page.insert_text(text_point, label, fontsize=self.font_size - 1, 
                                color=self.colors["text_color"])
    
    def _draw_text_regions_only(self, page: fitz.Page, text_data: Dict, 
                              target_page: int, original_page: fitz.Page):
        """Draw only OCR text regions with proper coordinate mapping"""
        
        # Get page text regions
        page_text_regions = [tr for tr in text_data["text_regions"] if tr["page"] == target_page]
        
        for text_region in page_text_regions:
            confidence = text_region.get("confidence", 0.0)
            
            # Only draw if confidence meets threshold
            if confidence < self.text_confidence_threshold:
                continue
            
            # Get text region bbox - these appear to be in a different coordinate system
            bbox = text_region["bbox"]
            text = text_region["text"]
            source = text_region["source"]
            
            # Try to map coordinates properly
            # The text coordinates seem to be scaled down significantly
            x1, y1, x2, y2 = bbox
            
            # Check if we have an associated symbol for coordinate reference
            associated_symbol = text_region.get("associated_symbol")
            if associated_symbol and "bbox_global" in associated_symbol:
                # Use the associated symbol's coordinates as reference
                symbol_bbox = associated_symbol["bbox_global"]
                if isinstance(symbol_bbox, dict):
                    symbol_bbox = [symbol_bbox["x1"], symbol_bbox["y1"], 
                                 symbol_bbox["x2"], symbol_bbox["y2"]]
                
                # Position text near the associated symbol
                sx1, sy1, sx2, sy2 = symbol_bbox
                
                # Calculate text box size based on text length
                text_width = len(text) * (self.font_size - 1) * 0.6
                text_height = self.font_size + 4
                
                # Position text box near symbol (slightly offset)
                text_x1 = sx1
                text_y1 = sy2 + 5  # Below the symbol
                text_x2 = text_x1 + text_width
                text_y2 = text_y1 + text_height
                
                # Ensure text box is within page bounds
                page_rect = original_page.rect
                if text_x2 > page_rect.width:
                    text_x1 = page_rect.width - text_width - 10
                    text_x2 = text_x1 + text_width
                
                if text_y2 > page_rect.height:
                    text_y1 = sy1 - text_height - 5  # Above the symbol instead
                    text_y2 = text_y1 + text_height
                
                # Draw text box
                text_rect = fitz.Rect(text_x1, text_y1, text_x2, text_y2)
                page.draw_rect(text_rect, color=self.colors["text_box"], 
                             width=1.0, fill=None)
                
                # Insert text
                text_point = fitz.Point(text_x1 + 2, text_y1 + self.font_size)
                page.insert_text(text_point, text, fontsize=self.font_size - 1, 
                                color=self.colors["text_color"])
                
                # Draw connection line
                symbol_center = fitz.Point((sx1 + sx2) / 2, (sy1 + sy2) / 2)
                text_center = fitz.Point((text_x1 + text_x2) / 2, (text_y1 + text_y2) / 2)
                page.draw_line(symbol_center, text_center, 
                              color=self.colors["connection"], width=0.5)
            else:
                # Fallback: try to scale the coordinates
                # Assume text coordinates are in a different scale
                # This is a rough estimation - may need adjustment
                scale_factor = 3.0  # Rough estimate based on coordinate differences
                
                scaled_x1 = x1 * scale_factor
                scaled_y1 = y1 * scale_factor
                scaled_x2 = x2 * scale_factor
                scaled_y2 = y2 * scale_factor
                
                # Ensure within page bounds
                if (scaled_x2 <= page_rect.width and scaled_y2 <= page_rect.height and
                    scaled_x1 >= 0 and scaled_y1 >= 0):
                    
                    text_rect = fitz.Rect(scaled_x1, scaled_y1, scaled_x2, scaled_y2)
                    page.draw_rect(text_rect, color=self.colors["text_box"], 
                                 width=1.0, fill=None)
                    
                    # Insert text
                    text_point = fitz.Point(scaled_x1 + 2, scaled_y1 + self.font_size)
                    page.insert_text(text_point, text, fontsize=self.font_size - 1, 
                                    color=self.colors["text_color"])
    
    def _add_detection_legend(self, page: fitz.Page, detection_data: Dict, target_page: int):
        """Add legend for detection-only page"""
        legend_x = 10
        legend_y = 50
        
        # Detection boxes legend
        page.draw_rect(fitz.Rect(legend_x, legend_y, legend_x + 20, legend_y + 20), 
                      color=self.colors["detection_box"], fill=self.colors["detection_box"])
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 15), 
                        f"YOLO Detections (≥{self.detection_confidence_threshold:.0%})", 
                        fontsize=self.font_size, color=self.colors["text_color"])
        
        # Statistics
        page_detections = []
        for page_data in detection_data.get("pages", []):
            if page_data.get("page", page_data.get("page_num", 1)) == target_page:
                page_detections = page_data.get("detections", [])
                break
        
        total_detections = len(page_detections)
        high_conf_detections = sum(1 for det in page_detections 
                                 if det.get("confidence", 0) >= self.detection_confidence_threshold)
        
        stats_text = f"Detections: {high_conf_detections}/{total_detections} shown"
        page.insert_text(fitz.Point(legend_x, legend_y + 40), 
                        stats_text, fontsize=self.font_size, color=self.colors["text_color"])
    
    def _add_text_legend(self, page: fitz.Page, text_data: Dict, target_page: int):
        """Add legend for text-only page"""
        legend_x = 10
        legend_y = 50
        
        # Text boxes legend
        page.draw_rect(fitz.Rect(legend_x, legend_y, legend_x + 20, legend_y + 20), 
                      color=self.colors["text_box"], fill=self.colors["text_box"])
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 15), 
                        f"OCR Text Regions (≥{self.text_confidence_threshold:.0%})", 
                        fontsize=self.font_size, color=self.colors["text_color"])
        
        # Connection lines legend
        page.draw_line(fitz.Point(legend_x, legend_y + 35), fitz.Point(legend_x + 20, legend_y + 35), 
                      color=self.colors["connection"], width=1)
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 40), 
                        "Text-Symbol Connections", fontsize=self.font_size, color=self.colors["text_color"])
        
        # Statistics
        page_text_regions = [tr for tr in text_data["text_regions"] if tr["page"] == target_page]
        total_texts = len(page_text_regions)
        high_conf_texts = sum(1 for tr in page_text_regions 
                            if tr.get("confidence", 0) >= self.text_confidence_threshold)
        
        stats_text = f"Text Regions: {high_conf_texts}/{total_texts} shown"
        page.insert_text(fitz.Point(legend_x, legend_y + 60), 
                        stats_text, fontsize=self.font_size, color=self.colors["text_color"])
    
    def _add_combined_legend(self, page: fitz.Page, detection_data: Dict, 
                           text_data: Dict, target_page: int):
        """Add legend for combined page"""
        legend_x = 10
        legend_y = 50
        
        # Detection boxes legend
        page.draw_rect(fitz.Rect(legend_x, legend_y, legend_x + 20, legend_y + 20), 
                      color=self.colors["detection_box"], fill=self.colors["detection_box"])
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 15), 
                        f"YOLO Detections (≥{self.detection_confidence_threshold:.0%})", 
                        fontsize=self.font_size, color=self.colors["text_color"])
        
        # Text boxes legend
        page.draw_rect(fitz.Rect(legend_x, legend_y + 30, legend_x + 20, legend_y + 50), 
                      color=self.colors["text_box"], fill=self.colors["text_box"])
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 45), 
                        f"OCR Text Regions (≥{self.text_confidence_threshold:.0%})", 
                        fontsize=self.font_size, color=self.colors["text_color"])
        
        # Connection lines legend
        page.draw_line(fitz.Point(legend_x, legend_y + 65), fitz.Point(legend_x + 20, legend_y + 65), 
                      color=self.colors["connection"], width=1)
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 70), 
                        "Text-Symbol Connections", fontsize=self.font_size, color=self.colors["text_color"])
        
        # Statistics
        page_detections = []
        for page_data in detection_data.get("pages", []):
            if page_data.get("page", page_data.get("page_num", 1)) == target_page:
                page_detections = page_data.get("detections", [])
                break
        
        page_text_regions = [tr for tr in text_data["text_regions"] if tr["page"] == target_page]
        
        total_detections = len(page_detections)
        high_conf_detections = sum(1 for det in page_detections 
                                 if det.get("confidence", 0) >= self.detection_confidence_threshold)
        
        total_texts = len(page_text_regions)
        high_conf_texts = sum(1 for tr in page_text_regions 
                            if tr.get("confidence", 0) >= self.text_confidence_threshold)
        
        stats_text = f"Detections: {high_conf_detections}/{total_detections} | Text: {high_conf_texts}/{total_texts}"
        page.insert_text(fitz.Point(legend_x, legend_y + 90), 
                        stats_text, fontsize=self.font_size, color=self.colors["text_color"])

def main():
    """CLI interface for Enhanced PDF creation"""
    parser = argparse.ArgumentParser(description='Create Enhanced PDFs with proper coordinate mapping')
    parser.add_argument('--detection-file', '-d', type=str, required=True,
                       help='Detection JSON file')
    parser.add_argument('--text-file', '-t', type=str, required=True,
                       help='Text extraction JSON file')
    parser.add_argument('--pdf-file', '-p', type=str, required=True,
                       help='Original PDF file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output PDF file')
    parser.add_argument('--version', '-v', type=str, choices=['long', 'short'], 
                       default='short', help='PDF version: long (4 pages) or short (1 page)')
    parser.add_argument('--detection-threshold', type=float, default=0.8,
                       help='Detection confidence threshold (default: 0.8)')
    parser.add_argument('--text-threshold', type=float, default=0.5,
                       help='Text confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    try:
        creator = EnhancedPDFCreator(
            detection_confidence_threshold=args.detection_threshold,
            text_confidence_threshold=args.text_threshold
        )
        
        enhanced_pdf = creator.create_enhanced_pdf(
            Path(args.detection_file),
            Path(args.text_file),
            Path(args.pdf_file),
            Path(args.output),
            args.version
        )
        
        print(f"\nEnhanced PDF created successfully!")
        print(f"Version: {args.version}")
        print(f"Output: {enhanced_pdf}")
        return 0
        
    except Exception as e:
        print(f"Enhanced PDF creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
