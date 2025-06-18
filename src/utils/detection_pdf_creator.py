"""
Detection PDF Creator - Simplified Version
Creates enhanced PDFs following the working pdf_enhancer.py pattern
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import argparse

class DetectionPDFCreator:
    """
    Creates enhanced PDFs with detection and text overlays
    Follows the working pdf_enhancer.py pattern for reliability
    """
    
    def __init__(self, font_size: float = 8, line_width: float = 1.0, 
                 detection_confidence_threshold: float = 0.8):
        """
        Initialize Detection PDF Creator
        
        Args:
            font_size: Font size for text annotations
            line_width: Line width for boxes and lines
            detection_confidence_threshold: Minimum confidence for showing detection boxes
        """
        self.font_size = font_size
        self.line_width = line_width
        self.detection_confidence_threshold = detection_confidence_threshold
        
        # Color definitions (RGB format for PyMuPDF)
        self.colors = {
            "yolo_detection": (1, 0, 1),        # Magenta
            "text_box": (0, 1, 0),              # Green for text boxes
            "connection": (1, 1, 0),            # Yellow for text-symbol connections
            "background": (1, 1, 1),            # White
            "text_color": (0, 0, 0),            # Black
            "confidence_high": (0, 1, 0),       # Green for high confidence
            "confidence_medium": (1, 0.6, 0),   # Orange for medium confidence
            "confidence_low": (1, 0, 0),        # Red for low confidence
            "page_title": (0.2, 0.2, 0.2)      # Dark gray for page titles
        }
    
    def create_enhanced_pdf(self,
                           detection_file: Path,
                           pdf_file: Path,
                           output_file: Path,
                           version: str = 'short',
                           text_extraction_file: Optional[Path] = None) -> Path:
        """
        Create enhanced PDF following pdf_enhancer.py pattern
        
        Args:
            detection_file: Path to detection JSON file
            pdf_file: Path to original PDF file
            output_file: Output PDF file path
            version: 'long' (4 pages) or 'short' (1 page)
            text_extraction_file: Optional path to text extraction JSON file
            
        Returns:
            Path to enhanced PDF file
        """
        print(f"Creating {version} version detection PDF: {pdf_file.name}")
        
        # Load detection data
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        # Load text extraction data if provided
        text_data = None
        if text_extraction_file and text_extraction_file.exists():
            with open(text_extraction_file, 'r') as f:
                text_data = json.load(f)
        
        if version == 'long':
            return self._create_long_version(detection_data, pdf_file, output_file, text_data)
        else:
            return self._create_short_version(detection_data, pdf_file, output_file, text_data)
    
    def _create_short_version(self, detection_data: Dict, pdf_file: Path, output_file: Path, text_data: Optional[Dict] = None) -> Path:
        """Create 1-page short version - direct modification like pdf_enhancer.py"""
        
        # Open original PDF (like pdf_enhancer.py)
        doc = fitz.open(str(pdf_file))
        
        # Process each page in the original PDF
        for page_data in detection_data.get("pages", []):
            page_num = page_data.get("page", page_data.get("page_num", 1)) - 1
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            
            print(f"Processing page {page_num + 1}")
            print(f"Page rect: {page.rect}")
            print(f"Page rotation: {page.rotation}")
            
            # Add title
            self._add_page_title(page, "Enhanced PLC Diagram", 1)
            
            # Draw detection boxes (using scaled coordinates)
            self._draw_detections_scaled(page, page_data)
            
            # Draw text regions if available
            if text_data:
                self._draw_text_regions_scaled(page, text_data, page_num + 1)
                self._draw_text_symbol_connections_scaled(page, text_data, page_num + 1)
            
            # Add legend
            self._add_legend(page, page_data, text_data)
        
        # Save enhanced PDF
        doc.save(str(output_file))
        doc.close()
        
        print(f"Detection PDF saved to: {output_file}")
        return output_file
    
    def _create_long_version(self, detection_data: Dict, pdf_file: Path, output_file: Path, text_data: Optional[Dict] = None) -> Path:
        """Create 4-page long version by copying and modifying pages"""
        
        # Open original PDF
        original_doc = fitz.open(str(pdf_file))
        
        # Create new document for enhanced version
        enhanced_doc = fitz.open()
        
        # Process each page in the original PDF
        for page_data in detection_data.get("pages", []):
            page_num = page_data.get("page", page_data.get("page_num", 1)) - 1
            if page_num >= len(original_doc):
                continue
                
            original_page = original_doc[page_num]
            
            print(f"Creating 4-page version for PDF page {page_num + 1}")
            
            # Create 4 pages for this PDF page
            for page_idx in range(4):
                print(f"  Creating enhanced page {page_idx + 1}")
                
                # Copy original page to new document
                enhanced_doc.insert_pdf(original_doc, from_page=page_num, to_page=page_num)
                new_page = enhanced_doc[-1]  # Get the last added page
                
                # Add content based on page type
                if page_idx == 0:
                    # Page 1: Original PDF (clean)
                    self._add_page_title(new_page, "Page 1: Original PDF", 1)
                    
                elif page_idx == 1:
                    # Page 2: PDF + YOLO Detections
                    self._add_page_title(new_page, "Page 2: YOLO Detections", 2)
                    self._draw_detections_scaled(new_page, page_data)
                    self._add_detection_legend(new_page, page_data)
                    
                elif page_idx == 2:
                    # Page 3: OCR Text (with fallback to detections if no text data)
                    if text_data and text_data.get("total_text_regions", 0) > 0:
                        self._add_page_title(new_page, "Page 3: OCR Text", 3)
                        self._draw_text_regions_scaled(new_page, text_data, page_num + 1)
                        self._add_text_legend(new_page, text_data)
                    else:
                        # Fallback: Show detections when no text data available
                        self._add_page_title(new_page, "Page 3: YOLO Detections (No Text Data)", 3)
                        self._draw_detections_scaled(new_page, page_data)
                        self._add_detection_legend(new_page, page_data)
                    
                elif page_idx == 3:
                    # Page 4: Combined (detections + text)
                    self._add_page_title(new_page, "Page 4: Combined", 4)
                    self._draw_detections_scaled(new_page, page_data)
                    if text_data:
                        self._draw_text_regions_scaled(new_page, text_data, page_num + 1)
                        self._draw_text_symbol_connections_scaled(new_page, text_data, page_num + 1)
                        self._add_combined_legend(new_page, page_data, text_data)
                    else:
                        self._add_detection_legend(new_page, page_data)
        
        # Close original document
        original_doc.close()
        
        # Save enhanced PDF
        enhanced_doc.save(str(output_file))
        enhanced_doc.close()
        
        print(f"Detection PDF saved to: {output_file}")
        return output_file
    
    def _scale_coordinates_detection(self, x: float, y: float, detection_width: float, detection_height: float, pdf_width: float, pdf_height: float) -> Tuple[float, float]:
        """Scale coordinates for detection boxes with 90-degree rotation"""
        # Step 1: Scale coordinates
        scale_x = pdf_width / detection_width
        scale_y = pdf_height / detection_height
        
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        
        # Step 2: Apply 90-degree counterclockwise rotation to align with rotated PDF
        rotated_x = scaled_y
        rotated_y = pdf_width - scaled_x
        
        return rotated_x, rotated_y
    
    def _scale_coordinates_text(self, x: float, y: float, detection_width: float, detection_height: float, pdf_width: float, pdf_height: float) -> Tuple[float, float]:
        """Use coordinates directly like the working pdf_enhancer.py - NO transformation needed"""
        # The working pdf_enhancer.py uses coordinates directly without any transformation!
        # Text coordinates are already in the correct PDF coordinate system
        return x, y
    
    def _draw_detections_scaled(self, page: fitz.Page, page_data: Dict):
        """Draw YOLO detection boxes with coordinate scaling (like pdf_enhancer.py)"""
        
        detections = page_data.get("detections", [])
        print(f"Found {len(detections)} detections")
        
        # Get detection coordinate system dimensions
        detection_width = page_data.get("original_width", 9362)
        detection_height = page_data.get("original_height", 6623)
        
        # Get PDF dimensions
        pdf_width = page.rect.width
        pdf_height = page.rect.height
        
        print(f"Detection dimensions: {detection_width} x {detection_height}")
        print(f"PDF dimensions: {pdf_width} x {pdf_height}")
        
        drawn_count = 0
        for detection in detections:
            confidence = detection.get("confidence", 0.0)
            
            # Only draw if confidence meets threshold
            if confidence < self.detection_confidence_threshold:
                continue
            
            # Get global bbox
            bbox = detection.get("global_bbox", detection.get("bbox_global", []))
            if not bbox:
                continue
            
            # Handle both dict and list formats
            if isinstance(bbox, dict):
                x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
            elif isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            else:
                continue
            
            # Scale coordinates to PDF dimensions (with rotation for detection boxes)
            new_x1, new_y1 = self._scale_coordinates_detection(x1, y1, detection_width, detection_height, pdf_width, pdf_height)
            new_x2, new_y2 = self._scale_coordinates_detection(x2, y2, detection_width, detection_height, pdf_width, pdf_height)
            
            # Create rectangle
            rect = fitz.Rect(min(new_x1, new_x2), min(new_y1, new_y2), 
                           max(new_x1, new_x2), max(new_y1, new_y2))
            
            # Choose color based on confidence level
            if confidence >= 0.95:
                box_color = self.colors["confidence_high"]
            elif confidence >= 0.85:
                box_color = self.colors["confidence_medium"]
            else:
                box_color = self.colors["yolo_detection"]
            
            # Draw box
            page.draw_rect(rect, color=box_color, width=self.line_width, fill=None)
            
            # Add label
            class_name = detection.get("class_name", "unknown")
            label = f"{class_name} ({confidence:.2f})"
            
            # Position label above box
            text_point = fitz.Point(rect.x0, rect.y0 - 5)
            page.insert_text(text_point, label, fontsize=self.font_size, 
                            color=self.colors["text_color"])
            
            drawn_count += 1
        
        print(f"Drew {drawn_count} detection boxes")
    
    def _draw_text_regions_scaled(self, page: fitz.Page, text_data: Dict, target_page: int):
        """Draw text extraction regions with coordinate scaling"""
        
        # Get text regions for this page
        page_text_regions = [tr for tr in text_data.get("text_regions", []) 
                           if tr.get("page") == target_page]
        
        print(f"Found {len(page_text_regions)} text regions for page {target_page}")
        
        # Get PDF dimensions
        pdf_width = page.rect.width
        pdf_height = page.rect.height
        
        # Use same detection dimensions for text scaling
        detection_width = 9362  # Default from detection data
        detection_height = 6623  # Default from detection data
        
        drawn_count = 0
        for text_region in page_text_regions:
            bbox = text_region.get("bbox", [])
            if not bbox or len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            text = text_region.get("text", "")
            
            # Scale coordinates (without rotation for text regions)
            new_x1, new_y1 = self._scale_coordinates_text(x1, y1, detection_width, detection_height, pdf_width, pdf_height)
            new_x2, new_y2 = self._scale_coordinates_text(x2, y2, detection_width, detection_height, pdf_width, pdf_height)
            
            # Create rectangle
            rect = fitz.Rect(min(new_x1, new_x2), min(new_y1, new_y2), 
                           max(new_x1, new_x2), max(new_y1, new_y2))
            
            # Draw rectangle (green)
            page.draw_rect(rect, color=self.colors["text_box"], width=self.line_width, fill=None)
            
            # Add text inside the box (simplified)
            text_point = fitz.Point(rect.x0 + 2, rect.y0 + (rect.height) / 2 + self.font_size / 3)
            display_text = text.strip()[:15]  # Limit text length
            page.insert_text(text_point, display_text, fontsize=self.font_size - 1, 
                            color=self.colors["text_color"])
            
            drawn_count += 1
        
        print(f"Drew {drawn_count} text regions")
    
    def _draw_text_symbol_connections_scaled(self, page: fitz.Page, text_data: Dict, target_page: int):
        """Draw connections between text regions and associated symbols with coordinate scaling"""
        
        page_text_regions = [tr for tr in text_data.get("text_regions", []) 
                           if tr.get("page") == target_page]
        
        # Get PDF dimensions
        pdf_width = page.rect.width
        pdf_height = page.rect.height
        
        # Use same detection dimensions for scaling
        detection_width = 9362  # Default from detection data
        detection_height = 6623  # Default from detection data
        
        connection_count = 0
        for text_region in page_text_regions:
            associated_symbol = text_region.get("associated_symbol")
            if not associated_symbol:
                continue
            
            text_bbox = text_region.get("bbox", [])
            symbol_bbox = associated_symbol.get("bbox_global", {})
            
            if not text_bbox or not symbol_bbox:
                continue
            
            # Scale text center (without rotation for text coordinates)
            text_center_x = (text_bbox[0] + text_bbox[2]) / 2
            text_center_y = (text_bbox[1] + text_bbox[3]) / 2
            new_text_x, new_text_y = self._scale_coordinates_text(text_center_x, text_center_y, detection_width, detection_height, pdf_width, pdf_height)
            text_center = fitz.Point(new_text_x, new_text_y)
            
            # Scale symbol center (with rotation for detection coordinates)
            if isinstance(symbol_bbox, dict):
                symbol_center_x = (symbol_bbox.get("x1", 0) + symbol_bbox.get("x2", 0)) / 2
                symbol_center_y = (symbol_bbox.get("y1", 0) + symbol_bbox.get("y2", 0)) / 2
                new_symbol_x, new_symbol_y = self._scale_coordinates_detection(symbol_center_x, symbol_center_y, detection_width, detection_height, pdf_width, pdf_height)
                symbol_center = fitz.Point(new_symbol_x, new_symbol_y)
            else:
                continue
            
            # Draw connection line (yellow)
            page.draw_line(text_center, symbol_center, 
                          color=self.colors["connection"], width=0.5)
            
            connection_count += 1
        
        print(f"Drew {connection_count} text-symbol connections")
    
    def _add_page_title(self, page: fitz.Page, title: str, page_num: int):
        """Add title to page"""
        # Add title with white background
        title_rect = fitz.Rect(10, 10, min(600, page.rect.width - 10), 35)
        page.draw_rect(title_rect, color=self.colors["background"], 
                      fill=self.colors["background"], width=1)
        page.insert_text(fitz.Point(15, 28), title, fontsize=self.font_size + 2, 
                        color=self.colors["page_title"])
        
        # Add page info
        page_info = f"Page {page_num} | Size: {page.rect.width:.0f}x{page.rect.height:.0f}"
        page.insert_text(fitz.Point(15, 45), page_info, fontsize=self.font_size - 2, 
                        color=self.colors["page_title"])
    
    def _add_legend(self, page: fitz.Page, page_data: Dict, text_data: Optional[Dict]):
        """Add comprehensive legend"""
        legend_x = 10
        legend_y = 60
        
        # YOLO detections (magenta)
        page.draw_rect(fitz.Rect(legend_x, legend_y, legend_x + 20, legend_y + 20), 
                      color=self.colors["yolo_detection"], fill=self.colors["yolo_detection"])
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 15), 
                        "YOLO Detections", fontsize=self.font_size, color=self.colors["text_color"])
        
        if text_data:
            # Text regions (green)
            page.draw_rect(fitz.Rect(legend_x, legend_y + 25, legend_x + 20, legend_y + 45), 
                          color=self.colors["text_box"], fill=self.colors["text_box"])
            page.insert_text(fitz.Point(legend_x + 25, legend_y + 40), 
                            "Text Regions", fontsize=self.font_size, color=self.colors["text_color"])
            
            # Connections (yellow)
            page.draw_line(fitz.Point(legend_x, legend_y + 60), fitz.Point(legend_x + 20, legend_y + 60), 
                          color=self.colors["connection"], width=2)
            page.insert_text(fitz.Point(legend_x + 25, legend_y + 65), 
                            "Text-Symbol Connections", fontsize=self.font_size, color=self.colors["text_color"])
    
    def _add_detection_legend(self, page: fitz.Page, page_data: Dict):
        """Add legend for detection boxes only"""
        self._add_legend(page, page_data, None)
    
    def _add_text_legend(self, page: fitz.Page, text_data: Dict):
        """Add legend for text regions only"""
        self._add_legend(page, {}, text_data)
    
    def _add_combined_legend(self, page: fitz.Page, page_data: Dict, text_data: Dict):
        """Add comprehensive legend for combined view"""
        self._add_legend(page, page_data, text_data)

def main():
    """CLI interface for Detection PDF creation"""
    parser = argparse.ArgumentParser(description='Create Enhanced PDFs following pdf_enhancer.py pattern')
    parser.add_argument('--detection-file', '-d', type=str, required=True,
                       help='Detection JSON file')
    parser.add_argument('--pdf-file', '-p', type=str, required=True,
                       help='Original PDF file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output PDF file')
    parser.add_argument('--text-file', '-t', type=str,
                       help='Text extraction JSON file (optional)')
    parser.add_argument('--version', '-v', type=str, choices=['long', 'short'], 
                       default='short', help='PDF version: long (4 pages) or short (1 page)')
    parser.add_argument('--detection-threshold', type=float, default=0.8,
                       help='Detection confidence threshold (default: 0.8)')
    
    args = parser.parse_args()
    
    try:
        creator = DetectionPDFCreator(
            detection_confidence_threshold=args.detection_threshold
        )
        
        text_file = Path(args.text_file) if args.text_file else None
        
        enhanced_pdf = creator.create_enhanced_pdf(
            Path(args.detection_file),
            Path(args.pdf_file),
            Path(args.output),
            args.version,
            text_file
        )
        
        print(f"\nDetection PDF created successfully!")
        print(f"Version: {args.version}")
        print(f"Output: {enhanced_pdf}")
        if text_file:
            print(f"Text extraction: {text_file}")
        return 0
        
    except Exception as e:
        print(f"Detection PDF creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
