"""
PDF Annotator
Creates enhanced PDFs using native PDF annotations instead of image overlays
Much simpler and more professional approach
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import argparse

class PDFAnnotator:
    """
    Creates enhanced PDFs using native PDF annotations
    Adds YOLO detection boxes as colored rectangles and OCR text as annotations
    """
    
    def __init__(self, detection_confidence_threshold: float = 0.8,
                 text_confidence_threshold: float = 0.5):
        """
        Initialize PDF Annotator
        
        Args:
            detection_confidence_threshold: Minimum confidence for showing detection boxes
            text_confidence_threshold: Minimum confidence for showing text annotations
        """
        self.detection_confidence_threshold = detection_confidence_threshold
        self.text_confidence_threshold = text_confidence_threshold
        
        # Color definitions for different detection classes
        self.detection_colors = {
            "Tag-ID": (0, 0, 1),      # Blue
            "C0082": (0, 0.8, 0),     # Green  
            "X8117": (1, 0, 0),       # Red
            "X8017": (1, 0.5, 0),     # Orange
            "xxxx": (0.5, 0, 0.5),    # Purple
            "default": (0, 0, 1)      # Default blue
        }
        
        # Text annotation color
        self.text_color = (0, 0.6, 0)  # Dark green
    
    def create_annotated_pdf(self,
                           detection_file: Path,
                           text_extraction_file: Path,
                           pdf_file: Path,
                           output_file: Path) -> Path:
        """
        Create annotated PDF with detection boxes and text annotations
        
        Args:
            detection_file: Path to detection JSON file
            text_extraction_file: Path to text extraction JSON file
            pdf_file: Path to original PDF file
            output_file: Output PDF file path
            
        Returns:
            Path to annotated PDF file
        """
        print(f"Creating annotated PDF: {pdf_file.name}")
        
        # Load data
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        with open(text_extraction_file, 'r') as f:
            text_data = json.load(f)
        
        # Open original PDF
        doc = fitz.open(str(pdf_file))
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            print(f"Processing page {page_num + 1}")
            print(f"  Page dimensions: {page.rect.width} x {page.rect.height}")
            print(f"  Page rotation: {page.rotation}°")
            
            # Add detection annotations
            self._add_detection_annotations(page, detection_data, page_num + 1)
            
            # Add text annotations
            self._add_text_annotations(page, text_data, page_num + 1)
        
        # Save annotated PDF
        doc.save(str(output_file))
        doc.close()
        
        print(f"Annotated PDF saved to: {output_file}")
        return output_file
    
    def _add_detection_annotations(self, page: fitz.Page, detection_data: Dict, target_page: int):
        """Add YOLO detection boxes as PDF annotations"""
        
        detection_count = 0
        
        for page_data in detection_data.get("pages", []):
            page_num = page_data.get("page", page_data.get("page_num", 1))
            if page_num != target_page:
                continue
            
            for detection in page_data.get("detections", []):
                confidence = detection.get("confidence", 0.0)
                
                # Only add if confidence meets threshold
                if confidence < self.detection_confidence_threshold:
                    continue
                
                # Get detection information
                class_name = detection.get("class_name", "unknown")
                snippet_position = detection.get("snippet_position", {})
                snippet_bbox = detection.get("bbox_snippet", {})
                
                if not snippet_position or not snippet_bbox:
                    continue
                
                # Transform coordinates to PDF space
                pdf_coords = self._transform_detection_to_pdf(
                    snippet_bbox, snippet_position, page
                )
                
                if pdf_coords:
                    x1, y1, x2, y2 = pdf_coords
                    
                    # Create rectangle annotation
                    rect = fitz.Rect(x1, y1, x2, y2)
                    
                    # Get color for this detection class
                    color = self.detection_colors.get(class_name, self.detection_colors["default"])
                    
                    # Enlarge the rectangle slightly for better visibility
                    enlarged_rect = fitz.Rect(x1-5, y1-5, x2+5, y2+5)
                    # Ensure it stays within page bounds
                    enlarged_rect = enlarged_rect & page.rect
                    
                    # Add rectangle annotation
                    annot = page.add_rect_annot(enlarged_rect)
                    annot.set_colors(stroke=color, fill=color)
                    annot.set_border(width=3)
                    annot.set_opacity(0.3)  # Semi-transparent fill
                    
                    # Add content (tooltip text)
                    content = f"{class_name}\nConfidence: {confidence:.1%}\nPosition: r{snippet_position.get('row', 0)}c{snippet_position.get('col', 0)}"
                    annot.set_info(content=content, title="YOLO Detection")
                    
                    # Update annotation
                    annot.update()
                    
                    detection_count += 1
            
            break
        
        print(f"  Added {detection_count} detection annotations")
    
    def _add_text_annotations(self, page: fitz.Page, text_data: Dict, target_page: int):
        """Add OCR text as PDF annotations"""
        
        text_count = 0
        page_text_regions = [tr for tr in text_data.get("text_regions", []) if tr.get("page") == target_page]
        
        for text_region in page_text_regions:
            confidence = text_region.get("confidence", 0.0)
            
            # Only add if confidence meets threshold
            if confidence < self.text_confidence_threshold:
                continue
            
            text_content = text_region.get("text", "").strip()
            if not text_content:
                continue
            
            # Try to get coordinates from associated symbol
            associated_symbol = text_region.get("associated_symbol")
            if associated_symbol and "bbox_global" in associated_symbol:
                symbol_bbox = associated_symbol["bbox_global"]
                
                # Transform symbol coordinates to PDF space
                pdf_coords = self._transform_symbol_to_pdf(symbol_bbox, page)
                
                if pdf_coords:
                    x1, y1, x2, y2 = pdf_coords
                    
                    # Create a small point for the text annotation
                    # Position it near the symbol
                    point = fitz.Point(x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
                    
                    # Add text annotation
                    annot = page.add_text_annot(point, text_content)
                    annot.set_info(title="OCR Text", content=f"Extracted text: {text_content}\nConfidence: {confidence:.1%}")
                    annot.set_colors(stroke=self.text_color)
                    
                    # Update annotation
                    annot.update()
                    
                    text_count += 1
        
        print(f"  Added {text_count} text annotations")
    
    def _transform_detection_to_pdf(self, snippet_bbox: Dict, snippet_position: Dict, page: fitz.Page) -> Optional[Tuple[float, float, float, float]]:
        """Transform detection coordinates from snippet space to PDF space using proper coordinate system conversion"""
        
        # Original image dimensions (from YOLO training)
        original_width = 9362
        original_height = 6623
        
        # Grid configuration: 6 rows x 4 columns (rows 0-5, cols 2-5)
        grid_rows = 6
        grid_cols = 4
        
        # Get snippet coordinates (ML detection format: top-left origin)
        sx1 = snippet_bbox.get("x1", 0)
        sy1 = snippet_bbox.get("y1", 0)
        sx2 = snippet_bbox.get("x2", 0)
        sy2 = snippet_bbox.get("y2", 0)
        
        # Get grid position
        row = snippet_position.get("row", 0)
        col = snippet_position.get("col", 0)
        
        # Validate grid position
        grid_col = col - 2  # Map columns 2-5 to 0-3
        if grid_col < 0 or grid_col >= grid_cols or row < 0 or row >= grid_rows:
            print(f"    Warning: Invalid grid position r{row}c{col}")
            return None
        
        # Calculate cell dimensions in original image space
        cell_width = original_width / grid_cols
        cell_height = original_height / grid_rows
        
        # Calculate cell base position
        cell_base_x = grid_col * cell_width
        cell_base_y = row * cell_height
        
        # Convert to original image coordinates (still top-left origin)
        orig_x1 = cell_base_x + sx1
        orig_y1 = cell_base_y + sy1
        orig_x2 = cell_base_x + sx2
        orig_y2 = cell_base_y + sy2
        
        # Get PDF dimensions
        pdf_width = page.rect.width
        pdf_height = page.rect.height
        
        # Simplified approach: Let PyMuPDF handle the rotation, we just need to scale properly
        # The key insight: PyMuPDF coordinate system matches ML detection coordinates (top-left origin)
        
        if page.rotation == 90:
            # For 90° rotated PDFs, we need to map coordinates correctly
            # Original image is landscape (9362×6623), PDF is portrait (3370×2384) with 90° rotation
            
            # Direct scaling without manual rotation - let PyMuPDF handle the rotation
            # Scale based on the actual image-to-PDF mapping
            scale_x = pdf_width / original_width   # 3370 / 9362
            scale_y = pdf_height / original_height # 2384 / 6623
            
            # Apply scaling directly
            pdf_x1 = orig_x1 * scale_x
            pdf_y1 = orig_y1 * scale_y
            pdf_x2 = orig_x2 * scale_x
            pdf_y2 = orig_y2 * scale_y
            
            print(f"    Detection r{row}c{col}: ({sx1:.0f},{sy1:.0f})-({sx2:.0f},{sy2:.0f}) -> Global: ({orig_x1:.0f},{orig_y1:.0f})-({orig_x2:.0f},{orig_y2:.0f}) -> PDF: ({pdf_x1:.0f},{pdf_y1:.0f})-({pdf_x2:.0f},{pdf_y2:.0f})")
        else:
            # For non-rotated PDFs: Direct scaling
            scale_x = pdf_width / original_width
            scale_y = pdf_height / original_height
            
            pdf_x1 = orig_x1 * scale_x
            pdf_y1 = orig_y1 * scale_y
            pdf_x2 = orig_x2 * scale_x
            pdf_y2 = orig_y2 * scale_y
            
            print(f"    Detection r{row}c{col}: ({sx1:.0f},{sy1:.0f})-({sx2:.0f},{sy2:.0f}) -> PDF: ({pdf_x1:.0f},{pdf_y1:.0f})-({pdf_x2:.0f},{pdf_y2:.0f})")
        
        # Validate coordinates are within PDF bounds
        if (pdf_x1 < 0 or pdf_y1 < 0 or pdf_x2 > pdf_width or pdf_y2 > pdf_height or
            pdf_x1 >= pdf_x2 or pdf_y1 >= pdf_y2):
            print(f"    Warning: Invalid PDF coordinates ({pdf_x1:.0f},{pdf_y1:.0f})-({pdf_x2:.0f},{pdf_y2:.0f}) for page {pdf_width}x{pdf_height}")
            return None
        
        return (pdf_x1, pdf_y1, pdf_x2, pdf_y2)
    
    def _transform_symbol_to_pdf(self, symbol_bbox: Dict, page: fitz.Page) -> Optional[Tuple[float, float, float, float]]:
        """Transform symbol coordinates to PDF space"""
        
        # Original image dimensions
        original_width = 9362
        original_height = 6623
        
        # Get symbol coordinates
        if isinstance(symbol_bbox, dict):
            sx1 = symbol_bbox.get("x1", 0)
            sy1 = symbol_bbox.get("y1", 0)
            sx2 = symbol_bbox.get("x2", 0)
            sy2 = symbol_bbox.get("y2", 0)
        else:
            # Handle list format
            sx1, sy1, sx2, sy2 = symbol_bbox[:4]
        
        # Transform to PDF coordinates
        pdf_width = page.rect.width
        pdf_height = page.rect.height
        
        # Check if PDF is rotated (same logic as detection transformation)
        if page.rotation == 90:
            # Apply 90° rotation transformation: (x, y) -> (y, original_width - x)
            rot_x1 = sy1
            rot_y1 = original_width - sx2
            rot_x2 = sy2
            rot_y2 = original_width - sx1
            
            # Ensure coordinates are in correct order
            if rot_x1 > rot_x2:
                rot_x1, rot_x2 = rot_x2, rot_x1
            if rot_y1 > rot_y2:
                rot_y1, rot_y2 = rot_y2, rot_y1
            
            # Scale to PDF dimensions
            scale_x = pdf_width / original_height
            scale_y = pdf_height / original_width
            
            pdf_x1 = rot_x1 * scale_x
            pdf_y1 = rot_y1 * scale_y
            pdf_x2 = rot_x2 * scale_x
            pdf_y2 = rot_y2 * scale_y
        else:
            # For non-rotated PDFs, simple scaling
            scale_x = pdf_width / original_width
            scale_y = pdf_height / original_height
            
            pdf_x1 = sx1 * scale_x
            pdf_y1 = sy1 * scale_y
            pdf_x2 = sx2 * scale_x
            pdf_y2 = sy2 * scale_y
        
        return (pdf_x1, pdf_y1, pdf_x2, pdf_y2)

def main():
    """CLI interface for PDF annotation"""
    parser = argparse.ArgumentParser(description='Create annotated PDFs using native PDF annotations')
    parser.add_argument('--detection-file', '-d', type=str, required=True,
                       help='Detection JSON file')
    parser.add_argument('--text-file', '-t', type=str, required=True,
                       help='Text extraction JSON file')
    parser.add_argument('--pdf-file', '-p', type=str, required=True,
                       help='Original PDF file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output PDF file')
    parser.add_argument('--detection-threshold', type=float, default=0.8,
                       help='Detection confidence threshold (default: 0.8)')
    parser.add_argument('--text-threshold', type=float, default=0.5,
                       help='Text confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    try:
        annotator = PDFAnnotator(
            detection_confidence_threshold=args.detection_threshold,
            text_confidence_threshold=args.text_threshold
        )
        
        annotated_pdf = annotator.create_annotated_pdf(
            Path(args.detection_file),
            Path(args.text_file),
            Path(args.pdf_file),
            Path(args.output)
        )
        
        print(f"\nAnnotated PDF created successfully!")
        print(f"Output: {annotated_pdf}")
        return 0
        
    except Exception as e:
        print(f"PDF annotation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
