"""
Enhanced PDF Creator using Relative Coordinate System
Creates PDFs by positioning text relative to YOLO symbol boxes
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Any, Tuple
from src.ocr.relative_coordinate_system import RelativeCoordinateSystem

class RelativeCoordinatePDFCreator:
    """
    Creates enhanced PDFs using relative coordinate positioning
    """
    
    def __init__(self, font_size: int = 8, min_conf: float = 0.8):
        self.font_size = font_size
        self.min_conf = min_conf
        self.rel_coord_system = RelativeCoordinateSystem()
        
        # Color scheme for different text types
        self.colors = {
            'variable': (0, 0.8, 0),      # Green for variables
            'numeric': (0, 0, 0.8),       # Blue for numbers
            'label': (0.6, 0.3, 0),       # Brown for labels
            'default': (0, 0, 0)          # Black for others
        }
        
        # Symbol box colors
        self.symbol_colors = {
            'high_conf': (0, 0.7, 0),     # Green for high confidence
            'medium_conf': (0.8, 0.5, 0), # Orange for medium confidence
            'low_conf': (0.8, 0, 0)       # Red for low confidence
        }
    
    def create_enhanced_pdf_from_relative_data(self, 
                                             original_pdf: Path,
                                             relative_data: Dict[str, Any],
                                             output_file: Path,
                                             version: str = 'short') -> Path:
        """
        Create enhanced PDF using relative coordinate data
        
        Args:
            original_pdf: Path to original PDF
            relative_data: Text extraction data with relative coordinates
            output_file: Output PDF path
            version: 'short' (1 page) or 'long' (4 pages)
            
        Returns:
            Path to created PDF
        """
        
        # Open original PDF
        doc = fitz.open(str(original_pdf))
        
        if not doc:
            raise ValueError(f"Could not open PDF: {original_pdf}")
        
        # Create new PDF document
        new_doc = fitz.open()
        
        try:
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Create new page with same dimensions
                new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
                
                # Copy original page content
                new_page.show_pdf_page(page.rect, doc, page_num)
                
                # Add enhanced annotations using relative coordinates
                self._add_relative_annotations(new_page, relative_data, page_num + 1)
                
                # For short version, only process first page
                if version == 'short' and page_num == 0:
                    break
            
            # Save the enhanced PDF
            output_file.parent.mkdir(parents=True, exist_ok=True)
            new_doc.save(str(output_file))
            
            return output_file
            
        finally:
            doc.close()
            new_doc.close()
    
    def _add_relative_annotations(self, page: fitz.Page, relative_data: Dict[str, Any], page_num: int):
        """
        Add annotations using relative coordinate positioning
        """
        
        # Group text regions by symbols for efficient processing
        layout_data = self.rel_coord_system.create_pdf_layout_data(relative_data)
        
        # Process each symbol and its associated text
        for symbol_id, symbol_data in layout_data['symbols'].items():
            symbol_info = symbol_data['symbol_info']
            text_regions = symbol_data['text_regions']
            
            # Get symbol bounding box
            symbol_bbox_global = symbol_info.get('bbox_global', {})
            if isinstance(symbol_bbox_global, dict):
                symbol_bbox = [
                    symbol_bbox_global.get('x1', 0),
                    symbol_bbox_global.get('y1', 0),
                    symbol_bbox_global.get('x2', 0),
                    symbol_bbox_global.get('y2', 0)
                ]
            else:
                symbol_bbox = symbol_bbox_global
            
            if len(symbol_bbox) < 4:
                continue
            
            # Draw symbol bounding box
            self._draw_symbol_box(page, symbol_bbox, symbol_info)
            
            # Render text regions relative to symbol
            rendered_texts = self.rel_coord_system.render_symbol_with_text(symbol_bbox, text_regions)
            
            # Add text annotations
            for text_region in rendered_texts:
                self._add_text_annotation(page, text_region)
    
    def _draw_symbol_box(self, page: fitz.Page, symbol_bbox: List[float], symbol_info: Dict[str, Any]):
        """
        Draw symbol bounding box with confidence-based coloring
        """
        confidence = symbol_info.get('confidence', 0)
        
        # Choose color based on confidence
        if confidence >= 0.8:
            color = self.symbol_colors['high_conf']
        elif confidence >= 0.6:
            color = self.symbol_colors['medium_conf']
        else:
            color = self.symbol_colors['low_conf']
        
        # Create rectangle
        rect = fitz.Rect(symbol_bbox[0], symbol_bbox[1], symbol_bbox[2], symbol_bbox[3])
        
        # Draw symbol box outline
        page.draw_rect(rect, color=color, width=2)
        
        # Add symbol class label
        class_name = symbol_info.get('class_name', 'unknown')
        conf_text = f"{class_name} ({confidence:.2f})"
        
        # Position label above the symbol box
        label_point = fitz.Point(symbol_bbox[0], symbol_bbox[1] - 5)
        page.insert_text(label_point, conf_text, fontsize=self.font_size - 1, color=color)
    
    def _add_text_annotation(self, page: fitz.Page, text_region: Dict[str, Any]):
        """
        Add text annotation with pattern-based coloring
        """
        text = text_region.get('text', '')
        confidence = text_region.get('confidence', 0)
        bbox_absolute = text_region.get('bbox_absolute', [])
        patterns = text_region.get('patterns', [])
        
        if not text or len(bbox_absolute) < 4 or confidence < self.min_conf:
            return
        
        # Determine text color based on patterns
        color = self._get_text_color(patterns)
        
        # Create text rectangle
        text_rect = fitz.Rect(bbox_absolute[0], bbox_absolute[1], bbox_absolute[2], bbox_absolute[3])
        
        # Draw text background (semi-transparent)
        page.draw_rect(text_rect, color=(1, 1, 0), fill=(1, 1, 0), width=0.5, fill_opacity=0.3)
        
        # Add text
        text_point = fitz.Point(bbox_absolute[0], bbox_absolute[1] + self.font_size)
        page.insert_text(text_point, text, fontsize=self.font_size, color=color)
        
        # Add confidence indicator
        conf_text = f"({confidence:.2f})"
        conf_point = fitz.Point(bbox_absolute[2] + 2, bbox_absolute[1] + self.font_size)
        page.insert_text(conf_point, conf_text, fontsize=self.font_size - 2, color=(0.5, 0.5, 0.5))
    
    def _get_text_color(self, patterns: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """
        Get text color based on matched patterns
        """
        if not patterns:
            return self.colors['default']
        
        # Use the highest priority pattern
        highest_priority = max(patterns, key=lambda p: p.get('priority', 0))
        pattern_name = highest_priority.get('pattern', 'default')
        
        return self.colors.get(pattern_name, self.colors['default'])
    
    def create_enhanced_pdf_from_files(self,
                                     original_pdf: Path,
                                     text_extraction_file: Path,
                                     output_file: Path,
                                     version: str = 'short') -> Path:
        """
        Create enhanced PDF from text extraction file (converts to relative coordinates first)
        
        Args:
            original_pdf: Path to original PDF
            text_extraction_file: Path to text extraction JSON file
            output_file: Output PDF path
            version: 'short' (1 page) or 'long' (4 pages)
            
        Returns:
            Path to created PDF
        """
        
        # Load text extraction data
        with open(text_extraction_file, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
        
        # Convert to relative coordinates if needed
        if text_data.get('coordinate_system') != 'relative_to_symbol':
            print(f"Converting to relative coordinates...")
            relative_data = self.rel_coord_system.convert_to_relative_coordinates(text_data)
        else:
            relative_data = text_data
        
        # Create enhanced PDF
        return self.create_enhanced_pdf_from_relative_data(
            original_pdf, relative_data, output_file, version
        )

def main():
    """Test the relative coordinate PDF creator"""
    
    # Initialize PDF creator
    pdf_creator = RelativeCoordinatePDFCreator(font_size=10, min_conf=0.8)
    
    # Test files
    original_pdf = Path("D:/MarMe/github/0.3/plc-data/raw/pdfs/1150.pdf")
    text_file = Path("D:/MarMe/github/0.3/plc-data/processed/text_extraction/1150_text_extraction.json")
    output_file = Path("D:/MarMe/github/0.3/plc-data/processed/enhanced_pdfs/1150_relative_enhanced.pdf")
    
    if not original_pdf.exists():
        print(f"Original PDF not found: {original_pdf}")
        return
    
    if not text_file.exists():
        print(f"Text extraction file not found: {text_file}")
        return
    
    print("Creating enhanced PDF using relative coordinate system...")
    
    try:
        result_pdf = pdf_creator.create_enhanced_pdf_from_files(
            original_pdf=original_pdf,
            text_extraction_file=text_file,
            output_file=output_file,
            version='short'
        )
        
        print(f"✓ Enhanced PDF created successfully: {result_pdf}")
        print(f"  Original PDF: {original_pdf}")
        print(f"  Text data: {text_file}")
        print(f"  Enhanced PDF: {result_pdf}")
        
    except Exception as e:
        print(f"✗ Failed to create enhanced PDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
