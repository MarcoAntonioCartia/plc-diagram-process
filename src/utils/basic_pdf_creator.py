"""
Basic PDF Creator - Phase 1
Creates enhanced PDFs with correct structure and orientation
No overlays - just proper PDF copying for debugging
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import argparse

class BasicPDFCreator:
    """
    Creates basic enhanced PDFs with correct structure
    Phase 1: Just copy PDFs correctly without overlays
    """
    
    def __init__(self, font_size: float = 12):
        """
        Initialize Basic PDF Creator
        
        Args:
            font_size: Font size for text annotations
        """
        self.font_size = font_size
        
        # Color definitions (RGB format for PyMuPDF)
        self.colors = {
            "background": (1, 1, 1),            # White
            "text_color": (0, 0, 0),            # Black
            "page_title": (0.2, 0.2, 0.2)      # Dark gray for page titles
        }
    
    def create_enhanced_pdf(self,
                           pdf_file: Path,
                           output_file: Path,
                           version: str = 'short') -> Path:
        """
        Create enhanced PDF structure without overlays
        
        Args:
            pdf_file: Path to original PDF file
            output_file: Output PDF file path
            version: 'long' (4 pages) or 'short' (1 page)
            
        Returns:
            Path to enhanced PDF file
        """
        print(f"Creating {version} version basic PDF: {pdf_file.name}")
        
        # Open original PDF
        original_doc = fitz.open(str(pdf_file))
        
        if version == 'long':
            enhanced_doc = self._create_long_version(original_doc)
        else:
            enhanced_doc = self._create_short_version(original_doc)
        
        # Save enhanced PDF
        enhanced_doc.save(str(output_file))
        enhanced_doc.close()
        original_doc.close()
        
        print(f"Basic PDF saved to: {output_file}")
        return output_file
    
    def _create_long_version(self, original_doc: fitz.Document) -> fitz.Document:
        """Create 4-page long version - just copies for now"""
        enhanced_doc = fitz.open()
        
        for page_num in range(len(original_doc)):
            original_page = original_doc[page_num]
            
            print(f"Processing page {page_num + 1}")
            print(f"Original page rect: {original_page.rect}")
            print(f"Original page rotation: {original_page.rotation}")
            
            # Page 1: Original PDF (clean)
            page1 = enhanced_doc.new_page(width=original_page.rect.width, 
                                        height=original_page.rect.height)
            page1.show_pdf_page(original_page.rect, original_doc, page_num)
            self._add_page_title(page1, "Page 1: Original PDF", 1)
            
            # Page 2: Copy for YOLO Detections (placeholder)
            page2 = enhanced_doc.new_page(width=original_page.rect.width, 
                                        height=original_page.rect.height)
            page2.show_pdf_page(original_page.rect, original_doc, page_num)
            self._add_page_title(page2, "Page 2: YOLO Detections (Placeholder)", 2)
            
            # Page 3: Copy for OCR Text (placeholder)
            page3 = enhanced_doc.new_page(width=original_page.rect.width, 
                                        height=original_page.rect.height)
            page3.show_pdf_page(original_page.rect, original_doc, page_num)
            self._add_page_title(page3, "Page 3: OCR Text (Placeholder)", 3)
            
            # Page 4: Copy for Combined (placeholder)
            page4 = enhanced_doc.new_page(width=original_page.rect.width, 
                                        height=original_page.rect.height)
            page4.show_pdf_page(original_page.rect, original_doc, page_num)
            self._add_page_title(page4, "Page 4: Combined (Placeholder)", 4)
        
        return enhanced_doc
    
    def _create_short_version(self, original_doc: fitz.Document) -> fitz.Document:
        """Create 1-page short version - just copy for now"""
        enhanced_doc = fitz.open()
        
        for page_num in range(len(original_doc)):
            original_page = original_doc[page_num]
            
            print(f"Processing page {page_num + 1}")
            print(f"Original page rect: {original_page.rect}")
            print(f"Original page rotation: {original_page.rotation}")
            
            # Single page: Copy of original
            page = enhanced_doc.new_page(width=original_page.rect.width, 
                                       height=original_page.rect.height)
            page.show_pdf_page(original_page.rect, original_doc, page_num)
            self._add_page_title(page, "Enhanced PLC Diagram (Placeholder)", 1)
        
        return enhanced_doc
    
    def _add_page_title(self, page: fitz.Page, title: str, page_num: int):
        """Add title to page"""
        # Add title with white background
        title_rect = fitz.Rect(10, 10, 500, 35)
        page.draw_rect(title_rect, color=self.colors["background"], 
                      fill=self.colors["background"], width=1)
        page.insert_text(fitz.Point(15, 28), title, fontsize=self.font_size, 
                        color=self.colors["page_title"])
        
        # Add page info for debugging
        page_info = f"Page {page_num} | Size: {page.rect.width:.0f}x{page.rect.height:.0f} | Rotation: {page.rotation}"
        page.insert_text(fitz.Point(15, 45), page_info, fontsize=self.font_size - 2, 
                        color=self.colors["page_title"])

def main():
    """CLI interface for Basic PDF creation"""
    parser = argparse.ArgumentParser(description='Create Basic Enhanced PDFs for debugging')
    parser.add_argument('--pdf-file', '-p', type=str, required=True,
                       help='Original PDF file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output PDF file')
    parser.add_argument('--version', '-v', type=str, choices=['long', 'short'], 
                       default='short', help='PDF version: long (4 pages) or short (1 page)')
    
    args = parser.parse_args()
    
    try:
        creator = BasicPDFCreator()
        
        enhanced_pdf = creator.create_enhanced_pdf(
            Path(args.pdf_file),
            Path(args.output),
            args.version
        )
        
        print(f"\nBasic PDF created successfully!")
        print(f"Version: {args.version}")
        print(f"Output: {enhanced_pdf}")
        return 0
        
    except Exception as e:
        print(f"Basic PDF creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
