"""
Detection PDF Creator
Creates enhanced PDFs with overlays from detection and text extraction.
This version uses a PNG-first approach to solve coordinate system issues
with rotated PDFs.
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Any, Optional

from .basic_pdf_creator import BasicPDFCreator

class DetectionPDFCreator(BasicPDFCreator):
    """
    Creates enhanced PDFs with detection and text overlays using a PNG-first approach.
    Inherits from BasicPDFCreator to reuse title and page structure logic.
    """
    def __init__(self, font_size: float = 8):
        """
        Initialize Detection PDF Creator
        """
        super().__init__(font_size)
        self.colors.update({
            "detection_box": (1, 0, 1),      # Magenta
            "text_region_box": (0, 1, 0),    # Green
            "connection_line": (1, 1, 0)     # Yellow
        })
        self.line_width = 1.0

    def create_enhanced_pdf(self,
                           image_file: Path,
                           detections_file: Path,
                           text_file: Path,
                           output_file: Path,
                           version: str = 'long') -> Path:
        """
        Create an enhanced PDF with overlays from a source image and JSON data.

        Args:
            image_file: Path to the master PNG image.
            detections_file: Path to the detections JSON file.
            text_file: Path to the text extraction JSON file.
            output_file: Path for the new output PDF.
            version: 'long' (4 pages) or 'short' (1 page).

        Returns:
            Path to the generated enhanced PDF.
        """
        print(f"Creating '{version}' version enhanced PDF for {image_file.name}")

        try:
            with open(detections_file) as f:
                detections = json.load(f)
            with open(text_file) as f:
                text_data = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: Could not open required file: {e}")
            return

        pix = fitz.Pixmap(str(image_file))
        pdf_width, pdf_height = pix.width, pix.height
        pix = None  # Release memory

        enhanced_doc = fitz.open()

        if version == 'long':
            self._create_long_version(enhanced_doc, image_file, pdf_width, pdf_height, detections, text_data)
        else:
            self._create_short_version(enhanced_doc, image_file, pdf_width, pdf_height, detections, text_data)

        enhanced_doc.save(str(output_file))
        enhanced_doc.close()

        print(f"Enhanced PDF saved to: {output_file}")
        return output_file

    def _create_long_version(self, doc: fitz.Document, image_file: Path, w: float, h: float, detections: Dict, text: Dict):
        """Creates the 4-page detailed view."""
        # Page 1: Original Image
        page1 = doc.new_page(width=w, height=h)
        page1.insert_image(page1.rect, filename=str(image_file))
        self._add_page_title(page1, "Page 1: Original Image", 1)

        # Page 2: YOLO Detections
        page2 = doc.new_page(width=w, height=h)
        page2.insert_image(page2.rect, filename=str(image_file))
        self._draw_detections(page2, detections.get('detection_results', []))
        self._add_page_title(page2, "Page 2: YOLO Detections", 2)

        # Page 3: OCR Text Regions
        page3 = doc.new_page(width=w, height=h)
        page3.insert_image(page3.rect, filename=str(image_file))
        self._draw_text_regions(page3, text.get('text_regions', []))
        self._add_page_title(page3, "Page 3: OCR Text Regions", 3)
        
        # Page 4: Combined View
        page4 = doc.new_page(width=w, height=h)
        page4.insert_image(page4.rect, filename=str(image_file))
        self._draw_detections(page4, detections.get('detection_results', []))
        self._draw_text_regions(page4, text.get('text_regions', []))
        # self._draw_connections(page4, ...) # Placeholder for future connection drawing
        self._add_page_title(page4, "Page 4: Combined View", 4)

    def _create_short_version(self, doc: fitz.Document, image_file: Path, w: float, h: float, detections: Dict, text: Dict):
        """Creates a single-page combined view."""
        page = doc.new_page(width=w, height=h)
        page.insert_image(page.rect, filename=str(image_file))
        self._draw_detections(page, detections.get('detection_results', []))
        self._draw_text_regions(page, text.get('text_regions', []))
        self._add_page_title(page, "Enhanced PLC Diagram - Combined View", 1)

    def _draw_detections(self, page: fitz.Page, detections: List[Dict]):
        """Draws detection boxes directly onto the page."""
        for det in detections:
            bbox = det.get('bbox')
            if not bbox: continue
            
            # No scaling or rotation needed. Coordinates are used directly.
            rect = fitz.Rect(bbox)
            page.draw_rect(rect, color=self.colors['detection_box'], width=self.line_width)

    def _draw_text_regions(self, page: fitz.Page, text_regions: List[Dict]):
        """Draws text region boxes directly onto the page."""
        for region in text_regions:
            bbox = region.get('bbox')
            if not bbox: continue

            # No scaling or rotation needed. Coordinates are used directly.
            rect = fitz.Rect(bbox)
            page.draw_rect(rect, color=self.colors['text_region_box'], width=self.line_width)
