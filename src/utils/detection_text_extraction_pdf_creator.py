"""
Detection PDF Creator
Creates enhanced PDFs with overlays from detection and text extraction.
This version uses a PNG-first approach to solve coordinate system issues
with rotated PDFs and integrates relative coordinate positioning.
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Any, Optional

from .basic_pdf_creator import BasicPDFCreator
from ..ocr.relative_coordinate_system import RelativeCoordinateSystem

class DetectionPDFCreator(BasicPDFCreator):
    """
    Creates enhanced PDFs with detection and text overlays using a PNG-first approach.
    Inherits from BasicPDFCreator to reuse title and page structure logic.
    """
    def __init__(self, font_size: float = 8, min_conf: float = 0.0):
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
        self.min_conf = float(min_conf)
        self.rel_coord_system = RelativeCoordinateSystem()

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

        # ------------------------------------------------------------------
        # Extract detection list from the JSON structure
        # The detection JSON has a "pages" structure with detections per page
        # ------------------------------------------------------------------
        det_list = []
        pages = detections.get('pages', [])
        
        # Flatten all detections from all pages
        for page_data in pages:
            page_detections = page_data.get('detections', [])
            for det in page_detections:
                # Convert detection format to what the PDF creator expects
                bbox_global = det.get('bbox_global', {})
                if isinstance(bbox_global, dict):
                    bbox = [bbox_global.get('x1', 0), bbox_global.get('y1', 0), 
                           bbox_global.get('x2', 0), bbox_global.get('y2', 0)]
                else:
                    bbox = bbox_global if isinstance(bbox_global, list) else [0, 0, 0, 0]
                
                det_list.append({
                    'bbox': bbox,
                    'label': det.get('class_name', 'unknown'),
                    'confidence': det.get('confidence', 0.0)
                })
        
        print(f"Extracted {len(det_list)} detections from {len(pages)} pages")
        
        # Get original PDF dimensions from the detection data
        if pages:
            page_data = pages[0]  # Use first page
            original_width = page_data.get('original_width', pdf_width)
            original_height = page_data.get('original_height', pdf_height)
        else:
            original_width = pdf_width
            original_height = pdf_height
        
        # Calculate scale factor from original PDF size to rendered image size
        scale = (pdf_width / original_width, pdf_height / original_height)
        self._offset = (0, 0)  # No offset needed since we're scaling directly
        
        print(f"PDF dimensions: {pdf_width}x{pdf_height}, Original: {original_width}x{original_height}, Scale: {scale}")

        enhanced_doc = fitz.open()

        if version == 'long':
            self._create_long_version(enhanced_doc, image_file, pdf_width, pdf_height, det_list, text_data, scale)
        else:
            self._create_short_version(enhanced_doc, image_file, pdf_width, pdf_height, det_list, text_data, scale)

        enhanced_doc.save(str(output_file))
        enhanced_doc.close()

        print(f"Enhanced PDF saved to: {output_file}")
        return output_file

    def create_enhanced_pdf_from_original(self,
                                        original_pdf: Path,
                                        detections_file: Path,
                                        text_file: Path,
                                        output_file: Path,
                                        version: str = 'short') -> Path:
        """
        Create an enhanced PDF with overlays from an original PDF and JSON data.
        This method converts PDF pages to clean images on-the-fly.

        Args:
            original_pdf: Path to the original PDF file.
            detections_file: Path to the detections JSON file.
            text_file: Path to the text extraction JSON file.
            output_file: Path for the new output PDF.
            version: 'long' (4 pages) or 'short' (1 page).

        Returns:
            Path to the generated enhanced PDF.
        """
        print(f"Creating '{version}' version enhanced PDF from original PDF: {original_pdf.name}")

        try:
            with open(detections_file) as f:
                detections = json.load(f)
            with open(text_file) as f:
                text_data = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: Could not open required file: {e}")
            return output_file

        # Convert first page of PDF to image
        doc = fitz.open(str(original_pdf))
        page = doc[0]  # Use first page
        
        # Convert to high-resolution image (2x scale for quality)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        
        # Get image dimensions
        pdf_width, pdf_height = pix.width, pix.height
        
        # Save as temporary PNG for processing
        temp_image_path = output_file.parent / f"temp_{original_pdf.stem}.png"
        pix.save(str(temp_image_path))
        pix = None  # Release memory
        doc.close()

        try:
            # Extract detection list from the JSON structure
            det_list = []
            pages = detections.get('pages', [])
            
            # Flatten all detections from all pages
            for page_data in pages:
                page_detections = page_data.get('detections', [])
                for det in page_detections:
                    # Convert detection format to what the PDF creator expects
                    bbox_global = det.get('bbox_global', {})
                    if isinstance(bbox_global, dict):
                        bbox = [bbox_global.get('x1', 0), bbox_global.get('y1', 0), 
                               bbox_global.get('x2', 0), bbox_global.get('y2', 0)]
                    else:
                        bbox = bbox_global if isinstance(bbox_global, list) else [0, 0, 0, 0]
                    
                    det_list.append({
                        'bbox': bbox,
                        'label': det.get('class_name', 'unknown'),
                        'confidence': det.get('confidence', 0.0)
                    })
            
            print(f"Extracted {len(det_list)} detections from {len(pages)} pages")
            
            # Get original PDF dimensions from the detection data
            if pages:
                page_data = pages[0]  # Use first page
                original_width = page_data.get('original_width', pdf_width // 2)  # Divide by 2 because of 2x scale
                original_height = page_data.get('original_height', pdf_height // 2)
            else:
                original_width = pdf_width // 2
                original_height = pdf_height // 2
            
            # Calculate scale factor from original PDF size to rendered image size (accounting for 2x scale)
            scale = (pdf_width / original_width, pdf_height / original_height)
            self._offset = (0, 0)  # No offset needed since we're scaling directly
            
            print(f"PDF dimensions: {pdf_width}x{pdf_height}, Original: {original_width}x{original_height}, Scale: {scale}")

            enhanced_doc = fitz.open()

            if version == 'long':
                self._create_long_version(enhanced_doc, temp_image_path, pdf_width, pdf_height, det_list, text_data, scale)
            else:
                self._create_short_version(enhanced_doc, temp_image_path, pdf_width, pdf_height, det_list, text_data, scale)

            enhanced_doc.save(str(output_file))
            enhanced_doc.close()

            print(f"Enhanced PDF saved to: {output_file}")
            
        finally:
            # Clean up temporary image
            try:
                temp_image_path.unlink()
                print(f"Cleaned up temporary image: {temp_image_path}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary image: {e}")

        return output_file

    def _create_long_version(self, doc: fitz.Document, image_file: Path, w: float, h: float, det_list: list, text: Dict, scale: tuple[float, float]):
        """Creates the 4-page detailed view."""
        # Page 1: Original Image
        page1 = doc.new_page(width=w, height=h)
        page1.insert_image(page1.rect, filename=str(image_file))
        self._add_page_title(page1, "Page 1: Original Image", 1)

        # Page 2: YOLO Detections
        page2 = doc.new_page(width=w, height=h)
        page2.insert_image(page2.rect, filename=str(image_file))
        self._draw_detections(page2, det_list, scale)
        self._add_page_title(page2, "Page 2: YOLO Detections", 2)

        # Page 3: OCR Text Regions
        page3 = doc.new_page(width=w, height=h)
        page3.insert_image(page3.rect, filename=str(image_file))
        self._draw_text_regions(page3, text.get('text_regions', []), scale)
        self._add_page_title(page3, "Page 3: OCR Text Regions", 3)
        
        # Page 4: Combined View
        page4 = doc.new_page(width=w, height=h)
        page4.insert_image(page4.rect, filename=str(image_file))
        self._draw_detections(page4, det_list, scale)
        self._draw_text_regions(page4, text.get('text_regions', []), scale)
        # self._draw_connections(page4, ...) # Placeholder for future connection drawing
        self._add_page_title(page4, "Page 4: Combined View", 4)

    def _create_short_version(self, doc: fitz.Document, image_file: Path, w: float, h: float, det_list: list, text: Dict, scale: tuple[float, float]):
        """Creates a single-page combined view."""
        page = doc.new_page(width=w, height=h)
        page.insert_image(page.rect, filename=str(image_file))
        self._draw_detections(page, det_list, scale)
        self._draw_text_regions(page, text.get('text_regions', []), scale)
        self._add_page_title(page, "Enhanced PLC Diagram - Combined View", 1)

    def _draw_detections(self, page: fitz.Page, detections: List[Dict], scale: tuple[float, float]):
        """Draws detection boxes and their labels onto the page."""
        filtered_count = 0
        total_count = len(detections)
        
        for det in detections:
            bbox = det.get('bbox')
            label = det.get('label', '')
            confidence = det.get('confidence', 0.0)
            
            # Apply confidence filtering
            if not bbox or confidence < self.min_conf:
                continue
            
            filtered_count += 1

            ox, oy = self._offset
            x1, y1, x2, y2 = bbox
            x1 += ox; y1 += oy; x2 += ox; y2 += oy
            rect = fitz.Rect(x1 * scale[0], y1 * scale[1], x2 * scale[0], y2 * scale[1])
            
            page.draw_rect(rect, color=self.colors['detection_box'], width=self.line_width)
            
            label_text = f"{label} ({confidence:.2f})"
            
            # Insert text at the top-left corner, inside the box
            text_insertion_point = fitz.Point(rect.x0 + 2, rect.y0 + self.font_size)
            
            # Draw a background for the text for better readability
            text_len = fitz.get_text_length(label_text, fontsize=self.font_size)
            text_bg_rect = fitz.Rect(
                text_insertion_point.x, 
                text_insertion_point.y - self.font_size,
                text_insertion_point.x + text_len + 4,
                text_insertion_point.y + 3
            )
            page.draw_rect(text_bg_rect, color=(1, 1, 1), fill=(1, 1, 1, 0.7))

            page.insert_text(
                text_insertion_point,
                label_text,
                fontsize=self.font_size,
                color=self.colors['detection_box']
            )
        
        print(f"Detection filtering: {total_count} total → {filtered_count} displayed (≥{self.min_conf:.1f} confidence)")

    def _draw_text_regions(self, page: fitz.Page, text_regions: List[Dict], scale: tuple[float, float]):
        """
        Draws text region boxes and the extracted text using relative coordinate positioning.
        This method fixes coordinate transformation issues by positioning text relative to YOLO boxes.
        """
        print(f"Drawing {len(text_regions)} text regions using relative coordinate system...")
        
        # Group text regions by their associated symbols for efficient processing
        symbol_groups = {}
        unassociated_regions = []
        
        for region in text_regions:
            text_content = region.get('text', '')
            if not text_content:
                continue
                
            # Check if region has an associated symbol
            symbol = region.get('associated_symbol', {})
            if symbol:
                # Create unique symbol identifier
                symbol_id = f"{symbol.get('class_name', 'unknown')}_{symbol.get('snippet_source', 'unknown')}"
                
                if symbol_id not in symbol_groups:
                    symbol_groups[symbol_id] = {
                        'symbol_info': symbol,
                        'text_regions': []
                    }
                symbol_groups[symbol_id]['text_regions'].append(region)
            else:
                # Handle regions without symbol association (fallback to old method)
                unassociated_regions.append(region)
        
        print(f"  Grouped into {len(symbol_groups)} symbols with {len(unassociated_regions)} unassociated regions")
        
        # Draw text regions grouped by symbols using relative coordinates
        for symbol_id, symbol_data in symbol_groups.items():
            self._draw_symbol_text_group(page, symbol_data, scale)
        
        # Draw unassociated regions using the old method as fallback
        for region in unassociated_regions:
            self._draw_single_text_region_fallback(page, region, scale)
    
    def _draw_symbol_text_group(self, page: fitz.Page, symbol_data: Dict, scale: tuple[float, float]):
        """
        Draw all text regions associated with a single symbol using relative coordinates.
        """
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
            print(f"  Warning: Invalid symbol bbox for {symbol_info.get('class_name', 'unknown')}")
            return
        
        # Convert each text region from relative to absolute coordinates
        for region in text_regions:
            text_content = region.get('text', '')
            confidence = region.get('confidence', 0)
            
            # Check if region already has relative coordinates
            if region.get('coordinate_system') == 'relative_to_symbol':
                # Use existing relative coordinates
                relative_bbox = region.get('bbox', [])
            else:
                # Convert absolute coordinates to relative coordinates
                absolute_bbox = region.get('bbox', [])
                if len(absolute_bbox) >= 4:
                    relative_bbox = [
                        absolute_bbox[0] - symbol_bbox[0],  # rel_x1
                        absolute_bbox[1] - symbol_bbox[1],  # rel_y1
                        absolute_bbox[2] - symbol_bbox[0],  # rel_x2
                        absolute_bbox[3] - symbol_bbox[1]   # rel_y2
                    ]
                else:
                    continue
            
            if len(relative_bbox) < 4:
                continue
            
            # Convert relative coordinates back to absolute coordinates for rendering
            absolute_bbox = [
                symbol_bbox[0] + relative_bbox[0],  # abs_x1
                symbol_bbox[1] + relative_bbox[1],  # abs_y1
                symbol_bbox[0] + relative_bbox[2],  # abs_x2
                symbol_bbox[1] + relative_bbox[3]   # abs_y2
            ]
            
            # Apply offset and scaling
            ox, oy = self._offset
            x1, y1, x2, y2 = absolute_bbox
            x1 += ox; y1 += oy; x2 += ox; y2 += oy
            rect = fitz.Rect(x1 * scale[0], y1 * scale[1], x2 * scale[0], y2 * scale[1])
            
            # Draw the bounding box for the text region with a dashed line
            page.draw_rect(rect, color=self.colors['text_region_box'], width=self.line_width, dashes="[2 2] 0")
            
            # Add confidence indicator to text
            display_text = f"{text_content} ({confidence:.2f})" if confidence > 0 else text_content
            
            # Insert the extracted text inside the box
            page.insert_textbox(
                rect,
                display_text,
                fontname="helv",
                fontsize=self.font_size,
                color=(0, 0.4, 0)  # Darker green for readability
            )
    
    def _draw_single_text_region_fallback(self, page: fitz.Page, region: Dict, scale: tuple[float, float]):
        """
        Fallback method for drawing text regions without symbol association.
        Uses the original coordinate system.
        """
        bbox = region.get('bbox')
        text_content = region.get('text', '')
        
        if not bbox or not text_content:
            return
        
        # Use original method for unassociated regions
        ox, oy = self._offset
        x1, y1, x2, y2 = bbox
        x1 += ox; y1 += oy; x2 += ox; y2 += oy
        rect = fitz.Rect(x1 * scale[0], y1 * scale[1], x2 * scale[0], y2 * scale[1])
        
        # Draw with different color to indicate fallback method
        page.draw_rect(rect, color=(1, 0.5, 0), width=self.line_width, dashes="[4 2] 0")  # Orange dashed
        
        page.insert_textbox(
            rect,
            f"{text_content} [fallback]",
            fontname="helv",
            fontsize=self.font_size,
            color=(0.8, 0.4, 0)  # Orange text
        )
