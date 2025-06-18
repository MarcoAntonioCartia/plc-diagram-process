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
    
    def __init__(self, font_size: float = 10, line_width: float = 1.5, 
                 confidence_threshold: float = 0.8):
        """
        Initialize PDF enhancer
        
        Args:
            font_size: Font size for text annotations
            line_width: Line width for boxes and lines
            confidence_threshold: Minimum confidence for showing detection boxes
        """
        self.font_size = font_size
        self.line_width = line_width
        self.confidence_threshold = confidence_threshold
        
        # Color definitions (RGB format for PyMuPDF) - Updated for cleaner look
        self.colors = {
            "yolo_detection": (0, 0, 1),        # Blue for detection boxes
            "text_box": (0, 0.8, 0),            # Green for text boxes (like your example)
            "text_label": (0, 0.6, 0),          # Darker green for text labels
            "connection": (0.7, 0.7, 0.7),      # Light gray for connections
            "background": (1, 1, 1),            # White
            "text_color": (0, 0, 0),            # Black
            "confidence_high": (0, 0.8, 0),     # Green for high confidence
            "confidence_medium": (1, 0.6, 0),   # Orange for medium confidence
            "confidence_low": (1, 0, 0)         # Red for low confidence
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
    def enhance_folder_batch(self,
                           detection_folder: Path,
                           text_extraction_folder: Path,
                           pdf_folder: Path,
                           output_folder: Optional[Path] = None,
                           mode: str = 'complete') -> Dict[str, Any]:
        """
        Enhance all PDFs in a folder with batch processing
        
        Args:
            detection_folder: Folder containing detection JSON files
            text_extraction_folder: Folder containing text extraction JSON files
            pdf_folder: Folder containing original PDF files
            output_folder: Output folder for enhanced PDFs (optional)
            mode: Enhancement mode ('detections', 'text', 'complete')
            
        Returns:
            Summary of processing results
        """
        print(f"Starting batch PDF enhancement...")
        print(f"Detection folder: {detection_folder}")
        print(f"Text extraction folder: {text_extraction_folder}")
        print(f"PDF folder: {pdf_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Mode: {mode}")
        
        if output_folder:
            output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all detection files
        detection_files = list(detection_folder.glob("*_detections.json"))
        if not detection_files:
            print(f"No detection files found in {detection_folder}")
            return {"processed": 0, "errors": [], "results": []}
        
        print(f"Found {len(detection_files)} detection files")
        
        results = []
        errors = []
        processed = 0
        
        for detection_file in detection_files:
            try:
                # Extract base name
                base_name = detection_file.stem.replace("_detections", "").replace("_converted", "")
                
                # Find corresponding files
                pdf_file = pdf_folder / f"{base_name}.pdf"
                text_extraction_file = text_extraction_folder / f"{base_name}_text_extraction.json"
                
                if not pdf_file.exists():
                    print(f"Warning: PDF not found for {detection_file.name}: {pdf_file}")
                    continue
                
                if mode in ['text', 'complete'] and not text_extraction_file.exists():
                    print(f"Warning: Text extraction file not found for {detection_file.name}: {text_extraction_file}")
                    continue
                
                print(f"\nProcessing: {detection_file.name}")
                
                # Determine output file
                if output_folder:
                    output_file = output_folder / f"{base_name}_enhanced.pdf"
                else:
                    output_file = pdf_file.parent / f"{base_name}_enhanced.pdf"
                
                # Enhance PDF based on mode
                if mode == 'detections':
                    enhanced_pdf = self.enhance_pdf_with_detections(
                        detection_file, pdf_file, output_file
                    )
                elif mode == 'text':
                    enhanced_pdf = self.enhance_pdf_with_text_extraction(
                        text_extraction_file, pdf_file, output_file
                    )
                elif mode == 'complete':
                    enhanced_pdf = self.enhance_pdf_complete(
                        detection_file, text_extraction_file, pdf_file, output_file
                    )
                
                results.append({
                    "detection_file": str(detection_file),
                    "pdf_file": str(pdf_file),
                    "text_extraction_file": str(text_extraction_file) if text_extraction_file.exists() else None,
                    "enhanced_pdf": str(enhanced_pdf),
                    "mode": mode
                })
                
                processed += 1
                print(f"✓ Enhanced: {enhanced_pdf.name}")
                
            except Exception as e:
                error_msg = f"Error processing {detection_file.name}: {e}"
                print(f"✗ {error_msg}")
                errors.append(error_msg)
                continue
        
        # Generate summary
        summary = {
            "processed": processed,
            "total_files": len(detection_files),
            "errors": errors,
            "results": results,
            "success_rate": (processed / len(detection_files) * 100) if detection_files else 0
        }
        
        # Save summary
        if output_folder:
            summary_file = output_folder / "enhancement_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to: {summary_file}")
        
        print(f"\nBatch enhancement completed!")
        print(f"Processed: {processed}/{len(detection_files)} files")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        if errors:
            print(f"Errors: {len(errors)}")
        
        return summary
    
    def _draw_detection_box(self, page: fitz.Page, detection: Dict):
        """Draw a YOLO detection box on the page (only if confidence >= threshold)"""
        confidence = detection.get("confidence", 0.0)
        
        # Only draw if confidence meets threshold
        if confidence < self.confidence_threshold:
            return
            
        bbox = detection.get("global_bbox", detection.get("bbox_global", None))
        if isinstance(bbox, dict):
            bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
        if not (isinstance(bbox, list) and len(bbox) == 4):
            return
        
        x1, y1, x2, y2 = bbox
        
        # Create rectangle
        rect = fitz.Rect(x1, y1, x2, y2)
        
        # Choose color based on confidence level
        if confidence >= 0.95:
            box_color = self.colors["confidence_high"]
        elif confidence >= 0.85:
            box_color = self.colors["confidence_medium"]
        else:
            box_color = self.colors["yolo_detection"]
        
        # Draw box with thinner line for cleaner look
        page.draw_rect(rect, color=box_color, width=1.0, fill=None)
        
        # Add label with cleaner formatting
        class_name = detection.get("class_name", "unknown")
        label = f"{class_name} {confidence:.0%}"  # Show as percentage
        
        # Position label above box with better spacing
        text_point = fitz.Point(x1, y1 - 8)
        page.insert_text(text_point, label, fontsize=self.font_size - 1, 
                        color=self.colors["text_color"])
    
    def _draw_text_region(self, page: fitz.Page, text_region: Dict):
        """Draw a text region on the page with green boxes like the example"""
        bbox = text_region["bbox"]
        text = text_region["text"]
        confidence = text_region["confidence"]
        source = text_region["source"]
        associated_symbol = text_region.get("associated_symbol")
        
        x1, y1, x2, y2 = bbox
        
        # Use green color for all text boxes (like your example)
        text_color = self.colors["text_box"]
        
        # Draw text box with green outline
        rect = fitz.Rect(x1, y1, x2, y2)
        page.draw_rect(rect, color=text_color, width=1.5, fill=None)
        
        # Position text inside the box (centered vertically)
        text_height = y2 - y1
        text_y = y1 + (text_height / 2) + (self.font_size / 3)  # Center vertically
        text_x = x1 + 2  # Small left margin
        
        # Clean text formatting - just show the text without confidence
        display_text = text.strip()
        
        # Insert text inside the box
        text_point = fitz.Point(text_x, text_y)
        page.insert_text(text_point, display_text, fontsize=self.font_size - 1, 
                        color=self.colors["text_color"])
        
        # Optional: Draw subtle connection to associated symbol (very light)
        if associated_symbol and "global_bbox" in associated_symbol:
            symbol_bbox = associated_symbol.get("global_bbox", associated_symbol.get("bbox_global"))
            if isinstance(symbol_bbox, list) and len(symbol_bbox) == 4:
                sx1, sy1, sx2, sy2 = symbol_bbox
                
                # Calculate centers
                text_center = fitz.Point((x1 + x2) / 2, (y1 + y2) / 2)
                symbol_center = fitz.Point((sx1 + sx2) / 2, (sy1 + sy2) / 2)
                
                # Draw very subtle connection line
                page.draw_line(text_center, symbol_center, 
                              color=(0.9, 0.9, 0.9), width=0.5)  # Very light gray
    
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
        
        # Text boxes (green)
        page.draw_rect(fitz.Rect(legend_x, legend_y + 30, legend_x + 20, legend_y + 50), 
                      color=self.colors["text_box"], fill=self.colors["text_box"])
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 45), 
                        "Text Regions", fontsize=self.font_size, color=self.colors["text_color"])
        
        # High confidence detections (blue)
        page.draw_rect(fitz.Rect(legend_x, legend_y + 60, legend_x + 20, legend_y + 80), 
                      color=self.colors["yolo_detection"], fill=self.colors["yolo_detection"])
        page.insert_text(fitz.Point(legend_x + 25, legend_y + 75), 
                        f"High Conf Detections (≥{self.confidence_threshold:.0%})", fontsize=self.font_size, color=self.colors["text_color"])
        
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
    parser.add_argument('--detection-folder', type=str,
                       help='Folder containing detection JSON files (batch mode)')
    parser.add_argument('--text-extraction-folder', type=str,
                       help='Folder containing text extraction JSON files (batch mode)')
    parser.add_argument('--pdf-folder', type=str,
                       help='Folder containing original PDF files (batch mode)')
    parser.add_argument('--output-folder', type=str,
                       help='Output folder for enhanced PDFs (batch mode)')
    parser.add_argument('--batch', action='store_true',
                       help='Enable batch processing mode')
    
    args = parser.parse_args()
    
    enhancer = PDFEnhancer()
    
    try:
        if args.batch:
            # Batch processing mode
            if not all([args.detection_folder, args.text_extraction_folder, args.pdf_folder]):
                print("Error: For batch mode, all folder arguments are required")
                return 1
            
            summary = enhancer.enhance_folder_batch(
                Path(args.detection_folder),
                Path(args.text_extraction_folder),
                Path(args.pdf_folder),
                Path(args.output_folder) if args.output_folder else None,
                args.mode
            )
            
            print(f"\nBatch processing completed successfully!")
            return 0
            
        else:
            # Single file mode
            if args.mode == 'detections' and args.detection_file and args.pdf_file:
                enhancer.enhance_pdf_with_detections(
                    Path(args.detection_file), Path(args.pdf_file), 
                    Path(args.output) if args.output else None
                )
            elif args.mode == 'text' and args.text_file and args.pdf_file:
                enhancer.enhance_pdf_with_text_extraction(
                    Path(args.text_file), Path(args.pdf_file), 
                    Path(args.output) if args.output else None
                )
            elif args.mode == 'complete' and args.detection_file and args.text_file and args.pdf_file:
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
