#!/usr/bin/env python3
"""OCR worker – runs PaddleOCR inside the *ocr_env*.

Receives detection results + PDF path via JSON, produces extracted text and
metadata as JSON. Runs in a Paddle-only virtual environment, so there is zero
risk of DLL clashes with PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    try:
        # Get input parameters
        detection_file_path = input_data.get('detection_file')
        pdf_folder_path = input_data.get('pdf_folder')  # Optional - will auto-detect if not provided
        output_dir_path = input_data.get('output_dir')
        
        if not detection_file_path:
            raise ValueError("Missing required parameter: detection_file")
        
        if not output_dir_path:
            raise ValueError("Missing required parameter: output_dir")
        
        detection_file = Path(detection_file_path)
        output_dir = Path(output_dir_path)
        
        if not detection_file.exists():
            raise FileNotFoundError(f"Detection file not found: {detection_file}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing OCR for detection file: {detection_file}")
        print(f"Output directory: {output_dir}")
        
        # Auto-detect PDF folder if not provided
        if pdf_folder_path:
            pdf_folder = Path(pdf_folder_path)
        else:
            # Use config to get PDF folder
            from src.config import get_config
            config = get_config()
            pdf_folder = Path(config.config["data_root"]) / "raw" / "pdfs"
        
        if not pdf_folder.exists():
            raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")
        
        print(f"PDF folder: {pdf_folder}")
        
        # Find corresponding PDF file using the same logic as run_text_extraction.py
        pdf_name = get_pdf_name_from_detection_file(detection_file.name)
        pdf_file = pdf_folder / pdf_name
        
        if not pdf_file.exists():
            raise FileNotFoundError(f"Corresponding PDF not found: {pdf_file}")
        
        print(f"PDF file: {pdf_file}")
        
        # Initialize text extraction pipeline with proper parameters
        from src.ocr.text_extraction_pipeline import TextExtractionPipeline
        
        # Get OCR parameters from input
        confidence_threshold = input_data.get('confidence_threshold', 0.7)
        ocr_lang = input_data.get('language', 'en')
        device = input_data.get('device', None)  # Let pipeline auto-detect
        
        print(f"Initializing TextExtractionPipeline...")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  OCR language: {ocr_lang}")
        print(f"  Device: {device or 'auto-detect'}")
        
        pipeline = TextExtractionPipeline(
            confidence_threshold=confidence_threshold,
            ocr_lang=ocr_lang,
            device=device
        )
        
        # Run text extraction using the original pipeline logic
        result = pipeline.extract_text_from_detection_results(
            detection_file, pdf_file, output_dir
        )
        
        print(f"Text extraction completed successfully!")
        print(f"Found {result['total_text_regions']} text regions")
        
        out = {"status": "success", "results": result}
        exit_code = 0
        
    except Exception as exc:
        print(f"OCR worker error: {exc}")
        import traceback
        traceback.print_exc()
        
        out = {"status": "error", "error": str(exc)}
        exit_code = 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    sys.exit(exit_code)


def get_pdf_name_from_detection_file(detection_filename: str) -> str:
    """
    Extract PDF name from detection filename
    
    Handles various naming patterns:
    - 1150_detections.json -> 1150.pdf
    - 1150_detections_converted.json -> 1150.pdf
    - diagram_detections.json -> diagram.pdf
    """
    # Remove .json extension first
    name = detection_filename
    if name.endswith(".json"):
        name = name[:-5]
    
    # Remove detection suffixes in order (longest first)
    if name.endswith("_detections_converted"):
        name = name[:-20]  # Remove "_detections_converted"
    elif name.endswith("_detections"):
        name = name[:-11]  # Remove "_detections"
    elif name.endswith("_converted"):
        name = name[:-10]  # Remove "_converted"
    
    # Remove trailing underscore if present
    if name.endswith("_"):
        name = name[:-1]
    
    # Add .pdf extension
    result = f"{name}.pdf"
    
    return result


if __name__ == "__main__":
    main()
