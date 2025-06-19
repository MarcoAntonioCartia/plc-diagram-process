#!/usr/bin/env python3
"""
Complete PLC Diagram Processing Pipeline
Combines YOLO detection, text extraction, and PDF enhancement in one command
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.ocr.text_extraction_pipeline import TextExtractionPipeline
from src.utils.pdf_enhancer import PDFEnhancer
from src.config import get_config

def main():
    parser = argparse.ArgumentParser(description='Complete PLC Diagram Processing Pipeline')
    parser.add_argument('--pdf', '-p', type=str, required=True,
                       help='PDF file to process (e.g., 1150.pdf)')
    parser.add_argument('--confidence-threshold', '-c', type=float, default=0.8,
                       help='Confidence threshold for detection boxes (default: 0.8)')
    parser.add_argument('--ocr-confidence', type=float, default=0.5,
                       help='OCR confidence threshold (default: 0.5)')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Output directory (default: auto-detect)')
    parser.add_argument('--data-root', type=str,
                       help='Data root directory (default: from config)')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Set up paths
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(config.config['data_root'])
    
    pdf_file = Path(args.pdf)
    if not pdf_file.is_absolute():
        # Look for PDF in the raw/pdfs folder
        pdf_file = data_root / 'raw' / 'pdfs' / pdf_file.name
    
    if not pdf_file.exists():
        print(f"Error: PDF file not found: {pdf_file}")
        return 1
    
    # Extract base name (e.g., "1150" from "1150.pdf")
    base_name = pdf_file.stem
    
    # Set up folder paths
    detection_folder = data_root / 'processed' / 'detdiagrams2'
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_root / 'processed' / 'enhanced_pdfs'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    text_extraction_dir = data_root / 'processed' / 'text_extraction'
    text_extraction_dir.mkdir(parents=True, exist_ok=True)
    
    # Find detection file
    detection_file = detection_folder / f"{base_name}_detections.json"
    if not detection_file.exists():
        print(f"Error: Detection file not found: {detection_file}")
        print(f"Available detection files in {detection_folder}:")
        for det_file in detection_folder.glob("*_detections.json"):
            print(f"  {det_file.name}")
        return 1
    
    print("=" * 60)
    print("🚀 Complete PLC Diagram Processing Pipeline")
    print("=" * 60)
    print(f"📄 Processing: {pdf_file.name}")
    print(f"🎯 Detection file: {detection_file.name}")
    print(f"📊 Confidence threshold: {args.confidence_threshold:.0%}")
    print(f"🔍 OCR confidence: {args.ocr_confidence:.0%}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    try:
        # Step 1: Text Extraction
        print("🔤 Step 1: Running Text Extraction Pipeline...")
        pipeline = TextExtractionPipeline(
            confidence_threshold=args.ocr_confidence,
            ocr_lang="en",
            enable_nms=True,
            nms_iou_threshold=0.5
        )
        
        text_result = pipeline.extract_text_from_detection_results(
            detection_file, pdf_file, text_extraction_dir
        )
        
        text_extraction_file = text_extraction_dir / f"{base_name}_text_extraction.json"
        
        print(f"✅ Text extraction completed!")
        print(f"   📝 Found {text_result['total_text_regions']} text regions")
        print(f"   📊 Average confidence: {text_result['statistics']['average_confidence']:.1%}")
        print(f"   🔗 Association rate: {text_result['statistics']['association_rate']:.1f}%")
        print()
        
        # Step 2: Enhanced PDF Creation
        print("🎨 Step 2: Creating Enhanced PDF...")
        enhancer = PDFEnhancer(
            font_size=10,
            line_width=1.5,
            confidence_threshold=args.confidence_threshold
        )
        
        enhanced_pdf = enhancer.enhance_pdf_complete(
            detection_file,
            text_extraction_file,
            pdf_file,
            output_dir / f"{base_name}_enhanced.pdf"
        )
        
        print(f"✅ Enhanced PDF created!")
        print(f"   📄 Output: {enhanced_pdf}")
        print()
        
        # Step 3: Summary
        print("📋 Processing Summary:")
        print("=" * 40)
        
        # Count high-confidence detections
        import json
        with open(detection_file, 'r') as f:
            det_data = json.load(f)
        
        total_detections = sum(len(page['detections']) for page in det_data['pages'])
        high_conf_detections = sum(
            1 for page in det_data['pages'] 
            for det in page['detections'] 
            if det.get('confidence', 0) >= args.confidence_threshold
        )
        
        print(f"🎯 Detection boxes shown: {high_conf_detections}/{total_detections} (≥{args.confidence_threshold:.0%})")
        print(f"🔤 Text regions extracted: {text_result['total_text_regions']}")
        print(f"   📄 PDF text: {text_result['statistics']['pdf_extracted']}")
        print(f"   🔍 OCR text: {text_result['statistics']['ocr_extracted']}")
        
        # Show PLC patterns found
        patterns = text_result['plc_patterns_found']
        if patterns:
            print(f"🏷️  PLC patterns found:")
            for pattern, count in patterns.items():
                print(f"   {pattern}: {count}")
        
        print()
        print("🎉 Pipeline completed successfully!")
        print(f"📄 Enhanced PDF: {enhanced_pdf}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
