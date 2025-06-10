"""
Text Extraction Pipeline Runner
Runs text extraction on detection results from the PLC diagram processor
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.ocr.text_extraction_pipeline import TextExtractionPipeline
from src.config import get_config

def main():
    parser = argparse.ArgumentParser(description='Run Text Extraction Pipeline on Detection Results')
    parser.add_argument('--detection-folder', '-d', type=str,
                       help='Folder containing detection JSON files (default: auto-detect from config)')
    parser.add_argument('--pdf-folder', '-p', type=str,
                       help='Folder containing original PDF files (default: auto-detect from config)')
    parser.add_argument('--output-folder', '-o', type=str,
                       help='Output folder for text extraction results (default: auto-detect from config)')
    parser.add_argument('--confidence', '-c', type=float, default=0.7,
                       help='OCR confidence threshold (default: 0.7)')
    parser.add_argument('--lang', '-l', type=str, default='en',
                       help='OCR language (default: en)')
    parser.add_argument('--single-file', '-s', type=str,
                       help='Process single detection file instead of folder')
    parser.add_argument('--list-files', action='store_true',
                       help='List available detection files and exit')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Set up paths
    if args.detection_folder:
        detection_folder = Path(args.detection_folder)
    else:
        # Auto-detect from processed folder
        detection_folder = Path(config.config['data_root']) / 'processed'
        # Look for detdiagrams subfolder or use processed directly
        if (detection_folder / 'detdiagrams').exists():
            detection_folder = detection_folder / 'detdiagrams'
    
    if args.pdf_folder:
        pdf_folder = Path(args.pdf_folder)
    else:
        pdf_folder = Path(config.config['data_root']) / 'raw' / 'pdfs'
    
    if args.output_folder:
        output_folder = Path(args.output_folder)
    else:
        output_folder = Path(config.config['data_root']) / 'processed' / 'text_extraction'
    
    # Validate paths
    if not detection_folder.exists():
        print(f"Error: Detection folder not found: {detection_folder}")
        print("Available options:")
        processed_base = Path(config.config['data_root']) / 'processed'
        if processed_base.exists():
            for item in processed_base.iterdir():
                if item.is_dir():
                    detection_files = list(item.glob("*_detections.json"))
                    if detection_files:
                        print(f"  {item} ({len(detection_files)} detection files)")
        return 1
    
    if not pdf_folder.exists():
        print(f"Error: PDF folder not found: {pdf_folder}")
        return 1
    
    # List files if requested
    if args.list_files:
        detection_files = list(detection_folder.glob("*_detections.json"))
        print(f"Detection files in {detection_folder}:")
        if detection_files:
            for i, file in enumerate(detection_files, 1):
                pdf_name = file.name.replace("_detections.json", ".pdf")
                pdf_exists = (pdf_folder / pdf_name).exists()
                status = "✓" if pdf_exists else "✗ (PDF missing)"
                print(f"  {i:2d}. {file.name} {status}")
        else:
            print("  No detection files found")
        return 0
    
    # Initialize text extraction pipeline
    print("Initializing Text Extraction Pipeline...")
    print(f"OCR Confidence Threshold: {args.confidence}")
    print(f"OCR Language: {args.lang}")
    
    pipeline = TextExtractionPipeline(
        confidence_threshold=args.confidence,
        ocr_lang=args.lang
    )
    
    try:
        if args.single_file:
            # Process single file
            detection_file = Path(args.single_file)
            if not detection_file.exists():
                print(f"Error: Detection file not found: {detection_file}")
                return 1
            
            # Find corresponding PDF
            pdf_name = detection_file.name.replace("_detections.json", ".pdf")
            pdf_file = pdf_folder / pdf_name
            
            if not pdf_file.exists():
                print(f"Error: Corresponding PDF not found: {pdf_file}")
                return 1
            
            print(f"Processing single file: {detection_file.name}")
            output_folder.mkdir(parents=True, exist_ok=True)
            
            result = pipeline.extract_text_from_detection_results(
                detection_file, pdf_file, output_folder
            )
            
            print(f"Text extraction completed successfully!")
            print(f"Found {result['total_text_regions']} text regions")
            print(f"Results saved to: {output_folder}")
            
        else:
            # Process entire folder
            print(f"Processing detection folder: {detection_folder}")
            print(f"PDF folder: {pdf_folder}")
            print(f"Output folder: {output_folder}")
            
            summary = pipeline.process_detection_folder(
                detection_folder, pdf_folder, output_folder
            )
            
            print(f"\nText extraction pipeline completed!")
            print(f"Processed files: {summary['processed_files']}")
            print(f"Total text regions: {summary['total_text_regions']}")
            
            if summary['processed_files'] > 0:
                avg_texts = summary['total_text_regions'] / summary['processed_files']
                print(f"Average text regions per file: {avg_texts:.1f}")
            
            print(f"Results saved to: {output_folder}")
        
        return 0
        
    except Exception as e:
        print(f"Text extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
