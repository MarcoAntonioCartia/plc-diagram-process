"""
Enhanced Text Extraction Runner
Integrates detection preprocessing before text extraction
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.ocr.text_extraction_pipeline import TextExtractionPipeline
from src.ocr.detection_preprocessor import DetectionPreprocessor
from src.config import get_config

def main():
    parser = argparse.ArgumentParser(description='Enhanced Text Extraction with Detection Preprocessing')
    parser.add_argument('--detection-folder', '-d', type=str,
                       help='Folder containing detection JSON files')
    parser.add_argument('--pdf-folder', '-p', type=str,
                       help='Folder containing original PDF files')
    parser.add_argument('--output-folder', '-o', type=str,
                       help='Output folder for text extraction results')
    parser.add_argument('--confidence', '-c', type=float, default=0.7,
                       help='OCR confidence threshold (default: 0.7)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for detection preprocessing (default: 0.5)')
    parser.add_argument('--preprocess-detections', action='store_true', default=True,
                       help='Preprocess detection results to remove overlaps')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip detection preprocessing')
    parser.add_argument('--single-file', '-s', type=str,
                       help='Process single detection file instead of folder')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Set up paths
    if args.detection_folder:
        detection_folder = Path(args.detection_folder)
    else:
        detection_folder = Path(config.config['data_root']) / 'processed'
        if (detection_folder / 'detdiagrams').exists():
            detection_folder = detection_folder / 'detdiagrams'
    
    if args.pdf_folder:
        pdf_folder = Path(args.pdf_folder)
    else:
        pdf_folder = Path(config.config['data_root']) / 'raw' / 'pdfs'
    
    if args.output_folder:
        output_folder = Path(args.output_folder)
    else:
        output_folder = Path(config.config['data_root']) / 'processed' / 'enhanced_text_extraction'
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Validate paths
    if not detection_folder.exists():
        print(f"Error: Detection folder not found: {detection_folder}")
        return 1
    
    if not pdf_folder.exists():
        print(f"Error: PDF folder not found: {pdf_folder}")
        return 1
    
    # Initialize components
    print("Initializing Enhanced Text Extraction Pipeline...")
    print(f"Detection preprocessing: {'Enabled' if args.preprocess_detections and not args.skip_preprocessing else 'Disabled'}")
    print(f"OCR confidence threshold: {args.confidence}")
    print(f"IoU threshold for preprocessing: {args.iou_threshold}")
    
    # Initialize detection preprocessor
    detection_preprocessor = None
    if args.preprocess_detections and not args.skip_preprocessing:
        detection_preprocessor = DetectionPreprocessor(
            iou_threshold=args.iou_threshold,
            confidence_threshold=0.25
        )
    
    # Initialize text extraction pipeline
    text_pipeline = TextExtractionPipeline(
        confidence_threshold=args.confidence,
        ocr_lang='en'
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
            
            # Preprocess detection results if enabled
            processed_detection_file = detection_file
            if detection_preprocessor:
                print("Preprocessing detection results...")
                processed_detection_file = output_folder / f"{detection_file.stem}_processed.json"
                detection_preprocessor.preprocess_detection_file(detection_file, processed_detection_file)
            
            # Run text extraction
            result = text_pipeline.extract_text_from_detection_results(
                processed_detection_file, pdf_file, output_folder
            )
            
            print(f"Text extraction completed successfully!")
            print(f"Found {result['total_text_regions']} text regions")
            
        else:
            # Process entire folder
            print(f"Processing detection folder: {detection_folder}")
            
            # Find all detection files
            detection_files = list(detection_folder.glob("*_detections.json"))
            if not detection_files:
                print(f"No detection files found in {detection_folder}")
                return 1
            
            print(f"Found {len(detection_files)} detection files")
            
            # Process each file
            total_processed = 0
            total_text_regions = 0
            
            for detection_file in detection_files:
                try:
                    # Find corresponding PDF
                    pdf_name = detection_file.name.replace("_detections.json", ".pdf")
                    pdf_file = pdf_folder / pdf_name
                    
                    if not pdf_file.exists():
                        print(f"Warning: PDF not found for {detection_file.name}, skipping...")
                        continue
                    
                    print(f"Processing: {detection_file.name}")
                    
                    # Preprocess detection results if enabled
                    processed_detection_file = detection_file
                    if detection_preprocessor:
                        processed_detection_file = output_folder / f"{detection_file.stem}_processed.json"
                        if not processed_detection_file.exists():
                            detection_preprocessor.preprocess_detection_file(detection_file, processed_detection_file)
                    
                    # Run text extraction
                    result = text_pipeline.extract_text_from_detection_results(
                        processed_detection_file, pdf_file, output_folder
                    )
                    
                    total_processed += 1
                    total_text_regions += result['total_text_regions']
                    
                    print(f"  Found {result['total_text_regions']} text regions")
                    
                except Exception as e:
                    print(f"Error processing {detection_file.name}: {e}")
                    continue
            
            print(f"\nEnhanced text extraction completed!")
            print(f"Files processed: {total_processed}")
            print(f"Total text regions: {total_text_regions}")
            
            if total_processed > 0:
                avg_texts = total_text_regions / total_processed
                print(f"Average text regions per file: {avg_texts:.1f}")
        
        print(f"Results saved to: {output_folder}")
        return 0
        
    except Exception as e:
        print(f"Enhanced text extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 