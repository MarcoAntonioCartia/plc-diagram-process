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

from src.utils.pdf_enhancer import PDFEnhancer


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
    parser.add_argument('--enhance-pdf', action='store_true',
                       help='Enhance PDF with detection boxes and text extraction results')
    parser.add_argument('--enhance-pdf-batch', action='store_true',
                   help='Create enhanced PDFs for all processed files')
    
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
                pdf_name = get_pdf_name_from_detection_file(file.name)
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
            
            # Find corresponding PDF using improved naming logic
            pdf_name = get_pdf_name_from_detection_file(detection_file.name)
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
            
            if args.enhance_pdf:
                print("Creating enhanced PDF...")
                enhancer = PDFEnhancer()
                
                # Find the original detection file (non-converted)
                original_detection_file = detection_file.parent / detection_file.name.replace("_converted.json", ".json")
                if not original_detection_file.exists():
                    # Try to find any detection file for this PDF
                    pdf_stem = pdf_file.stem
                    possible_detection_files = list(detection_file.parent.glob(f"{pdf_stem}*.json"))
                    if possible_detection_files:
                        original_detection_file = possible_detection_files[0]
                
                if original_detection_file.exists():
                    enhanced_pdf = enhancer.enhance_pdf_complete(
                        original_detection_file,
                        Path(output_folder) / f"{pdf_file.stem}_text_extraction.json",
                        pdf_file,
                        Path(output_folder) / f"{pdf_file.stem}_enhanced.pdf"
                    )
                    print(f"Enhanced PDF created: {enhanced_pdf}")
                else:
                    print("Warning: Could not find original detection file for PDF enhancement")

            if args.enhance_pdf_batch:
                print("\nCreating enhanced PDFs for all processed files...")
                enhancer = PDFEnhancer()
                
                # Find original detection folder (non-converted files)
                original_detection_folder = detection_folder.parent / detection_folder.name.replace("texttoextract", "detdiagrams2")
                if not original_detection_folder.exists():
                    # Try to find detection folder
                    possible_folders = list(detection_folder.parent.glob("*detdiagrams*"))
                    if possible_folders:
                        original_detection_folder = possible_folders[0]
                
                if original_detection_folder.exists():
                    enhanced_output_folder = output_folder.parent / "enhanced_pdfs"
                    summary = enhancer.enhance_folder_batch(
                        original_detection_folder,
                        output_folder,
                        pdf_folder,
                        enhanced_output_folder,
                        'complete'
                    )
                    print(f"Enhanced PDFs created in: {enhanced_output_folder}")
                else:
                    print("Warning: Could not find original detection folder for PDF enhancement")

            print(f"Results saved to: {output_folder}")
        
        return 0
        
    except Exception as e:
        print(f"Text extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
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
    
    # Debug: show intermediate steps
    print(f"Debug: After removing .json: '{name}'")
    
    # Remove detection suffixes in order (longest first)
    if name.endswith("_detections_converted"):
        name = name[:-20]  # Remove "_detections_converted"
        print(f"Debug: After removing _detections_converted: '{name}'")
    elif name.endswith("_detections"):
        name = name[:-11]  # Remove "_detections"
    elif name.endswith("_converted"):
        name = name[:-10]  # Remove "_converted"
    
    # Remove trailing underscore if present
    if name.endswith("_"):
        name = name[:-1]
        print(f"Debug: After removing trailing underscore: '{name}'")
    
    # Add .pdf extension
    result = f"{name}.pdf"
    
    # Debug output
    print(f"Debug: Final result: '{detection_filename}' -> '{result}'")
    
    return result


if __name__ == "__main__":
    exit(main())
