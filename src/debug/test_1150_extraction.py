"""
Focused text extraction test for 1150.pdf
Generates ROIs for detailed analysis
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.ocr.text_extraction_pipeline import TextExtractionPipeline

def main():
    print("=== Focused Text Extraction Test for 1150.pdf ===")
    
    # Clear debug ROIs folder
    debug_dir = Path("debug_rois")
    if debug_dir.exists():
        import shutil
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(exist_ok=True)
    print(f"Cleared debug ROIs folder: {debug_dir}")
    
    # Initialize pipeline
    print("\nInitializing Text Extraction Pipeline...")
    pipeline = TextExtractionPipeline(confidence_threshold=0.7, ocr_lang="en")
    
    # Define paths for 1150.pdf specifically
    detection_file = Path("../plc-data/processed/detdiagrams2/1150_detections.json")
    pdf_file = Path("../plc-data/processed/detdiagrams2/1150_detected.pdf")
    output_dir = Path("../plc-data/processed/text_extraction")
    
    # Verify files exist
    if not detection_file.exists():
        print(f"ERROR: Detection file not found: {detection_file}")
        return 1
    
    if not pdf_file.exists():
        print(f"ERROR: PDF file not found: {pdf_file}")
        return 1
    
    print(f"\nProcessing:")
    print(f"  Detection file: {detection_file}")
    print(f"  PDF file: {pdf_file}")
    print(f"  Output dir: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Process the single file
        result = pipeline.extract_text_from_detection_results(
            detection_file, pdf_file, output_dir
        )
        
        print(f"\n=== RESULTS ===")
        print(f"Total text regions found: {result['total_text_regions']}")
        print(f"Statistics: {result['statistics']}")
        print(f"PLC patterns found: {result['plc_patterns_found']}")
        
        # Count debug ROIs generated
        roi_files = list(debug_dir.glob("*.png"))
        print(f"\nDebug ROIs generated: {len(roi_files)}")
        print(f"ROI files saved in: {debug_dir}")
        
        if roi_files:
            print("\nFirst few ROI files:")
            for roi_file in roi_files[:5]:
                print(f"  {roi_file.name}")
            if len(roi_files) > 5:
                print(f"  ... and {len(roi_files) - 5} more")
        
        return 0
        
    except Exception as e:
        print(f"ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
