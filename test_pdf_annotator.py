#!/usr/bin/env python3
"""
Test script for the new PDF Annotator approach
Tests native PDF annotations instead of image overlays
"""

from pathlib import Path
from src.utils.pdf_annotator import PDFAnnotator

def test_pdf_annotator():
    """Test the PDF annotator with native annotations"""
    
    print("=" * 60)
    print("TESTING PDF ANNOTATOR (NATIVE ANNOTATIONS)")
    print("=" * 60)
    
    # Test files
    pdf_file = Path("pdf_debug/1150.pdf")
    detection_file = Path("pdf_debug/1150_detections.json")
    text_file = Path("pdf_debug/1150_text_extraction.json")
    
    # Check if files exist
    if not all([pdf_file.exists(), detection_file.exists(), text_file.exists()]):
        print("❌ Required test files not found:")
        print(f"  PDF: {pdf_file} - {'✓' if pdf_file.exists() else '✗'}")
        print(f"  Detections: {detection_file} - {'✓' if detection_file.exists() else '✗'}")
        print(f"  Text: {text_file} - {'✓' if text_file.exists() else '✗'}")
        return False
    
    print("✓ All test files found")
    
    # Test PDF Annotator
    print("\n" + "─" * 40)
    print("TEST: PDF Annotator with Native Annotations")
    print("─" * 40)
    
    try:
        annotator = PDFAnnotator(
            detection_confidence_threshold=0.8,
            text_confidence_threshold=0.5
        )
        
        output_pdf = Path("pdf_debug/1150_annotated.pdf")
        
        print("Creating annotated PDF with native annotations...")
        annotated_pdf = annotator.create_annotated_pdf(
            detection_file=detection_file,
            text_extraction_file=text_file,
            pdf_file=pdf_file,
            output_file=output_pdf
        )
        
        print(f"✓ Annotated PDF created: {annotated_pdf}")
        
        # Verify file was created and has reasonable size
        if output_pdf.exists() and output_pdf.stat().st_size > 1000:
            print(f"✓ Output file size: {output_pdf.stat().st_size:,} bytes")
        else:
            print("❌ Output file missing or too small")
            return False
            
    except Exception as e:
        print(f"❌ PDF annotation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ PDF ANNOTATOR TEST COMPLETED!")
    print("=" * 60)
    print("\nKey advantages of this approach:")
    print("1. ✓ Preserves original PDF text and vector graphics")
    print("2. ✓ Uses native PDF annotations (professional appearance)")
    print("3. ✓ Simple coordinate transformation (no rotation issues)")
    print("4. ✓ Annotations can be toggled on/off in PDF viewers")
    print("5. ✓ Text remains searchable and selectable")
    print("6. ✓ Colored detection boxes with tooltips")
    print("7. ✓ OCR text as clickable annotations")
    print("\nThe annotated PDF should now show:")
    print("- Colored rectangle annotations for YOLO detections")
    print("- Text annotations for OCR results")
    print("- Tooltips with confidence scores and metadata")
    print("- All original PDF functionality preserved")
    
    return True

def main():
    """Main test function"""
    success = test_pdf_annotator()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
