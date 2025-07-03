#!/usr/bin/env python3
"""
Test the corrected Enhanced PDF Creator with proper snippet coordinate transformation
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def test_corrected_enhanced_pdf():
    """Test the corrected Enhanced PDF Creator"""
    print("Testing Corrected Enhanced PDF Creator")
    print("=" * 50)
    
    # Define paths
    debug_folder = Path("D:/MarMe/github/0.4/plc-diagram-processor/pdf_debug")
    pdf_file = debug_folder / "1150.pdf"
    output_file = debug_folder / "1150_enhanced_corrected.pdf"
    
    # Check if debug folder and PDF exist
    if not debug_folder.exists():
        print(f"❌ Debug folder not found: {debug_folder}")
        return False
    
    if not pdf_file.exists():
        print(f"❌ Debug PDF not found: {pdf_file}")
        return False
    
    print(f"✅ Debug folder: {debug_folder}")
    print(f"✅ Debug PDF: {pdf_file}")
    
    try:
        from src.utils.enhanced_pdf_creator import EnhancedPDFCreator
        
        # Use existing detection and text extraction files
        detection_file = Path("D:/MarMe/github/0.4/plc-data/processed/detdiagrams/1150_detections.json")
        text_file = Path("D:/MarMe/github/0.4/plc-data/processed/text_extraction/1150_text_extraction.json")
        
        print(f"✅ Detection file: {detection_file}")
        print(f"✅ Text extraction file: {text_file}")
        
        # Create enhanced PDF with corrected coordinate transformation
        print(f"\nCreating enhanced PDF with corrected snippet-based coordinate transformation...")
        
        creator = EnhancedPDFCreator(
            detection_confidence_threshold=0.8,
            text_confidence_threshold=0.5
        )
        
        enhanced_pdf = creator.create_enhanced_pdf(
            detection_file=detection_file,
            text_extraction_file=text_file,
            pdf_file=pdf_file,
            output_file=output_file,
            version='short'
        )
        
        print(f"\n✅ Enhanced PDF created: {enhanced_pdf}")
        print(f"  File size: {enhanced_pdf.stat().st_size} bytes")
        
        # Check PDF properties
        try:
            import fitz
            doc = fitz.open(str(enhanced_pdf))
            page = doc[0]
            print(f"\nCorrected Enhanced PDF properties:")
            print(f"  MediaBox: {page.mediabox}")
            print(f"  Rotation: {page.rotation}°")
            print(f"  Rect: {page.rect}")
            print(f"  Orientation: {'Landscape' if page.rect.width > page.rect.height else 'Portrait'}")
            doc.close()
        except Exception as e:
            print(f"  Could not read PDF properties: {e}")
        
        print(f"\n🎯 Expected Results:")
        print(f"  - PDF content should be upright and readable")
        print(f"  - Detection boxes should align with diagram symbols")
        print(f"  - Boxes should be properly sized and positioned")
        print(f"  - Text should be positioned near associated symbols")
        
        print(f"\n✅ Please check the PDF at: {output_file}")
        print(f"  Compare with previous versions to see the improvement!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating enhanced PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🔧 Testing Corrected Enhanced PDF Creator")
    print("This test uses the new snippet-based coordinate transformation")
    print()
    
    success = test_corrected_enhanced_pdf()
    
    if success:
        print(f"\n✅ Corrected Enhanced PDF test completed!")
        print(f"Check the 1150_enhanced_corrected.pdf file to verify the fix.")
    else:
        print(f"\n❌ Corrected Enhanced PDF test failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
