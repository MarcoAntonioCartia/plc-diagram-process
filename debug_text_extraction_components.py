"""
Debug script to test individual components of text extraction
"""
import json
import fitz
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_pdf_text_extraction():
    """Test basic PDF text extraction without any pipeline"""
    print("=" * 50)
    print("TESTING: Basic PDF Text Extraction")
    print("=" * 50)
    
    # Find a test PDF
    pdf_folder = Path("D:/MarMe/github/0.3/plc-data/raw")
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in raw folder")
        return False
    
    test_pdf = pdf_files[0]
    print(f"Testing with: {test_pdf.name}")
    
    try:
        doc = fitz.open(str(test_pdf))
        total_text_blocks = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            page_blocks = 0
            for block in text_dict["blocks"]:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text and len(text) > 0:
                            page_blocks += 1
                            total_text_blocks += 1
                            if page_blocks <= 3:  # Show first 3 examples
                                print(f"  Found text: '{text}' at {span['bbox']}")
            
            print(f"Page {page_num + 1}: {page_blocks} text blocks")
        
        doc.close()
        print(f"✓ Total text blocks found: {total_text_blocks}")
        return total_text_blocks > 0
        
    except Exception as e:
        print(f"❌ PDF text extraction failed: {e}")
        return False

def test_detection_file_loading():
    """Test loading detection files"""
    print("\n" + "=" * 50)
    print("TESTING: Detection File Loading")
    print("=" * 50)
    
    detection_folder = Path("D:/MarMe/github/0.3/plc-data/processed/detdiagrams2")
    detection_files = list(detection_folder.glob("*_detections.json"))
    
    if not detection_files:
        print("❌ No detection files found")
        return False
    
    test_file = detection_files[0]
    print(f"Testing with: {test_file.name}")
    
    try:
        with open(test_file, 'r') as f:
            detection_data = json.load(f)
        
        total_detections = 0
        for page_data in detection_data.get("pages", []):
            page_detections = len(page_data.get("detections", []))
            total_detections += page_detections
            print(f"Page {page_data.get('page', '?')}: {page_detections} detections")
        
        print(f"✓ Total detections found: {total_detections}")
        return total_detections > 0
        
    except Exception as e:
        print(f"❌ Detection file loading failed: {e}")
        return False

def test_ocr_initialization():
    """Test OCR initialization in isolation"""
    print("\n" + "=" * 50)
    print("TESTING: OCR Initialization")
    print("=" * 50)
    
    try:
        from paddleocr import PaddleOCR
        
        # Try the corrected initialization methods
        print("Attempting PaddleOCR with device='cpu'...")
        try:
            ocr = PaddleOCR(lang='en', device='cpu', use_textline_orientation=True)
            print("✓ PaddleOCR initialized successfully with device='cpu'!")
            return True
        except Exception as e:
            print(f"Device parameter failed: {e}")
            
        print("Attempting PaddleOCR without device specification...")
        try:
            ocr = PaddleOCR(lang='en', use_textline_orientation=True)
            print("✓ PaddleOCR initialized successfully without device!")
            return True
        except Exception as e:
            print(f"No device specification failed: {e}")
            
        print("Attempting PaddleOCR with minimal parameters...")
        try:
            ocr = PaddleOCR(lang='en')
            print("✓ PaddleOCR initialized successfully with minimal parameters!")
            return True
        except Exception as e:
            print(f"Minimal parameters failed: {e}")
            
        return False
        
    except Exception as e:
        print(f"❌ PaddleOCR import failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🔍 DIAGNOSTIC: Text Extraction Components")
    print("Testing each component individually...\n")
    
    pdf_ok = test_pdf_text_extraction()
    detection_ok = test_detection_file_loading()
    ocr_ok = test_ocr_initialization()
    
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"PDF Text Extraction: {'✓ PASS' if pdf_ok else '❌ FAIL'}")
    print(f"Detection File Loading: {'✓ PASS' if detection_ok else '❌ FAIL'}")
    print(f"OCR Initialization: {'✓ PASS' if ocr_ok else '❌ FAIL'}")
    
    if pdf_ok and detection_ok:
        print("\n🎯 NEXT STEP: The basic components work. The issue is in the pipeline integration.")
    else:
        print("\n🚨 CRITICAL: Basic components are failing. Fix these first.")

if __name__ == "__main__":
    main() 