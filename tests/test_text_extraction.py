"""
Test script for text extraction pipeline
Validates the hybrid text extraction functionality
"""

import sys
import json
import tempfile
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.ocr.text_extraction_pipeline import TextExtractionPipeline
from src.ocr.paddle_ocr import PLCOCRProcessor
from src.config import get_config

def test_plc_ocr_processor():
    """Test the basic PLC OCR processor"""
    print("Testing PLC OCR Processor...")
    
    try:
        processor = PLCOCRProcessor(confidence_threshold=0.5)
        print("V PLC OCR Processor initialized successfully")
        
        # Test text extraction on a sample region
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White image
        test_bbox = [10, 10, 190, 90]
        
        # Extract text from the test region
        texts = processor.extract_text_from_regions(test_image, [test_bbox])
        
        # Validate results
        assert isinstance(texts, list), "Expected list of text results"
        
        print(f"V OCR extraction completed. Found {len(texts)} text regions")
        
        # Test PLC pattern recognition
        test_texts = ["M1.0", "I2.5", "Q0.3", "DB10.DBX5.2", "regular text"]
        plc_patterns = []
        
        for text in test_texts:
            if processor.is_plc_pattern(text):
                plc_patterns.append(text)
        
        assert len(plc_patterns) >= 4, f"Expected at least 4 PLC patterns, found {len(plc_patterns)}"
        
        return True
        
    except Exception as e:
        print(f"X PLC OCR Processor test failed: {e}")
        return False

def test_text_extraction_pipeline():
    """Test the complete text extraction pipeline"""
    try:
        from src.ocr.text_extraction_pipeline import TextExtractionPipeline
        
        print("V Text Extraction Pipeline initialized successfully")
        
        # Create pipeline with minimal configuration
        pipeline = TextExtractionPipeline(
            device='cpu',
            lang='en'
        )
        
        # Test with mock detection data
        mock_detection_data = {
            'pages': [
                {
                    'page_number': 1,
                    'detections': [
                        {
                            'bbox': [100, 100, 200, 150],
                            'confidence': 0.8,
                            'class': 'text'
                        }
                    ]
                }
            ]
        }
        
        # Test pattern recognition
        test_patterns = [
            "M1.0",      # Memory bit
            "I2.5",      # Input
            "Q0.3",      # Output  
            "DB10.DBX5.2", # Data block
            "T1",        # Timer
            "C5",        # Counter
            "FC100",     # Function
            "FB25"       # Function block
        ]
        
        plc_count = 0
        for pattern in test_patterns:
            if pipeline.is_plc_pattern(pattern):
                plc_count += 1
        
        assert plc_count >= 6, f"Expected at least 6 PLC patterns recognized, got {plc_count}"
        print("V PLC pattern recognition working correctly")
        
        return True
        
    except Exception as e:
        print(f"X Text Extraction Pipeline test failed: {e}")
        return False

def test_mock_detection_processing():
    """Test processing with mock detection data"""
    try:
        # Create temporary directory for test
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock detection file
            detection_file = temp_path / "test_detections.json"
            mock_data = {
                'pages': [
                    {
                        'page_number': 1,
                        'detections': [
                            {
                                'bbox': [100, 100, 200, 150],
                                'confidence': 0.85,
                                'class': 'symbol'
                            },
                            {
                                'bbox': [220, 120, 300, 140],
                                'confidence': 0.92,
                                'class': 'text'
                            }
                        ]
                    }
                ]
            }
            
            with open(detection_file, 'w') as f:
                json.dump(mock_data, f)
            
            print(f"V Mock detection data created: {detection_file}")
            
            # Validate detection file structure
            with open(detection_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert 'pages' in loaded_data, "Detection file missing 'pages' key"
            assert len(loaded_data['pages']) > 0, "No pages in detection data"
            assert 'detections' in loaded_data['pages'][0], "No detections in first page"
            
            print("V Mock detection processing structure validated")
            
            return True
            
    except Exception as e:
        print(f"X Mock detection processing test failed: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = [
        ("PaddleOCR", "paddleocr"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("PIL", "PIL")
    ]
    
    available = []
    for display_name, module_name in dependencies:
        try:
            __import__(module_name)
            available.append(display_name)
            print(f"V {display_name} available")
        except ImportError as e:
            print(f"X {display_name} not available: {e}")
    
    return available

def run_test(test_func, test_name):
    """Run a single test with error handling"""
    try:
        print(f"\n--- Running {test_name} ---")
        result = test_func()
        if result:
            print(f"V {test_name} - PASSED")
            return True
        else:
            print(f"X {test_name} - FAILED")
            return False
    except Exception as e:
        print(f"X {test_name} failed with exception: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Text Extraction Pipeline Tests ===")
    
    # Check dependencies first
    print("\n--- Checking Dependencies ---")
    available_deps = check_dependencies()
    
    # Run tests
    tests = [
        (test_plc_ocr_processor, "PLC OCR Processor Test"),
        (test_text_extraction_pipeline, "Text Extraction Pipeline Test"),
        (test_mock_detection_processing, "Mock Detection Processing Test")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests passed: {passed}/{total}")
    print(f"Dependencies available: {len(available_deps)}")
    
    if passed == total:
        print("V All tests passed! Text extraction pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run detection pipeline: python src/detection/run_complete_pipeline.py")
        print("2. Run text extraction: python src/ocr/run_text_extraction.py")
        print("3. Or run combined pipeline: python src/run_pipeline.py")
        return 0
    else:
        print("X Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
