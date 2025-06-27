"""
Text Extraction Pipeline Usage Example
Demonstrates how to use the PLC text extraction pipeline
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def example_basic_usage():
    """Basic usage example for text extraction pipeline"""
    print("=== Basic Text Extraction Usage ===")
    
    try:
        from src.ocr.text_extraction_pipeline import TextExtractionPipeline
        from src.config import get_config
        
        # Initialize the pipeline
        pipeline = TextExtractionPipeline(
            confidence_threshold=0.7,  # OCR confidence threshold
            ocr_lang="en"             # Language for OCR
        )
        
        print("✓ Text extraction pipeline initialized")
        
        # Get configuration
        config = get_config()
        data_root = Path(config.config['data_root'])
        
        # Set up paths
        detection_folder = data_root / "processed" / "detdiagrams"
        pdf_folder = data_root / "raw" / "pdfs"
        output_folder = data_root / "processed" / "text_extraction"
        
        print(f"Detection folder: {detection_folder}")
        print(f"PDF folder: {pdf_folder}")
        print(f"Output folder: {output_folder}")
        
        # Check if detection results exist
        if detection_folder.exists():
            detection_files = list(detection_folder.glob("*_detections.json"))
            print(f"Found {len(detection_files)} detection files")
            
            if detection_files and pdf_folder.exists():
                print("\nReady to run text extraction!")
                print("Run: python src/ocr/run_text_extraction.py")
            else:
                print("\nNo detection files found. Run detection pipeline first:")
                print("python src/detection/run_complete_pipeline.py")
        else:
            print("\nDetection folder not found. Run detection pipeline first:")
            print("python src/detection/run_complete_pipeline.py")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install missing dependencies:")
        print("pip install opencv-python PyMuPDF paddleocr")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def example_plc_pattern_recognition():
    """Example of PLC pattern recognition"""
    print("\n=== PLC Pattern Recognition Example ===")
    
    try:
        from src.ocr.text_extraction_pipeline import TextExtractionPipeline, TextRegion
        
        pipeline = TextExtractionPipeline()
        
        # Test various PLC text patterns
        test_texts = [
            "I0.1",           # Input address
            "Q2.3",           # Output address  
            "M1.5",           # Memory address
            "T10",            # Timer
            "C5",             # Counter
            "FB12",           # Function block
            "DB3",            # Data block
            "AI1",            # Analog input
            "AO2",            # Analog output
            "MOTOR_START",    # Variable name
            "VALVE_OPEN",     # Variable name
            "123.45",         # Numeric value
            "SENSOR_1",       # Label
            "random_text",    # Non-PLC text
        ]
        
        print("Testing PLC pattern recognition:")
        print("-" * 40)
        
        for text in test_texts:
            # Create mock text region
            text_region = TextRegion(
                text=text,
                confidence=0.9,
                bbox=(0, 0, 100, 20),
                source="test",
                page=1
            )
            
            # Apply pattern filtering
            results = pipeline._apply_plc_pattern_filtering([text_region])
            
            if results:
                result = results[0]
                patterns = [p["pattern"] for p in result["matched_patterns"]]
                score = result["relevance_score"]
                
                if patterns:
                    print(f"'{text:12}' -> {patterns} (score: {score})")
                else:
                    print(f"'{text:12}' -> No patterns matched (score: {score})")
        
        return True
        
    except Exception as e:
        print(f"✗ Pattern recognition test failed: {e}")
        return False

def example_ocr_processor():
    """Example of using the OCR processor directly"""
    print("\n=== OCR Processor Example ===")
    
    try:
        from src.ocr.paddle_ocr import PLCOCRProcessor
        from src.config import get_config
        
        # Initialize OCR processor
        processor = PLCOCRProcessor(confidence_threshold=0.6)
        print("✓ OCR processor initialized")
        
        # Try to find test images
        config = get_config()
        test_images_dir = Path(config.config['data_root']) / "datasets" / "test" / "images"
        
        if test_images_dir.exists():
            image_files = list(test_images_dir.glob("*.jpg"))[:2]  # Test with first 2 images
            
            if image_files:
                print(f"Testing with {len(image_files)} images:")
                
                for i, image_file in enumerate(image_files):
                    print(f"\nImage {i+1}: {image_file.name}")
                    
                    texts = processor.extract_text(str(image_file))
                    print(f"Found {len(texts)} text regions")
                    
                    # Show first few results
                    for j, text in enumerate(texts[:3]):
                        print(f"  {j+1}. '{text['text']}' (confidence: {text['confidence']:.3f})")
                    
                    if len(texts) > 3:
                        print(f"  ... and {len(texts) - 3} more")
            else:
                print("No test images found")
        else:
            print("Test images directory not found")
            print("This is normal if you haven't set up the dataset yet")
        
        return True
        
    except Exception as e:
        print(f"✗ OCR processor test failed: {e}")
        return False

def main():
    """Run all examples"""
    print("PLC Text Extraction Pipeline Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("PLC Pattern Recognition", example_plc_pattern_recognition),
        ("OCR Processor", example_ocr_processor),
    ]
    
    results = []
    
    for name, example_func in examples:
        try:
            result = example_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Example Results:")
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} examples completed successfully")
    
    if passed == total:
        print("\n✓ All examples completed! The text extraction pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run detection pipeline: python src/detection/run_complete_pipeline.py")
        print("2. Run text extraction: python src/ocr/run_text_extraction.py")
        print("3. Or run combined: python src/run_pipeline.py")
    else:
        print("\n⚠ Some examples had issues. Check the output above for details.")
    
    return 0

if __name__ == "__main__":
    exit(main())
