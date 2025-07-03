#!/usr/bin/env python3
"""
Test OCR worker in isolation to identify where it hangs
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_ocr_worker_direct():
    """Test OCR worker directly with minimal input"""
    print("=" * 60)
    print("Testing OCR worker directly")
    print("=" * 60)
    
    # Set environment variables
    os.environ["PLCDP_MULTI_ENV"] = "1"
    os.environ["PLCDP_VERBOSE"] = "1"
    
    # Create minimal test input
    test_input = {
        "action": "extract_text",
        "detection_file": "test_detection.json",
        "output_dir": "test_output",
        "language": "en",
        "confidence_threshold": 0.7,
        "config": {}
    }
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
        json.dump(test_input, input_file, indent=2)
        input_path = input_file.name
    
    output_path = input_path.replace('.json', '_output.json')
    
    try:
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        print("Running OCR worker...")
        
        # Get OCR environment python
        ocr_env_python = project_root / "environments" / "ocr_env" / "Scripts" / "python.exe"
        if not ocr_env_python.exists():
            print(f"ERROR: OCR environment not found: {ocr_env_python}")
            return False
        
        # Run OCR worker with timeout
        worker_script = project_root / "src" / "workers" / "ocr_worker.py"
        
        print(f"Command: {ocr_env_python} {worker_script} --input {input_path} --output {output_path}")
        
        # Run with 60 second timeout
        result = subprocess.run(
            [str(ocr_env_python), str(worker_script), "--input", input_path, "--output", output_path],
            timeout=60,
            capture_output=True,
            text=True
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode == 0:
            print("SUCCESS: OCR worker completed successfully")
            
            # Check output
            if Path(output_path).exists():
                with open(output_path, 'r') as f:
                    output_data = json.load(f)
                print(f"Output: {output_data}")
            
            return True
        else:
            print("ERROR: OCR worker failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: OCR worker timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"ERROR: Error running OCR worker: {e}")
        return False
    finally:
        # Cleanup
        try:
            if Path(input_path).exists():
                Path(input_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()
        except:
            pass

def test_paddleocr_import():
    """Test PaddleOCR import in OCR environment"""
    print("\n" + "=" * 60)
    print("Testing PaddleOCR import in OCR environment")
    print("=" * 60)
    
    # Get OCR environment python
    ocr_env_python = project_root / "environments" / "ocr_env" / "Scripts" / "python.exe"
    if not ocr_env_python.exists():
        print(f"ERROR: OCR environment not found: {ocr_env_python}")
        return False
    
    # Test import
    test_script = """
import sys
print(f"Python: {sys.executable}")
print("Importing PaddleOCR...")

try:
    from paddleocr import PaddleOCR
    print("SUCCESS: PaddleOCR imported successfully")
    
    print("Creating PaddleOCR instance...")
    ocr = PaddleOCR(
        lang='en',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    print("SUCCESS: PaddleOCR instance created successfully")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
"""
    
    try:
        print("Running PaddleOCR import test...")
        
        result = subprocess.run(
            [str(ocr_env_python), "-c", test_script],
            timeout=120,  # 2 minute timeout
            capture_output=True,
            text=True
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode == 0:
            print("SUCCESS: PaddleOCR import test passed")
            return True
        else:
            print("ERROR: PaddleOCR import test failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: PaddleOCR import test timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"ERROR: Error running import test: {e}")
        return False

if __name__ == "__main__":
    print("Testing OCR worker in isolation")
    
    # Test 1: PaddleOCR import
    import_success = test_paddleocr_import()
    
    # Test 2: OCR worker direct
    if import_success:
        worker_success = test_ocr_worker_direct()
    else:
        print("Skipping worker test due to import failure")
        worker_success = False
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"PaddleOCR import: {'SUCCESS' if import_success else 'ERROR'}")
    print(f"OCR worker: {'SUCCESS' if worker_success else 'ERROR'}")
    
    if not import_success:
        print("\nDIAGNOSIS: PaddleOCR import/initialization is hanging")
        print("This suggests the issue is with PaddleOCR 3.0 initialization, not our code")
    elif not worker_success:
        print("\nDIAGNOSIS: OCR worker logic has issues")
    else:
        print("\nSUCCESS: OCR worker works in isolation - issue is elsewhere")
    
    print("=" * 60)
