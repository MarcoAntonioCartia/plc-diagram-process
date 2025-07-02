#!/usr/bin/env python3
"""
Test detection stage with single PDF
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_detection_single_pdf():
    """Test detection with single PDF"""
    
    # Set multi-env mode
    os.environ["PLCDP_MULTI_ENV"] = "1"
    
    print("Testing detection stage with single PDF...")
    
    try:
        from src.utils.multi_env_manager import MultiEnvironmentManager
        from src.config import get_config
        
        config = get_config()
        env_manager = MultiEnvironmentManager(project_root)
        
        # Test with single PDF
        test_pdf_dir = Path("D:/MarMe/github/0.4/plc-data/raw/pdfs_test")
        output_dir = Path("D:/MarMe/github/0.4/plc-data/processed/detdiagrams_test")
        output_dir.mkdir(exist_ok=True)
        
        print(f"Input directory: {test_pdf_dir}")
        print(f"Output directory: {output_dir}")
        
        # Check if test PDF exists
        pdf_files = list(test_pdf_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files: {[f.name for f in pdf_files]}")
        
        if not pdf_files:
            print("No PDF files found in test directory")
            return False
        
        # Prepare detection payload
        detection_payload = {
            'action': 'detect',
            'pdf_folder': str(test_pdf_dir),
            'output_dir': str(output_dir),
            'config': {}
        }
        
        print("Starting detection worker...")
        print(f"Payload: {detection_payload}")
        
        # Run detection worker
        result = env_manager.run_detection_pipeline(detection_payload)
        
        print(f"Worker result: {result}")
        
        if result.get('status') == 'success':
            print("✓ Detection worker completed successfully")
            
            detection_data = result.get('results', {})
            if isinstance(detection_data, dict):
                total_detections = detection_data.get('total_detections', 0)
                output_directory = detection_data.get('output_directory', '')
                detection_files = detection_data.get('detection_files_created', [])
                
                print(f"  Total detections: {total_detections}")
                print(f"  Output directory: {output_directory}")
                print(f"  Detection files: {len(detection_files)}")
                
                return True
            else:
                print(f"✗ Unexpected result format: {type(detection_data)}")
                return False
        else:
            print(f"✗ Detection worker failed: {result}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_detection_single_pdf()
    if success:
        print("\n✓ Single PDF detection test passed")
    else:
        print("\n✗ Single PDF detection test failed")
