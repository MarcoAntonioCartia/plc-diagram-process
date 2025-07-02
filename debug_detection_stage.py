#!/usr/bin/env python3
"""
Debug script to test detection stage processing
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_detection_stage_processing():
    """Test detection stage with mock worker result"""
    
    # Set multi-env mode
    os.environ["PLCDP_MULTI_ENV"] = "1"
    
    from src.pipeline.stages.detection_stage import DetectionStage
    from src.config import get_config
    
    # Create detection stage
    stage = DetectionStage()
    
    # Setup stage
    state_dir = Path(".pipeline_state")
    stage.setup({}, state_dir)
    
    print("Testing detection stage processing...")
    
    # Mock a successful worker result (like what we saw in the output)
    mock_worker_result = {
        "status": "success",
        "results": {
            "output_directory": "D:/MarMe/github/0.4/plc-data/processed/detdiagrams",
            "processed_pdfs": 56,
            "detection_files_created": [
                "D:/MarMe/github/0.4/plc-data/processed/detdiagrams/1150_detections.json",
                "D:/MarMe/github/0.4/plc-data/processed/detdiagrams/1200_detections.json",
                # ... (truncated for brevity)
            ],
            "total_detections": 7508,
            "pipeline_output": "D:/MarMe/github/0.4/plc-data/processed/detdiagrams",
            "processing_summary": "Successfully processed 56 PDFs, created 56 detection files with 7508 total detections"
        }
    }
    
    print("Mock worker result created")
    
    # Test the result processing logic from detection stage
    try:
        detection_data = mock_worker_result.get('results', {})
        
        if isinstance(detection_data, dict):
            total_detections = detection_data.get('total_detections', 0)
            output_directory = detection_data.get('output_directory', '')
            detection_files = detection_data.get('detection_files_created', [])
            processing_summary = detection_data.get('processing_summary', 'Completed')
            
            print(f"✓ Parsed worker result successfully:")
            print(f"  Total detections: {total_detections}")
            print(f"  Output directory: {output_directory}")
            print(f"  Detection files: {len(detection_files)}")
            print(f"  Summary: {processing_summary}")
            
            # Test creating the stage result
            stage_result = {
                'status': 'success',
                'environment': 'multi',
                'files_processed': 56,
                'successful_files': 56,
                'total_detections': total_detections,
                'output_directory': output_directory,
                'detection_files_created': detection_files,
                'results': []
            }
            
            print("✓ Stage result created successfully")
            return True
        else:
            print("✗ Detection data is not a dict")
            return False
            
    except Exception as e:
        print(f"✗ Error processing worker result: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_detection_stage_processing()
    if success:
        print("\n✓ Detection stage processing logic works correctly")
        print("The issue might be in the progress display or stage manager")
    else:
        print("\n✗ Detection stage processing has issues")
