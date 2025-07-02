#!/usr/bin/env python3
"""
Create detection state file for existing detection results
This allows the OCR stage to run on already processed detection data
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.pipeline.base_stage import StageResult

def create_detection_state():
    """Create detection state file from existing detection results"""
    
    # Paths
    detection_dir = Path("D:/MarMe/github/0.4/plc-data/processed/detdiagrams")
    state_dir = Path(".pipeline_state")
    state_dir.mkdir(exist_ok=True)
    
    # Find all detection files
    detection_files = list(detection_dir.glob("*_detections.json"))
    
    if not detection_files:
        print("No detection files found!")
        return False
    
    print(f"Found {len(detection_files)} detection files")
    
    # Count total detections
    total_detections = 0
    processed_files = 0
    
    for det_file in detection_files:
        try:
            with open(det_file, 'r') as f:
                det_data = json.load(f)
                # Count detections across all pages
                for page in det_data.get("pages", []):
                    total_detections += len(page.get("detections", []))
                processed_files += 1
        except Exception as e:
            print(f"Warning: Could not read {det_file}: {e}")
    
    print(f"Total detections: {total_detections}")
    print(f"Processed files: {processed_files}")
    
    # Create detection stage result
    detection_result = StageResult(
        success=True,
        data={
            'status': 'success',
            'environment': 'multi',
            'files_processed': processed_files,
            'successful_files': processed_files,
            'total_detections': total_detections,
            'output_directory': str(detection_dir),
            'detection_files_created': [str(f) for f in detection_files],
            'results': [
                {
                    'pdf_file': f.name.replace('_detections.json', '.pdf'),
                    'success': True,
                    'detections': 0,  # We could count per file but not critical
                    'output_directory': str(detection_dir),
                    'detection_files': [str(f)]
                }
                for f in detection_files
            ]
        }
    )
    
    # Save detection state
    detection_state_file = state_dir / "detection_state.json"
    with open(detection_state_file, 'w') as f:
        json.dump(detection_result.to_dict(), f, indent=2)
    
    print(f"✓ Created detection state file: {detection_state_file}")
    print(f"✓ OCR stage can now run on existing detection data")
    
    return True

if __name__ == "__main__":
    success = create_detection_state()
    if success:
        print("\nNow you can run:")
        print("python launch.py src/run_pipeline.py --stages ocr --verbose")
    else:
        print("Failed to create detection state")
        sys.exit(1)
