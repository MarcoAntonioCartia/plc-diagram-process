#!/usr/bin/env python3
"""
Test the complete pipeline with the fixed OCR stage
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def test_complete_pipeline():
    """Test the complete pipeline including the fixed OCR stage"""
    print("Testing complete pipeline with fixed OCR...")
    print("=" * 60)
    
    # Set multi-environment mode
    os.environ["PLCDP_MULTI_ENV"] = "1"
    
    try:
        from src.pipeline.stage_manager import StageManager
        
        # Initialize stage manager
        stage_manager = StageManager()
        
        print("✓ Stage manager initialized")
        
        # Get pipeline status
        status = stage_manager.get_pipeline_status()
        print(f"Total stages: {status['total_stages']}")
        print(f"Completed stages: {status['completed_stages']}")
        
        # Run only the OCR stage (assuming detection is already completed)
        print("\nRunning OCR stage...")
        
        config = {
            'ocr': {
                'ocr_confidence_threshold': 0.5,
                'ocr_language': 'en',
                'ocr_device': None  # Auto-detect
            }
        }
        
        # Run OCR stage specifically
        result = stage_manager.run_stages(
            stage_names=['ocr'],
            config=config,
            skip_completed=False,
            force_stages=['ocr']
        )
        
        print(f"\nPipeline execution result:")
        print(f"Success: {result['success']}")
        print(f"Stages run: {result['stages_run']}")
        print(f"Duration: {result['total_duration']:.2f}s")
        
        if result['success']:
            ocr_result = result['results'].get('ocr', {})
            if ocr_result:
                # OCR stage data is nested under 'data' key
                ocr_data = ocr_result.get('data', ocr_result)
                print(f"OCR stage results:")
                print(f"  Status: {ocr_data.get('status', 'unknown')}")
                print(f"  Files processed: {ocr_data.get('files_processed', 0)}")
                print(f"  Successful files: {ocr_data.get('successful_files', 0)}")
                print(f"  Total text regions: {ocr_data.get('total_text_regions', 0)}")
                print(f"  Environment: {ocr_data.get('environment', 'unknown')}")
        else:
            print(f"Pipeline failed: {result.get('error', 'Unknown error')}")
        
        return result['success']
        
    except Exception as e:
        print(f"✗ Error testing complete pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    
    if success:
        print("\n✓ Complete pipeline test passed!")
    else:
        print("\n✗ Complete pipeline test failed!")
        sys.exit(1)
