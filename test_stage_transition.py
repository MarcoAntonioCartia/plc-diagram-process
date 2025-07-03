#!/usr/bin/env python3
"""
Test stage transition to identify where the hang occurs
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_detection_only():
    """Test running only detection stage"""
    print("=" * 60)
    print("Testing DETECTION stage only")
    print("=" * 60)
    
    # Set environment variables
    os.environ["PLCDP_MULTI_ENV"] = "1"
    os.environ["PLCDP_VERBOSE"] = "1"
    
    try:
        from src.pipeline.stage_manager import StageManager
        
        manager = StageManager(project_root=project_root)
        
        print("Running detection stage only...")
        
        # Run only detection stage
        summary = manager.run_stages(
            stage_names=['detection'],
            config={},
            skip_completed=False,  # Force re-run
            force_stages=['detection']
        )
        
        print(f"Detection stage result: {summary['success']}")
        if summary['success']:
            print("✅ Detection stage completed successfully")
        else:
            print("❌ Detection stage failed")
            print(f"Error: {summary.get('error', 'Unknown error')}")
        
        return summary['success']
        
    except Exception as e:
        print(f"❌ Error testing detection stage: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detection_to_ocr():
    """Test running detection then OCR to see where it hangs"""
    print("\n" + "=" * 60)
    print("Testing DETECTION → OCR transition")
    print("=" * 60)
    
    # Set environment variables
    os.environ["PLCDP_MULTI_ENV"] = "1"
    os.environ["PLCDP_VERBOSE"] = "1"
    
    try:
        from src.pipeline.stage_manager import StageManager
        
        manager = StageManager(project_root=project_root)
        
        print("Running detection then OCR stages...")
        print("This is where the hang likely occurs...")
        print("DEBUG: About to call run_stages...")
        
        # Run detection and OCR stages
        summary = manager.run_stages(
            stage_names=['detection', 'ocr'],
            config={},
            skip_completed=True,  # Skip detection if already completed
            force_stages=[]
        )
        
        print("DEBUG: run_stages returned successfully")
        print(f"Pipeline result: {summary['success']}")
        if summary['success']:
            print("SUCCESS: Both stages completed successfully")
        else:
            print("ERROR: Pipeline failed")
            print(f"Error: {summary.get('error', 'Unknown error')}")
        
        return summary['success']
        
    except Exception as e:
        print(f"ERROR: Error testing stage transition: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ocr_stage_initialization():
    """Test OCR stage initialization separately"""
    print("\n" + "=" * 60)
    print("Testing OCR stage initialization")
    print("=" * 60)
    
    try:
        from src.pipeline.stages.ocr_stage import OcrStage
        
        print("Creating OCR stage...")
        ocr_stage = OcrStage('ocr', 'Test OCR stage', 'ocr_env', ['detection'])
        
        print("Setting up OCR stage...")
        state_dir = project_root / ".pipeline_state"
        state_dir.mkdir(exist_ok=True)
        
        ocr_stage.setup({}, state_dir)
        
        print("✅ OCR stage initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing OCR stage: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing stage transitions to identify hang location")
    
    # Test 1: Detection only (should work)
    detection_success = test_detection_only()
    
    # Test 2: OCR initialization (might fail)
    ocr_init_success = test_ocr_stage_initialization()
    
    # Test 3: Detection to OCR transition (likely hangs)
    if detection_success and ocr_init_success:
        print("\nBoth detection and OCR initialization work, testing transition...")
        transition_success = test_detection_to_ocr()
    else:
        print("\nSkipping transition test due to previous failures")
        transition_success = False
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Detection stage only: {'✅' if detection_success else '❌'}")
    print(f"OCR initialization: {'✅' if ocr_init_success else '❌'}")
    print(f"Stage transition: {'✅' if transition_success else '❌'}")
    
    if detection_success and not ocr_init_success:
        print("\n🔍 DIAGNOSIS: OCR stage initialization is the problem")
    elif detection_success and ocr_init_success and not transition_success:
        print("\n🔍 DIAGNOSIS: Stage transition logic is the problem")
    elif not detection_success:
        print("\n🔍 DIAGNOSIS: Detection stage itself is the problem")
    
    print("=" * 60)
