#!/usr/bin/env python3
"""
Test script to verify the pipeline fixes
Tests the modular pipeline data flow without running actual detection/OCR
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_detection_worker_output_format():
    """Test that detection worker returns structured data"""
    print("Testing detection worker output format...")
    
    # Create a temporary input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_input = {
            "pdf_path": str(project_root / "test_file.pdf"),  # Non-existent file for testing
            "output_dir": str(project_root / "test_output"),
            "confidence_threshold": 0.25
        }
        json.dump(test_input, f)
        input_file = f.name
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_file = f.name
    
    try:
        # Import and test the detection worker
        from src.workers.detection_worker import main as detection_main
        
        # Mock sys.argv for the worker
        original_argv = sys.argv
        sys.argv = ['detection_worker.py', '--input', input_file, '--output', output_file]
        
        # This should fail gracefully and create structured output
        try:
            detection_main()
        except SystemExit:
            pass  # Expected due to file not existing
        
        # Check the output format
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                result = json.load(f)
            
            print(f"✓ Detection worker output format: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'error':
                print("  Expected error due to non-existent test file")
                return True
            elif result.get('status') == 'success':
                results = result.get('results', {})
                if isinstance(results, dict):
                    print("  ✓ Structured results format detected")
                    print(f"    Keys: {list(results.keys())}")
                    return True
                else:
                    print("  ✗ Results not in structured format")
                    return False
        else:
            print("  ✗ No output file created")
            return False
            
    except Exception as e:
        print(f"  ✗ Error testing detection worker: {e}")
        return False
    finally:
        # Cleanup
        sys.argv = original_argv
        for temp_file in [input_file, output_file]:
            try:
                Path(temp_file).unlink()
            except:
                pass

def test_stage_state_management():
    """Test stage state management and dependency checking"""
    print("\nTesting stage state management...")
    
    try:
        from src.pipeline.stage_manager import StageManager
        from src.pipeline.base_stage import StageResult
        
        # Create a temporary state directory
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir)
            
            # Create mock detection stage state
            detection_state = StageResult(
                success=True,
                data={
                    'output_directory': str(project_root / 'test_output'),
                    'detection_files_created': ['test1_detections.json', 'test2_detections.json'],
                    'total_detections': 42,
                    'files_processed': 2
                }
            )
            
            # Save mock detection state
            detection_state_file = state_dir / "detection_state.json"
            with open(detection_state_file, 'w') as f:
                json.dump(detection_state.to_dict(), f, indent=2)
            
            print("  ✓ Created mock detection state")
            
            # Test OCR stage dependency checking
            from src.pipeline.stages.ocr_stage import OcrStage
            
            ocr_stage = OcrStage()
            ocr_stage.setup({}, state_dir)
            
            # Test dependency state reading
            dep_state = ocr_stage.get_dependency_state('detection')
            if dep_state and dep_state.success:
                print("  ✓ OCR stage can read detection state")
                print(f"    Detection output dir: {dep_state.data.get('output_directory')}")
                print(f"    Detection files: {len(dep_state.data.get('detection_files_created', []))}")
                return True
            else:
                print("  ✗ OCR stage cannot read detection state")
                return False
                
    except Exception as e:
        print(f"  ✗ Error testing stage state management: {e}")
        return False

def test_stage_manager_logging():
    """Test stage manager logging and execution flow"""
    print("\nTesting stage manager logging...")
    
    try:
        from src.pipeline.stage_manager import StageManager
        
        # Create stage manager
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir)
            manager = StageManager(project_root, state_dir)
            
            # List available stages
            stages = manager.list_stages()
            print(f"  ✓ Found {len(stages)} registered stages:")
            for stage in stages:
                print(f"    - {stage['name']} ({stage['environment']})")
            
            # Test execution order
            print(f"  ✓ Execution order: {manager.execution_order}")
            
            return True
            
    except Exception as e:
        print(f"  ✗ Error testing stage manager: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Pipeline Fixes")
    print("=" * 60)
    
    tests = [
        test_detection_worker_output_format,
        test_stage_state_management,
        test_stage_manager_logging
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Pipeline fixes should work correctly.")
        print("\nNext steps:")
        print("1. Run the pipeline with: python src/run_pipeline.py --run-all")
        print("2. Check for improved logging and stage transitions")
        print("3. Verify OCR stage starts after detection completes")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())
