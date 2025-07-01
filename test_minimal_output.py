#!/usr/bin/env python3
"""
Test script to demonstrate the minimal output mode
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_minimal_progress_display():
    """Test the minimal progress display functionality"""
    print("Testing minimal progress display...")
    
    # Set minimal mode
    os.environ["PLCDP_MINIMAL_OUTPUT"] = "1"
    
    from src.utils.progress_display import create_stage_progress
    
    # Test training stage progress
    progress = create_stage_progress("training")
    
    progress.start_stage("Checking for existing models...")
    time.sleep(1)
    
    progress.update_progress("Validating dataset structure")
    time.sleep(1)
    
    progress.update_progress("Starting model training")
    time.sleep(1)
    
    progress.complete_stage("Model training completed")
    
    print("\nMinimal progress display test completed")

def test_normal_progress_display():
    """Test the normal progress display functionality"""
    print("\nTesting normal progress display...")
    
    # Disable minimal mode
    os.environ["PLCDP_MINIMAL_OUTPUT"] = "0"
    
    from src.utils.progress_display import create_stage_progress
    
    # Test training stage progress
    progress = create_stage_progress("training")
    
    progress.start_stage("Checking for existing models...")
    time.sleep(1)
    
    progress.start_file("model1.pt")
    progress.update_progress("Validating model")
    time.sleep(1)
    progress.complete_file("model1.pt", "validation successful")
    
    progress.start_file("dataset.yaml")
    progress.update_progress("Checking dataset structure")
    time.sleep(1)
    progress.complete_file("dataset.yaml", "structure valid")
    
    progress.complete_stage("All validations completed")
    
    print("\nNormal progress display test completed")

if __name__ == "__main__":
    print("=== Progress Display Mode Comparison ===")
    
    print("\n1. MINIMAL MODE (single-line updates):")
    test_minimal_progress_display()
    
    print("\n" + "="*50)
    
    print("\n2. NORMAL MODE (detailed progress):")
    test_normal_progress_display()
    
    print("\n=== Test completed ===")
    print("\nTo use minimal mode with the pipeline:")
    print("python src/run_pipeline.py --run-all --minimal")
    print("or")
    print("python src/run_pipeline.py --run-all --quiet")
