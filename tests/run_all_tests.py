#!/usr/bin/env python3
"""
Run all tests in the tests directory
"""

import sys
import subprocess
from pathlib import Path

def run_test(test_file: Path, description: str) -> bool:
    """Run a single test file and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {test_file.name}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"\nV {description} - PASSED")
            return True
        else:
            print(f"\nX {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\nX {description} - ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("PLC Diagram Processor - Test Suite")
    print("="*60)
    
    # Get tests directory
    tests_dir = Path(__file__).parent
    
    # Define tests to run
    tests = [
        ("validate_setup.py", "Setup Validation"),
        ("test_network_drive.py", "Network Drive Connectivity"),
        ("test_wsl_poppler.py", "WSL Poppler Integration"),
        ("test_pipeline.py", "Pipeline Structure Validation"),
    ]
    
    # Track results
    results = []
    
    # Run each test
    for test_file, description in tests:
        test_path = tests_dir / test_file
        if test_path.exists():
            success = run_test(test_path, description)
            results.append((description, success))
        else:
            print(f"\nX {description} - NOT FOUND: {test_file}")
            results.append((description, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{description:<40}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nV All tests passed!")
        return 0
    else:
        print(f"\nX {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
