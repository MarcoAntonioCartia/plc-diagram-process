#!/usr/bin/env python3
"""
Test script to validate CI setup functionality
"""

import sys
import subprocess
from pathlib import Path

def test_setup_imports():
    """Test that setup.py can be imported without missing dependencies"""
    try:
        # Add setup directory to path
        setup_dir = Path(__file__).resolve().parent.parent / 'setup'
        sys.path.insert(0, str(setup_dir))
        
        # Try importing the main setup module
        import setup
        print("V Setup script imports successfully")
        return True
    except ImportError as e:
        print(f"X Import error: {e}")
        return False
    except Exception as e:
        print(f"X Unexpected error: {e}")
        return False

def test_setup_dry_run():
    """Test that setup.py can run in dry-run mode"""
    try:
        setup_script = Path(__file__).resolve().parent.parent / 'setup' / 'setup.py'
        result = subprocess.run([
            sys.executable, str(setup_script), '--dry-run'
        ], capture_output=True, text=True, timeout=1800)  # 30 minutes for slower CI
        
        if result.returncode == 0:
            print("V Setup script dry-run completed successfully")
            return True
        else:
            print(f"X Setup script failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("X Setup script timed out")
        return False
    except Exception as e:
        print(f"X Error running setup script: {e}")
        return False

def test_yaml_availability():
    """Test that required YAML libraries are available"""
    try:
        import yaml
        print("V PyYAML is available")
    except ImportError:
        print("X PyYAML is not available")
        return False
    
    try:
        import ruamel.yaml
        print("V ruamel.yaml is available")
    except ImportError:
        print("X ruamel.yaml is not available")
        return False
    
    return True

if __name__ == "__main__":
    print("Running CI setup validation tests...")
    
    tests = [
        test_yaml_availability,
        test_setup_imports,
        test_setup_dry_run
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\n--- Running {test.__name__} ---")
        if test():
            passed += 1
        else:
            print(f"Test {test.__name__} failed")
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("All tests passed! CI setup should work correctly.")
        sys.exit(0)
    else:
        print("Some tests failed. CI may have issues.")
        sys.exit(1) 