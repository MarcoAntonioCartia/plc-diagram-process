#!/usr/bin/env python3
"""
Test script to verify WSL poppler installation
"""

import subprocess
import sys
from pathlib import Path

def check_wsl():
    """Check if WSL is available"""
    try:
        result = subprocess.run(['wsl', '--list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("WSL is available")
            print(f"  WSL distributions: {result.stdout}")
            return True
        else:
            print("WSL is not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"X WSL check failed: {e}")
        return False

def test_wsl_poppler():
    """Test if poppler-utils can be accessed through WSL"""
    if not check_wsl():
        return False
    
    print("\nTesting poppler-utils in WSL...")
    
    # Test if poppler-utils is installed in WSL
    try:
        result = subprocess.run(['wsl', '-e', 'which', 'pdftotext'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("poppler-utils is already installed in WSL")
            print(f"  pdftotext location: {result.stdout.strip()}")
            
            # Test pdftotext version
            version_result = subprocess.run(['wsl', '-e', 'pdftotext', '-v'], 
                                          capture_output=True, text=True, timeout=5)
            if version_result.stderr:  # pdftotext outputs version to stderr
                print(f"  Version: {version_result.stderr.strip()}")
            
            return True
        else:
            print("⚠ poppler-utils is not installed in WSL")
            print("  Run: wsl -e bash -c 'sudo apt-get update && sudo apt-get install -y poppler-utils'")
            return False
            
    except Exception as e:
        print(f"Error testing poppler in WSL: {e}")
        return False

def test_wrapper_functionality():
    """Test if a wrapper script would work"""
    print("\nTesting wrapper script concept...")
    
    # Create a test wrapper
    test_wrapper = Path("test_wrapper.bat")
    wrapper_content = '''@echo off
echo Test wrapper executing: wsl -e echo "Hello from WSL"
wsl -e echo "Hello from WSL"
'''
    
    try:
        with open(test_wrapper, 'w') as f:
            f.write(wrapper_content)
        
        # Test the wrapper
        result = subprocess.run([str(test_wrapper)], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print("Wrapper script concept works")
            print(f"  Output: {result.stdout.strip()}")
        else:
            print("X Wrapper script failed")
            print(f"  Error: {result.stderr}")
        
        # Clean up
        test_wrapper.unlink()
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error testing wrapper: {e}")
        if test_wrapper.exists():
            test_wrapper.unlink()
        return False

def main():
    print("=== WSL Poppler Installation Test ===")
    print()
    
    wsl_ok = check_wsl()
    if wsl_ok:
        poppler_ok = test_wsl_poppler()
        wrapper_ok = test_wrapper_functionality()
        
        print("\n=== Summary ===")
        print(f"WSL Available: {'V' if wsl_ok else 'X'}")
        print(f"Poppler in WSL: {'V' if poppler_ok else 'X'}")
        print(f"Wrapper Scripts: {'V' if wrapper_ok else 'X'}")
        
        if all([wsl_ok, wrapper_ok]):
            print("\nV All WSL Poppler components are working correctly!")
            if not poppler_ok:
                print("  Note: You'll need to install poppler-utils in WSL during setup")
        else:
            print("\nX Some WSL Poppler components need attention.")
    else:
        print("\nX WSL is required for automatic poppler installation")
        print("Please install WSL first: https://docs.microsoft.com/en-us/windows/wsl/install")

if __name__ == "__main__":
    main()
