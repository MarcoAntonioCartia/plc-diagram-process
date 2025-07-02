#!/usr/bin/env python3
"""Test yolo11_train.py directly in yolo_env"""

import subprocess
import sys
from pathlib import Path

def test_yolo_direct():
    """Test yolo11_train.py directly"""
    
    yolo_python = Path("environments/yolo_env/Scripts/python.exe")
    training_script = Path("src/detection/yolo11_train.py")
    
    if not yolo_python.exists():
        print(f"ERROR: YOLO environment not found at {yolo_python}")
        return
    
    if not training_script.exists():
        print(f"ERROR: Training script not found at {training_script}")
        return
    
    # Test just the help command first
    print("Testing yolo11_train.py --help...")
    cmd = [str(yolo_python), str(training_script), "--help"]
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # Short timeout for help
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out after 30 seconds")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_yolo_direct()
