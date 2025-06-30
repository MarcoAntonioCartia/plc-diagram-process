"""
GPU Environment Launcher for the PLC Diagram Processor.

This script fixes the Windows PATH environment to prioritize Paddle's
bundled CUDA libraries, preventing conflicts with system-wide installations.
It then executes the main application script.

Usage: python launch.py [arguments for main script]
Example: python launch.py --skip-detection --create-enhanced-pdf
"""

import os
import sys
import runpy
import argparse
from pathlib import Path
import platform

def configure_gpu_environment():
    """
    Forcefully configure the environment to use bundled GPU libraries from the venv.
    This is the modern, robust way for Python 3.8+ on Windows and is critical
    to prevent conflicts with system-wide CUDA installations.
    """
    if platform.system() != "Windows":
        return

    print("--- Launcher: Configuring GPU environment for Windows...")
    
    venv_path = Path(sys.executable).parent.parent
    # Known optional sub-folders vary between wheel versions (some ship only /bin)
    required_dll_paths = [
        venv_path / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        venv_path / "Lib" / "site-packages" / "nvidia" / "cuda_nvrtc" / "bin",
        # At least **one** of the runtime locations must exist (lib is optional on Windows wheels)
        venv_path / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
        venv_path / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        venv_path / "Lib" / "site-packages" / "nvidia" / "cusparse" / "bin",
        venv_path / "Lib" / "site-packages" / "nvidia" / "cusolver" / "bin",
    ]

    optional_dll_paths = [
        venv_path / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "lib",  # present on some wheels, absent on others
    ]

    # Validate the required paths
    missing_required = False
    for path in required_dll_paths:
        if not path.exists():
            print(f"--- Launcher: WARNING - Expected DLL path does not exist: {path}")
            missing_required = True

    if missing_required:
        print("--- Launcher: CRITICAL - Essential GPU library folders missing – falling back to PATH manipulation.")
        original_path = os.environ.get("PATH", "")
        search_paths = [str(p) for p in required_dll_paths + optional_dll_paths if p.exists()]
        os.environ["PATH"] = ";".join(search_paths) + ";" + original_path
        return

    dll_paths = required_dll_paths + [p for p in optional_dll_paths if p.exists()]

    print("--- Launcher: All required GPU library paths found.")
    try:
        # Use the robust os.add_dll_directory method
        for path in dll_paths:
            os.add_dll_directory(str(path))
        print("--- Launcher: DLL search paths configured successfully.")
    except Exception as e:
        print(f"--- Launcher: ERROR - Failed to configure DLL search paths with os.add_dll_directory: {e}")
        # Fallback to PATH manipulation if os.add_dll_directory fails for some reason
        original_path = os.environ.get("PATH", "")
        path_str = ";".join(str(p) for p in dll_paths)
        os.environ["PATH"] = path_str + ";" + original_path
        print("--- Launcher: Fell back to PATH environment variable manipulation.")


def detect_required_environment(script_path):
    """Detect which environment a script needs based on its path and imports"""
    script_path_str = str(script_path).lower()
    
    # Detection/YOLO scripts need yolo_env
    if 'detection' in script_path_str or 'yolo' in script_path_str:
        return 'yolo_env'
    
    # OCR scripts need ocr_env
    if 'ocr' in script_path_str or 'paddle' in script_path_str:
        return 'ocr_env'
    
    # Check imports in the script
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for key imports
        if 'from ultralytics import' in content or 'import torch' in content:
            return 'yolo_env'
        elif 'import paddleocr' in content or 'import paddle' in content:
            return 'ocr_env'
    except Exception:
        pass
    
    # Default to main environment
    return 'main'

def get_environment_python(env_name, project_root):
    """Get the Python executable for a specific environment"""
    if env_name == 'main':
        return sys.executable
    
    env_path = project_root / "environments" / env_name / "Scripts" / "python.exe"
    if env_path.exists():
        return str(env_path)
    
    # Try Linux/Mac path
    env_path = project_root / "environments" / env_name / "bin" / "python"
    if env_path.exists():
        return str(env_path)
    
    return None

def main():
    """
    Main entry point. Configures environment and runs the target script.
    """
    # Parse arguments to determine target script
    project_root = Path(__file__).parent
    
    # Check if first argument is a script path
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.py') or '/' in sys.argv[1] or '\\' in sys.argv[1]):
        # First argument is the target script
        target_script_arg = sys.argv[1]
        remaining_args = sys.argv[2:]
        
        # Handle relative paths from project root
        if target_script_arg.startswith('src/') or target_script_arg.startswith('src\\'):
            target_script_path = project_root / target_script_arg
        else:
            # Assume it's relative to src/ if no path separator
            target_script_path = project_root / "src" / target_script_arg
            
        target_script_name = target_script_path.name
    else:
        # Default behavior - run the stage-based pipeline
        target_script_name = "run_pipeline.py"
        target_script_path = project_root / "src" / target_script_name
        remaining_args = sys.argv[1:]

    if not target_script_path.exists():
        print(f"--- Launcher: ERROR - Target script not found at {target_script_path}")
        print(f"--- Launcher: Available scripts in src/:")
        src_dir = project_root / "src"
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                rel_path = py_file.relative_to(project_root)
                print(f"  {rel_path}")
        sys.exit(1)

    # Detect required environment
    required_env = detect_required_environment(target_script_path)
    required_python = get_environment_python(required_env, project_root)
    
    print(f"\n--- Launcher: Detected required environment: {required_env}")
    
    if required_env != 'main' and required_python:
        # Need to run in a specific environment
        print(f"--- Launcher: Switching to {required_env} environment")
        print(f"--- Launcher: Using Python: {required_python}")
        
        # Build command to run in the specific environment
        cmd = [required_python, str(target_script_path)] + remaining_args
        
        print(f"--- Launcher: Executing: {' '.join(cmd)}")
        
        try:
            import subprocess
            result = subprocess.run(cmd, cwd=str(project_root))
            sys.exit(result.returncode)
        except Exception as e:
            print(f"--- Launcher: ERROR running in {required_env}: {e}")
            sys.exit(1)
    
    elif required_env != 'main':
        print(f"--- Launcher: WARNING - {required_env} environment not found, running in main environment")
        print(f"--- Launcher: This may cause import errors")
    
    # Run in current environment (main)
    # CRITICAL: Configure the environment BEFORE any deep learning libraries are imported.
    configure_gpu_environment()
    
    # Set up sys.argv for the target script
    sys.argv = [str(target_script_path)] + remaining_args

    print(f"--- Launcher: Environment configured. Executing {target_script_name} ---")
    print(f"--- Launcher: Script path: {target_script_path}")
    print(f"--- Launcher: Arguments: {remaining_args}")
    
    try:
        # Execute the target script in the current process
        # This is simpler and more reliable now with os.add_dll_directory
        runpy.run_path(str(target_script_path), run_name="__main__")
    except Exception as e:
        print(f"\n--- Launcher: Target script '{target_script_name}' encountered an error. ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
