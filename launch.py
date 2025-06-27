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


def main():
    """
    Main entry point. Configures environment and runs the target script.
    """
    # CRITICAL: Configure the environment BEFORE any deep learning libraries are imported.
    configure_gpu_environment()

    # The main application script to run
    # Assuming launch.py is in the root and the target is in src/
    project_root = Path(__file__).parent
    target_script_name = "run_pipeline.py"
    target_script_path = project_root / "src" / target_script_name

    if not target_script_path.exists():
        print(f"--- Launcher: ERROR - Target script not found at {target_script_path}")
        sys.exit(1)

    # We need to pass through all command-line arguments to the target script.
    # We remove the launcher's own arguments if we add any in the future.
    # For now, we pass everything.
    original_sys_argv = sys.argv.copy()
    sys.argv = [str(target_script_path)] + original_sys_argv[1:]

    print(f"\n--- Launcher: Environment configured. Executing {target_script_name} ---")
    
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