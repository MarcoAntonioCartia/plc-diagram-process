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
from pathlib import Path
import subprocess

def get_venv_path() -> Path:
    """Dynamically finds and returns the path to the virtual environment."""
    if sys.platform != "win32":
        return

    try:
        venv_path = Path(sys.executable).parent.parent
        site_packages = venv_path / "Lib" / "site-packages"

        if not site_packages.is_dir():
            return

        bundled_paths = [
            site_packages / "nvidia" / "cuda_runtime" / "bin",
            site_packages / "nvidia" / "cudnn" / "bin",
            site_packages / "nvidia" / "cublas" / "bin",
            site_packages / "paddle" / "libs",
        ]
        
        found_paths = [str(p) for p in bundled_paths if p.is_dir()]
        
        if not found_paths:
            return

        original_path = os.environ.get("PATH", "")
        new_path = os.pathsep.join(found_paths) + os.pathsep + original_path
        os.environ["PATH"] = new_path
        
    except Exception:
        # Silently fail if something goes wrong, the app will run with default paths.
        pass

    return venv_path

def configure_environment(venv_path: Path) -> dict:
    """
    Configures the environment variables for the application.
    Returns the modified environment dictionary.
    """
    print("--- Launcher: Configuring GPU environment...")
    
    # Define paths to the CUDA/cuDNN binaries within the virtual environment
    base_path = venv_path / "Lib" / "site-packages"
    nvidia_paths = [
        base_path / "nvidia" / "cudnn" / "bin",
        base_path / "nvidia" / "cuda_runtime" / "bin",
        base_path / "nvidia" / "cublas" / "bin",
    ]
    
    # Filter out paths that don't exist
    existing_paths = [str(p) for p in nvidia_paths if p.exists()]
    
    if not existing_paths:
        print("--- Launcher: Warning - No NVIDIA library paths found in venv. Skipping PATH modification.")
        return

    # Prepend the NVIDIA paths to the system PATH
    original_path = os.environ.get('PATH', '')
    path_separator = ';' if sys.platform == 'win32' else ':'
    new_path = path_separator.join(existing_paths) + path_separator + original_path
    
    # Create a copy of the current environment and update the PATH
    env = os.environ.copy()
    env['PATH'] = new_path
    
    print("--- Launcher: PATH configured for bundled GPU libraries.")
    return env

def main():
    """
    Main entry point for the application.
    Sets up the environment and runs the target script.
    """
    venv_path = get_venv_path()
    # Get the correctly configured environment
    configured_env = configure_environment(venv_path)

    # The target script is now relative to the project root
    target_script_name = "src/run_complete_pipeline_with_text.py"

    print(f"\n--- Launcher: Environment configured. Executing {target_script_name} via subprocess ---")
    
    # The command to execute: python.exe [target_script] [arg1] [arg2] ...
    # We get the python executable from our current venv to ensure consistency.
    python_executable = sys.executable
    command = [python_executable, target_script_name] + sys.argv[1:]

    try:
        # Execute the target script as a new process with the modified environment
        # This is the most reliable way to ensure the PATH is inherited correctly.
        result = subprocess.run(command, env=configured_env, check=True)
        print(f"--- Launcher: Script finished with exit code {result.returncode} ---")
        sys.exit(result.returncode)

    except subprocess.CalledProcessError as e:
        print(f"--- Launcher: Target script failed with a non-zero exit code {e.returncode}. ---")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"--- Launcher: ERROR - Could not find python executable at '{python_executable}' or script at '{target_script_name}'.")
        sys.exit(1)
    except Exception as e:
        print(f"--- Launcher: An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 