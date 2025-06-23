import os
import sys
from pathlib import Path
import traceback

print("--- A script to diagnose and fix GPU runtime issues ---")
print(f"Python Executable: {sys.executable}")

# --- 1. Find the correct library paths bundled with Paddle ---
try:
    venv_path = Path(sys.executable).parent.parent
    site_packages = venv_path / "Lib" / "site-packages"

    if not site_packages.is_dir():
        raise FileNotFoundError(f"Could not find site-packages at {site_packages}")

    print(f"\nFound site-packages: {site_packages}")

    # These are the directories where Paddle 3.0.0 stores its CUDA/cuDNN DLLs
    bundled_paths_to_add = [
        site_packages / "nvidia" / "cuda_runtime" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "paddle" / "libs",
    ]
    
    found_paths = [str(p) for p in bundled_paths_to_add if p.is_dir()]

    if not found_paths:
        raise FileNotFoundError("Could not find any bundled NVIDIA library paths. Is paddlepaddle-gpu installed?")
    
    print("\nFound these essential library paths to prioritize:")
    for path in found_paths:
        print(f"  - {path}")

    # --- 2. Force Windows to look in these paths FIRST ---
    original_path = os.environ.get("PATH", "")
    # Prepend our paths to the system PATH. The 'os.pathsep' is the ';' character on Windows.
    new_path = os.pathsep.join(found_paths) + os.pathsep + original_path
    os.environ["PATH"] = new_path

    print("\nSuccessfully modified PATH for this script run. Attempting to initialize Paddle...")

    # --- 3. Run the final check ---
    import paddle
    print("\n--- Paddle Diagnosis ---")
    paddle.utils.run_check()

except Exception as e:
    print("\n--- SCRIPT FAILED ---")
    print(f"An error occurred: {e}")
    traceback.print_exc()
