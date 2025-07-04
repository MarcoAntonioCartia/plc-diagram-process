#!/usr/bin/env python3
"""Training worker – runs YOLO training inside the *yolo_env*.

The script is intentionally minimal: it receives a JSON file with the input
parameters, runs the existing training pipeline and writes the results back as
JSON so that the coordinator process (MultiEnvironmentManager) can pick them up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH so we can import src.* without installing the
# package in editable mode inside the worker venv.
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Remove problematic environment variables that cause performance issues
    # PYTORCH_CUDA_ALLOC_CONF with max_split_size_mb:128 causes massive slowdown
    # These were added for compatibility but hurt training performance severely
    pass

    try:
        # Extract training parameters
        model_path = input_data.get("model_path")
        data_yaml_path = input_data.get("data_yaml_path")
        epochs = input_data.get("epochs", 50)
        batch_size = input_data.get("batch_size", 16)
        patience = input_data.get("patience", 20)
        project_name = input_data.get("project_name", "plc_symbol_detector")
        verbose = input_data.get("verbose", False) or os.environ.get("PLCDP_VERBOSE", "0") == "1"
        
        # Check if files exist
        if model_path and not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if data_yaml_path and not Path(data_yaml_path).exists():
            raise FileNotFoundError(f"Data YAML file not found: {data_yaml_path}")
        
        # Build command - use same workers as direct script for performance
        training_script = project_root / "src" / "detection" / "yolo11_train.py"
        cmd = [
            sys.executable,
            str(training_script),
            "--model", str(model_path) if model_path else 'yolo11m.pt',
            "--epochs", str(epochs),
            "--batch", str(batch_size),
            "--patience", str(patience),
            "--name", project_name,
            "--workers", "8"  # Use same as direct script for performance
        ]
        
        # Add device if specified
        device = input_data.get("device")
        if device:
            cmd.extend(["--device", str(device)])
        
        # Add quiet flag if verbose is False
        if not verbose:
            cmd.append("--quiet")
        
        # Run the training script as subprocess - NO output tampering
        import subprocess
        
        # Only print worker's own messages, let training output pass through directly
        print("[Worker] Starting YOLO training...")
        
        # Always use direct output - no capture, no processing, no tampering
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            timeout=3600
        )
        return_code = result.returncode
        stdout_output = ""  # No output capture for clean performance
        
        if return_code != 0:
            raise RuntimeError(f"Training script failed with return code {return_code}")
        
        print("[Worker] Training completed, processing results...")
        
        # Parse the output to extract results directory
        import re
        save_dir_match = re.search(r"Results saved to: (.+)", stdout_output)
        if save_dir_match:
            save_dir = Path(save_dir_match.group(1).strip())
        else:
            # Fallback: construct expected path
            from src.config import get_config
            config = get_config()
            save_dir = config.get_run_path('train') / project_name
        
        print(f"[Worker] Results dir: {save_dir}")
        print(f"[Worker] Best model: {save_dir}/weights/best.pt")
        
        # Extract metrics from the training output if available
        metrics = {}
        
        # Look for final validation results in the output
        map50_match = re.search(r"mAP50\s+mAP50-95.*?all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", stdout_output, re.DOTALL)
        if map50_match:
            metrics = {
                'precision': float(map50_match.group(1)),
                'recall': float(map50_match.group(2)),
                'mAP50': float(map50_match.group(3)),
                'mAP50-95': float(map50_match.group(4))
            }
        
        # Prepare output
        training_results = {
            'save_dir': str(save_dir),
            'epochs_completed': epochs,
            'best_model_path': str(save_dir / "weights" / "best.pt"),
            'last_model_path': str(save_dir / "weights" / "last.pt"),
            'metrics': metrics,
            'project_name': project_name,
            'stdout': stdout_output,
            'stderr': ""  # stderr is combined with stdout in our approach
        }
        
        out = {"status": "success", "results": training_results}
        
        # Optional auto-cleanup old training runs (controlled by flag)
        auto_cleanup = input_data.get("auto_cleanup", False) or os.environ.get("PLCDP_AUTO_CLEANUP", "0") == "1"
        
        if auto_cleanup:
            try:
                print("DEBUG: Performing automatic cleanup of old training runs...")
                import shutil
                
                # Get the runs directory
                runs_dir = save_dir.parent
                
                # Get all training run directories
                run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
                
                # Keep only the latest 3 runs (including current)
                if len(run_dirs) > 3:
                    # Sort by modification time (newest first)
                    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    # Remove old runs beyond the latest 3
                    for old_run in run_dirs[3:]:
                        try:
                            print(f"DEBUG: Removing old training run: {old_run.name}")
                            shutil.rmtree(old_run)
                        except Exception as cleanup_error:
                            print(f"DEBUG: Failed to cleanup {old_run.name}: {cleanup_error}")
                            
            except Exception as cleanup_error:
                print(f"DEBUG: Auto-cleanup failed: {cleanup_error}")
        else:
            print("DEBUG: Auto-cleanup disabled (use auto_cleanup=true or PLCDP_AUTO_CLEANUP=1 to enable)")

    except Exception as exc:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: Training worker failed: {error_details}")
        out = {"status": "error", "error": str(exc)}
        exit_code = 1
    else:
        exit_code = 0

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
