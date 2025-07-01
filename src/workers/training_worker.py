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
sys.path.append(str(project_root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Ensure Torch is loaded first – not strictly required inside dedicated env
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    try:
        print(f"DEBUG: Starting training worker with input: {input_data}")
        
        # Apply PyTorch 2.6 compatibility fix for YOLO model loading
        try:
            import torch
            original_torch_load = torch.load
            
            def safe_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
                # Force weights_only=False for YOLO models to avoid security restrictions
                return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                                         weights_only=False, **kwargs)
            
            # Apply the patch globally
            torch.load = safe_torch_load
            print("DEBUG: Applied PyTorch 2.6 compatibility fix")
        except ImportError:
            print("DEBUG: PyTorch not available, skipping compatibility fix")
        
        # Import compatibility classes first
        from src.detection.yolo_compatibility import register_compatibility_classes
        register_compatibility_classes()
        
        # Import training functions from yolo11_train.py
        from src.detection.yolo11_train import train_yolo11, validate_dataset
        
        # Validate dataset first
        print("DEBUG: Validating dataset structure...")
        if not validate_dataset():
            raise RuntimeError("Dataset validation failed")
        
        # Extract training parameters
        model_path = input_data.get("model_path")
        data_yaml_path = input_data.get("data_yaml_path")
        epochs = input_data.get("epochs", 50)
        batch_size = input_data.get("batch_size", 16)
        patience = input_data.get("patience", 20)
        project_name = input_data.get("project_name", "plc_symbol_detector")
        
        print(f"DEBUG: Training parameters:")
        print(f"  Model: {model_path}")
        print(f"  Data YAML: {data_yaml_path}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Project: {project_name}")
        
        # Check if model file exists
        if model_path and not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Check if data YAML exists
        if data_yaml_path and not Path(data_yaml_path).exists():
            raise FileNotFoundError(f"Data YAML file not found: {data_yaml_path}")
        
        # Run training using yolo11_train.py functions
        print("DEBUG: Starting YOLO training...")
        results = train_yolo11(
            model_name=Path(model_path).name if model_path else 'yolo11m.pt',
            data_yaml_path=data_yaml_path,  # Pass the correct data yaml path
            epochs=epochs,
            batch=batch_size,
            patience=patience,
            project_name=project_name
        )
        
        print(f"DEBUG: Training completed successfully")
        print(f"  Results dir: {results.save_dir}")
        print(f"  Best model: {results.save_dir}/weights/best.pt")
        
        # Extract metrics from results
        metrics = {}
        if hasattr(results, 'results_dict'):
            metrics = {
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0))
            }
        
        # Prepare output
        training_results = {
            'save_dir': str(results.save_dir),
            'epochs_completed': epochs,
            'best_model_path': str(results.save_dir / "weights" / "best.pt"),
            'last_model_path': str(results.save_dir / "weights" / "last.pt"),
            'metrics': metrics,
            'project_name': project_name
        }
        
        out = {"status": "success", "results": training_results}

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
