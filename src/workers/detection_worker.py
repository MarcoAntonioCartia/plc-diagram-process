#!/usr/bin/env python3
"""Detection worker – runs YOLO detection inside the *detection_env*.

The script is intentionally minimal: it receives a JSON file with the input
parameters, runs the existing detection pipeline and writes the results back as
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
        print(f"DEBUG: Starting detection worker with input: {input_data}")
        
        # Check if model file exists
        model_path = input_data.get("model_path")
        if model_path and not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        from src.detection.detect_pipeline import PLCDetectionPipeline  # type: ignore

        pipeline = PLCDetectionPipeline(
            model_path=input_data.get("model_path"),
            confidence_threshold=input_data.get("confidence_threshold", 0.25),
        )

        # Check if we're processing a single PDF or a folder
        pdf_path = input_data.get("pdf_path")
        pdf_folder = input_data.get("pdf_folder")
        
        print(f"DEBUG: pdf_path={pdf_path}, pdf_folder={pdf_folder}")
        
        if pdf_path:
            # Single PDF mode
            pdf_parent = Path(pdf_path).parent
            output_dir = Path(input_data.get("output_dir", "detection_out"))
            print(f"DEBUG: Processing single PDF - diagrams_folder={pdf_parent}, output_folder={output_dir}")
            
            results = pipeline.process_pdf_folder(
                diagrams_folder=pdf_parent,
                output_folder=output_dir,
                snippet_size=tuple(input_data.get("snippet_size", [1500, 1200])),
                overlap=input_data.get("overlap", 500),
                skip_pdf_conversion=False
            )
        elif pdf_folder:
            # Folder mode
            diagrams_folder = Path(pdf_folder)
            output_dir = Path(input_data.get("output_dir", "detection_out"))
            print(f"DEBUG: Processing folder - diagrams_folder={diagrams_folder}, output_folder={output_dir}")
            
            results = pipeline.process_pdf_folder(
                diagrams_folder=diagrams_folder,
                output_folder=output_dir,
                snippet_size=tuple(input_data.get("snippet_size", [1500, 1200])),
                overlap=input_data.get("overlap", 500),
                skip_pdf_conversion=False
            )
        else:
            raise ValueError("Either pdf_path or pdf_folder must be provided")

        print(f"DEBUG: Detection completed successfully, results: {results}")
        out = {"status": "success", "results": str(results)}

    except Exception as exc:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: Detection worker failed: {error_details}")
        out = {"status": "error", "error": str(exc)}
        exit_code = 1
    else:
        exit_code = 0

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    sys.exit(exit_code)


if __name__ == "__main__":
    main() 