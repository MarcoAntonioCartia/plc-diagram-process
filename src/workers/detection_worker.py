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
        from src.detection.detect_pipeline import PLCDetectionPipeline  # type: ignore

        pipeline = PLCDetectionPipeline(
            model_name=input_data.get("model_name"),
            confidence_threshold=input_data.get("confidence_threshold", 0.25),
        )

        results = pipeline.process_pdf(
            pdf_path=Path(input_data["pdf_path"]),
            output_dir=Path(input_data.get("output_dir", "detection_out")),
        )

        out = {"status": "success", "results": results}

    except Exception as exc:
        out = {"status": "error", "error": str(exc)}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main() 