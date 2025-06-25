#!/usr/bin/env python3
"""OCR worker – runs PaddleOCR inside the *ocr_env*.

Receives detection results + PDF path via JSON, produces extracted text and
metadata as JSON. Runs in a Paddle-only virtual environment, so there is zero
risk of DLL clashes with PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

    # Paddle-specific env tweaks
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
    os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.5")

    try:
        from src.ocr.text_extraction_pipeline import TextExtractionPipeline  # type: ignore

        pipeline = TextExtractionPipeline(
            confidence_threshold=input_data.get("confidence_threshold", 0.7),
            language=input_data.get("language", "en"),
        )

        results = pipeline.extract_text_from_detections(
            detection_results=input_data["detection_results"],
            pdf_path=Path(input_data["pdf_path"]),
            output_dir=Path(input_data.get("output_dir", "ocr_out")),
        )

        out = {"status": "success", "results": results}
    except Exception as exc:
        out = {"status": "error", "error": str(exc)}
        exit_code = 1
    else:
        exit_code = 0

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    sys.exit(exit_code)


if __name__ == "__main__":
    main() 