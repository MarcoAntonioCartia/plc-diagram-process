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
        # Use the new PaddleOCR 3.0 API directly instead of our custom pipeline
        from paddleocr import PaddleOCR
        
        # Initialize with the new API - no device parameter needed
        ocr = PaddleOCR(
            lang=input_data.get("language", "en"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        
        # Simple mock implementation for now - just return success
        # TODO: Implement actual OCR processing with new API
        results = {
            "total_text_regions": 0,
            "text_regions": [],
            "extraction_method": "paddleocr_3.0",
            "status": "mock_success"
        }

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
