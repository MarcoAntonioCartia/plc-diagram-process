#!/usr/bin/env python3
"""Detection Pipeline Subprocess Runner

This script is called by DetectionManager to run detection in an isolated process.
It CAN import ultralytics and other heavy dependencies because it runs in a 
subprocess, not in the main process.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# NOW we can import heavy dependencies
from src.detection.detect_pipeline import PLCDetectionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    
    # Read input
    with open(args.input, 'r') as f:
        input_data = json.load(f)
    
    try:
        # Initialize pipeline
        pipeline = PLCDetectionPipeline(
            model_path=input_data.get("model_path"),
            confidence_threshold=input_data.get("confidence_threshold", 0.25)
        )
        
        # Run detection
        result = pipeline.process_pdf_folder(
            diagrams_folder=Path(input_data["pdf_path"]).parent,
            output_folder=Path(input_data["output_dir"]).parent,
            snippet_size=tuple(input_data.get("snippet_size", [1500, 1200])),
            overlap=input_data.get("overlap", 500),
            skip_pdf_conversion=False
        )
        
        output_data = {
            "status": "success",
            "result_folder": str(result),
            "detection_files": [str(f) for f in Path(result).glob("*_detections.json")]
        }
        
    except Exception as e:
        output_data = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }
    
    # Write output
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return 0 if output_data["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main()) 