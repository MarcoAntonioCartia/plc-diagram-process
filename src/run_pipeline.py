"""
PLC Pipeline Runner
Main entry point for running the PLC diagram processing pipeline.
Supports multiple modes: Training → Detection → Text Extraction → PDF Enhancement
"""

def main():
    parser = argparse.ArgumentParser(description='PLC Pipeline Runner - Process PLC diagrams with detection and text extraction')
    # ... existing code ...
    -            print("2. Run text extraction: python src/run_complete_pipeline_with_text.py --skip-detection")
    +            print("2. Run text extraction: python src/run_pipeline.py --skip-detection")
    # ... existing code ...