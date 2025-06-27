"""
Detection Format Converter
Converts detection files from coordinate-transformed format to text extraction format
"""

import json
from pathlib import Path
from typing import Dict, List, Any

def convert_detection_format(input_file: Path, output_file: Path = None) -> Dict[str, Any]:
    """
    Convert detection file from coordinate-transformed format to text extraction format
    
    Args:
        input_file: Path to input detection file
        output_file: Path to output file (optional, will create with _converted suffix)
    
    Returns:
        Converted detection data
    """
    print(f"Converting detection file: {input_file}")
    
    # Load original detection data
    with open(input_file, 'r') as f:
        original_data = json.load(f)
    
    # Create converted structure
    converted_data = {
        "original_pdf": original_data.get("original_pdf", ""),
        "source_file": str(input_file),
        "conversion_info": {
            "original_format": "coordinate_transformed",
            "converted_format": "text_extraction_compatible"
        },
        "pages": []
    }
    
    # Convert each page
    for page_data in original_data.get("pages", []):
        converted_page = {
            "page": page_data.get("page_num", 1),  # Convert page_num to page
            "original_width": page_data.get("original_width", 0),
            "original_height": page_data.get("original_height", 0),
            "detections": []
        }
        
        # Convert each detection
        for detection in page_data.get("detections", []):
            # Convert bbox_global to global_bbox
            bbox_global = detection.get("bbox_global", {})
            if isinstance(bbox_global, dict):
                global_bbox = [
                    bbox_global.get("x1", 0),
                    bbox_global.get("y1", 0),
                    bbox_global.get("x2", 0),
                    bbox_global.get("y2", 0)
                ]
            else:
                global_bbox = bbox_global  # Already a list
            
            converted_detection = {
                "class_id": detection.get("class_id", 0),
                "class_name": detection.get("class_name", "unknown"),
                "confidence": detection.get("confidence", 0.0),
                "global_bbox": global_bbox,  # Converted from bbox_global
                "snippet_source": detection.get("snippet_source", ""),
                "snippet_position": detection.get("snippet_position", {"row": 0, "col": 0}),
                "bbox_snippet": detection.get("bbox_snippet", {})  # Keep original for reference
            }
            
            converted_page["detections"].append(converted_detection)
        
        converted_data["pages"].append(converted_page)
    
    # Save converted data
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_converted.json"
    
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"V Conversion completed:")
    print(f"  Input file: {input_file}")
    print(f"  Output file: {output_file}")
    print(f"  Pages processed: {len(converted_data['pages'])}")
    print(f"  Total detections: {sum(len(p.get('detections', [])) for p in converted_data.get('pages', []))}")
    
    return converted_data

def convert_detection_folder(input_folder: Path, output_folder: Path = None) -> List[Path]:
    """
    Convert all detection files in a folder
    
    Args:
        input_folder: Folder containing detection files
        output_folder: Output folder (optional, will use input folder)
    
    Returns:
        List of converted file paths
    """
    if output_folder is None:
        output_folder = input_folder
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all detection files
    detection_files = list(input_folder.glob("*_detections.json"))
    
    if not detection_files:
        print(f"No detection files found in {input_folder}")
        return []
    
    print(f"Found {len(detection_files)} detection files to convert")
    
    converted_files = []
    
    for detection_file in detection_files:
        try:
            output_file = output_folder / f"{detection_file.stem}_converted.json"
            convert_detection_format(detection_file, output_file)
            converted_files.append(output_file)
            
        except Exception as e:
            print(f"Error converting {detection_file}: {e}")
            continue
    
    print(f"\nConversion completed: {len(converted_files)} files converted")
    return converted_files

def main():
    """CLI interface for detection format conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert detection files to text extraction format')
    parser.add_argument('--input-file', '-i', type=str,
                       help='Single detection file to convert')
    parser.add_argument('--input-folder', '-f', type=str,
                       help='Folder containing detection files to convert')
    parser.add_argument('--output-folder', '-o', type=str,
                       help='Output folder for converted files')
    
    args = parser.parse_args()
    
    if args.input_file:
        # Convert single file
        input_file = Path(args.input_file)
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}")
            return 1
        
        convert_detection_format(input_file)
        
    elif args.input_folder:
        # Convert folder
        input_folder = Path(args.input_folder)
        if not input_folder.exists():
            print(f"Error: Input folder not found: {input_folder}")
            return 1
        
        output_folder = Path(args.output_folder) if args.output_folder else None
        convert_detection_folder(input_folder, output_folder)
        
    else:
        print("Please specify either --input-file or --input-folder")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())