"""
Diagnostic script to check detection file structure
"""

import json
from pathlib import Path

def check_detection_structure(detection_file_path):
    """Check the structure of a detection file"""
    
    print(f"Checking detection file: {detection_file_path}")
    
    try:
        with open(detection_file_path, 'r') as f:
            data = json.load(f)
        
        print(f"✓ File loaded successfully")
        print(f"Top-level keys: {list(data.keys())}")
        
        # Check if it has the expected structure
        if "pages" in data:
            print(f"✓ Has 'pages' key")
            pages = data["pages"]
            print(f"  Number of pages: {len(pages)}")
            
            if pages:
                first_page = pages[0]
                print(f"  First page keys: {list(first_page.keys())}")
                
                if "detections" in first_page:
                    detections = first_page["detections"]
                    print(f"  Number of detections on first page: {len(detections)}")
                    
                    if detections:
                        first_detection = detections[0]
                        print(f"  First detection keys: {list(first_detection.keys())}")
                        
                        # Check for bbox formats
                        bbox_keys = [k for k in first_detection.keys() if 'bbox' in k.lower()]
                        print(f"  Bbox-related keys: {bbox_keys}")
                        
                        if "global_bbox" in first_detection:
                            print(f"  ✓ Has 'global_bbox' - compatible with text extraction")
                        else:
                            print(f"  ✗ Missing 'global_bbox' - needs conversion")
                else:
                    print(f"  ✗ Missing 'detections' key in first page")
        else:
            print(f"✗ Missing 'pages' key - incompatible structure")
            print(f"Available keys: {list(data.keys())}")
            
            # Check if it's a different format
            if "original_pdf" in data:
                print(f"  Has 'original_pdf' key - might be coordinate-transformed format")
            
            # Check for snippet-based format
            snippet_keys = [k for k in data.keys() if 'snippet' in k.lower()]
            if snippet_keys:
                print(f"  Has snippet-related keys: {snippet_keys}")
        
        return data
        
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return None

if __name__ == "__main__":
    # Test with your detection file
    detection_file = "D:/MarMe/github/0.3/plc-data/processed/detdiagrams2/1150_detections.json"
    
    if Path(detection_file).exists():
        check_detection_structure(detection_file)
    else:
        print(f"File not found: {detection_file}")
        print("Please update the path in the script")