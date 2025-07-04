"""
Diagnostic script to check detection file structure
"""

import json
from pathlib import Path

def check_detection_file(file_path):
    """Check if detection file has the correct structure for text extraction"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"V File loaded successfully")
        
        # Check top-level structure
        if 'pages' not in data:
            print(f"X Missing 'pages' key - incompatible structure")
            return False
            
        print(f"V Has 'pages' key")
        
        # Check first page structure
        if len(data['pages']) == 0:
            print(f"X No pages found")
            return False
            
        first_page = data['pages'][0]
        
        # Check if page has detections
        if 'detections' not in first_page:
            print(f"X Missing 'detections' key in first page")
            return False
        
        # Check if detections have global_bbox (required for text extraction)
        detections = first_page['detections']
        if len(detections) > 0:
            first_detection = detections[0]
            if 'global_bbox' in first_detection:
                print(f"  V Has 'global_bbox' - compatible with text extraction")
            else:
                print(f"  X Missing 'global_bbox' - needs conversion")
                return False
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"X JSON decode error: {e}")
        return False
    except FileNotFoundError:
        print(f"X File not found: {file_path}")
        return False
    except Exception as e:
        print(f"X Error reading file: {e}")
        return False

if __name__ == "__main__":
    # Test with your detection file
    detection_file = "D:/MarMe/github/0.3/plc-data/processed/detdiagrams2/1150_detections.json" # NExt step update this path as a parameter to the pipelines config location
    
    if Path(detection_file).exists():
        check_detection_file(detection_file)
    else:
        print(f"File not found: {detection_file}")
        print("Please update the path in the script")