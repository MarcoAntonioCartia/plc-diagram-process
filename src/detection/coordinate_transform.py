"""
Coordinate Transformation Module
Converts detection coordinates from snippet-relative to global PDF coordinates
"""

import json
from pathlib import Path
from typing import Dict, List, Any

def transform_detections_to_global(detection_results: Dict, metadata: Dict) -> Dict:
    """
    Transform detection coordinates from snippet-relative to global PDF coordinates
    
    Args:
        detection_results: Dictionary of detection results per snippet
        metadata: PDF metadata containing snippet positioning information
    
    Returns:
        Dictionary with global coordinates for all detections
    """
    global_detections = {
        "original_pdf": metadata["original_pdf"],
        "pages": []
    }
    
    # Process each page
    for page_info in metadata["pages"]:
        page_num = page_info["page_num"]
        page_detections = {
            "page_num": page_num,
            "original_width": page_info["original_width"],
            "original_height": page_info["original_height"],
            "detections": []
        }
        
        # Process each snippet in this page
        for snippet_info in page_info["snippets"]:
            snippet_filename = snippet_info["filename"]
            
            # Check if we have detection results for this snippet
            if snippet_filename in detection_results:
                snippet_detections = detection_results[snippet_filename].get("detections", [])
                
                # Transform each detection
                for detection in snippet_detections:
                    global_detection = transform_single_detection(detection, snippet_info)
                    if global_detection:
                        page_detections["detections"].append(global_detection)
        
        global_detections["pages"].append(page_detections)
    
    return global_detections

def transform_single_detection(detection: Dict, snippet_info: Dict) -> Dict:
    """
    Transform a single detection from snippet coordinates to global coordinates
    
    Args:
        detection: Single detection with snippet-relative coordinates
        snippet_info: Snippet metadata with global positioning
    
    Returns:
        Detection with global coordinates
    """
    try:
        # Extract snippet-relative coordinates
        snippet_bbox = detection["bbox"]
        snippet_x1 = snippet_bbox["x1"]
        snippet_y1 = snippet_bbox["y1"]
        snippet_x2 = snippet_bbox["x2"]
        snippet_y2 = snippet_bbox["y2"]
        
        # Extract snippet global position
        snippet_global_x1 = snippet_info["x1"]
        snippet_global_y1 = snippet_info["y1"]
        
        # Transform to global coordinates
        global_x1 = snippet_global_x1 + snippet_x1
        global_y1 = snippet_global_y1 + snippet_y1
        global_x2 = snippet_global_x1 + snippet_x2
        global_y2 = snippet_global_y1 + snippet_y2
        
        # Create global detection
        global_detection = {
            "class_id": detection["class_id"],
            "class_name": detection["class_name"],
            "confidence": detection["confidence"],
            "snippet_source": snippet_info["filename"],
            "snippet_position": {
                "row": snippet_info["row"],
                "col": snippet_info["col"]
            },
            "bbox_snippet": {
                "x1": snippet_x1,
                "y1": snippet_y1,
                "x2": snippet_x2,
                "y2": snippet_y2
            },
            "bbox_global": {
                "x1": global_x1,
                "y1": global_y1,
                "x2": global_x2,
                "y2": global_y2
            }
        }
        
        return global_detection
        
    except Exception as e:
        print(f"Error transforming detection: {e}")
        return None

def save_coordinate_mapping(global_detections: Dict, output_file: Path):
    """
    Save coordinate mapping to a human-readable text file
    
    Args:
        global_detections: Global detection results
        output_file: Path to output text file
    """
    with open(output_file, 'w') as f:
        f.write(f"PLC Detection Coordinate Mapping\n")
        f.write(f"PDF: {global_detections['original_pdf']}\n")
        f.write("=" * 50 + "\n\n")
        
        total_detections = 0
        
        for page in global_detections["pages"]:
            page_num = page["page_num"]
            page_detections = page["detections"]
            
            f.write(f"Page {page_num} ({page['original_width']}x{page['original_height']})\n")
            f.write("-" * 30 + "\n")
            
            if not page_detections:
                f.write("No detections found\n\n")
                continue
            
            for i, detection in enumerate(page_detections, 1):
                bbox = detection["bbox_global"]
                snippet_bbox = detection["bbox_snippet"]
                
                f.write(f"Detection {i}:\n")
                f.write(f"  Class: {detection['class_name']} (confidence: {detection['confidence']:.3f})\n")
                f.write(f"  Global coordinates: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) -> ({bbox['x2']:.1f}, {bbox['y2']:.1f})\n")
                f.write(f"  Snippet coordinates: ({snippet_bbox['x1']:.1f}, {snippet_bbox['y1']:.1f}) -> ({snippet_bbox['x2']:.1f}, {snippet_bbox['y2']:.1f})\n")
                f.write(f"  Source snippet: {detection['snippet_source']} (row {detection['snippet_position']['row']}, col {detection['snippet_position']['col']})\n")
                f.write("\n")
                
                total_detections += 1
            
            f.write(f"Page {page_num} total: {len(page_detections)} detections\n\n")
        
        f.write(f"Total detections across all pages: {total_detections}\n")

def validate_coordinates(global_detections: Dict) -> bool:
    """
    Validate that all global coordinates are within page bounds
    
    Args:
        global_detections: Global detection results
    
    Returns:
        True if all coordinates are valid, False otherwise
    """
    valid = True
    
    for page in global_detections["pages"]:
        page_width = page["original_width"]
        page_height = page["original_height"]
        
        for detection in page["detections"]:
            bbox = detection["bbox_global"]
            
            # Check if coordinates are within page bounds
            if (bbox["x1"] < 0 or bbox["y1"] < 0 or 
                bbox["x2"] > page_width or bbox["y2"] > page_height):
                
                print(f"Warning: Detection outside page bounds in {detection['snippet_source']}")
                print(f"  Coordinates: ({bbox['x1']}, {bbox['y1']}) -> ({bbox['x2']}, {bbox['y2']})")
                print(f"  Page size: {page_width}x{page_height}")
                valid = False
    
    return valid

def get_detection_statistics(global_detections: Dict) -> Dict:
    """
    Generate statistics about the detections
    
    Args:
        global_detections: Global detection results
    
    Returns:
        Dictionary with detection statistics
    """
    stats = {
        "total_pages": len(global_detections["pages"]),
        "total_detections": 0,
        "detections_per_page": [],
        "class_counts": {},
        "confidence_stats": {
            "min": float('inf'),
            "max": 0,
            "avg": 0
        }
    }
    
    all_confidences = []
    
    for page in global_detections["pages"]:
        page_count = len(page["detections"])
        stats["detections_per_page"].append(page_count)
        stats["total_detections"] += page_count
        
        for detection in page["detections"]:
            # Count classes
            class_name = detection["class_name"]
            stats["class_counts"][class_name] = stats["class_counts"].get(class_name, 0) + 1
            
            # Track confidence
            confidence = detection["confidence"]
            all_confidences.append(confidence)
            stats["confidence_stats"]["min"] = min(stats["confidence_stats"]["min"], confidence)
            stats["confidence_stats"]["max"] = max(stats["confidence_stats"]["max"], confidence)
    
    # Calculate average confidence
    if all_confidences:
        stats["confidence_stats"]["avg"] = sum(all_confidences) / len(all_confidences)
    else:
        stats["confidence_stats"]["min"] = 0
    
    return stats

if __name__ == "__main__":
    # Example usage for testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Test coordinate transformation')
    parser.add_argument('--detections', required=True, help='Detection results JSON file')
    parser.add_argument('--metadata', required=True, help='Metadata JSON file')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Load detection results
    with open(args.detections, 'r') as f:
        detection_results = json.load(f)
    
    # Load metadata
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    
    # Transform coordinates
    global_detections = transform_detections_to_global(detection_results, metadata)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    json_file = output_dir / f"{metadata['original_pdf']}_global_detections.json"
    with open(json_file, 'w') as f:
        json.dump(global_detections, f, indent=2)
    
    # Save coordinate mapping
    txt_file = output_dir / f"{metadata['original_pdf']}_coordinates.txt"
    save_coordinate_mapping(global_detections, txt_file)
    
    # Validate and show statistics
    valid = validate_coordinates(global_detections)
    stats = get_detection_statistics(global_detections)
    
    print(f"Coordinate transformation completed")
    print(f"Valid coordinates: {valid}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Results saved to: {output_dir}")
