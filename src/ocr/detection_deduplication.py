"""
Detection Deduplication Module
Implements Non-Maximum Suppression (NMS) to remove overlapping detections
"""

import numpy as np
from typing import List, Dict, Any, Tuple

def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1, box2: Bounding boxes with keys 'x1', 'y1', 'x2', 'y2'
        
    Returns:
        IoU value between 0 and 1
    """
    # Get coordinates
    x1_1, y1_1, x2_1, y2_1 = box1['x1'], box1['y1'], box1['x2'], box1['y2']
    x1_2, y1_2, x2_2, y2_2 = box2['x1'], box2['y1'], box2['x2'], box2['y2']
    
    # Calculate intersection area
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union

def non_maximum_suppression(detections: List[Dict[str, Any]], 
                          iou_threshold: float = 0.5,
                          class_specific: bool = True) -> List[Dict[str, Any]]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for considering boxes as overlapping
        class_specific: If True, only suppress within same class
        
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    # Sort detections by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Group by class if class_specific is True
    if class_specific:
        class_groups = {}
        for detection in sorted_detections:
            class_name = detection.get('class_name', 'unknown')
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(detection)
        
        # Apply NMS to each class separately
        filtered_detections = []
        for class_name, class_detections in class_groups.items():
            filtered_detections.extend(_apply_nms_to_group(class_detections, iou_threshold))
        
        return filtered_detections
    else:
        # Apply NMS to all detections together
        return _apply_nms_to_group(sorted_detections, iou_threshold)

def _apply_nms_to_group(detections: List[Dict[str, Any]], 
                       iou_threshold: float) -> List[Dict[str, Any]]:
    """
    Apply NMS to a group of detections
    
    Args:
        detections: List of detection dictionaries (should be sorted by confidence)
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    filtered = []
    suppressed = set()
    
    for i, detection in enumerate(detections):
        if i in suppressed:
            continue
        
        # This detection is kept
        filtered.append(detection)
        
        # Suppress overlapping detections with lower confidence
        bbox1 = detection.get('bbox_global', detection.get('global_bbox', {}))
        if not bbox1:
            continue
            
        for j, other_detection in enumerate(detections[i+1:], start=i+1):
            if j in suppressed:
                continue
            
            bbox2 = other_detection.get('bbox_global', other_detection.get('global_bbox', {}))
            if not bbox2:
                continue
            
            # Calculate IoU
            iou = calculate_iou(bbox1, bbox2)
            
            # Suppress if IoU is above threshold
            if iou > iou_threshold:
                suppressed.add(j)
    
    return filtered

def deduplicate_detections(detection_data: Dict[str, Any], 
                         iou_threshold: float = 0.5,
                         class_specific: bool = True) -> Dict[str, Any]:
    """
    Remove duplicate detections from detection data using NMS
    
    Args:
        detection_data: Detection data dictionary
        iou_threshold: IoU threshold for NMS
        class_specific: Whether to apply NMS per class
        
    Returns:
        Detection data with duplicates removed
    """
    print(f"Applying NMS with IoU threshold: {iou_threshold}, class_specific: {class_specific}")
    
    # Create a copy of the detection data
    deduplicated_data = detection_data.copy()
    deduplicated_data['pages'] = []
    
    total_before = 0
    total_after = 0
    
    for page_data in detection_data['pages']:
        original_detections = page_data['detections']
        total_before += len(original_detections)
        
        print(f"Page {page_data.get('page_num', 1)}: {len(original_detections)} detections before NMS")
        
        # Apply NMS
        filtered_detections = non_maximum_suppression(
            original_detections, 
            iou_threshold=iou_threshold,
            class_specific=class_specific
        )
        
        total_after += len(filtered_detections)
        print(f"Page {page_data.get('page_num', 1)}: {len(filtered_detections)} detections after NMS")
        
        # Create new page data with filtered detections
        new_page_data = page_data.copy()
        new_page_data['detections'] = filtered_detections
        deduplicated_data['pages'].append(new_page_data)
    
    print(f"Total detections: {total_before} -> {total_after} (removed {total_before - total_after})")
    
    return deduplicated_data

def analyze_detection_overlaps(detections: List[Dict[str, Any]], 
                             iou_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Analyze overlaps in detections for debugging purposes
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for considering overlaps
        
    Returns:
        Analysis results
    """
    if not detections:
        return {"total_detections": 0, "overlapping_pairs": 0, "overlap_groups": []}
    
    overlapping_pairs = []
    
    for i, det1 in enumerate(detections):
        bbox1 = det1.get('bbox_global', det1.get('global_bbox', {}))
        if not bbox1:
            continue
            
        for j, det2 in enumerate(detections[i+1:], start=i+1):
            bbox2 = det2.get('bbox_global', det2.get('global_bbox', {}))
            if not bbox2:
                continue
            
            iou = calculate_iou(bbox1, bbox2)
            if iou > iou_threshold:
                overlapping_pairs.append({
                    "detection1": {
                        "class": det1.get('class_name', 'unknown'),
                        "confidence": det1.get('confidence', 0),
                        "bbox": bbox1
                    },
                    "detection2": {
                        "class": det2.get('class_name', 'unknown'),
                        "confidence": det2.get('confidence', 0),
                        "bbox": bbox2
                    },
                    "iou": iou
                })
    
    return {
        "total_detections": len(detections),
        "overlapping_pairs": len(overlapping_pairs),
        "overlap_details": overlapping_pairs[:10]  # Show first 10 for debugging
    }
