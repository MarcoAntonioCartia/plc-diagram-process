"""
Area Grouper for PLC Pipeline
Groups text regions by spatial areas and symbol associations
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Bounding box representation"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def distance_to(self, other: 'BoundingBox') -> float:
        """Calculate distance between centers"""
        x1, y1 = self.center
        x2, y2 = other.center
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def overlaps_with(self, other: 'BoundingBox', threshold: float = 0.1) -> bool:
        """Check if bounding boxes overlap"""
        # Calculate intersection
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        
        if ix1 >= ix2 or iy1 >= iy2:
            return False
        
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        min_area = min(self.area, other.area)
        
        return intersection_area / min_area > threshold


class AreaGrouper:
    """Groups text regions into spatial areas"""
    
    def __init__(self, 
                 proximity_threshold: float = 100.0,
                 symbol_association_threshold: float = 200.0,
                 grid_size: float = 100.0):
        """
        Initialize area grouper
        
        Args:
            proximity_threshold: Distance threshold for grouping nearby text
            symbol_association_threshold: Distance threshold for symbol-text association
            grid_size: Grid size for spatial partitioning
        """
        self.proximity_threshold = proximity_threshold
        self.symbol_association_threshold = symbol_association_threshold
        self.grid_size = grid_size
    
    def group_text_regions(self, 
                          text_regions: List[Dict[str, Any]], 
                          detection_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Group text regions into spatial areas
        
        Args:
            text_regions: List of text region dictionaries
            detection_results: Optional detection results for symbol association
            
        Returns:
            Dict with grouped areas
        """
        print(f"X Grouping {len(text_regions)} text regions into areas")
        
        # Convert to bounding box objects
        text_boxes = []
        for i, region in enumerate(text_regions):
            bbox = region.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                text_boxes.append({
                    'id': i,
                    'bbox': BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3]),
                    'text': region.get('text', ''),
                    'confidence': region.get('confidence', 0.0),
                    'source': region.get('source', 'ocr'),
                    'page': region.get('page', 1),
                    'original': region
                })
        
        # Convert detection results if provided
        symbol_boxes = []
        if detection_results:
            for detection in detection_results:
                bbox_global = detection.get('bbox_global', {})
                if isinstance(bbox_global, dict):
                    symbol_boxes.append({
                        'bbox': BoundingBox(
                            bbox_global.get('x1', 0),
                            bbox_global.get('y1', 0),
                            bbox_global.get('x2', 0),
                            bbox_global.get('y2', 0)
                        ),
                        'class_name': detection.get('class_name', 'unknown'),
                        'confidence': detection.get('confidence', 0.0),
                        'original': detection
                    })
        
        # Group by spatial proximity
        spatial_groups = self._group_by_proximity(text_boxes)
        
        # Associate with symbols if available
        if symbol_boxes:
            symbol_associations = self._associate_with_symbols(text_boxes, symbol_boxes)
        else:
            symbol_associations = {}
        
        # Create final area groups
        areas = self._create_area_groups(spatial_groups, symbol_associations)
        
        return {
            'total_areas': len(areas),
            'total_text_regions': len(text_regions),
            'symbol_associations': len(symbol_associations),
            'areas': areas,
            'grouping_stats': {
                'proximity_threshold': self.proximity_threshold,
                'symbol_association_threshold': self.symbol_association_threshold,
                'grid_size': self.grid_size
            }
        }
    
    def _group_by_proximity(self, text_boxes: List[Dict[str, Any]]) -> List[List[int]]:
        """Group text boxes by spatial proximity"""
        if not text_boxes:
            return []
        
        # Create adjacency matrix based on proximity
        n = len(text_boxes)
        adjacency = [[False] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = text_boxes[i]['bbox'].distance_to(text_boxes[j]['bbox'])
                if distance <= self.proximity_threshold:
                    adjacency[i][j] = True
                    adjacency[j][i] = True
        
        # Find connected components using DFS
        visited = [False] * n
        groups = []
        
        def dfs(node: int, current_group: List[int]):
            visited[node] = True
            current_group.append(node)
            
            for neighbor in range(n):
                if adjacency[node][neighbor] and not visited[neighbor]:
                    dfs(neighbor, current_group)
        
        for i in range(n):
            if not visited[i]:
                group = []
                dfs(i, group)
                groups.append(group)
        
        print(f"  V Created {len(groups)} spatial groups")
        return groups
    
    def _associate_with_symbols(self, 
                               text_boxes: List[Dict[str, Any]], 
                               symbol_boxes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Associate text regions with nearby symbols"""
        associations = {}
        
        for i, text_box in enumerate(text_boxes):
            closest_symbol = None
            closest_distance = float('inf')
            
            for symbol_box in symbol_boxes:
                distance = text_box['bbox'].distance_to(symbol_box['bbox'])
                
                if distance <= self.symbol_association_threshold and distance < closest_distance:
                    closest_distance = distance
                    closest_symbol = symbol_box
            
            if closest_symbol:
                associations[i] = {
                    'symbol': closest_symbol,
                    'distance': closest_distance
                }
        
        print(f"  V Associated {len(associations)} text regions with symbols")
        return associations
    
    def _create_area_groups(self, 
                           spatial_groups: List[List[int]], 
                           symbol_associations: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create final area groups with metadata"""
        areas = []
        
        for group_idx, group in enumerate(spatial_groups):
            # Calculate group bounding box
            if not group:
                continue
            
            # Find all text boxes in this group
            group_text_boxes = [text_boxes[i] for i in group if i < len(text_boxes)]
            
            if not group_text_boxes:
                continue
            
            # Calculate combined bounding box
            min_x = min(box['bbox'].x1 for box in group_text_boxes)
            min_y = min(box['bbox'].y1 for box in group_text_boxes)
            max_x = max(box['bbox'].x2 for box in group_text_boxes)
            max_y = max(box['bbox'].y2 for box in group_text_boxes)
            
            # Determine area type based on associated symbols
            area_type = self._determine_area_type(group, symbol_associations)
            
            # Create area ID
            area_id = f"area_{area_type}_{int(min_x//self.grid_size):03d}_{int(min_y//self.grid_size):03d}"
            
            # Collect text content
            text_content = []
            for box in group_text_boxes:
                if box['text'].strip():
                    text_content.append({
                        'text': box['text'],
                        'confidence': box['confidence'],
                        'source': box['source'],
                        'bbox': [box['bbox'].x1, box['bbox'].y1, box['bbox'].x2, box['bbox'].y2]
                    })
            
            # Get associated symbols
            associated_symbols = []
            for text_idx in group:
                if text_idx in symbol_associations:
                    symbol_info = symbol_associations[text_idx]['symbol']
                    if symbol_info not in associated_symbols:
                        associated_symbols.append(symbol_info)
            
            area = {
                'area_id': area_id,
                'area_type': area_type,
                'bounding_box': {
                    'x1': min_x,
                    'y1': min_y,
                    'x2': max_x,
                    'y2': max_y
                },
                'text_regions': text_content,
                'associated_symbols': associated_symbols,
                'text_count': len(text_content),
                'symbol_count': len(associated_symbols)
            }
            
            areas.append(area)
        
        # Sort areas by position (top-left to bottom-right)
        areas.sort(key=lambda a: (a['bounding_box']['y1'], a['bounding_box']['x1']))
        
        return areas
    
    def _determine_area_type(self, 
                           group: List[int], 
                           symbol_associations: Dict[int, Dict[str, Any]]) -> str:
        """Determine area type based on associated symbols"""
        symbol_classes = []
        
        for text_idx in group:
            if text_idx in symbol_associations:
                symbol_class = symbol_associations[text_idx]['symbol']['class_name']
                symbol_classes.append(symbol_class)
        
        if not symbol_classes:
            return 'text'
        
        # Determine type based on most common symbol class
        symbol_counts = {}
        for symbol_class in symbol_classes:
            symbol_counts[symbol_class] = symbol_counts.get(symbol_class, 0) + 1
        
        most_common_symbol = max(symbol_counts, key=symbol_counts.get)
        
        # Map symbol classes to area types
        if most_common_symbol in ['resistor', 'capacitor', 'inductor', 'diode']:
            return 'component'
        elif most_common_symbol in ['connector', 'terminal']:
            return 'connection'
        elif most_common_symbol in ['label', 'text']:
            return 'label'
        else:
            return 'symbol'
    
    def create_grid_based_areas(self, 
                              text_regions: List[Dict[str, Any]], 
                              grid_size: Optional[float] = None) -> Dict[str, Any]:
        """Create areas based on regular grid"""
        if grid_size is None:
            grid_size = self.grid_size
        
        print(f"X Creating grid-based areas with grid size {grid_size}")
        
        grid_areas = {}
        
        for region in text_regions:
            bbox = region.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                # Calculate grid cell
                grid_x = int(bbox[0] // grid_size)
                grid_y = int(bbox[1] // grid_size)
                grid_key = f"grid_{grid_x:03d}_{grid_y:03d}"
                
                if grid_key not in grid_areas:
                    grid_areas[grid_key] = {
                        'area_id': grid_key,
                        'area_type': 'grid',
                        'grid_x': grid_x,
                        'grid_y': grid_y,
                        'text_regions': [],
                        'bounding_box': {
                            'x1': grid_x * grid_size,
                            'y1': grid_y * grid_size,
                            'x2': (grid_x + 1) * grid_size,
                            'y2': (grid_y + 1) * grid_size
                        }
                    }
                
                grid_areas[grid_key]['text_regions'].append({
                    'text': region.get('text', ''),
                    'confidence': region.get('confidence', 0.0),
                    'source': region.get('source', 'ocr'),
                    'bbox': bbox
                })
        
        return {
            'total_areas': len(grid_areas),
            'grid_size': grid_size,
            'areas': list(grid_areas.values())
        }


# Global reference for text_boxes (needed for the grouping functions)
text_boxes = []
