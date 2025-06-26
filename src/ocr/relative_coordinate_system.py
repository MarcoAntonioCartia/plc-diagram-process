"""
Relative Coordinate System for Text Extraction
Stores text coordinates relative to their associated YOLO symbol boxes
"""

import json
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path

class RelativeCoordinateSystem:
    """
    Manages text coordinates relative to their associated symbol boxes
    """
    
    def __init__(self):
        self.coordinate_system = "relative_to_symbol"
        
    def convert_to_relative_coordinates(self, text_extraction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert absolute text coordinates to relative coordinates based on associated symbols
        
        Args:
            text_extraction_data: Original text extraction data with absolute coordinates
            
        Returns:
            Modified data with relative coordinates
        """
        relative_data = text_extraction_data.copy()
        relative_regions = []
        
        for region in text_extraction_data.get('text_regions', []):
            relative_region = region.copy()
            
            # Get text and symbol bounding boxes
            text_bbox = region.get('bbox', [])
            symbol = region.get('associated_symbol', {})
            symbol_bbox_global = symbol.get('bbox_global', {})
            
            if not text_bbox or not symbol_bbox_global:
                # Keep original if no symbol association
                relative_regions.append(relative_region)
                continue
            
            # Convert symbol bbox to list format
            if isinstance(symbol_bbox_global, dict):
                symbol_bbox = [
                    symbol_bbox_global.get('x1', 0),
                    symbol_bbox_global.get('y1', 0),
                    symbol_bbox_global.get('x2', 0),
                    symbol_bbox_global.get('y2', 0)
                ]
            else:
                symbol_bbox = symbol_bbox_global
            
            if len(text_bbox) >= 4 and len(symbol_bbox) >= 4:
                # Calculate relative coordinates (relative to symbol's top-left corner)
                relative_bbox = [
                    text_bbox[0] - symbol_bbox[0],  # rel_x1
                    text_bbox[1] - symbol_bbox[1],  # rel_y1
                    text_bbox[2] - symbol_bbox[0],  # rel_x2
                    text_bbox[3] - symbol_bbox[1]   # rel_y2
                ]
                
                # Store both absolute and relative coordinates
                relative_region['bbox_absolute'] = text_bbox
                relative_region['bbox'] = relative_bbox
                relative_region['coordinate_system'] = 'relative_to_symbol'
                
                # Add symbol dimensions for reference
                symbol_width = symbol_bbox[2] - symbol_bbox[0]
                symbol_height = symbol_bbox[3] - symbol_bbox[1]
                relative_region['symbol_dimensions'] = {
                    'width': symbol_width,
                    'height': symbol_height
                }
                
                # Calculate relative position as percentages (useful for analysis)
                relative_region['relative_position_percent'] = {
                    'x1_percent': (relative_bbox[0] / symbol_width * 100) if symbol_width > 0 else 0,
                    'y1_percent': (relative_bbox[1] / symbol_height * 100) if symbol_height > 0 else 0,
                    'x2_percent': (relative_bbox[2] / symbol_width * 100) if symbol_width > 0 else 0,
                    'y2_percent': (relative_bbox[3] / symbol_height * 100) if symbol_height > 0 else 0
                }
            
            relative_regions.append(relative_region)
        
        relative_data['text_regions'] = relative_regions
        relative_data['coordinate_system'] = 'relative_to_symbol'
        relative_data['conversion_metadata'] = {
            'original_system': 'absolute',
            'converted_system': 'relative_to_symbol',
            'total_regions': len(relative_regions),
            'regions_with_relative_coords': len([r for r in relative_regions if r.get('coordinate_system') == 'relative_to_symbol'])
        }
        
        return relative_data
    
    def convert_to_absolute_coordinates(self, relative_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert relative coordinates back to absolute coordinates
        
        Args:
            relative_data: Text extraction data with relative coordinates
            
        Returns:
            Data with absolute coordinates restored
        """
        absolute_data = relative_data.copy()
        absolute_regions = []
        
        for region in relative_data.get('text_regions', []):
            absolute_region = region.copy()
            
            # Check if this region has relative coordinates
            if region.get('coordinate_system') != 'relative_to_symbol':
                absolute_regions.append(absolute_region)
                continue
            
            # Get relative bbox and symbol information
            relative_bbox = region.get('bbox', [])
            symbol = region.get('associated_symbol', {})
            symbol_bbox_global = symbol.get('bbox_global', {})
            
            if not relative_bbox or not symbol_bbox_global:
                absolute_regions.append(absolute_region)
                continue
            
            # Convert symbol bbox to list format
            if isinstance(symbol_bbox_global, dict):
                symbol_bbox = [
                    symbol_bbox_global.get('x1', 0),
                    symbol_bbox_global.get('y1', 0),
                    symbol_bbox_global.get('x2', 0),
                    symbol_bbox_global.get('y2', 0)
                ]
            else:
                symbol_bbox = symbol_bbox_global
            
            if len(relative_bbox) >= 4 and len(symbol_bbox) >= 4:
                # Convert relative coordinates back to absolute
                absolute_bbox = [
                    symbol_bbox[0] + relative_bbox[0],  # abs_x1
                    symbol_bbox[1] + relative_bbox[1],  # abs_y1
                    symbol_bbox[0] + relative_bbox[2],  # abs_x2
                    symbol_bbox[1] + relative_bbox[3]   # abs_y2
                ]
                
                # Update the region
                absolute_region['bbox'] = absolute_bbox
                absolute_region['coordinate_system'] = 'absolute'
                
                # Keep relative coordinates for reference
                absolute_region['bbox_relative'] = relative_bbox
            
            absolute_regions.append(absolute_region)
        
        absolute_data['text_regions'] = absolute_regions
        absolute_data['coordinate_system'] = 'absolute'
        
        return absolute_data
    
    def create_pdf_layout_data(self, relative_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create layout data optimized for PDF creation using relative coordinates
        
        Args:
            relative_data: Text extraction data with relative coordinates
            
        Returns:
            Layout data grouped by symbols for efficient PDF rendering
        """
        layout_data = {
            'coordinate_system': 'relative_to_symbol',
            'symbols': {},
            'metadata': {
                'total_symbols': 0,
                'total_text_regions': 0
            }
        }
        
        # Group text regions by their associated symbols
        for region in relative_data.get('text_regions', []):
            symbol = region.get('associated_symbol', {})
            
            if not symbol:
                continue
            
            # Create a unique symbol identifier
            symbol_id = f"{symbol.get('class_name', 'unknown')}_{symbol.get('snippet_source', 'unknown')}"
            
            if symbol_id not in layout_data['symbols']:
                layout_data['symbols'][symbol_id] = {
                    'symbol_info': symbol,
                    'text_regions': []
                }
                layout_data['metadata']['total_symbols'] += 1
            
            # Add text region with relative coordinates
            text_info = {
                'text': region.get('text', ''),
                'confidence': region.get('confidence', 0),
                'bbox_relative': region.get('bbox', []),
                'bbox_absolute': region.get('bbox_absolute', []),
                'relative_position_percent': region.get('relative_position_percent', {}),
                'source': region.get('source', 'unknown'),
                'patterns': region.get('matched_patterns', []),
                'relevance_score': region.get('relevance_score', 0)
            }
            
            layout_data['symbols'][symbol_id]['text_regions'].append(text_info)
            layout_data['metadata']['total_text_regions'] += 1
        
        return layout_data
    
    def render_symbol_with_text(self, symbol_bbox: List[float], text_regions: List[Dict]) -> List[Dict]:
        """
        Render a symbol with its associated text regions using relative coordinates
        
        Args:
            symbol_bbox: Absolute coordinates of the symbol [x1, y1, x2, y2]
            text_regions: List of text regions with relative coordinates
            
        Returns:
            List of text regions with absolute coordinates for rendering
        """
        rendered_texts = []
        
        for text_region in text_regions:
            relative_bbox = text_region.get('bbox_relative', [])
            
            if len(relative_bbox) >= 4 and len(symbol_bbox) >= 4:
                # Convert relative to absolute for rendering
                absolute_bbox = [
                    symbol_bbox[0] + relative_bbox[0],
                    symbol_bbox[1] + relative_bbox[1],
                    symbol_bbox[0] + relative_bbox[2],
                    symbol_bbox[1] + relative_bbox[3]
                ]
                
                rendered_text = text_region.copy()
                rendered_text['bbox_absolute'] = absolute_bbox
                rendered_texts.append(rendered_text)
        
        return rendered_texts

def main():
    """Test the relative coordinate system"""
    
    # Initialize the system
    rel_coord_system = RelativeCoordinateSystem()
    
    # Load test data
    input_file = Path("D:/MarMe/github/0.3/plc-data/processed/text_extraction/1150_text_extraction.json")
    
    if not input_file.exists():
        print(f"Test file not found: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print("Testing Relative Coordinate System")
    print("=" * 50)
    
    # Convert to relative coordinates
    print("1. Converting to relative coordinates...")
    relative_data = rel_coord_system.convert_to_relative_coordinates(original_data)
    
    # Save relative coordinate version
    relative_file = input_file.parent / f"{input_file.stem}_relative.json"
    with open(relative_file, 'w', encoding='utf-8') as f:
        json.dump(relative_data, f, indent=2, ensure_ascii=False)
    
    print(f"   Saved relative coordinate version: {relative_file}")
    print(f"   Regions with relative coords: {relative_data['conversion_metadata']['regions_with_relative_coords']}")
    
    # Test conversion back to absolute
    print("\n2. Converting back to absolute coordinates...")
    reconstructed_data = rel_coord_system.convert_to_absolute_coordinates(relative_data)
    
    # Verify accuracy
    print("\n3. Verifying accuracy...")
    original_regions = original_data.get('text_regions', [])
    reconstructed_regions = reconstructed_data.get('text_regions', [])
    
    total_error = 0
    max_error = 0
    
    for i, (orig, recon) in enumerate(zip(original_regions, reconstructed_regions)):
        orig_bbox = orig.get('bbox', [])
        recon_bbox = recon.get('bbox', [])
        
        if len(orig_bbox) >= 4 and len(recon_bbox) >= 4:
            error = sum(abs(o - r) for o, r in zip(orig_bbox, recon_bbox))
            total_error += error
            max_error = max(max_error, error)
    
    avg_error = total_error / len(original_regions) if original_regions else 0
    
    print(f"   Average coordinate error: {avg_error:.6f}")
    print(f"   Maximum coordinate error: {max_error:.6f}")
    
    if avg_error < 0.001:
        print("   ✓ Perfect reconstruction! Relative coordinate system works flawlessly.")
    else:
        print("   ✗ Reconstruction has errors. Need investigation.")
    
    # Create layout data
    print("\n4. Creating PDF layout data...")
    layout_data = rel_coord_system.create_pdf_layout_data(relative_data)
    
    layout_file = input_file.parent / f"{input_file.stem}_layout.json"
    with open(layout_file, 'w', encoding='utf-8') as f:
        json.dump(layout_data, f, indent=2, ensure_ascii=False)
    
    print(f"   Saved layout data: {layout_file}")
    print(f"   Total symbols: {layout_data['metadata']['total_symbols']}")
    print(f"   Total text regions: {layout_data['metadata']['total_text_regions']}")
    
    print("\n✓ Relative coordinate system test completed successfully!")

if __name__ == "__main__":
    main()
