"""
Coordinate Calibration Module for Text Extraction
Detects and corrects coordinate transformation errors
"""

import json
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path

class CoordinateCalibrator:
    """
    Detects coordinate transformation patterns and applies reverse corrections
    """
    
    def __init__(self):
        self.transformation_detected = False
        self.offset_pattern = None
        self.scale_pattern = None
        
    def analyze_coordinate_pattern(self, text_extraction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the coordinate transformation pattern from text extraction results
        
        Args:
            text_extraction_data: The text extraction JSON data
            
        Returns:
            Analysis results with detected patterns
        """
        text_regions = text_extraction_data.get('text_regions', [])
        
        if not text_regions:
            return {"error": "No text regions found"}
        
        # Collect offset samples
        offsets_x = []
        offsets_y = []
        
        for region in text_regions:
            text_bbox = region.get('bbox', [])
            symbol = region.get('associated_symbol', {})
            symbol_bbox = symbol.get('bbox_global', {})
            
            if not text_bbox or not symbol_bbox:
                continue
                
            # Convert symbol bbox to list if it's a dict
            if isinstance(symbol_bbox, dict):
                symbol_bbox = [
                    symbol_bbox.get('x1', 0), 
                    symbol_bbox.get('y1', 0),
                    symbol_bbox.get('x2', 0), 
                    symbol_bbox.get('y2', 0)
                ]
            
            if len(text_bbox) >= 4 and len(symbol_bbox) >= 4:
                offset_x = text_bbox[0] - symbol_bbox[0]
                offset_y = text_bbox[1] - symbol_bbox[1]
                
                offsets_x.append(offset_x)
                offsets_y.append(offset_y)
        
        if not offsets_x:
            return {"error": "No valid coordinate pairs found"}
        
        # Analyze patterns
        analysis = {
            "sample_count": len(offsets_x),
            "offset_x_stats": {
                "mean": np.mean(offsets_x),
                "std": np.std(offsets_x),
                "min": np.min(offsets_x),
                "max": np.max(offsets_x),
                "median": np.median(offsets_x)
            },
            "offset_y_stats": {
                "mean": np.mean(offsets_y),
                "std": np.std(offsets_y),
                "min": np.min(offsets_y),
                "max": np.max(offsets_y),
                "median": np.median(offsets_y)
            }
        }
        
        # Detect if there's a consistent transformation error
        # Large offsets with high variation suggest double transformation
        x_variation = np.std(offsets_x) / (np.abs(np.mean(offsets_x)) + 1e-6)
        y_variation = np.std(offsets_y) / (np.abs(np.mean(offsets_y)) + 1e-6)
        
        analysis["transformation_detected"] = (
            np.abs(np.mean(offsets_x)) > 1000 or  # Large X offset
            np.abs(np.mean(offsets_y)) > 500 or   # Large Y offset
            x_variation > 0.5 or                  # High X variation
            y_variation > 0.5                     # High Y variation
        )
        
        analysis["correction_needed"] = analysis["transformation_detected"]
        
        return analysis
    
    def detect_snippet_based_transformation(self, text_regions: List[Dict]) -> Dict[str, Any]:
        """
        Detect if the transformation is based on snippet positions
        """
        snippet_offsets = {}
        
        for region in text_regions:
            symbol = region.get('associated_symbol', {})
            snippet_pos = symbol.get('snippet_position', {})
            
            if not snippet_pos:
                continue
                
            row = snippet_pos.get('row', 0)
            col = snippet_pos.get('col', 0)
            snippet_key = f"r{row}_c{col}"
            
            text_bbox = region.get('bbox', [])
            symbol_bbox = symbol.get('bbox_global', {})
            
            if isinstance(symbol_bbox, dict):
                symbol_bbox = [
                    symbol_bbox.get('x1', 0), 
                    symbol_bbox.get('y1', 0),
                    symbol_bbox.get('x2', 0), 
                    symbol_bbox.get('y2', 0)
                ]
            
            if len(text_bbox) >= 4 and len(symbol_bbox) >= 4:
                offset_x = text_bbox[0] - symbol_bbox[0]
                offset_y = text_bbox[1] - symbol_bbox[1]
                
                if snippet_key not in snippet_offsets:
                    snippet_offsets[snippet_key] = {'x': [], 'y': [], 'row': row, 'col': col}
                
                snippet_offsets[snippet_key]['x'].append(offset_x)
                snippet_offsets[snippet_key]['y'].append(offset_y)
        
        # Calculate average offsets per snippet
        snippet_analysis = {}
        for snippet_key, data in snippet_offsets.items():
            snippet_analysis[snippet_key] = {
                'row': data['row'],
                'col': data['col'],
                'avg_offset_x': np.mean(data['x']),
                'avg_offset_y': np.mean(data['y']),
                'count': len(data['x'])
            }
        
        return snippet_analysis
    
    def calculate_reverse_transformation(self, analysis: Dict[str, Any], text_extraction_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate the reverse transformation to correct coordinates
        """
        if not analysis.get("transformation_detected", False):
            return {"correction_type": "none"}
        
        # If we have the full data, try snippet-based correction
        if text_extraction_data:
            snippet_analysis = self.detect_snippet_based_transformation(text_extraction_data.get('text_regions', []))
            
            if snippet_analysis:
                # Use snippet-based correction
                return {
                    "correction_type": "snippet_based",
                    "snippet_corrections": snippet_analysis,
                    "description": "Apply snippet-specific coordinate corrections"
                }
        
        # Fallback to simple offset correction
        mean_offset_x = analysis["offset_x_stats"]["mean"]
        mean_offset_y = analysis["offset_y_stats"]["mean"]
        
        # The correction is to subtract the detected offset
        correction = {
            "correction_type": "offset",
            "offset_x": -mean_offset_x,
            "offset_y": -mean_offset_y,
            "description": f"Subtract offset: X={-mean_offset_x:.1f}, Y={-mean_offset_y:.1f}"
        }
        
        return correction
    
    def apply_coordinate_correction(self, text_extraction_data: Dict[str, Any], 
                                  correction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply coordinate correction to text extraction data
        """
        if correction.get("correction_type") == "none":
            return text_extraction_data
        
        corrected_data = text_extraction_data.copy()
        corrected_regions = []
        
        for region in text_extraction_data.get('text_regions', []):
            corrected_region = region.copy()
            
            if correction.get("correction_type") == "offset":
                offset_x = correction.get("offset_x", 0)
                offset_y = correction.get("offset_y", 0)
                
                # Apply correction to text bbox
                if 'bbox' in corrected_region:
                    bbox = corrected_region['bbox']
                    corrected_region['bbox'] = [
                        bbox[0] + offset_x,
                        bbox[1] + offset_y,
                        bbox[2] + offset_x,
                        bbox[3] + offset_y
                    ]
            
            elif correction.get("correction_type") == "snippet_based":
                # Apply snippet-specific correction
                symbol = corrected_region.get('associated_symbol', {})
                snippet_pos = symbol.get('snippet_position', {})
                
                if snippet_pos:
                    row = snippet_pos.get('row', 0)
                    col = snippet_pos.get('col', 0)
                    snippet_key = f"r{row}_c{col}"
                    
                    snippet_corrections = correction.get('snippet_corrections', {})
                    if snippet_key in snippet_corrections:
                        snippet_correction = snippet_corrections[snippet_key]
                        offset_x = -snippet_correction['avg_offset_x']
                        offset_y = -snippet_correction['avg_offset_y']
                        
                        # Apply correction to text bbox
                        if 'bbox' in corrected_region:
                            bbox = corrected_region['bbox']
                            corrected_region['bbox'] = [
                                bbox[0] + offset_x,
                                bbox[1] + offset_y,
                                bbox[2] + offset_x,
                                bbox[3] + offset_y
                            ]
            
            corrected_regions.append(corrected_region)
        
        corrected_data['text_regions'] = corrected_regions
        corrected_data['coordinate_correction_applied'] = correction
        
        return corrected_data
    
    def calibrate_text_extraction_file(self, input_file: Path, output_file: Path = None) -> Dict[str, Any]:
        """
        Calibrate coordinates in a text extraction file
        
        Args:
            input_file: Path to input text extraction JSON file
            output_file: Path to output corrected file (optional)
            
        Returns:
            Calibration results
        """
        # Load data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Analyze coordinate pattern
        analysis = self.analyze_coordinate_pattern(data)
        print(f"Coordinate analysis: {analysis}")
        
        # Calculate correction
        correction = self.calculate_reverse_transformation(analysis, data)
        print(f"Correction calculated: {correction}")
        
        # Apply correction
        corrected_data = self.apply_coordinate_correction(data, correction)
        
        # Save corrected data
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(corrected_data, f, indent=2, ensure_ascii=False)
            print(f"Corrected data saved to: {output_file}")
        
        return {
            "analysis": analysis,
            "correction": correction,
            "corrected_data": corrected_data,
            "input_file": str(input_file),
            "output_file": str(output_file) if output_file else None
        }

def main():
    """Test the calibration on the 1150 file"""
    calibrator = CoordinateCalibrator()
    
    input_file = Path("D:/MarMe/github/0.3/plc-data/processed/text_extraction/1150_text_extraction.json")
    output_file = Path("D:/MarMe/github/0.3/plc-data/processed/text_extraction/1150_text_extraction_corrected.json")
    
    if input_file.exists():
        result = calibrator.calibrate_text_extraction_file(input_file, output_file)
        print("\nCalibration completed!")
        print(f"Original regions: {len(result['corrected_data']['text_regions'])}")
        print(f"Correction applied: {result['correction']['description']}")
    else:
        print(f"Input file not found: {input_file}")

if __name__ == "__main__":
    main()
