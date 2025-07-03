"""
CSV Formatter for PLC Pipeline
Creates CSV output with alphanumeric ordering and area-based grouping
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .area_grouper import AreaGrouper


@dataclass
class TextRegion:
    """Data class for text region information"""
    document: str
    page: int
    area_id: str
    area_type: str
    sequence: str
    text_content: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    source: str = "ocr"  # "ocr" or "pdf"


class CSVFormatter:
    """Formats text extraction results into CSV with area-based grouping"""
    
    def __init__(self, area_grouping: bool = True, alphanumeric_sort: bool = True):
        """
        Initialize CSV formatter
        
        Args:
            area_grouping: Enable area-based text grouping
            alphanumeric_sort: Enable alphanumeric sorting within areas
        """
        self.area_grouping = area_grouping
        self.alphanumeric_sort = alphanumeric_sort
    
    def format_text_extraction_results(self, 
                                     text_extraction_files: List[Path],
                                     output_file: Path) -> Dict[str, Any]:
        """
        Format text extraction results into CSV
        
        Args:
            text_extraction_files: List of text extraction JSON files
            output_file: Output CSV file path
            
        Returns:
            Dict with formatting results
        """
        print(f"X Formatting {len(text_extraction_files)} text extraction files to CSV")
        
        results = {
            'csv_files': [],
            'summary_files': [],
            'errors': []
        }
        
        for text_file in text_extraction_files:
            try:
                # Load text extraction data
                with open(text_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"  V Loaded {len(data.get('text_regions', []))} regions from {text_file.name}")
                
                # Extract text regions
                text_regions = data.get('text_regions', [])
                if not text_regions:
                    continue
                
                # Convert to TextRegion objects for processing
                text_region_objects = self._convert_to_text_regions(text_regions, text_file)
                
                if not text_region_objects:
                    continue
                
                # Group by areas if enabled
                if self.area_grouping:
                    grouped_regions = self._group_by_areas(text_region_objects)
                else:
                    # Single group with all regions
                    grouped_regions = {'all_text': text_region_objects}
                
                # Sort within each area if enabled
                if self.alphanumeric_sort:
                    for area_id in grouped_regions:
                        grouped_regions[area_id] = self._sort_alphanumeric(grouped_regions[area_id])
                
                # Assign sequences
                final_regions = self._assign_sequences(grouped_regions)
                
                # Format to CSV
                document_name = text_file.stem.replace('_text_extraction', '')
                csv_file = output_file.parent / f"{document_name}_text_regions.csv"
                
                # Write CSV
                self._write_csv(final_regions, csv_file)
                results['csv_files'].append(str(csv_file))
                
                # Create summary
                self.create_summary_report(final_regions, csv_file)
                results['summary_files'].append(str(csv_file.with_suffix('.summary.json')))
                
            except Exception as e:
                error_msg = f"Error loading {text_file.name}: {e}"
                results['errors'].append(error_msg)
                print(f"  X Error loading {text_file.name}: {e}")
        
        return results
    
    def _convert_to_text_regions(self, text_regions: List[Dict[str, Any]], text_file: Path) -> List[TextRegion]:
        """Convert text region dictionaries to TextRegion objects"""
        regions = []
        
        document_name = text_file.stem.replace('_text_extraction', '')
        
        for region_data in text_regions:
            try:
                # Extract bounding box
                bbox = region_data.get('bbox', [0, 0, 0, 0])
                if len(bbox) >= 4:
                    x, y, x2, y2 = bbox[:4]
                    width = x2 - x
                    height = y2 - y
                else:
                    x = y = width = height = 0
                
                # Determine area information
                area_info = self._determine_area_info(region_data)
                
                region = TextRegion(
                    document=document_name,
                    page=region_data.get('page', 1),
                    area_id=area_info['area_id'],
                    area_type=area_info['area_type'],
                    sequence="",  # Will be assigned later
                    text_content=region_data.get('text', '').strip(),
                    confidence=region_data.get('confidence', 0.0),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    source=region_data.get('source', 'ocr')
                )
                
                # Only add if text content is not empty
                if region.text_content:
                    regions.append(region)
                    
            except Exception as e:
                print(f"    Warning: Error processing region: {e}")
                continue
        
        return regions
    
    def _load_text_regions(self, text_file: Path) -> List[TextRegion]:
        """Load text regions from JSON file"""
        regions = []
        
        with open(text_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        document_name = text_file.parent.name  # Use parent directory name as document name
        
        for region_data in data.get('text_regions', []):
            try:
                # Extract bounding box
                bbox = region_data.get('bbox', [0, 0, 0, 0])
                if len(bbox) >= 4:
                    x, y, x2, y2 = bbox[:4]
                    width = x2 - x
                    height = y2 - y
                else:
                    x = y = width = height = 0
                
                # Determine area information
                area_info = self._determine_area_info(region_data)
                
                region = TextRegion(
                    document=document_name,
                    page=region_data.get('page', 1),
                    area_id=area_info['area_id'],
                    area_type=area_info['area_type'],
                    sequence="",  # Will be assigned later
                    text_content=region_data.get('text', '').strip(),
                    confidence=region_data.get('confidence', 0.0),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    source=region_data.get('source', 'ocr')
                )
                
                # Only add if text content is not empty
                if region.text_content:
                    regions.append(region)
                    
            except Exception as e:
                print(f"    Warning: Error processing region: {e}")
                continue
        
        return regions
    
    def _determine_area_info(self, region_data: Dict[str, Any]) -> Dict[str, str]:
        """Determine area ID and type for a text region"""
        # Check if region has associated symbol information
        associated_symbol = region_data.get('associated_symbol')
        
        if associated_symbol:
            # Use symbol information to determine area
            symbol_class = associated_symbol.get('class_name', 'unknown')
            symbol_bbox = associated_symbol.get('bbox_global', {})
            
            # Create area ID based on symbol location and class
            if isinstance(symbol_bbox, dict):
                x1 = symbol_bbox.get('x1', 0)
                y1 = symbol_bbox.get('y1', 0)
                area_id = f"area_{symbol_class}_{int(x1//100):03d}_{int(y1//100):03d}"
            else:
                area_id = f"area_{symbol_class}_000_000"
            
            # Determine area type based on symbol class
            if symbol_class in ['resistor', 'capacitor', 'inductor', 'diode']:
                area_type = 'component'
            elif symbol_class in ['connector', 'terminal']:
                area_type = 'connection'
            elif symbol_class in ['label', 'text']:
                area_type = 'label'
            else:
                area_type = 'symbol'
        else:
            # No associated symbol - create area based on position
            bbox = region_data.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                x, y = bbox[0], bbox[1]
                area_id = f"area_text_{int(x//100):03d}_{int(y//100):03d}"
            else:
                area_id = "area_text_000_000"
            
            area_type = 'text'
        
        return {
            'area_id': area_id,
            'area_type': area_type
        }
    
    def _group_by_areas(self, regions: List[TextRegion]) -> Dict[str, List[TextRegion]]:
        """Group text regions by area"""
        grouped = {}
        
        for region in regions:
            area_id = region.area_id
            if area_id not in grouped:
                grouped[area_id] = []
            grouped[area_id].append(region)
        
        print(f"  V Grouped into {len(grouped)} areas")
        return grouped
    
    def _sort_alphanumeric(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Sort regions alphanumerically by text content"""
        def alphanumeric_key(region: TextRegion) -> Tuple:
            """Create sorting key for alphanumeric sorting"""
            text = region.text_content.upper()
            
            # Split text into parts (letters and numbers)
            parts = re.findall(r'[A-Z]+|\d+', text)
            
            # Convert to tuple of (string, int) for proper sorting
            key_parts = []
            for part in parts:
                if part.isdigit():
                    key_parts.append((0, int(part)))  # Numbers sort before letters
                else:
                    key_parts.append((1, part))  # Letters
            
            # Add position as secondary sort key
            key_parts.append((2, region.x, region.y))
            
            return tuple(key_parts)
        
        return sorted(regions, key=alphanumeric_key)
    
    def _assign_sequences(self, grouped_regions: Dict[str, List[TextRegion]]) -> List[TextRegion]:
        """Assign sequence numbers within each area"""
        final_regions = []
        
        for area_id, regions in grouped_regions.items():
            for i, region in enumerate(regions, 1):
                # Assign sequence number with zero padding
                region.sequence = f"{i:03d}"
                final_regions.append(region)
        
        return final_regions
    
    def _write_csv(self, regions: List[TextRegion], output_file: Path) -> None:
        """Write regions to CSV file"""
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'sequence_id',
                'area_id', 
                'text',
                'confidence',
                'bbox_x1',
                'bbox_y1', 
                'bbox_x2',
                'bbox_y2',
                'bbox_width',
                'bbox_height',
                'page_number',
                'symbol_association'
            ])
            
            # Write data rows
            for region in regions:
                writer.writerow([
                    region.sequence,
                    region.area_id,
                    region.text_content.replace('\n', ' ').strip(),
                    f"{region.confidence:.3f}",
                    f"{region.x:.1f}",
                    f"{region.y:.1f}",
                    f"{region.x + region.width:.1f}",
                    f"{region.y + region.height:.1f}",
                    f"{region.width:.1f}",
                    f"{region.height:.1f}",
                    region.page,
                    region.area_type
                ])
        
        print(f"  V CSV written to: {output_file}")
        print(f"  V Total rows: {len(regions)}")
    
    def create_summary_report(self, regions: List[TextRegion], output_file: Path) -> None:
        """Create a summary report of the CSV formatting"""
        summary = {
            'total_regions': len(regions),
            'documents': {},
            'areas': {},
            'area_types': {},
            'sources': {}
        }
        
        for region in regions:
            # Count by document
            if region.document not in summary['documents']:
                summary['documents'][region.document] = 0
            summary['documents'][region.document] += 1
            
            # Count by area
            if region.area_id not in summary['areas']:
                summary['areas'][region.area_id] = 0
            summary['areas'][region.area_id] += 1
            
            # Count by area type
            if region.area_type not in summary['area_types']:
                summary['area_types'][region.area_type] = 0
            summary['area_types'][region.area_type] += 1
            
            # Count by source
            if region.source not in summary['sources']:
                summary['sources'][region.source] = 0
            summary['sources'][region.source] += 1
        
        # Write summary
        summary_file = output_file.parent / f"{output_file.stem}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Summary report: {summary_file}")
