# Output Module Documentation

## Overview

The Output module handles the formatting and organization of final pipeline results. This module provides utilities for CSV formatting, area grouping, and result compilation to create structured output from the PLC diagram processing pipeline.

## Architecture

### Core Components

```
src/output/
├── Data Formatting
│   └── csv_formatter.py        # CSV output formatting and structure
├── Result Organization
│   └── area_grouper.py         # Spatial grouping and area analysis
└── Module Initialization
    └── __init__.py             # Output module initialization
```

## File-by-File Documentation

### Data Formatting

#### `csv_formatter.py`
**Purpose**: CSV output formatting and data structure management
**Functionality**:
- Formats detection and OCR results into structured CSV output
- Handles coordinate normalization and data validation
- Provides standardized column structure for analysis tools
- Includes data type conversion and error handling

**Key Functions**:
- `format_detection_results()`: Format detection data for CSV output
- `format_text_extraction_results()`: Format OCR results for CSV output
- `combine_detection_and_text()`: Merge detection and text data
- `validate_csv_structure()`: Validate output data structure

**CSV Output Structure**:
```csv
symbol_id,symbol_type,confidence,x_min,y_min,x_max,y_max,text_content,page_number,source_file
1,resistor,0.95,100,200,150,250,"R1",1,diagram_001.pdf
2,capacitor,0.87,300,400,350,450,"C2",1,diagram_001.pdf
```

**Data Processing**:
- **Coordinate Normalization**: Converts coordinates to consistent format
- **Confidence Filtering**: Filters results by confidence thresholds
- **Text Integration**: Combines detection boxes with extracted text
- **Metadata Addition**: Adds source file and page information

### Result Organization

#### `area_grouper.py`
**Purpose**: Spatial grouping and area analysis of detection results
**Functionality**:
- Groups nearby detections into logical areas or circuits
- Analyzes spatial relationships between detected symbols
- Provides hierarchical organization of detection results
- Includes area-based statistics and analysis

**Key Functions**:
- `group_detections_by_area()`: Group detections into spatial areas
- `analyze_spatial_relationships()`: Analyze symbol relationships
- `create_hierarchical_structure()`: Create nested result structure
- `calculate_area_statistics()`: Generate area-based statistics

**Grouping Algorithms**:
- **Proximity Grouping**: Groups symbols based on spatial proximity
- **Grid-Based Grouping**: Organizes symbols into grid-based areas
- **Connectivity Analysis**: Groups based on electrical connectivity
- **Hierarchical Clustering**: Creates nested grouping structures

**Output Structure**:
```json
{
  "areas": [
    {
      "area_id": "area_001",
      "bounds": {"x_min": 100, "y_min": 200, "x_max": 500, "y_max": 600},
      "symbols": [
        {"symbol_id": "1", "type": "resistor", "position": [125, 225]},
        {"symbol_id": "2", "type": "capacitor", "position": [175, 275]}
      ],
      "statistics": {
        "symbol_count": 2,
        "area_size": 160000,
        "density": 0.0000125
      }
    }
  ]
}
```

## Integration with Pipeline System

### Pipeline Integration

The output module integrates with the pipeline through:

1. **Enhancement Stage**: Final result formatting and compilation
2. **Detection Stage**: Intermediate result formatting
3. **OCR Stage**: Text extraction result formatting
4. **Configuration System**: Output format and structure configuration

### Data Flow

```
Detection Results → Output Formatting → Structured Output
     ↓                    ↓                   ↓
detection.json → csv_formatter.py → results.csv
     +                    +                   +
text_extraction.json → area_grouper.py → grouped_results.json
```

## Usage Examples

### CSV Formatting

```python
from src.output.csv_formatter import format_detection_results

# Format detection results
csv_data = format_detection_results(
    detection_file="detection_results.json",
    text_file="text_extraction.json",
    output_file="formatted_results.csv"
)

# Custom formatting options
csv_data = format_detection_results(
    detection_file="detection_results.json",
    confidence_threshold=0.5,
    include_text=True,
    normalize_coordinates=True
)
```

### Area Grouping

```python
from src.output.area_grouper import group_detections_by_area

# Group detections by spatial proximity
grouped_results = group_detections_by_area(
    detection_file="detection_results.json",
    grouping_method="proximity",
    proximity_threshold=100
)

# Grid-based grouping
grouped_results = group_detections_by_area(
    detection_file="detection_results.json",
    grouping_method="grid",
    grid_size=(200, 200)
)
```

### Combined Processing

```python
from src.output.csv_formatter import format_detection_results
from src.output.area_grouper import group_detections_by_area

# Process detection results
detection_file = "detection_results.json"
text_file = "text_extraction.json"

# Format as CSV
csv_output = format_detection_results(
    detection_file=detection_file,
    text_file=text_file,
    output_file="results.csv"
)

# Group by areas
grouped_output = group_detections_by_area(
    detection_file=detection_file,
    output_file="grouped_results.json"
)
```

## Configuration

### Output Format Configuration

- **csv_delimiter**: CSV delimiter character (default: ',')
- **coordinate_precision**: Decimal precision for coordinates (default: 2)
- **include_confidence**: Include confidence scores in output (default: True)
- **normalize_coordinates**: Normalize coordinates to page dimensions (default: False)

### Grouping Configuration

- **grouping_method**: Spatial grouping method ('proximity', 'grid', 'connectivity')
- **proximity_threshold**: Distance threshold for proximity grouping (pixels)
- **grid_size**: Grid cell size for grid-based grouping [width, height]
- **min_group_size**: Minimum symbols per group (default: 1)

### Output Structure Configuration

- **output_format**: Output format ('csv', 'json', 'both')
- **include_metadata**: Include source file and processing metadata
- **hierarchical_output**: Create hierarchical nested structure
- **statistics_enabled**: Include area and grouping statistics

## Error Handling and Troubleshooting

### Common Issues

1. **Invalid Detection Data**
   - **Cause**: Malformed or missing detection result files
   - **Solution**: Validate detection data structure and format
   - **Tool**: Use data validation functions

2. **Coordinate System Mismatches**
   - **Cause**: Inconsistent coordinate systems between detection and text
   - **Solution**: Use coordinate normalization and validation
   - **Parameter**: Enable `normalize_coordinates=True`

3. **Empty or Missing Results**
   - **Cause**: No detections found or processing failures
   - **Solution**: Check detection confidence thresholds and input data
   - **Parameter**: Lower confidence thresholds or validate input

4. **Grouping Algorithm Issues**
   - **Cause**: Inappropriate grouping parameters for data
   - **Solution**: Adjust grouping parameters based on diagram characteristics
   - **Parameter**: Modify proximity thresholds or grid sizes

### Debugging Tools

1. **Data Validation**: Validate input data structure and content
2. **Visualization**: Visualize grouping results for validation
3. **Statistics**: Use area statistics to validate grouping quality
4. **Format Checking**: Validate output format and structure

## Performance Optimization

### Processing Performance

- **Batch Processing**: Process multiple files efficiently
- **Memory Management**: Optimize memory usage for large datasets
- **Algorithm Selection**: Choose appropriate grouping algorithms
- **Data Filtering**: Filter data early to reduce processing overhead

### Output Optimization

- **Format Selection**: Choose optimal output format for use case
- **Compression**: Use compression for large output files
- **Streaming**: Stream output for large datasets
- **Indexing**: Create indexes for fast data access

## Development Guidelines

### Adding New Features

1. **Maintain Compatibility**: Preserve existing output formats and interfaces
2. **Data Validation**: Implement comprehensive data validation
3. **Error Handling**: Include robust error handling for data processing
4. **Documentation**: Document output formats and data structures

### Code Standards

1. **Documentation**: Comprehensive docstrings for output functions
2. **Error Messages**: Clear, actionable error messages for data issues
3. **Logging**: Appropriate logging for output processing steps
4. **Testing**: Unit tests for output formatting and validation

## Future Enhancements

### Planned Improvements

1. **Advanced Grouping**: More sophisticated spatial grouping algorithms
2. **Export Formats**: Support for additional output formats (XML, Excel, etc.)
3. **Interactive Visualization**: Interactive visualization of grouped results
4. **Statistical Analysis**: Advanced statistical analysis of detection results

### Research Directions

1. **Machine Learning Grouping**: ML-based intelligent grouping algorithms
2. **Semantic Analysis**: Semantic understanding of symbol relationships
3. **Quality Assessment**: Automatic quality assessment of output results
4. **Integration Standards**: Standard formats for CAD/engineering tool integration

## Conclusion

The Output module provides working result formatting and organization capabilities for the PLC Diagram Processor pipeline. The system handles CSV formatting, spatial grouping, and result compilation with focus on structured output generation.

The modular architecture supports different output formats and grouping strategies while maintaining data integrity and validation. This documentation covers the current implementation and provides guidance for usage and development.
