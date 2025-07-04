# OCR Module Documentation

## Overview

The OCR module handles text extraction from PLC diagrams using PaddleOCR. This module processes detection results to extract text content from identified symbol regions and provides coordinate calibration for accurate text positioning.

## Architecture

### Core Components

```
src/ocr/
├── Core Pipeline
│   ├── text_extraction_pipeline.py    # Main text extraction orchestrator
│   ├── paddle_ocr.py                  # PaddleOCR wrapper and utilities
│   └── run_text_extraction.py         # Standalone text extraction script
├── Detection Processing
│   ├── detection_preprocessor.py      # Detection data preprocessing
│   ├── detection_format_converter.py  # Format conversion utilities
│   ├── detection_deduplication.py     # Duplicate detection removal
│   └── check_detection_structure.py   # Detection data validation
├── Coordinate Management
│   ├── coordinate_calibration.py      # Coordinate system calibration
│   ├── relative_coordinate_system.py  # Relative positioning utilities
│   └── roi_preprocessing.py           # Region of interest processing
├── Visualization
│   └── visualize_text_extraction.py   # Text extraction visualization
└── Testing
    └── test_paddleocr_versions.py     # PaddleOCR version testing
```

## File-by-File Documentation

### Core Pipeline

#### `text_extraction_pipeline.py`
**Purpose**: Main orchestrator for text extraction from detection results
**Functionality**:
- Processes detection JSON files to extract text from symbol regions
- Coordinates between detection data and OCR processing
- Manages batch processing of multiple detection files
- Handles output formatting and result aggregation

**Key Classes**:
- `TextExtractionPipeline`: Main pipeline orchestrator

**Usage**:
```python
from src.ocr.text_extraction_pipeline import TextExtractionPipeline

pipeline = TextExtractionPipeline(confidence_threshold=0.7)
results = pipeline.process_detection_folder(detection_folder, pdf_folder, output_folder)
```

#### `paddle_ocr.py`
**Purpose**: PaddleOCR wrapper with optimized configuration for PLC diagrams
**Functionality**:
- Provides simplified interface to PaddleOCR functionality
- Handles OCR model initialization and configuration
- Manages text detection and recognition on image regions
- Includes error handling for OCR processing issues

**Key Functions**:
- `initialize_paddle_ocr()`: OCR model initialization
- `extract_text_from_region()`: Text extraction from image regions
- `process_image_batch()`: Batch processing capabilities

#### `run_text_extraction.py`
**Purpose**: Standalone script for text extraction operations
**Functionality**:
- Command-line interface for text extraction
- Batch processing of detection results
- Integration with existing pipeline workflows

**Usage**:
```bash
python src/ocr/run_text_extraction.py --detection-folder path/to/detections --output-folder path/to/output
```

### Detection Processing

#### `detection_preprocessor.py`
**Purpose**: Preprocessing of detection data for OCR processing
**Functionality**:
- Validates detection data structure and format
- Filters detections by confidence thresholds
- Prepares detection regions for OCR processing
- Handles coordinate system transformations

#### `detection_format_converter.py`
**Purpose**: Conversion between different detection data formats
**Functionality**:
- Converts between YOLO and custom detection formats
- Handles coordinate system conversions
- Provides format validation and error checking

#### `detection_deduplication.py`
**Purpose**: Removal of duplicate detections before OCR processing
**Functionality**:
- Identifies overlapping detection regions
- Removes duplicate detections based on IoU thresholds
- Preserves highest confidence detections
- Optimizes OCR processing efficiency

#### `check_detection_structure.py`
**Purpose**: Validation of detection data structure and integrity
**Functionality**:
- Validates detection JSON file structure
- Checks coordinate validity and bounds
- Verifies detection metadata consistency
- Provides diagnostic information for troubleshooting

### Coordinate Management

#### `coordinate_calibration.py`
**Purpose**: Calibration of coordinate systems between detection and text extraction
**Functionality**:
- Handles coordinate system mismatches between detection and OCR
- Provides transformation matrices for coordinate correction
- Calibrates text positioning relative to detection boxes
- Includes validation of coordinate transformations

**Key Classes**:
- `CoordinateCalibrator`: Main calibration functionality

#### `relative_coordinate_system.py`
**Purpose**: Management of relative coordinate positioning
**Functionality**:
- Converts between absolute and relative coordinate systems
- Handles coordinate scaling and transformation
- Provides utilities for coordinate system validation

#### `roi_preprocessing.py`
**Purpose**: Region of interest preprocessing for OCR
**Functionality**:
- Extracts regions of interest from detection results
- Preprocesses image regions for optimal OCR performance
- Handles image enhancement and noise reduction
- Manages batch processing of ROI extraction

### Visualization

#### `visualize_text_extraction.py`
**Purpose**: Visualization tools for text extraction results
**Functionality**:
- Creates visual overlays of text extraction results
- Generates comparison images showing detection vs text extraction
- Provides debugging visualizations for coordinate calibration
- Supports batch visualization processing

### Testing

#### `test_paddleocr_versions.py`
**Purpose**: Testing and validation of PaddleOCR installation and versions
**Functionality**:
- Tests PaddleOCR installation and functionality
- Validates OCR model loading and processing
- Provides diagnostic information for troubleshooting
- Tests compatibility with different PaddleOCR versions

## Integration with Pipeline System

### Stage Integration

The OCR module integrates with the main pipeline through:

1. **OCR Stage**: `src/pipeline/stages/ocr_stage.py`
2. **OCR Worker**: `src/workers/ocr_worker.py`
3. **Multi-Environment Manager**: Isolated execution in `ocr_env`

### Input/Output Flow

```
Detection Results → OCR Processing → Text Extraction Results
     ↓                    ↓                    ↓
detection_*.json → text_extraction_pipeline → *_text_extraction.json
```

### Dependencies

- **PaddleOCR**: Core OCR functionality
- **OpenCV**: Image processing
- **NumPy**: Numerical operations
- **PIL**: Image handling

## Configuration

### OCR Parameters

- **confidence_threshold**: OCR confidence threshold (default: 0.7)
- **language**: OCR language setting (default: 'en')
- **use_gpu**: Enable GPU acceleration for OCR (default: True if available)
- **batch_size**: Batch processing size (default: 8)

### Detection Processing

- **min_detection_confidence**: Minimum detection confidence for OCR processing
- **deduplication_threshold**: IoU threshold for duplicate removal
- **roi_padding**: Padding around detection regions for OCR

## Usage Examples

### Basic Text Extraction

```python
from src.ocr.text_extraction_pipeline import TextExtractionPipeline

# Initialize pipeline
pipeline = TextExtractionPipeline(
    confidence_threshold=0.7,
    ocr_lang='en'
)

# Process detection folder
results = pipeline.process_detection_folder(
    detection_folder="path/to/detections",
    pdf_folder="path/to/pdfs",
    output_folder="path/to/output"
)
```

### Standalone Processing

```bash
# Basic text extraction
python src/ocr/run_text_extraction.py --detection-folder detections/ --output-folder output/

# With custom parameters
python src/ocr/run_text_extraction.py \
    --detection-folder detections/ \
    --output-folder output/ \
    --confidence 0.8 \
    --language en
```

### Coordinate Calibration

```python
from src.ocr.coordinate_calibration import CoordinateCalibrator

calibrator = CoordinateCalibrator()
result = calibrator.calibrate_text_extraction_file(
    text_file="text_extraction.json",
    output_file="calibrated_text.json"
)
```

## Error Handling and Troubleshooting

### Common Issues

1. **PaddleOCR Installation Issues**
   - **Cause**: Missing or incompatible PaddleOCR installation
   - **Solution**: Install PaddleOCR in `ocr_env` environment
   - **Command**: `pip install paddlepaddle paddleocr`

2. **GPU Memory Issues**
   - **Cause**: Insufficient GPU memory for OCR processing
   - **Solution**: Reduce batch size or use CPU processing
   - **Parameter**: Set `use_gpu=False` or reduce `batch_size`

3. **Coordinate Misalignment**
   - **Cause**: Coordinate system mismatch between detection and OCR
   - **Solution**: Use coordinate calibration functionality
   - **Tool**: `coordinate_calibration.py`

4. **Poor OCR Accuracy**
   - **Cause**: Low quality detection regions or inappropriate OCR settings
   - **Solution**: Adjust confidence thresholds and preprocessing
   - **Parameters**: Increase `confidence_threshold`, enable ROI preprocessing

### Debugging Tools

1. **Visualization**: Use `visualize_text_extraction.py` for visual debugging
2. **Structure Validation**: Use `check_detection_structure.py` for data validation
3. **OCR Testing**: Use `test_paddleocr_versions.py` for OCR functionality testing
4. **Coordinate Validation**: Use coordinate calibration tools for positioning issues

## Performance Optimization

### OCR Performance

- **GPU Acceleration**: Enable GPU processing when available
- **Batch Processing**: Process multiple regions in batches
- **ROI Optimization**: Preprocess regions for better OCR accuracy
- **Confidence Filtering**: Filter low-confidence detections before OCR

### Memory Management

- **Batch Size Control**: Adjust batch size based on available memory
- **Image Preprocessing**: Optimize image sizes for OCR processing
- **Resource Cleanup**: Proper cleanup of OCR resources after processing

## Development Guidelines

### Adding New Features

1. **Maintain Modularity**: Keep OCR components focused and independent
2. **Error Handling**: Implement comprehensive error handling for OCR operations
3. **Configuration**: Use configurable parameters for OCR settings
4. **Testing**: Include validation and testing for new OCR functionality

### Code Standards

1. **Documentation**: Comprehensive docstrings for OCR functions
2. **Error Messages**: Clear, actionable error messages for OCR issues
3. **Logging**: Appropriate logging for OCR processing steps
4. **Performance**: Consider memory and processing efficiency

## Future Enhancements

### Planned Improvements

1. **Multi-Language Support**: Enhanced support for multiple languages
2. **OCR Model Selection**: Support for different OCR models and engines
3. **Advanced Preprocessing**: Improved image preprocessing for better accuracy
4. **Real-time Processing**: Support for real-time text extraction

### Research Directions

1. **Custom OCR Models**: Training custom OCR models for PLC-specific text
2. **Text Understanding**: Semantic understanding of extracted text content
3. **Layout Analysis**: Advanced layout analysis for complex diagrams
4. **Quality Assessment**: Automatic quality assessment of text extraction results

## Conclusion

The OCR module provides working text extraction capabilities for PLC diagrams using PaddleOCR. The system handles detection processing, coordinate management, and text extraction with basic error handling and performance considerations.

The modular architecture supports both standalone and pipeline-integrated usage, while coordinate calibration helps address common positioning issues. This documentation covers the current implementation and provides guidance for usage and troubleshooting.
