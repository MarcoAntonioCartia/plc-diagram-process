# Source Code Documentation

## Overview

The `src/` directory contains the core source code for the PLC Diagram Processor. This directory is organized into modular components that handle different aspects of the pipeline, from configuration management to final output generation.

## Directory Structure

```
src/
├── README.md                    # This file - source code overview
├── config.py                    # Centralized configuration management
├── run_pipeline.py              # Main pipeline runner with stage management
├── integration_guide.md         # Multi-environment integration guide
├── detection/                   # Symbol detection pipeline
├── ocr/                         # Text extraction and OCR processing
├── pipeline/                    # Stage-based pipeline management
├── postprocessing/              # Output formatting and CSV generation
├── preprocessing/               # PDF and image preprocessing
├── structuring/                 # Data structuring and organization
├── utils/                       # Shared utilities and helpers
├── verification/                # Quality assurance and validation
├── workers/                     # Multi-environment worker processes
├── legacy/                      # Legacy code and deprecated modules
└── environments/                # Environment configurations (legacy)
```

## Core Components

### Configuration Management
- **`config.py`**: Centralized configuration system with automatic path resolution and environment detection

### Pipeline Management
- **`run_pipeline.py`**: Modern stage-based pipeline runner with CLI interface
- **`pipeline/`**: Stage management system with dependency tracking and state persistence

### Processing Modules

#### Detection Pipeline (`detection/`)
**Purpose**: YOLO-based symbol detection in PLC diagrams
**Key Components**:
- Symbol detection using YOLO11 models
- Coordinate transformation from image snippets to PDF coordinates
- Training and inference pipelines
- GPU-optimized processing

#### OCR Pipeline (`ocr/`)
**Purpose**: Text extraction from detected regions and PDF content
**Key Components**:
- Hybrid text extraction (PDF + OCR)
- PaddleOCR integration for image-based text
- PyMuPDF for direct PDF text extraction
- PLC-specific pattern recognition

#### Postprocessing (`postprocessing/`)
**Purpose**: Output formatting and final result generation
**Key Components**:
- CSV formatting with area-based grouping
- Enhanced PDF creation with annotations
- Data aggregation and summary generation
- Quality metrics and statistics

### Support Systems

#### Preprocessing (`preprocessing/`)
**Purpose**: Input data preparation and format conversion
**Key Components**:
- PDF to image snippet conversion
- Image preprocessing and optimization
- Batch processing utilities

#### Utilities (`utils/`)
**Purpose**: Shared utilities and helper functions
**Key Components**:
- Multi-environment management
- GPU resource management
- PDF creation and annotation tools
- Storage management and cleanup
- Progress tracking and display

#### Workers (`workers/`)
**Purpose**: Multi-environment subprocess execution
**Key Components**:
- Detection worker (PyTorch environment)
- OCR worker (PaddleOCR environment)
- Training worker (YOLO training environment)
- Inter-process communication handling

#### Verification (`verification/`)
**Purpose**: Quality assurance and validation
**Key Components**:
- Result validation and quality checks
- Error detection and reporting
- Performance monitoring

#### Structuring (`structuring/`)
**Purpose**: Data organization and relationship mapping
**Key Components**:
- Symbol-text association algorithms
- Spatial relationship analysis
- Data structure optimization

## Architecture Principles

### Modular Design
Each directory represents a distinct functional area with clear responsibilities and minimal cross-dependencies.

### Multi-Environment Support
The system supports both single-environment and multi-environment execution modes to handle conflicting dependencies (PyTorch vs PaddlePaddle).

### Stage-Based Processing
The pipeline is organized into discrete stages with dependency tracking, state persistence, and resumable execution.

### Configuration-Driven
All components use centralized configuration management for consistent behavior and easy customization.

## Key Features

### Environment Isolation
- **Multi-Environment Mode**: Separate environments for YOLO and OCR to prevent CUDA conflicts
- **Worker Processes**: Subprocess-based execution for heavy dependencies
- **Automatic Management**: Transparent environment switching and resource management

### Pipeline Management
- **Stage Dependencies**: Automatic dependency resolution and execution ordering
- **State Persistence**: Resume interrupted pipelines from last completed stage
- **Error Recovery**: Robust error handling with detailed diagnostics
- **Progress Tracking**: Real-time progress monitoring and reporting

### Performance Optimization
- **GPU Management**: Intelligent GPU resource allocation and conflict resolution
- **Parallel Processing**: Multi-threaded and multi-process execution where beneficial
- **Memory Management**: Efficient memory usage and cleanup
- **Caching**: Intelligent caching of intermediate results

### Quality Assurance
- **CI/CD Integration**: CI-safe execution with mock dependencies
- **Validation**: Comprehensive input and output validation
- **Testing**: Extensive test coverage with automated testing
- **Monitoring**: Performance and quality monitoring

## Usage Patterns

### Basic Pipeline Execution
```bash
# Run complete pipeline
python src/run_pipeline.py --run-all

# Run specific stages
python src/run_pipeline.py --stages detection ocr postprocessing

# Show pipeline status
python src/run_pipeline.py --status
```

### Multi-Environment Mode
```bash
# Enable multi-environment mode
python src/run_pipeline.py --run-all --multi-env

# Single environment mode (legacy)
python src/run_pipeline.py --run-all --single-env
```

### Development and Debugging
```bash
# Verbose output for debugging
python src/run_pipeline.py --run-all --verbose

# Force re-run specific stages
python src/run_pipeline.py --stages detection --force

# Reset pipeline state
python src/run_pipeline.py --reset
```

## Development Guidelines

### Code Organization
1. **Single Responsibility**: Each module should have a clear, single purpose
2. **Minimal Dependencies**: Avoid unnecessary cross-module dependencies
3. **Configuration**: Use centralized configuration for all settings
4. **Error Handling**: Implement comprehensive error handling and logging

### Multi-Environment Compatibility
1. **Heavy Dependencies**: Place heavy dependencies in worker processes
2. **Lazy Imports**: Use lazy imports for optional dependencies
3. **CI Safety**: Provide mock implementations for CI environments
4. **Environment Detection**: Use runtime environment detection

### Performance Considerations
1. **GPU Usage**: Implement proper GPU resource management
2. **Memory Management**: Clean up resources and avoid memory leaks
3. **Parallel Processing**: Use parallel processing where beneficial
4. **Caching**: Cache expensive operations appropriately

### Testing and Quality
1. **Unit Tests**: Write comprehensive unit tests for all modules
2. **Integration Tests**: Test multi-module interactions
3. **CI Testing**: Ensure all code works in CI environments
4. **Documentation**: Maintain up-to-date documentation

## Integration Points

### External Dependencies
- **PyTorch**: YOLO detection and training (isolated in detection environment)
- **PaddlePaddle**: OCR processing (isolated in OCR environment)
- **OpenCV**: Image processing (shared across environments)
- **PyMuPDF**: PDF processing (core environment)

### Data Flow
```
Input PDFs → Preprocessing → Detection → OCR → Postprocessing → Output
     ↓            ↓           ↓        ↓         ↓           ↓
  Raw PDFs → Image Snippets → Symbols → Text → CSV/Enhanced PDFs
```

### Configuration Flow
```
config.py → Stage Configs → Worker Configs → Module Configs
    ↓           ↓              ↓              ↓
Central → Stage-specific → Environment → Component Settings
```

## Future Enhancements

### Planned Improvements
1. **Cloud Integration**: Support for cloud-based processing
2. **Distributed Processing**: Multi-machine pipeline execution
3. **Advanced Caching**: Intelligent result caching and reuse
4. **Real-time Processing**: Support for real-time diagram processing

### Research Directions
1. **AI Integration**: Advanced AI models for improved accuracy
2. **Optimization**: Performance optimization and resource efficiency
3. **Scalability**: Horizontal scaling and load balancing
4. **Automation**: Automated model training and optimization

## Troubleshooting

### Common Issues
1. **Import Errors**: Check environment setup and dependencies
2. **CUDA Conflicts**: Use multi-environment mode to resolve conflicts
3. **Memory Issues**: Monitor GPU and system memory usage
4. **Performance**: Check GPU utilization and parallel processing settings

### Debugging Tools
1. **Verbose Mode**: Enable detailed logging and progress tracking
2. **Stage Status**: Check individual stage completion and errors
3. **Environment Health**: Validate multi-environment setup
4. **Resource Monitoring**: Monitor GPU and memory usage

## Contributing

### Development Setup
1. **Environment Setup**: Use multi-environment setup for development
2. **Code Standards**: Follow PEP8 and project coding standards
3. **Testing**: Write tests for new functionality
4. **Documentation**: Update documentation for changes

### Code Review Process
1. **Functionality**: Ensure code works as intended
2. **Performance**: Check for performance implications
3. **Compatibility**: Verify multi-environment compatibility
4. **Documentation**: Review documentation updates

This source code architecture provides a robust, scalable, and maintainable foundation for the PLC Diagram Processor, with clear separation of concerns and support for complex processing workflows.
