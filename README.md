# PLC Diagram Processor

**End-to-end pipeline for industrial PLC diagram analysis** using open-source AI models, local deployment, and dual AI systems for processing and verification.

## Overview

The PLC Diagram Processor is a comprehensive solution for automated analysis of industrial PLC (Programmable Logic Controller) diagrams. It combines open source AI models to extract structured data from technical drawings.

### Key Features

| Component               | Model / Tool              | Purpose                                      |
|-------------------------|---------------------------|----------------------------------------------|
| **Symbol Detection**    | YOLO11 (Ultralytics)     | Custom-trained PLC symbol detection          |
| **OCR & Text Extraction** | PaddleOCR (PP-OCRv4)    | High-accuracy text region recognition        |
| **Data Structuring**    | LayoutLMv3 (Microsoft)    | Multimodal mapping of text to symbols        |
| **Logical Verification**| DeBERTa-v3 + Drools       | Industrial standards & rule-based validation |
| **Visual Verification** | Segment Anything Model    | Highlight errors overlayed on original diagram |
| **Deployment**          | Docker, ONNX, Triton      | Containerized, optimized local inference     |
- **Flexible Deployment**: Single-environment or multi-environment modes

## Detection Pipeline Architecture

The detection stage implements a complete end-to-end pipeline for processing PLC diagrams:

### Pipeline Flow
```
PDF Diagrams → Image Snippets → YOLO11 Detection → Coordinate Transform → Reconstructed PDFs
```

## Architecture

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Symbol Detection** | YOLO11 (Ultralytics) | Custom-trained PLC symbol detection |
| **Text Extraction** | PaddleOCR + PyMuPDF | Hybrid OCR and direct PDF text extraction |
| **Environment Management** | Multi-venv system | Isolated GPU environments |
| **Pipeline Coordination** | Worker processes | Subprocess-based task delegation |
| **PDF Processing** | Poppler + PIL | PDF-to-image conversion and reconstruction |
| **Data Management** | Network drive integration | Centralized dataset and model storage |

### Stage 1: PDF Processing
- **Input**: PDF files in `../plc-data/raw/pdfs/`
- **Process**: Convert PDFs to overlapping image snippets using `src/preprocessing/SnipPdfToPng.py`
- **Output**: PNG snippets with metadata in `../plc-data/processed/images/`

### Stage 2: Symbol Detection
- **Model**: YOLO11m fine-tuned on PLC symbols
- **Process**: Detect symbols in each image snippet
- **Output**: Detection results with snippet-relative coordinates

### Stage 3: Coordinate Transformation
- **Process**: Convert snippet coordinates to global PDF coordinates using metadata
- **Logic**: `global_x = snippet_global_x + snippet_relative_x`
- **Output**: Detection results with global PDF coordinates

### Stage 4: PDF Reconstruction
- **Process**: Reconstruct original PDFs with detection overlays
- **Features**: Bounding boxes, labels, confidence scores, detection numbering
- **Output**: Labeled PDFs and coordinate mapping files in `../plc-data/processed/`

### Output Files Generated
For each processed PDF:
- `{pdf_name}_detected.pdf`: PDF with detection overlays and labels
- `{pdf_name}_detections.json`: All detections with global coordinates
- `{pdf_name}_coordinates.txt`: Human-readable coordinate mapping
- `{pdf_name}_statistics.json`: Detection statistics and metrics
- `{pdf_name}_page_{n}_detected.png`: Individual page images with detections

## Repository Structure

```
plc-diagram-processor/
├── README.md
├── launch.py                    # GPU environment launcher
├── activate.bat / activate.sh   # Environment activation scripts
├── requirements*.txt            # Dependency specifications
├── setup/                       # Setup and configuration
│   ├── setup.py                 # Unified setup script
│   ├── manage_datasets.py       # Dataset management
│   ├── manage_models.py         # Model management
│   ├── gpu_detector.py          # GPU capability detection
│   ├── build_tools_installer.py # Build environment setup
│   └── config/
│       └── download_config.yaml # Storage configuration
├── bin/                         # Executable scripts
│   └── create_enhanced_pdf.py   # Complete pipeline script
├── src/
│   ├── run_pipeline.py          # Main pipeline runner
│   ├── config.py                # Configuration management
│   ├── detection/               # Symbol detection pipeline
│   │   ├── detection_manager.py      # Detection coordinator (no heavy imports)
│   │   ├── lightweight_pipeline_runner.py # Multi-env compatible runner
│   │   ├── detect_pipeline_subprocess.py # Subprocess detection runner
│   │   ├── yolo11_train.py            # YOLO11 training
│   │   ├── yolo11_infer.py            # YOLO11 inference
│   │   ├── detect_pipeline.py         # Core detection pipeline
│   │   ├── coordinate_transform.py    # Coordinate transformation
│   │   └── run_complete_pipeline.py   # Legacy single-env runner
│   ├── ocr/                     # Text extraction pipeline
│   │   ├── paddle_ocr.py              # PaddleOCR integration
│   │   ├── run_text_extraction.py     # Text extraction runner
│   │   └── text_extraction_pipeline.py # Text processing pipeline
│   ├── workers/                 # Multi-environment workers
│   │   ├── detection_worker.py        # Detection subprocess worker
│   │   └── ocr_worker.py              # OCR subprocess worker
│   ├── utils/                   # Utility modules
│   │   ├── runtime_flags.py           # Runtime environment flags
│   │   ├── multi_env_manager.py       # Multi-environment coordinator
│   │   ├── gpu_manager.py             # GPU resource management
│   │   ├── dataset_manager.py         # Dataset operations
│   │   ├── model_manager.py           # Model operations
│   │   ├── network_drive_manager.py   # Network storage backend
│   │   ├── pdf_annotator.py           # Native PDF annotation system
│   │   ├── cleanup_storage.py         # Storage management utility
│   │   └── fix_config_path.py         # Configuration path fixer
│   └── preprocessing/           # Data preprocessing
│       ├── SnipPdfToPng.py           # PDF to image conversion
│       └── SnipPngToPdf.py           # PDF reconstruction
├── tests/                       # Test suite
│   ├── validate_setup.py        # Setup validation
│   ├── test_pipeline.py         # Pipeline testing
│   └── test_text_extraction.py  # Text extraction testing
└── environments/                # Multi-environment setup
    ├── detection_env/           # Detection-specific environment
    └── ocr_env/                 # OCR-specific environment

../plc-data/                     # Data directory (sibling to project)
├── datasets/                    # Training datasets
├── models/                      # Model storage
│   ├── pretrained/              # Downloaded models
│   └── custom/                  # Trained models
├── processed/                   # Pipeline outputs
├── raw/                         # Input PDFs
└── runs/                        # Training runs
```

## Quickstart

### 1. Clone Repository
```bash
git clone https://github.com/your-org/plc-diagram-processor.git
cd plc-diagram-processor
```

# Complete setup with multi-environment support (recommended)
python setup/setup.py --multi-env
```

The setup script will:
- Install system dependencies (Poppler, build tools)
- Create virtual environments (`plcdp` core + `detection_env` + `ocr_env`)
- Install Python dependencies in appropriate environments
- Set up data directory structure
- Optionally download datasets and models from network drive

### 2. Activate Environment

```bash
# Windows
.\activate.bat

# Linux/macOS
source activate.sh
```

### 3. Run Pipeline

**Complete pipeline (detection + text extraction):**
```bash
python src/run_pipeline.py --mode multi --model yolo11m.pt --conf 0.8 --ocr-confidence 0.8
```

**Detection only:**
```bash
python src/run_pipeline.py --mode multi --model yolo11m.pt --conf 0.8 --skip-text
```

**Text extraction only (requires existing detections):**
```bash
python src/run_pipeline.py --skip-detection --ocr-confidence 0.8
```

## Pipeline Modes

### Multi-Environment Mode (Recommended)

Runs detection and OCR in isolated environments to prevent CUDA conflicts:

```bash
python src/run_pipeline.py --mode multi [options]
```

**Benefits:**
-  No CUDA library conflicts
-  Isolated dependencies
-  Better error isolation
-  Parallel processing capability

### Single-Environment Mode (Legacy)

Runs everything in the main environment:

```bash
python src/run_pipeline.py --mode single [options]
```

**Use cases:**
- CPU-only processing
- Development/debugging
- Minimal resource environments

## Data Flow

```
PDF Diagrams → Image Snippets → Symbol Detection → Text Extraction → Structured Output
     ↓              ↓                  ↓               ↓              ↓
  Poppler      Preprocessing      YOLO11 Worker    PaddleOCR      JSON Results
                                                   Worker
```

### Stage 1: PDF Processing
- **Input**: PDF files in `../plc-data/raw/pdfs/`
- **Process**: Convert to overlapping image snippets
- **Output**: PNG snippets with metadata

### Stage 2: Symbol Detection
- **Environment**: `detection_env` (multi-mode) or `plcdp` (single-mode)
- **Model**: YOLO11 (configurable variant)
- **Process**: Detect PLC symbols in image snippets
- **Output**: Detection results with coordinates

### Stage 3: Text Extraction
- **Environment**: `ocr_env` (multi-mode) or `plcdp` (single-mode)
- **Engine**: PaddleOCR + PyMuPDF hybrid
- **Process**: Extract text and associate with symbols
- **Output**: Structured text data with PLC patterns

### Stage 4: Output Generation
- **PDF**: Reconstructed diagrams with overlays
- **JSON**: Structured detection and text data
- **Statistics**: Comprehensive processing metrics

## Configuration

### Detection Parameters
```bash
--model yolo11m.pt           # Model file (in plc-data/models/)
--conf 0.8                   # Detection confidence threshold
--snippet-size 1500 1200     # Snippet dimensions
--overlap 500                # Snippet overlap
```

### Text Extraction Parameters
```bash
--ocr-confidence 0.8         # OCR confidence threshold
--ocr-lang en                # OCR language
--pdf-confidence 0.8         # PDF text confidence
```

### Environment Parameters
```bash
--mode multi                 # Multi-environment mode
--mode single                # Single-environment mode
--skip-detection             # Skip detection stage
--skip-text                  # Skip text extraction stage
```

## Output Files

For each processed PDF, the pipeline generates:

- `{pdf_name}_detected.pdf` - PDF with detection overlays
- `{pdf_name}_detections.json` - Detection results with coordinates
- `{pdf_name}_text_extraction.json` - Extracted text with metadata
- `{pdf_name}_statistics.json` - Processing statistics
- `complete_pipeline_summary.json` - Overall pipeline metrics

## Utility Scripts

### Complete Pipeline Script
```bash
# Run complete pipeline with enhanced PDF output
python bin/create_enhanced_pdf.py --pdf 1150.pdf --confidence-threshold 0.8 --ocr-confidence 0.5
```

### Storage Management
```bash
# Audit current storage usage
python src/utils/cleanup_storage.py --audit

# Clean up old training runs and cache files (dry run)
python src/utils/cleanup_storage.py --cleanup --keep-runs 2

# Actually perform cleanup
python src/utils/cleanup_storage.py --cleanup --force
```

### Configuration Management
```bash
# Fix configuration paths (e.g., update from 0.3 to 0.4)
python src/utils/fix_config_path.py
```

### PDF Annotation System
```bash
# Create annotated PDFs with native PDF annotations
python src/utils/pdf_annotator.py --detection-file detections.json --text-file text.json --pdf-file input.pdf --output output.pdf
```

## Advanced Usage

### Training Custom Models

```bash
# Train YOLO11 model
python src/detection/yolo11_train.py --epochs 100 --batch-size 16

# Validate dataset
python src/detection/yolo11_train.py --validate-only
```

### Dataset Management

```bash
# Interactive dataset management
python setup/manage_datasets.py --interactive

# Download specific dataset
python setup/manage_datasets.py --download latest_plc_symbols

# List available datasets
python setup/manage_datasets.py --list
```

### Model Management

```bash
# Interactive model management
python setup/manage_models.py --interactive

# Download specific model
python setup/manage_models.py --download yolo11x.pt

# List available models
python setup/manage_models.py --list
```

### Performance Optimization

```bash
# GPU optimization analysis
python src/detection/profile_gpu_pipeline.py

# Benchmark different pipelines
python src/detection/benchmark_all_pipelines.py

# Parallel processing
python src/detection/detect_pipeline_parallel.py
```

## Multi-Environment Architecture

The project uses a sophisticated multi-environment system to handle conflicting GPU dependencies:

### Core Environment (`plcdp`)
- **Purpose**: Coordination, file I/O, preprocessing
- **Dependencies**: Lightweight utilities, configuration management
- **Role**: Orchestrates pipeline execution

### Detection Environment (`detection_env`)
- **Purpose**: Symbol detection using YOLO11
- **Dependencies**: PyTorch, Ultralytics, CUDA libraries
- **Role**: Subprocess worker for detection tasks

### OCR Environment (`ocr_env`)
- **Purpose**: Text extraction using PaddleOCR
- **Dependencies**: PaddlePaddle, PaddleOCR, CUDA libraries
- **Role**: Subprocess worker for OCR tasks

### Communication
- **Method**: JSON-based inter-process communication
- **Coordination**: `MultiEnvironmentManager` class
- **Workers**: Dedicated worker scripts in `src/workers/`

## Testing

```bash
# Validate complete setup
python tests/validate_setup.py

# Test pipeline functionality
python tests/test_pipeline.py

# Test text extraction
python tests/test_text_extraction.py

# Test network drive connectivity
python tests/test_network_drive.py
```

## Troubleshooting

### CUDA Conflicts
The multi-environment architecture resolves most CUDA conflicts automatically. If issues persist:

```bash
# Check GPU status
python -m src.utils.gpu_sanity_checker --device auto

# Force CPU mode
python src/run_pipeline.py --device cpu
```

### Environment Issues
```bash
# Clean and recreate environments
python setup/setup.py --clean --multi-env

# Validate environment setup
python tests/validate_setup.py
```

### Performance Issues
```bash
# Profile GPU usage
python src/detection/profile_gpu_pipeline.py

# Check system capabilities
python setup/gpu_detector.py
```

## Development

### Adding New Features
1. Implement in appropriate module (`src/detection/`, `src/ocr/`, etc.)
2. Add worker support if GPU-dependent
3. Update configuration options
4. Add tests
5. Update documentation

### Code Organization
- **Lightweight coordinators**: No heavy imports (e.g., `detection_manager.py`)
- **Heavy workers**: Subprocess-based with full dependencies
- **Runtime flags**: Use `src/utils/runtime_flags.py` for environment detection
- **Configuration**: Centralized in `src/config.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow PEP8 style guidelines
4. Add comprehensive tests
5. Update documentation
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- **Ultralytics** for YOLO11 implementation
- **PaddlePaddle** team for PaddleOCR
- **Microsoft** for LayoutLM research
- **Poppler** team for PDF processing tools
