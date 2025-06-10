# PLC Diagram Processor

**End-to-end pipeline for industrial PLC diagram analysis** using open-source AI models, local deployment, and dual AI systems for processing and verification.

## Key Intentions

- **Automated Symbol & Text Extraction**: Detect PLC symbols (relays, sensors, valves) and extract alphanumeric labels (e.g., I0.1, Q2.3) from engineering diagrams.
- **Structured Data Output**: Convert raw diagram insights into JSON input/output lists per PLC module for seamless integration.
- **Dual-Phase AI Workflow**:
  1. **Processing AI**: YOLO11 for symbol detection, PaddleOCR for text extraction, LayoutLMv3 for context linking.
  2. **Verification AI**: DeBERTa-v3 with Drools rules engine for compliance checks, SAM for visual error overlays.
- **Local & Edge Deployment**: Fully self-hosted with Docker, ONNX/TensorRT optimizations, and NVIDIA Triton for sub-2s inference.

## Feature Overview

| Component               | Model / Tool              | Purpose                                      |
|-------------------------|---------------------------|----------------------------------------------|
| **Symbol Detection**    | YOLO11 (Ultralytics)     | Custom-trained PLC symbol detection          |
| **OCR & Text Extraction** | PaddleOCR (PP-OCRv4)    | High-accuracy text region recognition        |
| **Data Structuring**    | LayoutLMv3 (Microsoft)    | Multimodal mapping of text to symbols        |
| **Logical Verification**| DeBERTa-v3 + Drools       | Industrial standards & rule-based validation |
| **Visual Verification** | Segment Anything Model    | Highlight errors overlayed on original diagram |
| **Deployment**          | Docker, ONNX, Triton      | Containerized, optimized local inference     |

## Detection Pipeline Architecture

The detection stage implements a complete end-to-end pipeline for processing PLC diagrams:

### Pipeline Flow
```
PDF Diagrams → Image Snippets → YOLO11 Detection → Coordinate Transform → Reconstructed PDFs
```

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
├── NETWORK_DRIVE_MIGRATION.md   # Migration documentation
├── requirements.txt
├── preinstall.sh                # Cross-platform dependency installer
├── activate.sh                  # Virtual environment activation script
├── tests/                       # Test scripts directory
│   ├── test_network_drive.py    # Network drive connectivity test
│   ├── test_wsl_poppler.py      # WSL poppler integration test
│   ├── test_pipeline.py         # Pipeline validation test
│   ├── validate_setup.py        # Setup validation script
│   └── README.md                # Test documentation
├── docker/
│   └── Dockerfile
├── setup/                       # Setup and management scripts
│   ├── setup.py                 # Main setup script
│   ├── manage_datasets.py       # Dataset management utility
│   ├── manage_models.py         # Model management utility
│   ├── README.md                # Setup documentation
│   └── config/
│       └── download_config.yaml # Storage backend configuration
├── src/
│   ├── detection/
│   │   ├── yolo11_train.py              # YOLO11 training script
│   │   ├── yolo11_infer.py              # YOLO11 inference script
│   │   ├── detect_pipeline.py           # Main detection orchestrator
│   │   ├── coordinate_transform.py      # Coordinate transformation
│   │   ├── reconstruct_with_detections.py # PDF reconstruction
│   │   ├── run_complete_pipeline.py     # Complete pipeline runner
│   │   └── validate_pipeline_structure.py # Pipeline validation
│   ├── ocr/
│   │   └── paddle_ocr.py
│   ├── preprocessing/
│   │   ├── generate_synthetic.py
│   │   ├── SnipPdfToPng.py      # PDF to snippets converter
│   │   └── SnipPngToPdf.py      # PDF reconstruction utility
│   ├── structuring/
│   │   └── layoutlm_train.py
│   ├── utils/                   # Utility modules
│   │   ├── dataset_manager.py   # Dataset activation manager
│   │   ├── model_manager.py     # Model download manager
│   │   ├── network_drive_manager.py # Network drive storage backend
│   │   └── onedrive_manager.py  # OneDrive backend (legacy)
│   └── verification/
│       └── visual_verify.py
└── data/                        # Local data directory (optional)

../plc-data/                     # Main data directory (sibling to project)
├── datasets/
│   ├── downloaded/              # Downloaded datasets from network drive
│   ├── train/                   # Active training data (symlink/copy)
│   ├── valid/                   # Active validation data (symlink/copy)
│   ├── test/                    # Active test data (symlink/copy)
│   └── plc_symbols.yaml         # Active dataset configuration
├── models/
│   ├── pretrained/              # Downloaded YOLO models
│   └── custom/                  # Trained models
├── processed/                   # Processed outputs
├── raw/                         # Raw input PDFs
└── runs/                        # Training/inference outputs
```

## Quickstart

### 1. Clone Repository
```bash
git clone https://github.com/your-org/plc-diagram-processor.git
cd plc-diagram-processor
```

### 2. Complete Setup (Recommended)
```bash
# Run the complete setup script
python setup/setup.py
```

This will:
- Install system dependencies
- Create virtual environment
- Install Python dependencies
- Set up data directory structure
- Optionally download datasets and models from network drive

### 3. Configure Data Storage
Edit `setup/config/download_config.yaml` to set your network drive path:
```yaml
storage_backend: "network_drive"
network_drive:
  base_path: "S:\\99_Automation\\Datasets plc-diagram-processor"
```

### 4. Dataset Management
```bash
# Test network drive connectivity
python tests/test_network_drive.py

# Interactive dataset management
python setup/manage_datasets.py --interactive

# Download latest dataset
python setup/manage_datasets.py --download-latest
```

### 5. Run Complete Detection Pipeline

**Full pipeline with training:**
```bash
python src/detection/run_complete_pipeline.py --epochs 10
```

**Pipeline with existing model:**
```bash
python src/detection/run_complete_pipeline.py --skip-training
```

**Custom configuration:**
```bash
python src/detection/run_complete_pipeline.py --epochs 20 --conf 0.3 --snippet-size 1200 1000
```

### 6. Validate Setup
```bash
python tests/validate_setup.py
```

## Individual Component Usage

### Train YOLO11 Model
```bash
python src/detection/yolo11_train.py
```

### Run Detection Only
```bash
python src/detection/yolo11_infer.py --input ../plc-data/processed/images --output ../plc-data/processed/results
```

### Process PDFs Step-by-Step
```bash
# 1. Convert PDFs to snippets
python src/preprocessing/SnipPdfToPng.py

# 2. Run detection pipeline
python src/detection/detect_pipeline.py --diagrams ../plc-data/raw/pdfs --output ../plc-data/processed

# 3. View results in ../plc-data/processed/
```

## Pipeline Configuration

### YOLO11 Training Parameters
- **Model**: YOLO11m (medium variant for accuracy/speed balance)
- **Classes**: PLC symbols (configurable in `plc_symbols.yaml`)
- **Image Size**: 640x640 pixels
- **Batch Size**: 16
- **Default Epochs**: 10 (configurable)

### Detection Parameters
- **Confidence Threshold**: 0.25 (configurable)
- **Snippet Size**: 1500x1200 pixels (configurable)
- **Snippet Overlap**: 500 pixels (configurable)

### Output Formats
- **PDF**: Reconstructed diagrams with detection overlays
- **JSON**: Structured detection data with coordinates
- **TXT**: Human-readable coordinate mappings
- **PNG**: Individual page images with detections

## Performance Metrics

The pipeline tracks comprehensive metrics:
- Training time and model performance
- Detection success rate per PDF
- Total detections across all documents
- Average detections per PDF
- Processing time for each stage
- Coordinate transformation accuracy

## Text Extraction Pipeline

The project now includes a comprehensive text extraction pipeline that processes detected symbols to extract associated text labels and identifiers.

### Text Extraction Features

| Component               | Technology                | Purpose                                      |
|-------------------------|---------------------------|----------------------------------------------|
| **Hybrid Text Extraction** | PaddleOCR + PyMuPDF    | Combines OCR and direct PDF text extraction  |
| **PLC Pattern Recognition** | Regex + Priority System | Identifies PLC-specific text patterns        |
| **Symbol-Text Association** | Spatial Analysis       | Links text to nearby detected symbols        |
| **Smart Deduplication**     | Overlap Detection      | Removes duplicate text from different sources |

### Text Extraction Usage

**Run text extraction on existing detection results:**
```bash
python src/ocr/run_text_extraction.py
```

**Run complete pipeline with text extraction:**
```bash
python src/detection/run_complete_pipeline_with_text.py --epochs 10
```

**Custom text extraction parameters:**
```bash
python src/ocr/run_text_extraction.py --confidence 0.8 --lang en
```

**List available detection files:**
```bash
python src/ocr/run_text_extraction.py --list-files
```

### Text Extraction Configuration

#### OCR Parameters
- **Confidence Threshold**: 0.7 (minimum OCR confidence)
- **Language**: English (configurable)
- **Engine**: PaddleOCR PP-OCRv4

#### PLC Pattern Recognition
The system recognizes these PLC-specific patterns (by priority):
1. **Input addresses**: I0.1, I1.2, etc.
2. **Output addresses**: Q0.1, Q2.3, etc.
3. **Memory addresses**: M0.1, M1.2, etc.
4. **Timer addresses**: T1, T2, etc.
5. **Counter addresses**: C1, C2, etc.
6. **Function blocks**: FB1, FB2, etc.
7. **Data blocks**: DB1, DB2, etc.
8. **Analog I/O**: AI1, AO2, etc.
9. **Variable names**: MOTOR_START, VALVE_OPEN, etc.

### Text Extraction Output

For each processed PDF, the system generates:
- `{pdf_name}_text_extraction.json`: Extracted text with coordinates and metadata
- `text_extraction_summary.json`: Overall extraction statistics
- `complete_pipeline_summary.json`: Combined detection and text extraction metrics

### Text Extraction Pipeline Architecture

```
Detection Results → Hybrid Text Extraction → PLC Pattern Recognition → Symbol Association → Structured Output
                    ↓                        ↓                         ↓                    ↓
                    PyMuPDF (PDF text)       Regex Matching           Spatial Analysis     JSON Results
                    PaddleOCR (Image text)   Priority Scoring         Distance Calculation  Statistics
```

## Next Steps

After completing the detection and text extraction stages:
1. Review detection results in `../plc-data/processed/detdiagrams/`
2. Review text extraction results in `../plc-data/processed/text_extraction/`
3. Check `complete_pipeline_summary.json` for comprehensive metrics
4. Continue with data structuring using LayoutLM
5. Apply verification and validation rules

## Testing

**Test the text extraction pipeline:**
```bash
python tests/test_text_extraction.py
```

**Validate complete setup:**
```bash
python tests/validate_setup.py
```

## Contributing

Feel free to open issues or PRs. Please follow PEP8, include tests where possible, and update documentation.

## License

Distributed under the MIT License. See `LICENSE` for details.
