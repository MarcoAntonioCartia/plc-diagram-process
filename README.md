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
- **Input**: PDF files in `data/dataset/test/diagrams/`
- **Process**: Convert PDFs to overlapping image snippets using `SnipPdfToPng.py`
- **Output**: PNG snippets with metadata in `data/dataset/test/images/`

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
- **Output**: Labeled PDFs and coordinate mapping files in `data/dataset/test/detdiagrams/`

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
├── requirements.txt
├── preinstall.sh                # Cross-platform dependency installer
├── docker/
│   └── Dockerfile
├── data/
│   ├── plc_symbols.yaml         # YOLO11 training configuration
│   ├── yolo11m.pt              # YOLO11m pretrained model
│   ├── SnipPdfToPng.py         # PDF to snippets converter
│   ├── SnipPngToPdf.py         # PDF reconstruction utility
│   └── dataset/
│       └── test/
│           ├── diagrams/        # Original PDF files
│           ├── images/          # Image snippets + metadata
│           ├── detdiagrams/     # Reconstructed PDFs with detections
│           ├── train/           # YOLO training data
│           ├── valid/           # YOLO validation data
│           └── labels/          # YOLO label files
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
│   │   └── generate_synthetic.py
│   ├── structuring/
│   │   └── layoutlm_train.py
│   └── verification/
│       └── visual_verify.py
└── runs/                        # YOLO training outputs
    └── detect/
```

## Quickstart

### 1. Clone Repository
```bash
git clone https://github.com/your-org/plc-diagram-processor.git
cd plc-diagram-processor
```

### 2. Automated Setup (Recommended)
```bash
# Run the automated setup script
bash preinstall.sh
```

The `preinstall.sh` script will automatically:
- Detect your operating system (Linux/macOS/Windows)
- Install system dependencies (Python dev headers, build tools, Poppler)
- Create and activate a virtual environment
- Install all Python dependencies from requirements.txt

### 3. Manual Setup (Alternative)

**For Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3-dev python3-pip build-essential poppler-utils
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**For CentOS/RHEL/Fedora:**
```bash
sudo yum install python3-devel gcc gcc-c++ make poppler-utils
# or: sudo dnf install python3-devel gcc gcc-c++ make poppler-utils
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Run Complete Detection Pipeline

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

### 5. Validate Pipeline Setup
```bash
python src/detection/validate_pipeline_structure.py
```

## Individual Component Usage

### Train YOLO11 Model
```bash
python src/detection/yolo11_train.py
```

### Run Detection Only
```bash
python src/detection/yolo11_infer.py --input data/dataset/test/images --output results
```

### Process PDFs Step-by-Step
```bash
# 1. Convert PDFs to snippets
python data/SnipPdfToPng.py

# 2. Run detection pipeline
python src/detection/detect_pipeline.py --diagrams data/dataset/test/diagrams --output data/dataset/test

# 3. View results in data/dataset/test/detdiagrams/
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

## Next Steps

After completing the detection stage:
1. Review results in `data/dataset/test/detdiagrams/`
2. Check `pipeline_summary.json` for performance metrics
3. Proceed to OCR and text extraction stage
4. Continue with data structuring using LayoutLM
5. Apply verification and validation rules

## Contributing

Feel free to open issues or PRs. Please follow PEP8, include tests where possible, and update documentation.

## License

Distributed under the MIT License. See `LICENSE` for details.