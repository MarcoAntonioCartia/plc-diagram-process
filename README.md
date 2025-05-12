## Repository Structure

```
plc-diagram-processor/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── docker/
│   └── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/             # Unprocessed diagrams
│   ├── annotated/       # CVAT and RoboFlow annotations
│   └── synthetic/       # Generated SVG/AutoCAD diagrams
├── models/
│   ├── yolo/            # YOLOv8 config and weights
│   ├── ocr/             # PaddleOCR model files
│   ├── layoutlm/        # LayoutLMv3 checkpoints
│   ├── deberta/         # DeBERTa-v3 checkpoints
│   └── sam/             # SAM model artifacts
├── src/
│   ├── preprocessing/
│   │   ├── enhance.py           # OpenCV contrast, perspective
│   │   └── generate_synthetic.py# svgplc/AutoCAD generator
│   ├── detection/
│   │   ├── yolov8_train.py
│   │   └── yolov8_infer.py
│   ├── ocr/
│   │   └── paddle_ocr.py
│   ├── structuring/
│   │   └── layoutlm_parser.py
│   ├── verification/
│   │   ├── rules_engine.py      # Drools integration
│   │   └── visual_verify.py     # SAM overlay
│   └── interface/
│       └── app.py               # Employee review UI (e.g., Streamlit)
└── deployment/
    ├── triton_config/           # NVIDIA Triton configs
    └── onnx/                    # ONNX/TensorRT conversion scripts
```

---

## Key Files

### README.md

````markdown
# PLC Diagram Processor

**End-to-end pipeline for industrial PLC diagram analysis** using open-source AI models, local deployment, and dual AI systems for processing and verification.

## Key Intentions

- **Automated Symbol & Text Extraction**: Detect PLC symbols (relays, sensors, valves) and extract alphanumeric labels (I0.1, Q2.3) from engineering diagrams.
- **Structured Data Output**: Convert raw diagram insights into JSON input/output lists per PLC module, enabling downstream system integration.
- **Dual-Phase AI Workflow**:
  1. **Processing AI**: YOLOv8 for symbol detection, PaddleOCR for text extraction, and LayoutLMv3 for context linking.
  2. **Verification AI**: DeBERTa-v3 combined with Drools rules engine for standards compliance, plus SAM for visual error overlays.
- **Local & Edge Deployment**: Fully self-hosted solution with Docker, ONNX/TensorRT optimizations, and NVIDIA Triton serving for sub-2s inference per diagram.

## Feature Overview

| Component             | Model/Tool                | Purpose                                    |
|-----------------------|---------------------------|--------------------------------------------|
| **Symbol Detection**  | YOLOv8 (Ultralytics)      | Custom-trained detection of PLC symbols    |
| **OCR**               | PaddleOCR (PP-OCRv4)      | High-accuracy text region extraction       |
| **Data Structuring**  | LayoutLMv3 (Microsoft)    | Multimodal mapping of text to symbols     |
| **Logical Verification** | DeBERTa-v3 + Drools    | Standards compliance & rule-based checks   |
| **Visual Verification**  | SAM (Meta)             | Highlight errors on original diagrams      |
| **Deployment**        | Docker, ONNX, Triton      | Containerized, optimized local inference   |

## Quickstart

1. **Clone**:
   ```bash
   git clone https://github.com/your-org/plc-diagram-processor.git
   cd plc-diagram-processor
````

2. **Install**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Train Symbol Model**:

   ```bash
   python src/detection/yolov8_train.py --data data/plc_symbols.yaml
   ```
4. **Run Inference**:

   ```bash
   python src/detection/yolov8_infer.py --image data/raw/diagram.png
   ```
5. **Review & Verify**:

   * Launch UI: `streamlit run src/interface/app.py`
   * Inspect JSON outputs under `results/` and visual overlays under `results/overlays/`

````
markdown
# PLC Diagram Processor

End-to-end pipeline for industrial PLC diagram analysis, leveraging open-source AI models for symbol detection, OCR, data structuring, and verification.

## Features
- **Symbol Detection**: YOLOv8 with custom classes
- **OCR**: PaddleOCR (PP-OCRv4)
- **Structuring**: LayoutLMv3 for linking text to symbols
- **Verification**: DeBERTa-v3 + Drools rules engine
- **Visual Validation**: SAM overlays errors
- **Local Deployment**: Docker, ONNX, NVIDIA Triton

## Getting Started

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Build Docker image:
   ```bash
   docker build -t plc-processor .
````

4. Run inference:

   ```bash
   python src/detection/yolov8_infer.py --image path/to/diagram.png
   ```

## Repo Layout

See [Repository Structure](#repository-structure) above.

````

### Dockerfile (docker/Dockerfile)
```dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY deployment/onnx/ ./onnx/
CMD ["python", "src/detection/yolov8_infer.py", "--image", "/data/input.png"]
````

### requirements.txt

```
ultralytics
paddlepaddle
paddleocr
transformers
torch
opencv-python
ruamel.yaml
drools-jpy
streamlit
tritonclient[all]
onxtruntime
django     # or flask
```
