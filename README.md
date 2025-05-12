# PLC Diagram Processor

**End-to-end pipeline for industrial PLC diagram analysis** using open-source AI models, local deployment, and dual AI systems for processing and verification.

## Key Intentions

- **Automated Symbol & Text Extraction**: Detect PLC symbols (relays, sensors, valves) and extract alphanumeric labels (e.g., I0.1, Q2.3) from engineering diagrams.
- **Structured Data Output**: Convert raw diagram insights into JSON input/output lists per PLC module for seamless integration.
- **Dual-Phase AI Workflow**:
  1. **Processing AI**: YOLOv8 for symbol detection, PaddleOCR for text extraction, LayoutLMv3 for context linking.
  2. **Verification AI**: DeBERTa-v3 with Drools rules engine for compliance checks, SAM for visual error overlays.
- **Local & Edge Deployment**: Fully self-hosted with Docker, ONNX/TensorRT optimizations, and NVIDIA Triton for sub-2s inference.

## Feature Overview

| Component               | Model / Tool              | Purpose                                      |
|-------------------------|---------------------------|----------------------------------------------|
| **Symbol Detection**    | YOLOv8 (Ultralytics)      | Custom-trained PLC symbol detection          |
| **OCR & Text Extraction** | PaddleOCR (PP-OCRv4)    | High-accuracy text region recognition        |
| **Data Structuring**    | LayoutLMv3 (Microsoft)    | Multimodal mapping of text to symbols        |
| **Logical Verification**| DeBERTa-v3 + Drools       | Industrial standards & rule-based validation |
| **Visual Verification** | Segment Anything Model    | Highlight errors overlayed on original diagram |
| **Deployment**          | Docker, ONNX, Triton      | Containerized, optimized local inference     |

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
│   ├── yolo/            # YOLOv8 configs & weights
│   ├── ocr/             # PaddleOCR model files
│   ├── layoutlm/        # LayoutLMv3 checkpoints
│   ├── deberta/         # DeBERTa-v3 checkpoints
│   └── sam/             # SAM model artifacts
├── src/
│   ├── preprocessing/
│   │   ├── enhance.py           # Contrast & perspective fixes
│   │   └── generate_synthetic.py # Synthetic diagram generator
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
│       └── app.py               # Streamlit review UI
└── deployment/
    ├── triton_config/           # NVIDIA Triton configs
    └── onnx/                    # ONNX/TensorRT conversion scripts
```

## Quickstart

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-org/plc-diagram-processor.git
   cd plc-diagram-processor
   ```

2. **Set up your environment**  
   - *Conda:*  
     ```bash
     conda create -n plc-ai python=3.10
     conda activate plc-ai
     ```  
   - *Virtualenv:*  
     ```bash
     python3.10 -m venv .venv
     source .venv/bin/activate      # macOS/Linux
     .venv\Scripts\Activate.ps1   # Windows PowerShell
     ```

3. **Install dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Training**  
   ```bash
   python src/detection/yolov8_train.py --data data/plc_symbols.yaml
   ```

5. **Inference**  
   ```bash
   python src/detection/yolov8_infer.py --image data/raw/your_diagram.png
   ```

6. **Review & Verification**  
   ```bash
   streamlit run src/interface/app.py
   ```
   Inspect JSON outputs under `results/` and overlay images under `results/overlays/`.

## Contributing

Feel free to open issues or PRs. Please follow PEP8, include tests where possible, and update documentation.

## License

Distributed under the MIT License. See `LICENSE` for details.
