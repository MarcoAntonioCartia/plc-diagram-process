"""
Pipeline Structure Validation Script
Tests the pipeline components without requiring full dependencies
"""

import sys
import json
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config

def validate_pipeline_structure():
    """Validate that all pipeline components are properly structured"""
    
    print("Validating PLC Detection Pipeline Structure")
    print("=" * 50)
    
    # Get configuration
    try:
        config = get_config()
        print(f"Configuration loaded from: {config.config_path}")
        print(f"Data root: {config.config['data_root']}")
    except Exception as e:
        print(f"ERROR: Could not load configuration: {e}")
        return False
    
    # Check folder structure using new layout
    data_root = Path(config.config['data_root'])
    folder_mapping = {
        "raw/pdfs": "Input PDFs",
        "processed/images": "Processed image snippets", 
        "processed/detdiagrams": "Detection results"
    }
    
    print("1. Folder Structure:")
    all_folders_exist = True
    for folder_path, description in folder_mapping.items():
        full_path = data_root / folder_path
        if full_path.exists():
            print(f"   ✓ {folder_path}/ exists - {description}")
            
            # Count contents
            if folder_path == "raw/pdfs":
                pdf_count = len(list(full_path.glob("*.pdf")))
                print(f"     - Contains {pdf_count} PDF files")
            elif folder_path == "processed/images":
                png_count = len(list(full_path.glob("*.png")))
                json_count = len(list(full_path.glob("*.json")))
                print(f"     - Contains {png_count} PNG files, {json_count} JSON files")
            elif folder_path == "processed/detdiagrams":
                file_count = len(list(full_path.glob("*")))
                print(f"     - Contains {file_count} files")
        else:
            print(f"   ✗ {folder_path}/ missing - {description}")
            all_folders_exist = False
    
    # Check pipeline scripts
    print("\n2. Pipeline Scripts:")
    project_root = Path(__file__).resolve().parent.parent.parent
    detection_folder = project_root / "src" / "detection"
    required_scripts = [
        "yolo11_train.py",
        "yolo11_infer.py", 
        "detect_pipeline.py",
        "coordinate_transform.py",
        "reconstruct_with_detections.py",
        "run_complete_pipeline.py"
    ]
    
    all_scripts_exist = True
    for script in required_scripts:
        script_path = detection_folder / script
        if script_path.exists():
            print(f"   ✓ {script} exists")
        else:
            print(f"   ✗ {script} missing")
            all_scripts_exist = False
    
    # Check configuration files using config system
    print("\n3. Configuration Files:")
    config_checks = [
        (config.data_yaml_path, "YOLO training configuration"),
        (config.get_model_path('yolo11m.pt', 'pretrained'), "YOLO11m pretrained model"),
    ]
    
    # Check for preprocessing scripts in their new location
    preprocessing_folder = project_root / "src" / "preprocessing"
    config_checks.extend([
        (preprocessing_folder / "SnipPdfToPng.py", "PDF to snippets converter"),
        (preprocessing_folder / "SnipPngToPdf.py", "PDF reconstruction script")
    ])
    
    all_configs_exist = True
    for config_path, description in config_checks:
        if config_path.exists():
            print(f"   ✓ {config_path.relative_to(project_root)} - {description}")
        else:
            print(f"   ✗ {config_path.relative_to(project_root)} - {description}")
            all_configs_exist = False
    
    # Test import structure (without actually importing heavy dependencies)
    print("\n4. Import Structure Test:")
    try:
        # Test basic Python syntax of all scripts
        import ast
        
        for script in required_scripts:
            script_path = detection_folder / script
            if script_path.exists():
                try:
                    with open(script_path, 'r') as f:
                        content = f.read()
                    ast.parse(content)
                    print(f"   ✓ {script} - Valid Python syntax")
                except SyntaxError as e:
                    print(f"   ✗ {script} - Syntax error: {e}")
                    all_scripts_exist = False
        
    except Exception as e:
        print(f"   Warning: Could not validate syntax: {e}")
    
    # Generate test summary
    print("\n5. Pipeline Readiness Summary:")
    if all_folders_exist and all_scripts_exist and all_configs_exist:
        print("   ✓ Pipeline structure is complete and ready")
        print("   ✓ All required components are present")
        print("   ✓ Ready for full pipeline execution")
        
        # Show usage command
        print("\n6. Usage Commands:")
        print("   Full pipeline (with training):")
        print("   python src/detection/run_complete_pipeline.py --epochs 10")
        print("\n   Pipeline with existing model:")
        print("   python src/detection/run_complete_pipeline.py --skip-training")
        
        return True
    else:
        print("   ✗ Pipeline structure incomplete")
        print("   ✗ Some components are missing")
        print("   ✗ Please fix issues before running pipeline")
        return False

def create_test_metadata():
    """Create a sample metadata structure for testing"""
    
    sample_metadata = {
        "original_pdf": "test_diagram",
        "snippet_size": [1500, 1200],
        "overlap": 500,
        "pages": [
            {
                "page_num": 1,
                "original_width": 3000,
                "original_height": 2400,
                "rows": 2,
                "cols": 2,
                "snippets": [
                    {
                        "filename": "test_diagram_p1_r0_c0.png",
                        "row": 0,
                        "col": 0,
                        "x1": 0,
                        "y1": 0,
                        "x2": 1500,
                        "y2": 1200,
                        "width": 1500,
                        "height": 1200
                    },
                    {
                        "filename": "test_diagram_p1_r0_c1.png",
                        "row": 0,
                        "col": 1,
                        "x1": 1000,
                        "y1": 0,
                        "x2": 2500,
                        "y2": 1200,
                        "width": 1500,
                        "height": 1200
                    }
                ]
            }
        ]
    }
    
    sample_detections = {
        "test_diagram_p1_r0_c0.png": {
            "image_path": "test_diagram_p1_r0_c0.png",
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "PLC",
                    "confidence": 0.85,
                    "bbox": {
                        "x1": 100,
                        "y1": 150,
                        "x2": 200,
                        "y2": 250
                    }
                }
            ],
            "detection_count": 1
        }
    }
    
    print("\n7. Sample Data Structures:")
    print("   Sample metadata structure:")
    print(f"   {json.dumps(sample_metadata, indent=2)[:200]}...")
    
    print("\n   Sample detection structure:")
    print(f"   {json.dumps(sample_detections, indent=2)[:200]}...")
    
    return sample_metadata, sample_detections

if __name__ == "__main__":
    success = validate_pipeline_structure()
    create_test_metadata()
    
    if success:
        print("\n" + "=" * 50)
        print("Pipeline validation completed successfully!")
        print("The detection pipeline is ready for execution.")
        print("=" * 50)
        sys.exit(0)
    else:
        print("\n" + "=" * 50)
        print("Pipeline validation failed!")
        print("Please fix the issues above before proceeding.")
        print("=" * 50)
        sys.exit(1)
