"""
Test script for the complete PLC detection pipeline
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent  # Go up from tests to project root
sys.path.append(str(project_root))

from src.config import get_config

def test_pipeline_setup():
    """Test that all components are properly set up"""
    print("Testing PLC Detection Pipeline Setup")
    print("=" * 40)
    
    # Get configuration
    try:
        config = get_config()
        print(f"Configuration loaded from: {config.config_path}")
        print(f"Data root: {config.config['data_root']}")
    except Exception as e:
        print(f"ERROR: Could not load configuration: {e}")
        return False
    
    # Check folder structure using config
    dataset_path = config.get_dataset_path()
    test_base = dataset_path.parent / "test"  # Assuming test data is alongside dataset
    required_folders = ["diagrams", "images", "detdiagrams"]
    
    print("1. Checking folder structure...")
    for folder in required_folders:
        folder_path = test_base / folder
        if folder_path.exists():
            print(f"   ✓ {folder}/ exists")
        else:
            print(f"   ✗ {folder}/ missing")
            return False
    
    # Check for PDFs in diagrams folder
    diagrams_folder = test_base / "diagrams"
    pdf_files = list(diagrams_folder.glob("*.pdf"))
    print(f"   Found {len(pdf_files)} PDF files in diagrams/")
    
    # Check for existing snippets in images folder
    images_folder = test_base / "images"
    png_files = list(images_folder.glob("*.png"))
    metadata_files = list(images_folder.glob("*_metadata.json"))
    print(f"   Found {len(png_files)} PNG snippets in images/")
    print(f"   Found {len(metadata_files)} metadata files in images/")
    
    # Check model files using config
    print("\n2. Checking model files...")
    model_files = ["yolo11m.pt", "yolo11n.pt"]
    
    for model_file in model_files:
        model_path = config.get_model_path(model_file, 'pretrained')
        if model_path.exists():
            print(f"   ✓ {model_file} exists")
        else:
            print(f"   ✗ {model_file} missing")
    
    # Check configuration using config
    print("\n3. Checking configuration...")
    config_file = config.data_yaml_path
    if config_file.exists():
        print(f"   ✓ data.yaml exists")
    else:
        print(f"   ✗ data.yaml missing")
        return False
    
    # Check pipeline scripts
    print("\n4. Checking pipeline scripts...")
    detection_folder = project_root / "src" / "detection"
    required_scripts = [
        "yolo11_train.py",
        "yolo11_infer.py", 
        "detect_pipeline.py",
        "coordinate_transform.py",
        "reconstruct_with_detections.py"
    ]
    
    for script in required_scripts:
        script_path = detection_folder / script
        if script_path.exists():
            print(f"   ✓ {script} exists")
        else:
            print(f"   ✗ {script} missing")
            return False
    
    print("\n✓ Pipeline setup validation completed successfully!")
    return True

def show_usage_examples():
    """Show usage examples for the pipeline"""
    print("\nUsage Examples:")
    print("=" * 40)
    
    try:
        config = get_config()
        dataset_path = config.get_dataset_path()
        test_base = dataset_path.parent / "test"
        runs_path = config.get_run_path('detect')
    except:
        # Fallback to generic examples if config fails
        test_base = "plc-data/datasets/test"
        runs_path = "plc-data/runs/detect"
    
    print("\n1. Train YOLO11 model:")
    print("   python src/detection/yolo11_train.py")
    
    print("\n2. Run inference on test images:")
    print(f"   python src/detection/yolo11_infer.py --input {test_base}/images --output {runs_path}")
    
    print("\n3. Run complete detection pipeline:")
    print(f"   python src/detection/detect_pipeline.py --diagrams {test_base}/diagrams --output {test_base}")
    
    print("\n4. Test coordinate transformation:")
    print(f"   python src/detection/coordinate_transform.py --detections {runs_path}/all_detections.json --metadata {test_base}/images/1150_metadata.json --output {runs_path}")
    
    print("\n5. Test PDF reconstruction:")
    print(f"   python src/detection/reconstruct_with_detections.py --metadata {test_base}/images/1150_metadata.json --detections {runs_path}/1150_global_detections.json --images {test_base}/images --output {runs_path}")
    
    print("\n6. Run complete pipeline with training:")
    print("   python src/detection/run_complete_pipeline.py --epochs 10")
    
    print("\n7. Run pipeline with existing model:")
    print("   python src/detection/run_complete_pipeline.py --skip-training")

def show_pipeline_overview():
    """Show overview of the complete pipeline"""
    print("\nPipeline Overview:")
    print("=" * 40)
    
    print("\nComplete Detection Pipeline Flow:")
    print("1. PDF Files (diagrams/) → SnipPdfToPng.py → Image Snippets (images/)")
    print("2. Image Snippets → YOLO11 Detection → Detection Results")
    print("3. Detection Results + Metadata → Coordinate Transform → Global Coordinates")
    print("4. Global Coordinates + Snippets → PDF Reconstruction → Labeled PDFs (detdiagrams/)")
    
    print("\nOutput Files Generated:")
    print("- {pdf_name}_detected.pdf: PDF with detection overlays")
    print("- {pdf_name}_detections.json: All detections with coordinates")
    print("- {pdf_name}_coordinates.txt: Human-readable coordinate list")
    print("- {pdf_name}_statistics.json: Detection statistics")
    print("- {pdf_name}_page_{n}_detected.png: Individual page images")

if __name__ == "__main__":
    # Run setup validation
    success = test_pipeline_setup()
    
    if success:
        show_pipeline_overview()
        show_usage_examples()
        
        print("\n" + "=" * 50)
        print("PLC Detection Pipeline is ready to use!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("Pipeline setup incomplete. Please fix the issues above.")
        print("=" * 50)
        sys.exit(1)
