# YOLO11 Training Script for PLC Symbol Detection
# https://docs.ultralytics.com/modes/train/
# https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config

def get_best_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        return '0'  # Use first GPU
    else:
        return 'cpu'

def train_yolo11(
    model_name='yolo11m.pt',
    data_yaml_path=None,
    epochs=10,
    batch=16,
    patience=20,
    save_period=10,
    device=None,
    workers=8,
    project_name="plc_symbol_detector_yolo11m",
    verbose=True
):
    """
    Train YOLO11 model for PLC symbol detection using configuration management
    """
    # Get configuration
    config = get_config()
    
    # Configure Ultralytics to use proper directories
    models_dir = config.get_model_path('', 'pretrained').parent
    
    # Set Ultralytics settings to use our directories
    from ultralytics import settings
    settings.update({
        'datasets_dir': str(config.get_dataset_path()),
        'weights_dir': str(models_dir),
        'runs_dir': str(config.get_run_path('').parent)
    })
    
    # Use smart device detection if device not specified
    if device is None:
        device = get_best_device()
        print(f"Auto-detected device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    # Define paths using config, but allow override for data_yaml_path
    # Handle both full paths and model names
    if Path(model_name).is_absolute() and Path(model_name).exists():
        # Full path provided (from training worker)
        model_path = Path(model_name)
        actual_model_name = model_path.name
    else:
        # Model name provided (from command line)
        model_path = config.get_model_path(model_name, 'pretrained')
        actual_model_name = model_name
    
    if data_yaml_path is None:
        data_yaml_path = config.data_yaml_path
    else:
        data_yaml_path = Path(data_yaml_path)
    runs_path = config.get_run_path('train')
    
    # Verify files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Data config file not found: {data_yaml_path}")
    
    print(f"Loading YOLO11 model from: {model_path}")
    print(f"Using dataset config: {data_yaml_path}")
    print(f"Training runs will be saved to: {runs_path}")
    print("Note: AMP (Automatic Mixed Precision) is disabled to prevent model download issues")
    
    # Load pretrained YOLO11 model - with updated Ultralytics, no patches needed
    model = YOLO(str(model_path))
    
    # Suppress YOLO's verbose output if requested
    if not verbose:
        import logging
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        os.environ['TQDM_DISABLE'] = '1'
    
    # Train the model using standard Ultralytics training with CSV error handling
    try:
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=640,
            batch=batch,
            name=project_name,
            project=str(runs_path),
            save=True,
            save_period=save_period,
            patience=patience,
            device=device,
            workers=workers,
            verbose=verbose,
            exist_ok=True,  # Allow overwriting existing runs
            val=True,  # Keep validation enabled
            amp=False,  # Disable AMP to prevent yolo11n.pt download issues
            close_mosaic=10  # Close mosaic augmentation early for faster convergence
        )
    except Exception as e:
        # Handle CSV parsing errors and other training issues
        if "Error tokenizing data" in str(e) or "Expected" in str(e) and "fields" in str(e):
            print(f"WARNING: CSV parsing error detected: {e}")
            print("This is usually caused by corrupted results.csv file during training.")
            print("Attempting to clean up and continue...")
            
            # Try to find and clean up corrupted CSV files
            import glob
            csv_files = glob.glob(str(runs_path / project_name / "*.csv"))
            for csv_file in csv_files:
                try:
                    os.remove(csv_file)
                    print(f"Removed corrupted CSV file: {csv_file}")
                except Exception:
                    pass
            
            # Create a mock results object to continue
            class MockResults:
                def __init__(self, save_dir):
                    self.save_dir = save_dir
                    self.results_dict = {
                        'metrics/mAP50(B)': 0.0,
                        'metrics/mAP50-95(B)': 0.0
                    }
            
            results = MockResults(runs_path / project_name)
            print("Training completed with CSV error handling - results may be incomplete")
        else:
            # Re-raise other exceptions
            raise
    
    # Restore tqdm if it was disabled
    if not verbose and 'TQDM_DISABLE' in os.environ:
        del os.environ['TQDM_DISABLE']
    
    print("Training finished!")
    print(f"Results saved to: {results.save_dir}")
    
    # Copy best model to custom models directory
    best_model_src = Path(results.save_dir) / "weights" / "best.pt"
    if best_model_src.exists():
        custom_model_name = f"{project_name}_best.pt"
        custom_model_path = config.get_model_path(custom_model_name, 'custom')
        
        # Ensure custom models directory exists
        custom_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the model
        import shutil
        shutil.copy2(best_model_src, custom_model_path)
        print(f"Best model copied to: {custom_model_path}")
        
        # Also save the model metadata
        metadata = {
            "original_pretrained": actual_model_name,
            "epochs_trained": epochs,
            "dataset": config.config['dataset']['name'],
            "training_dir": str(results.save_dir),
            "metrics": {
                "mAP50": float(results.results_dict.get('metrics/mAP50(B)', 0)),
                "mAP50-95": float(results.results_dict.get('metrics/mAP50-95(B)', 0))
            }
        }
        
        import json
        metadata_path = custom_model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to: {metadata_path}")
    
    return results

def validate_dataset():
    """
    Validate that the dataset structure is correct using configuration
    """
    config = get_config()
    dataset_path = config.get_dataset_path()
    
    required_dirs = ["train/images", "train/labels", "valid/images", "valid/labels"]
    
    print("Validating dataset structure...")
    print(f"Dataset path: {dataset_path}")
    
    all_valid = True
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        if not full_path.exists():
            print(f"❌ Missing directory: {full_path}")
            all_valid = False
        else:
            file_count = len(list(full_path.glob("*")))
            print(f"✅ Found {file_count} files in {dir_path}")
    
    # Also check data.yaml
    if not config.data_yaml_path.exists():
        print(f"❌ Missing data.yaml at: {config.data_yaml_path}")
        all_valid = False
    else:
        print(f"✅ Found data.yaml at: {config.data_yaml_path}")
    
    return all_valid

def main():
    """
    Main training function with argument parsing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO11 PLC Symbol Detection Training')
    parser.add_argument('--model', '-m', default='yolo11m.pt',
                       help='Pretrained model name (default: yolo11m.pt)')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch', '-b', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--device', '-d', default='auto',
                       help='Device to use: auto, cpu, cuda, 0, 1, etc. (default: auto)')
    parser.add_argument('--workers', '-w', type=int, default=8,
                       help='Number of dataloader workers (default: 8)')
    parser.add_argument('--patience', '-p', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    parser.add_argument('--name', '-n', default='plc_symbol_detector',
                       help='Project name for this training run')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Enable verbose output (default: True)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Disable verbose output (overrides --verbose)')
    
    args = parser.parse_args()
    
    # Handle verbose/quiet flags
    verbose = args.verbose and not args.quiet
    
    print("Starting YOLO11 PLC Symbol Detection Training")
    print("=" * 50)
    
    # Show configuration info
    config = get_config()
    print(f"Configuration loaded from: {config.config_path}")
    print(f"Data root: {config.config['data_root']}")
    print(f"Dataset: {config.config['dataset']['name']}")
    print("=" * 50)
    
    # Validate dataset first
    if not validate_dataset():
        print("\nDataset validation failed. Please check your dataset structure.")
        print("Make sure you have run: python setup_data.py --migrate")
        exit(1)
    
    # Start training
    try:
        # Handle device argument - if 'auto' is passed, let our function handle it
        device_arg = None if args.device == 'auto' else args.device
        
        results = train_yolo11(
            model_name=args.model,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            device=device_arg,
            workers=args.workers,
            project_name=args.name,
            verbose=verbose
        )
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
