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
    """Get the best available device, working around Ultralytics' auto detection issues."""
    if torch.cuda.is_available():
        # Return '0' for first GPU instead of 'auto' which seems buggy
        return '0'
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
    project_name="plc_symbol_detector_yolo11m"
):
    """
    Train YOLO11 model for PLC symbol detection using configuration management
    """
    # Get configuration
    config = get_config()
    
    # Use smart device detection if device not specified
    if device is None:
        device = get_best_device()
        print(f"Auto-detected device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    # Define paths using config, but allow override for data_yaml_path
    model_path = config.get_model_path(model_name, 'pretrained')
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
    
    # Apply PyTorch 2.6 compatibility fix for YOLO model loading
    original_torch_load = torch.load
    
    def safe_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
        # Force weights_only=False for YOLO models to avoid security restrictions
        return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                                 weights_only=False, **kwargs)
    
    # Import compatibility classes from dedicated module
    try:
        from .yolo_compatibility import register_compatibility_classes
        register_compatibility_classes()
    except Exception as e:
        print(f"Warning: Could not import compatibility classes: {e}")
    
    # Apply global torch.load patch for all YOLO operations
    torch.load = safe_torch_load
    
    # Patch numpy.load to handle numpy._core compatibility issues
    import numpy as np
    original_np_load = np.load
    
    def safe_np_load(file, *args, **kwargs):
        """Safe numpy load that handles version compatibility issues"""
        try:
            return original_np_load(file, *args, **kwargs)
        except ModuleNotFoundError as e:
            if "numpy._core" in str(e):
                # If it's a numpy._core issue, try to return None which will force regeneration
                print("Warning: Skipping incompatible numpy cache file due to version mismatch")
                return None
            raise
    
    np.load = safe_np_load
    
    # Also patch the load_dataset_cache_file function directly
    try:
        import ultralytics.data.utils as data_utils
        original_load_cache = data_utils.load_dataset_cache_file
        
        def safe_load_cache(path):
            """Safe cache loader that handles numpy compatibility"""
            try:
                return original_load_cache(path)
            except Exception as e:
                if "numpy._core" in str(e) or "ModuleNotFoundError" in str(type(e)):
                    print(f"Warning: Cache file incompatible, will regenerate: {path}")
                    return None
                raise
        
        data_utils.load_dataset_cache_file = safe_load_cache
    except Exception:
        pass
    
    # Clear any problematic cache files that might cause numpy._core issues
    try:
        import glob
        cache_files = glob.glob(str(config.get_dataset_path()) + "/**/*.cache", recursive=True)
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                print(f"Removed problematic cache file: {cache_file}")
            except Exception:
                pass
    except Exception:
        pass
    
    try:
        # Load pretrained YOLO11 model
        model = YOLO(str(model_path))
        
        # Train the model (this will also use our patched torch.load)
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
            verbose=True,
            exist_ok=True,  # Allow overwriting existing runs
            cache=False  # Disable caching to avoid numpy._core compatibility issues
        )
        
    finally:
        # Always restore original torch.load
        torch.load = original_torch_load
    
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
            "original_pretrained": model_name,
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
    
    args = parser.parse_args()
    
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
            project_name=args.name
        )
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
