# YOLO11 Inference Script for PLC Symbol Detection
# https://docs.ultralytics.com/modes/predict/

import argparse
import json
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config

def load_model(model_path=None):
    """
    Load YOLO11 model for inference using configuration management
    """
    config = get_config()
    
    if model_path is None:
        # Try to find the best custom trained model
        custom_models_dir = config.get_model_path('', 'custom').parent
        
        if custom_models_dir.exists():
            # Look for models with metadata
            model_files = list(custom_models_dir.glob("*_best.pt"))
            
            if model_files:
                # Find the most recent model
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                
                # Check if metadata exists
                metadata_file = latest_model.with_suffix('.json')
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    print(f"Using trained model: {latest_model.name}")
                    print(f"  - Dataset: {metadata.get('dataset', 'unknown')}")
                    print(f"  - Epochs: {metadata.get('epochs_trained', 'unknown')}")
                    print(f"  - mAP50: {metadata.get('metrics', {}).get('mAP50', 'unknown')}")
                else:
                    print(f"Using trained model: {latest_model}")
                
                model_path = latest_model
        
        # Fallback to pretrained model
        if model_path is None:
            model_path = config.get_model_path('yolo11m.pt', 'pretrained')
            print(f"Using pretrained model: {model_path}")
    
    # Handle string paths
    model_path = Path(model_path)
    
    # If it's just a filename, look for it in the models directory
    if not model_path.is_absolute() and not model_path.exists():
        # Try custom models first
        custom_path = config.get_model_path(model_path.name, 'custom')
        if custom_path.exists():
            model_path = custom_path
        else:
            # Try pretrained models
            pretrained_path = config.get_model_path(model_path.name, 'pretrained')
            if pretrained_path.exists():
                model_path = pretrained_path
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load YOLO model - with updated Ultralytics, no patches needed
    model = YOLO(str(model_path))
    return model

def predict_image(model, image_path, conf_threshold=0.25, save_results=True, output_dir=None):
    """
    Run inference on a single image
    """
    config = get_config()
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"Processing image: {image_path}")
    
    # Set output directory
    if output_dir is None and save_results:
        output_dir = config.get_run_path('detect') / 'predict'
    
    # Run inference
    predict_kwargs = {
        'conf': conf_threshold,
        'save': save_results,
        'exist_ok': True
    }
    
    if save_results and output_dir:
        predict_kwargs['project'] = str(output_dir.parent)
        predict_kwargs['name'] = output_dir.name
    
    results = model(str(image_path), **predict_kwargs)
    
    # Extract detection results
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                detection = {
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                }
                detections.append(detection)
    
    return detections, results

def predict_folder(model, folder_path, output_dir=None, conf_threshold=0.25):
    """
    Run inference on all images in a folder
    """
    config = get_config()
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if output_dir is None:
        output_dir = config.get_run_path('detect') / f"{folder_path.name}_results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all images
    image_files = [f for f in folder_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    all_results = {}
    
    # Create visualizations directory
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    for idx, image_file in enumerate(image_files, 1):
        try:
            print(f"[{idx}/{len(image_files)}] Processing {image_file.name}...", end='', flush=True)
            
            detections, raw_results = predict_image(
                model, image_file, conf_threshold, 
                save_results=False
            )
            
            all_results[image_file.name] = {
                "image_path": str(image_file),
                "detections": detections,
                "detection_count": len(detections)
            }
            
            # Save annotated image
            for result in raw_results:
                annotated = result.plot()
                cv2.imwrite(str(vis_dir / f"{image_file.stem}_annotated{image_file.suffix}"), annotated)
            
            print(f" V {len(detections)} detections")
            
        except Exception as e:
            print(f" X Error: {e}")
            all_results[image_file.name] = {
                "image_path": str(image_file),
                "error": str(e)
            }
    
    # Save results to JSON
    results_file = output_dir / "detection_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary
    summary = {
        "total_images": len(image_files),
        "successful": sum(1 for r in all_results.values() if 'detections' in r),
        "failed": sum(1 for r in all_results.values() if 'error' in r),
        "total_detections": sum(r.get('detection_count', 0) for r in all_results.values()),
        "average_detections": sum(r.get('detection_count', 0) for r in all_results.values()) / len([r for r in all_results.values() if 'detections' in r]) if any('detections' in r for r in all_results.values()) else 0
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Visualizations saved to: {vis_dir}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='YOLO11 PLC Symbol Detection Inference')
    parser.add_argument('--input', '-i', required=True, 
                       help='Input image file or folder path')
    parser.add_argument('--model', '-m', default=None,
                       help='Path to model file or model name (default: auto-detect best trained model)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory for results')
    parser.add_argument('--save-images', action='store_true',
                       help='Save annotated images')
    parser.add_argument('--show-config', action='store_true',
                       help='Show configuration information')
    
    args = parser.parse_args()
    
    try:
        # Show configuration if requested
        if args.show_config:
            config = get_config()
            print("Configuration Information:")
            print("=" * 50)
            print(f"Config file: {config.config_path}")
            print(f"Data root: {config.config['data_root']}")
            print(f"Models directory: {config.config['paths']['models']}")
            print(f"Runs directory: {config.config['paths']['runs']}")
            print("=" * 50)
            print()
        
        # Load model
        print("Loading YOLO11 model...")
        model = load_model(args.model)
        print()
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single image inference
            detections, results = predict_image(
                model, input_path, args.conf, 
                args.save_images, args.output
            )
            
            print(f"\nDetection Results for {input_path.name}:")
            print("=" * 50)
            for i, det in enumerate(detections, 1):
                print(f"{i}. {det['class_name']} (confidence: {det['confidence']:.3f})")
                bbox = det['bbox']
                print(f"   BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) -> ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
            
            if not detections:
                print("No detections found.")
            
            # Save single image results
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                results_file = output_dir / f"{input_path.stem}_results.json"
                with open(results_file, 'w') as f:
                    json.dump({
                        "image_path": str(input_path),
                        "detections": detections,
                        "model_used": str(model.model_path) if hasattr(model, 'model_path') else "unknown"
                    }, f, indent=2)
                print(f"\nResults saved to: {results_file}")
        
        elif input_path.is_dir():
            # Folder inference
            results = predict_folder(model, input_path, args.output, args.conf)
            
            # Summary
            total_images = len(results)
            successful = sum(1 for r in results.values() if 'detections' in r)
            total_detections = sum(r.get('detection_count', 0) for r in results.values())
            
            print(f"\nSummary:")
            print("=" * 30)
            print(f"Total images processed: {total_images}")
            print(f"Successful: {successful}")
            print(f"Failed: {total_images - successful}")
            print(f"Total detections: {total_detections}")
            print(f"Average detections per image: {total_detections/successful if successful > 0 else 0:.2f}")
        
        else:
            print(f"Error: {input_path} is neither a file nor a directory")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
