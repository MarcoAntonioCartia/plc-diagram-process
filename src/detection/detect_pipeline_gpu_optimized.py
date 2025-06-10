"""
GPU-Optimized Detection Pipeline for PLC Diagrams
Properly utilizes GPU for batch processing with pre-loaded tensors
"""

import json
import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.detection.yolo11_infer import load_model
from src.detection.coordinate_transform import transform_detections_to_global
from src.detection.reconstruct_with_detections import reconstruct_pdf_with_detections
from src.detection.yolo11_train import get_best_device


class ImageDataset(Dataset):
    """Custom dataset for loading images efficiently"""
    
    def __init__(self, image_paths, img_size=640):
        self.image_paths = image_paths
        self.img_size = img_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert to tensor and normalize
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)  # HWC to CHW
        
        return img, str(img_path), img_path.name


class GPUOptimizedPLCDetectionPipeline:
    def __init__(self, model_path=None, confidence_threshold=0.25, batch_size=32, 
                 num_workers=4, img_size=640, use_amp=True):
        """
        Initialize the GPU-optimized PLC detection pipeline
        
        Args:
            model_path: Path to YOLO11 model (None for auto-detect)
            confidence_threshold: Detection confidence threshold
            batch_size: Number of images to process in each GPU batch
            num_workers: Number of parallel workers for data loading
            img_size: Input image size for model
            use_amp: Use automatic mixed precision for faster inference
        """
        self.model = load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.use_amp = use_amp
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Set device
        device_id = get_best_device()
        if device_id == 'cpu':
            self.device = 'cpu'
            self.use_amp = False  # Disable AMP for CPU
        else:
            self.device = f'cuda:{device_id}'
        
        print(f"Using device: {self.device}")
        print(f"Automatic Mixed Precision: {'Enabled' if self.use_amp else 'Disabled'}")
        
        # Move model to GPU and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Warm up GPU
        if self.device != 'cpu':
            self._warmup_gpu()
    
    def _warmup_gpu(self):
        """Warm up GPU with dummy inference"""
        print("Warming up GPU...")
        dummy_batch = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        
        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    _ = self.model(dummy_batch)
            else:
                _ = self.model(dummy_batch)
        
        if self.device != 'cpu':
            torch.cuda.synchronize()
        print("GPU warmup complete")
    
    def _detect_on_snippets_gpu_optimized(self, images_folder):
        """Run YOLO11 detection using optimized GPU batch processing"""
        images_folder = Path(images_folder)
        
        # Find all PNG files
        image_files = [f for f in images_folder.glob("*.png") if not f.name.endswith("_metadata.json")]
        
        if not image_files:
            print("No image snippets found for detection")
            return {}
        
        print(f"Found {len(image_files)} image snippets to process")
        
        # Create dataset and dataloader
        dataset = ImageDataset(image_files, self.img_size)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=(self.device != 'cpu'),
            prefetch_factor=2 if self.num_workers > 0 else None
        )
        
        all_detections = {}
        total_batches = len(dataloader)
        
        # Process batches
        with torch.no_grad():
            for batch_idx, (images, paths, names) in enumerate(tqdm(dataloader, desc="GPU Detection")):
                # Move batch to GPU
                images = images.to(self.device, non_blocking=True)
                
                # Run inference with AMP if enabled
                if self.use_amp and self.device != 'cpu':
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                # Process outputs
                for i, (output, path, name) in enumerate(zip(outputs, paths, names)):
                    detections = []
                    
                    if output.boxes is not None:
                        boxes = output.boxes
                        
                        # Filter by confidence
                        conf_mask = boxes.conf >= self.confidence_threshold
                        if conf_mask.any():
                            filtered_boxes = boxes[conf_mask]
                            
                            for box in filtered_boxes:
                                detection = {
                                    "class_id": int(box.cls[0].item()),
                                    "class_name": self.model.names[int(box.cls[0].item())],
                                    "confidence": float(box.conf[0].item()),
                                    "bbox": {
                                        "x1": float(box.xyxy[0][0].item()),
                                        "y1": float(box.xyxy[0][1].item()),
                                        "x2": float(box.xyxy[0][2].item()),
                                        "y2": float(box.xyxy[0][3].item())
                                    }
                                }
                                detections.append(detection)
                    
                    all_detections[name] = {
                        "image_path": path,
                        "detections": detections,
                        "detection_count": len(detections)
                    }
                
                # Synchronize GPU every few batches to prevent timeout
                if self.device != 'cpu' and (batch_idx + 1) % 10 == 0:
                    torch.cuda.synchronize()
        
        # Final synchronization
        if self.device != 'cpu':
            torch.cuda.synchronize()
        
        # Save detection results
        results_file = images_folder / "all_detections.json"
        with open(results_file, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        # Print summary
        total_detections = sum(d.get("detection_count", 0) for d in all_detections.values())
        print(f"Total detections: {total_detections}")
        print(f"Detection results saved to: {results_file}")
        
        return all_detections
    
    def process_pdf_folder(self, diagrams_folder, output_folder=None, snippet_size=(1500, 1200), 
                          overlap=500, skip_pdf_conversion=False):
        """
        Complete pipeline with GPU-optimized processing
        """
        diagrams_folder = Path(diagrams_folder)
        if not diagrams_folder.exists():
            raise FileNotFoundError(f"Diagrams folder not found: {diagrams_folder}")
        
        # Set up output folders
        if output_folder is None:
            output_folder = diagrams_folder.parent
        
        output_folder = Path(output_folder)
        images_folder = output_folder / "images"
        detdiagrams_folder = output_folder / "detdiagrams"
        
        # Create output directories
        images_folder.mkdir(exist_ok=True)
        detdiagrams_folder.mkdir(exist_ok=True)
        
        print("Starting GPU-Optimized PLC Detection Pipeline")
        print("=" * 50)
        print(f"Batch size: {self.batch_size}")
        print(f"Workers: {self.num_workers}")
        print(f"Device: {self.device}")
        print(f"Image size: {self.img_size}")
        print("=" * 50)
        
        # Step 1: Convert PDFs to snippets (if needed)
        if skip_pdf_conversion:
            print("Step 1: Skipping PDF conversion (using existing images)")
            image_count = len(list(images_folder.glob("*.png")))
            if image_count == 0:
                raise FileNotFoundError(f"No PNG images found in {images_folder}")
            print(f"Found {image_count} existing image snippets")
        else:
            print("Step 1: Converting PDFs to image snippets...")
            self._run_pdf_to_snippets_parallel(diagrams_folder, images_folder, snippet_size, overlap)
        
        # Step 2: Run GPU-optimized detection
        print("\nStep 2: Running GPU-optimized YOLO11 detection...")
        start_time = time.time()
        detection_results = self._detect_on_snippets_gpu_optimized(images_folder)
        detection_time = time.time() - start_time
        
        print(f"Detection completed in {detection_time:.2f} seconds")
        print(f"Speed: {len(detection_results)/detection_time:.2f} images/second")
        
        # Step 3: Transform coordinates and reconstruct PDFs
        print("\nStep 3: Transforming coordinates and reconstructing PDFs...")
        self._reconstruct_with_detections_parallel(images_folder, detdiagrams_folder, 
                                                  detection_results, diagrams_folder)
        
        print("\nPipeline completed successfully!")
        print(f"Results saved to: {detdiagrams_folder}")
        
        return detdiagrams_folder
    
    def _run_pdf_to_snippets_parallel(self, diagrams_folder, images_folder, snippet_size, overlap):
        """Run PDF to snippets conversion using parallel processing"""
        import sys
        src_path = str(self.project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from preprocessing.preprocessing_parallel import ParallelPDFProcessor
        
        # Use parallel PDF processor
        processor = ParallelPDFProcessor(
            num_workers=self.num_workers,
            snippet_size=snippet_size,
            overlap=overlap
        )
        
        # Process PDFs in parallel
        print(f"Processing PDFs with {self.num_workers} parallel workers...")
        processor.process_pdf_folder(
            input_folder=diagrams_folder,
            output_folder=images_folder,
            show_progress=True
        )
    
    def _reconstruct_with_detections_parallel(self, images_folder, detdiagrams_folder, 
                                            detection_results, diagrams_folder):
        """Reconstruct PDFs with detection overlays using parallel processing"""
        images_folder = Path(images_folder)
        
        # Find all metadata files
        metadata_files = list(images_folder.glob("*_metadata.json"))
        metadata_files = [f for f in metadata_files if f.name not in ["all_detections.json", "all_pdfs_metadata.json"]]
        
        if not metadata_files:
            print("No metadata files found for reconstruction")
            return
        
        def process_single_pdf_reconstruction(metadata_file):
            try:
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                pdf_name = metadata["original_pdf"]
                
                # Transform detections to global coordinates
                global_detections = transform_detections_to_global(
                    detection_results, metadata
                )
                
                # Find original PDF
                original_pdf = self._find_original_pdf(pdf_name, diagrams_folder)
                
                # Reconstruct PDF with detections
                output_files = reconstruct_pdf_with_detections(
                    metadata=metadata,
                    global_detections=global_detections,
                    images_folder=images_folder,
                    output_folder=detdiagrams_folder,
                    original_pdf=original_pdf
                )
                
                return pdf_name, len(output_files), None
                
            except Exception as e:
                return metadata_file.stem, 0, str(e)
        
        # Use ThreadPoolExecutor for I/O-bound reconstruction
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_pdf_reconstruction, metadata_files),
                total=len(metadata_files),
                desc="Reconstructing PDFs"
            ))
        
        # Report results
        successful = sum(1 for _, files, error in results if error is None)
        print(f"Reconstructed {successful}/{len(metadata_files)} PDFs successfully")
    
    def _find_original_pdf(self, base_name, diagrams_folder):
        """Find the original PDF file"""
        diagrams_folder = Path(diagrams_folder)
        
        exact_match = diagrams_folder / f"{base_name}.pdf"
        if exact_match.exists():
            return exact_match
        
        matches = list(diagrams_folder.glob(f"{base_name}*.pdf"))
        if matches:
            return matches[0]
        
        return None


def benchmark_gpu_detection(model_path, images_folder, batch_sizes=[8, 16, 32, 64], num_workers=4):
    """Benchmark GPU detection with different batch sizes"""
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING GPU DETECTION (batch_size={batch_size})")
        print(f"{'='*60}")
        
        pipeline = GPUOptimizedPLCDetectionPipeline(
            model_path=model_path,
            batch_size=batch_size,
            num_workers=num_workers,
            use_amp=True
        )
        
        start_time = time.time()
        detection_results = pipeline._detect_on_snippets_gpu_optimized(images_folder)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        image_count = len(detection_results)
        speed = image_count / elapsed_time if elapsed_time > 0 else 0
        
        results[f'batch_{batch_size}'] = {
            'time': elapsed_time,
            'images': image_count,
            'speed': speed
        }
        
        print(f"\nGPU Results (batch_size={batch_size}):")
        print(f"  - Time: {elapsed_time:.2f} seconds")
        print(f"  - Images processed: {image_count}")
        print(f"  - Speed: {speed:.2f} images/second")
        
        # Clear GPU cache between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='GPU-Optimized PLC Detection Pipeline')
    parser.add_argument('--diagrams', '-d', required=True,
                       help='Folder containing PDF diagrams')
    parser.add_argument('--output', '-o', default=None,
                       help='Output folder (default: parent of diagrams folder)')
    parser.add_argument('--model', '-m', default=None,
                       help='Path to YOLO11 model (default: auto-detect)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                       help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size for GPU inference (default: 32)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size for model (default: 640)')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')
    parser.add_argument('--skip-pdf-conversion', action='store_true',
                       help='Skip PDF to image conversion')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark mode with different batch sizes')
    
    args = parser.parse_args()
    
    try:
        if args.benchmark:
            # Run benchmark mode
            from src.config import get_config
            config = get_config()
            
            if args.model is None:
                # Auto-detect model
                runs_dir = config.get_run_path('train')
                if runs_dir.exists():
                    train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "plc_symbol_detector" in d.name]
                    if train_dirs:
                        latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
                        args.model = latest_dir / "weights" / "best.pt"
            
            images_folder = Path(args.output) / "images" if args.output else Path(config.config['data_root']) / "processed" / "images"
            
            results = benchmark_gpu_detection(
                args.model,
                images_folder,
                batch_sizes=[8, 16, 32, 64],
                num_workers=args.workers
            )
            
            print("\n" + "="*60)
            print("BENCHMARK SUMMARY")
            print("="*60)
            for config_name, result in results.items():
                print(f"{config_name}: {result['speed']:.2f} img/s ({result['time']:.2f}s)")
            
        else:
            # Normal pipeline mode
            pipeline = GPUOptimizedPLCDetectionPipeline(
                model_path=args.model,
                confidence_threshold=args.conf,
                batch_size=args.batch_size,
                num_workers=args.workers,
                img_size=args.img_size,
                use_amp=not args.no_amp
            )
            
            result_folder = pipeline.process_pdf_folder(
                diagrams_folder=args.diagrams,
                output_folder=args.output,
                skip_pdf_conversion=args.skip_pdf_conversion
            )
            
            print(f"\nPipeline completed successfully!")
            print(f"Results available in: {result_folder}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
