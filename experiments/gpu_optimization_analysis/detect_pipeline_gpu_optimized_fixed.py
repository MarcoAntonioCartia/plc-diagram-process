"""
Fixed GPU-Optimized Detection Pipeline for PLC Diagrams
Properly utilizes YOLO's native batch processing for maximum GPU efficiency
"""

import json
import argparse
from pathlib import Path
import sys
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import gc

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.detection.yolo11_infer import load_model
from src.detection.coordinate_transform import transform_detections_to_global
from src.detection.reconstruct_with_detections import reconstruct_pdf_with_detections
from src.detection.yolo11_train import get_best_device


class OptimizedGPUPipeline:
    """
    Properly optimized GPU pipeline using YOLO's native batch processing.
    Fixes the architectural issues in the original implementation.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.25, 
                 max_batch_size=32, num_workers=4, warmup_batches=3):
        """
        Initialize the optimized GPU pipeline
        
        Args:
            model_path: Path to YOLO11 model (None for auto-detect)
            confidence_threshold: Detection confidence threshold
            max_batch_size: Maximum batch size (will be optimized automatically)
            num_workers: Number of parallel workers for non-GPU tasks
            warmup_batches: Number of warmup batches for GPU optimization
        """
        self.confidence_threshold = confidence_threshold
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers
        self.warmup_batches = warmup_batches
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Set device and initialize model properly
        self.device = self._get_optimal_device()
        self.model = self._initialize_model(model_path)
        
        # Determine optimal batch size
        self.optimal_batch_size = self._determine_optimal_batch_size()
        
        print(f"GPU Pipeline Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Model: {self.model.model_name if hasattr(self.model, 'model_name') else 'YOLO11'}")
        print(f"  Optimal batch size: {self.optimal_batch_size}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Workers: {self.num_workers}")
        
        # Warm up the model
        self._warmup_model()
    
    def _get_optimal_device(self):
        """Get the optimal device for inference"""
        device_id = get_best_device()
        if device_id == 'cpu':
            print("Warning: No CUDA device available, using CPU")
            return 'cpu'
        else:
            device = f'cuda:{device_id}'
            print(f"Using CUDA device: {device}")
            return device
    
    def _initialize_model(self, model_path):
        """Initialize YOLO model properly"""
        if model_path is None:
            model = load_model(model_path)
        else:
            model = YOLO(model_path)
        
        # Let YOLO handle device placement internally
        # Don't manually move to device - YOLO manages this
        
        # Configure model for optimal inference
        if hasattr(model, 'model'):
            model.model.eval()  # Ensure eval mode
        
        return model
    
    def _determine_optimal_batch_size(self):
        """Determine optimal batch size based on GPU memory and model"""
        if self.device == 'cpu':
            return min(8, self.max_batch_size)  # Conservative for CPU
        
        try:
            # Get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            # Estimate memory per image (rough calculation)
            # YOLO11 with 640x640 input: ~4MB per image in FP32
            memory_per_image_mb = 4
            available_memory_mb = (gpu_memory_gb * 1024) * 0.7  # Use 70% of GPU memory
            
            estimated_batch_size = int(available_memory_mb / memory_per_image_mb)
            optimal_batch_size = min(estimated_batch_size, self.max_batch_size)
            
            # Ensure it's a reasonable size
            optimal_batch_size = max(1, min(optimal_batch_size, 64))
            
            print(f"GPU Memory: {gpu_memory_gb:.1f}GB")
            print(f"Estimated optimal batch size: {optimal_batch_size}")
            
            return optimal_batch_size
            
        except Exception as e:
            print(f"Could not determine optimal batch size: {e}")
            return min(16, self.max_batch_size)
    
    def _warmup_model(self):
        """Warm up the model with dummy inference"""
        print("Warming up GPU model...")
        
        try:
            # Create dummy images for warmup
            dummy_images = []
            import tempfile
            import os
            
            for i in range(self.warmup_batches):
                # Create a dummy image file path (YOLO expects file paths)
                dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                # Use proper temporary directory for Windows compatibility
                temp_dir = tempfile.gettempdir()
                dummy_path = os.path.join(temp_dir, f"dummy_warmup_{i}.jpg")
                cv2.imwrite(dummy_path, dummy_img)
                dummy_images.append(dummy_path)
            
            # Run warmup inference
            start_time = time.time()
            _ = self.model(dummy_images, 
                          conf=self.confidence_threshold,
                          device=self.device,
                          verbose=False)
            warmup_time = time.time() - start_time
            
            # Clean up dummy files
            for dummy_path in dummy_images:
                try:
                    Path(dummy_path).unlink()
                except:
                    pass
            
            print(f"GPU warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            print(f"Warning: GPU warmup failed: {e}")
    
    def _detect_on_snippets_optimized(self, images_folder):
        """Run optimized YOLO detection using native batch processing"""
        images_folder = Path(images_folder)
        
        # Find all PNG files
        image_files = [f for f in images_folder.glob("*.png") 
                      if not f.name.endswith("_metadata.json")]
        
        if not image_files:
            print("No image snippets found for detection")
            return {}
        
        print(f"Processing {len(image_files)} images with batch size {self.optimal_batch_size}")
        
        all_detections = {}
        total_batches = (len(image_files) + self.optimal_batch_size - 1) // self.optimal_batch_size
        
        # Process images in optimized batches
        with tqdm(total=len(image_files), desc="GPU Detection", unit="img") as pbar:
            for i in range(0, len(image_files), self.optimal_batch_size):
                batch_files = image_files[i:i + self.optimal_batch_size]
                batch_paths = [str(f) for f in batch_files]
                
                # Use YOLO's native batch inference - this is the key optimization
                try:
                    start_time = time.time()
                    
                    # YOLO handles all preprocessing, batching, and GPU optimization internally
                    results = self.model(batch_paths,
                                       conf=self.confidence_threshold,
                                       device=self.device,
                                       verbose=False,
                                       stream=False)  # Don't use streaming for batch processing
                    
                    batch_time = time.time() - start_time
                    
                    # Process results
                    for result, file_path in zip(results, batch_files):
                        detections = []
                        
                        if result.boxes is not None and len(result.boxes) > 0:
                            boxes = result.boxes
                            
                            for box in boxes:
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
                        
                        all_detections[file_path.name] = {
                            "image_path": str(file_path),
                            "detections": detections,
                            "detection_count": len(detections)
                        }
                    
                    # Update progress
                    pbar.update(len(batch_files))
                    
                    # Performance logging
                    images_per_second = len(batch_files) / batch_time
                    pbar.set_postfix({
                        'batch_speed': f'{images_per_second:.1f} img/s',
                        'batch_size': len(batch_files)
                    })
                    
                except Exception as e:
                    print(f"Error processing batch {i//self.optimal_batch_size + 1}: {e}")
                    # Add error entries for this batch
                    for file_path in batch_files:
                        all_detections[file_path.name] = {
                            "image_path": str(file_path),
                            "detections": [],
                            "detection_count": 0,
                            "error": str(e)
                        }
                    pbar.update(len(batch_files))
                
                # Periodic GPU memory cleanup
                if (i // self.optimal_batch_size + 1) % 10 == 0:
                    self._cleanup_gpu_memory()
        
        # Save detection results
        results_file = images_folder / "all_detections.json"
        with open(results_file, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        # Print summary
        total_detections = sum(d.get("detection_count", 0) for d in all_detections.values())
        print(f"Detection completed:")
        print(f"  Total images: {len(all_detections)}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average detections per image: {total_detections/len(all_detections):.2f}")
        print(f"  Results saved to: {results_file}")
        
        return all_detections
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory periodically"""
        if self.device != 'cpu':
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Warning: GPU memory cleanup failed: {e}")
    
    def process_pdf_folder(self, diagrams_folder, output_folder=None, 
                          snippet_size=(1500, 1200), overlap=500, 
                          skip_pdf_conversion=False):
        """
        Complete pipeline with properly optimized GPU processing
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
        
        print("Starting Optimized GPU Detection Pipeline")
        print("=" * 60)
        
        # Step 1: Convert PDFs to snippets (if needed)
        if skip_pdf_conversion:
            print("Step 1: Skipping PDF conversion (using existing images)")
            image_count = len(list(images_folder.glob("*.png")))
            if image_count == 0:
                raise FileNotFoundError(f"No PNG images found in {images_folder}")
            print(f"Found {image_count} existing image snippets")
        else:
            print("Step 1: Converting PDFs to image snippets...")
            self._run_pdf_to_snippets_parallel(diagrams_folder, images_folder, 
                                              snippet_size, overlap)
        
        # Step 2: Run optimized GPU detection
        print("\nStep 2: Running optimized GPU detection...")
        start_time = time.time()
        detection_results = self._detect_on_snippets_optimized(images_folder)
        detection_time = time.time() - start_time
        
        images_processed = len(detection_results)
        speed = images_processed / detection_time if detection_time > 0 else 0
        
        print(f"\nDetection Performance:")
        print(f"  Time: {detection_time:.2f} seconds")
        print(f"  Images: {images_processed}")
        print(f"  Speed: {speed:.2f} images/second")
        print(f"  Batch size used: {self.optimal_batch_size}")
        
        # Step 3: Transform coordinates and reconstruct PDFs
        print("\nStep 3: Transforming coordinates and reconstructing PDFs...")
        self._reconstruct_with_detections_parallel(images_folder, detdiagrams_folder, 
                                                  detection_results, diagrams_folder)
        
        # Final cleanup
        self._cleanup_gpu_memory()
        
        print("\nOptimized pipeline completed successfully!")
        print(f"Results saved to: {detdiagrams_folder}")
        
        return detdiagrams_folder
    
    def _run_pdf_to_snippets_parallel(self, diagrams_folder, images_folder, 
                                     snippet_size, overlap):
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
        metadata_files = [f for f in metadata_files 
                         if f.name not in ["all_detections.json", "all_pdfs_metadata.json"]]
        
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


def benchmark_optimized_gpu(model_path, images_folder, batch_sizes=[8, 16, 32, 64], 
                           num_workers=4):
    """Benchmark the optimized GPU pipeline with different batch sizes"""
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING OPTIMIZED GPU (batch_size={batch_size})")
        print(f"{'='*60}")
        
        try:
            pipeline = OptimizedGPUPipeline(
                model_path=model_path,
                max_batch_size=batch_size,
                num_workers=num_workers
            )
            
            start_time = time.time()
            detection_results = pipeline._detect_on_snippets_optimized(images_folder)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            image_count = len(detection_results)
            speed = image_count / elapsed_time if elapsed_time > 0 else 0
            
            results[f'optimized_batch_{batch_size}'] = {
                'time': elapsed_time,
                'images': image_count,
                'speed': speed,
                'actual_batch_size': pipeline.optimal_batch_size
            }
            
            print(f"\nOptimized GPU Results (requested batch_size={batch_size}):")
            print(f"  - Actual batch size used: {pipeline.optimal_batch_size}")
            print(f"  - Time: {elapsed_time:.2f} seconds")
            print(f"  - Images processed: {image_count}")
            print(f"  - Speed: {speed:.2f} images/second")
            
            # Clean up
            del pipeline
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except Exception as e:
            print(f"Error benchmarking batch size {batch_size}: {e}")
            results[f'optimized_batch_{batch_size}'] = {
                'error': str(e)
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Optimized GPU Detection Pipeline')
    parser.add_argument('--diagrams', '-d', required=True,
                       help='Folder containing PDF diagrams')
    parser.add_argument('--output', '-o', default=None,
                       help='Output folder (default: parent of diagrams folder)')
    parser.add_argument('--model', '-m', default=None,
                       help='Path to YOLO11 model (default: auto-detect)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                       help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--max-batch-size', '-b', type=int, default=32,
                       help='Maximum batch size (will be optimized automatically)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
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
                    train_dirs = [d for d in runs_dir.iterdir() 
                                 if d.is_dir() and "plc_symbol_detector" in d.name]
                    if train_dirs:
                        latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
                        args.model = latest_dir / "weights" / "best.pt"
            
            images_folder = (Path(args.output) / "images" if args.output 
                           else Path(config.config['data_root']) / "processed" / "images")
            
            results = benchmark_optimized_gpu(
                args.model,
                images_folder,
                batch_sizes=[8, 16, 32, 64],
                num_workers=args.workers
            )
            
            print("\n" + "="*60)
            print("OPTIMIZED GPU BENCHMARK SUMMARY")
            print("="*60)
            for config_name, result in results.items():
                if 'error' in result:
                    print(f"{config_name}: ERROR - {result['error']}")
                else:
                    print(f"{config_name}: {result['speed']:.2f} img/s "
                          f"(actual batch: {result.get('actual_batch_size', 'unknown')})")
            
        else:
            # Normal pipeline mode
            pipeline = OptimizedGPUPipeline(
                model_path=args.model,
                confidence_threshold=args.conf,
                max_batch_size=args.max_batch_size,
                num_workers=args.workers
            )
            
            result_folder = pipeline.process_pdf_folder(
                diagrams_folder=args.diagrams,
                output_folder=args.output,
                skip_pdf_conversion=args.skip_pdf_conversion
            )
            
            print(f"\nOptimized pipeline completed successfully!")
            print(f"Results available in: {result_folder}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
