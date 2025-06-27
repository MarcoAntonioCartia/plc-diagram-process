"""
Unified Parallel Pipeline for PLC Diagram Processing
Combines parallel PDF preprocessing with GPU batch detection in a single optimized pipeline
"""

import json
import argparse
from pathlib import Path
import sys
import time
from ultralytics import YOLO
import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import threading
import queue

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.detection.yolo11_infer import load_model
from src.detection.coordinate_transform import transform_detections_to_global
from src.detection.reconstruct_with_detections import reconstruct_pdf_with_detections
from src.detection.yolo11_train import get_best_device
from src.preprocessing.preprocessing_parallel import StreamingPDFProcessor, ParallelPDFProcessor


class UnifiedParallelPipeline:
    """
    Unified pipeline that processes PDFs and detects symbols in parallel.
    Optimizes resource usage by overlapping CPU-intensive preprocessing with GPU detection.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.25, 
                 batch_size=32, pdf_workers=4, detection_workers=2,
                 streaming_mode=False):
        """
        Initialize the unified parallel pipeline.
        
        Args:
            model_path: Path to YOLO11 model (None for auto-detect)
            confidence_threshold: Detection confidence threshold
            batch_size: Number of images to process in each GPU batch
            pdf_workers: Number of parallel workers for PDF processing
            detection_workers: Number of workers for parallel operations
            streaming_mode: Enable streaming mode for lower memory usage
        """
        self.model = load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.pdf_workers = pdf_workers
        self.detection_workers = detection_workers
        self.streaming_mode = streaming_mode
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Set device
        device_id = get_best_device()
        if device_id == 'cpu':
            self.device = 'cpu'
        else:
            self.device = f'cuda:{device_id}'
        print(f"Using device: {self.device}")
        
        # Move model to GPU if available
        if self.device != 'cpu':
            self.model.to(self.device)
        
        # Statistics tracking
        self.stats = {
            'pdfs_processed': 0,
            'images_processed': 0,
            'total_detections': 0,
            'preprocessing_time': 0,
            'detection_time': 0,
            'reconstruction_time': 0
        }
    
    def process_pdf_folder(self, diagrams_folder, output_folder=None, 
                          snippet_size=(1500, 1200), overlap=500,
                          skip_pdf_conversion=False):
        """
        Process all PDFs in a folder with unified parallel processing.
        """
        start_time = time.time()
        
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
        
        self._print_configuration()
        
        if skip_pdf_conversion:
            # Use existing images
            print("\nStep 1: Using existing images (skipping PDF conversion)")
            image_count = len(list(images_folder.glob("*.png")))
            if image_count == 0:
                raise FileNotFoundError(f"No PNG images found in {images_folder}")
            print(f"Found {image_count} existing image snippets")
            
            # Run detection on existing images
            detection_results = self._run_detection_batch(images_folder)
            
            # Reconstruct PDFs
            self._run_reconstruction_parallel(
                images_folder, detdiagrams_folder, detection_results, diagrams_folder
            )
        else:
            if self.streaming_mode:
                # Streaming mode: process PDFs and detect simultaneously
                self._run_streaming_pipeline(
                    diagrams_folder, images_folder, detdiagrams_folder,
                    snippet_size, overlap
                )
            else:
                # Standard mode: process all PDFs first, then detect
                self._run_standard_pipeline(
                    diagrams_folder, images_folder, detdiagrams_folder,
                    snippet_size, overlap
                )
        
        # Print final statistics
        total_time = time.time() - start_time
        self._print_statistics(total_time)
        
        return detdiagrams_folder
    
    def _print_configuration(self):
        """Print pipeline configuration."""
        print("\nUnified Parallel Pipeline Configuration")
        print("=" * 50)
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"PDF workers: {self.pdf_workers}")
        print(f"Detection workers: {self.detection_workers}")
        print(f"Streaming mode: {'Enabled' if self.streaming_mode else 'Disabled'}")
        print("=" * 50)
    
    def _run_standard_pipeline(self, diagrams_folder, images_folder, 
                              detdiagrams_folder, snippet_size, overlap):
        """Run standard pipeline: preprocess all, then detect all."""
        # Step 1: Parallel PDF preprocessing
        print("\nStep 1: Parallel PDF preprocessing...")
        preprocess_start = time.time()
        
        processor = ParallelPDFProcessor(
            num_workers=self.pdf_workers,
            snippet_size=snippet_size,
            overlap=overlap
        )
        
        all_metadata = processor.process_pdf_folder(
            input_folder=diagrams_folder,
            output_folder=images_folder,
            show_progress=True
        )
        
        self.stats['preprocessing_time'] = time.time() - preprocess_start
        self.stats['pdfs_processed'] = len(all_metadata)
        
        # Step 2: Batch GPU detection
        print("\nStep 2: Batch GPU detection...")
        detection_results = self._run_detection_batch(images_folder)
        
        # Step 3: Parallel reconstruction
        print("\nStep 3: Parallel PDF reconstruction...")
        self._run_reconstruction_parallel(
            images_folder, detdiagrams_folder, detection_results, diagrams_folder
        )
    
    def _run_streaming_pipeline(self, diagrams_folder, images_folder, 
                               detdiagrams_folder, snippet_size, overlap):
        """Run streaming pipeline: process and detect simultaneously."""
        print("\nRunning streaming pipeline (preprocessing + detection in parallel)...")
        
        # Find all PDFs
        pdf_files = list(diagrams_folder.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {diagrams_folder}")
            return
        
        # Create streaming processor
        stream_processor = StreamingPDFProcessor(
            num_pdf_workers=self.pdf_workers,
            snippet_size=snippet_size,
            overlap=overlap
        )
        
        # Start PDF processing in background
        producer_thread = stream_processor.process_pdfs_streaming(
            pdf_files, images_folder
        )
        
        # Process images as they become available
        all_detections = {}
        detection_start = time.time()
        
        with tqdm(desc="Processing images", unit="img") as pbar:
            while not stream_processor.stop_signal.is_set() or not stream_processor.image_queue.empty():
                # Get batch of images
                batch_paths = stream_processor.get_image_batch(self.batch_size, timeout=1.0)
                
                if batch_paths:
                    # Run detection on batch
                    batch_detections = self._detect_batch(batch_paths)
                    all_detections.update(batch_detections)
                    pbar.update(len(batch_paths))
                    self.stats['images_processed'] += len(batch_paths)
        
        # Wait for producer to finish
        producer_thread.join()
        
        self.stats['detection_time'] = time.time() - detection_start
        self.stats['pdfs_processed'] = len(stream_processor.metadata_dict)
        
        # Save detection results
        results_file = images_folder / "all_detections.json"
        with open(results_file, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        # Step 3: Reconstruction
        print("\nStep 3: Parallel PDF reconstruction...")
        self._run_reconstruction_parallel(
            images_folder, detdiagrams_folder, all_detections, diagrams_folder
        )
    
    def _run_detection_batch(self, images_folder):
        """Run batch detection on all images in folder."""
        detection_start = time.time()
        
        # Find all PNG files
        image_files = [f for f in images_folder.glob("*.png") 
                      if not f.name.endswith("_metadata.json")]
        
        if not image_files:
            print("No image snippets found for detection")
            return {}
        
        print(f"Found {len(image_files)} image snippets to process")
        
        all_detections = {}
        
        # Process in batches
        for i in tqdm(range(0, len(image_files), self.batch_size), desc="Detecting"):
            batch_files = image_files[i:i + self.batch_size]
            batch_detections = self._detect_batch(batch_files)
            all_detections.update(batch_detections)
        
        self.stats['detection_time'] = time.time() - detection_start
        self.stats['images_processed'] = len(image_files)
        
        # Save detection results
        results_file = images_folder / "all_detections.json"
        with open(results_file, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        # Count total detections
        self.stats['total_detections'] = sum(
            d.get("detection_count", 0) for d in all_detections.values()
        )
        
        return all_detections
    
    def _detect_batch(self, image_paths):
        """Detect objects in a batch of images."""
        batch_detections = {}
        batch_paths_str = [str(p) for p in image_paths]
        
        try:
            # Run batch inference
            results = self.model(batch_paths_str, conf=self.confidence_threshold, 
                               device=self.device, verbose=False)
            
            # Process results
            for image_path, result in zip(image_paths, results):
                detections = []
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        detection = {
                            "class_id": int(box.cls[0]),
                            "class_name": self.model.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": {
                                "x1": float(box.xyxy[0][0]),
                                "y1": float(box.xyxy[0][1]),
                                "x2": float(box.xyxy[0][2]),
                                "y2": float(box.xyxy[0][3])
                            }
                        }
                        detections.append(detection)
                
                image_name = Path(image_path).name
                batch_detections[image_name] = {
                    "image_path": str(image_path),
                    "detections": detections,
                    "detection_count": len(detections)
                }
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Add error entries
            for image_path in image_paths:
                image_name = Path(image_path).name
                batch_detections[image_name] = {
                    "image_path": str(image_path),
                    "error": str(e)
                }
        
        return batch_detections
    
    def _run_reconstruction_parallel(self, images_folder, detdiagrams_folder,
                                    detection_results, diagrams_folder):
        """Run parallel PDF reconstruction."""
        reconstruction_start = time.time()
        
        # Find all metadata files
        metadata_files = list(images_folder.glob("*_metadata.json"))
        metadata_files = [f for f in metadata_files if f.name not in 
                         ["all_detections.json", "all_pdfs_metadata.json"]]
        
        if not metadata_files:
            print("No metadata files found for reconstruction")
            return
        
        def reconstruct_single_pdf(metadata_file):
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
        
        # Use ThreadPoolExecutor for parallel reconstruction
        with ThreadPoolExecutor(max_workers=self.detection_workers) as executor:
            results = list(tqdm(
                executor.map(reconstruct_single_pdf, metadata_files),
                total=len(metadata_files),
                desc="Reconstructing PDFs"
            ))
        
        # Report results
        successful = sum(1 for _, files, error in results if error is None)
        print(f"Reconstructed {successful}/{len(metadata_files)} PDFs successfully")
        
        self.stats['reconstruction_time'] = time.time() - reconstruction_start
    
    def _find_original_pdf(self, base_name, diagrams_folder):
        """Find the original PDF file."""
        diagrams_folder = Path(diagrams_folder)
        
        exact_match = diagrams_folder / f"{base_name}.pdf"
        if exact_match.exists():
            return exact_match
        
        matches = list(diagrams_folder.glob(f"{base_name}*.pdf"))
        if matches:
            return matches[0]
        
        return None
    
    def _print_statistics(self, total_time):
        """Print pipeline statistics."""
        print("\n" + "=" * 50)
        print("Pipeline Statistics")
        print("=" * 50)
        print(f"Total time: {total_time:.2f}s")
        print(f"PDFs processed: {self.stats['pdfs_processed']}")
        print(f"Images processed: {self.stats['images_processed']}")
        print(f"Total detections: {self.stats['total_detections']}")
        print(f"\nTime breakdown:")
        print(f"  - Preprocessing: {self.stats['preprocessing_time']:.2f}s")
        print(f"  - Detection: {self.stats['detection_time']:.2f}s")
        print(f"  - Reconstruction: {self.stats['reconstruction_time']:.2f}s")
        
        if self.stats['images_processed'] > 0:
            img_per_sec = self.stats['images_processed'] / self.stats['detection_time']
            print(f"\nPerformance:")
            print(f"  - Detection speed: {img_per_sec:.2f} images/second")
            print(f"  - Avg detections/image: {self.stats['total_detections']/self.stats['images_processed']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Unified Parallel Pipeline for PLC Diagram Processing')
    parser.add_argument('--diagrams', '-d', required=True,
                       help='Folder containing PDF diagrams')
    parser.add_argument('--output', '-o', default=None,
                       help='Output folder (default: parent of diagrams folder)')
    parser.add_argument('--model', '-m', default=None,
                       help='Path to YOLO11 model (default: auto-detect)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                       help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--snippet-size', nargs=2, type=int, default=[1500, 1200],
                       help='Snippet size as width height (default: 1500 1200)')
    parser.add_argument('--overlap', type=int, default=500,
                       help='Overlap between snippets (default: 500)')
    parser.add_argument('--skip-pdf-conversion', action='store_true',
                       help='Skip PDF to image conversion (use existing images)')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size for GPU inference (default: 32)')
    parser.add_argument('--pdf-workers', type=int, default=4,
                       help='Number of parallel workers for PDF processing (default: 4)')
    parser.add_argument('--detection-workers', type=int, default=2,
                       help='Number of workers for detection operations (default: 2)')
    parser.add_argument('--streaming', action='store_true',
                       help='Enable streaming mode for lower memory usage')
    
    args = parser.parse_args()
    
    try:
        # Initialize unified pipeline
        pipeline = UnifiedParallelPipeline(
            model_path=args.model,
            confidence_threshold=args.conf,
            batch_size=args.batch_size,
            pdf_workers=args.pdf_workers,
            detection_workers=args.detection_workers,
            streaming_mode=args.streaming
        )
        
        # Run pipeline
        result_folder = pipeline.process_pdf_folder(
            diagrams_folder=args.diagrams,
            output_folder=args.output,
            snippet_size=tuple(args.snippet_size),
            overlap=args.overlap,
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
