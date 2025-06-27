"""
Parallel Detection Pipeline for PLC Diagrams with GPU Batch Processing
Optimized for speed using batch inference and multiprocessing
"""

import json
import argparse
from pathlib import Path
import sys
import subprocess
from ultralytics import YOLO
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.detection.yolo11_infer import load_model
from src.detection.coordinate_transform import transform_detections_to_global
from src.detection.reconstruct_with_detections import reconstruct_pdf_with_detections
from src.detection.yolo11_train import get_best_device


class ParallelPLCDetectionPipeline:
    def __init__(self, model_path=None, confidence_threshold=0.25, batch_size=32, num_workers=4):
        """
        Initialize the parallel PLC detection pipeline
        
        Args:
            model_path: Path to YOLO11 model (None for auto-detect)
            confidence_threshold: Detection confidence threshold
            batch_size: Number of images to process in each GPU batch
            num_workers: Number of parallel workers for I/O operations
        """
        self.model = load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Set device
        device_id = get_best_device()
        if device_id == 'cpu':
            self.device = 'cpu'
        else:
            self.device = f'cuda:{device_id}'
        print(f"Using device: {self.device}")
        
        # Move model to GPU if available and set to eval mode
        if self.device != 'cpu':
            self.model.to(self.device)
        
        # Set model to evaluation mode for inference
        self.model.eval()
        
        # Warm up GPU (important for accurate benchmarking)
        if self.device != 'cpu':
            print("Warming up GPU...")
            dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            torch.cuda.synchronize()
        
    def process_pdf_folder(self, diagrams_folder, output_folder=None, snippet_size=(1500, 1200), 
                          overlap=500, skip_pdf_conversion=False, parallel_pdfs=True):
        """
        Complete pipeline with parallel processing
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
        
        print("Starting Parallel PLC Detection Pipeline")
        print("=" * 50)
        print(f"Batch size: {self.batch_size}")
        print(f"Workers: {self.num_workers}")
        print(f"Device: {self.device}")
        print("=" * 50)
        
        # Step 1: Convert PDFs to snippets
        if skip_pdf_conversion:
            print("Step 1: Skipping PDF conversion (using existing images)")
            image_count = len(list(images_folder.glob("*.png")))
            if image_count == 0:
                raise FileNotFoundError(f"No PNG images found in {images_folder}. Run preprocessing first.")
            print(f"Found {image_count} existing image snippets")
        else:
            print("Step 1: Converting PDFs to image snippets...")
            if parallel_pdfs:
                self._run_pdf_to_snippets_parallel(diagrams_folder, images_folder, snippet_size, overlap)
            else:
                self._run_pdf_to_snippets(diagrams_folder, images_folder, snippet_size, overlap)
        
        # Step 2: Run batch detection on all snippets
        print("\nStep 2: Running YOLO11 batch detection on snippets...")
        detection_results = self._detect_on_snippets_batch(images_folder)
        
        # Step 3: Transform coordinates and reconstruct PDFs (parallel)
        print("\nStep 3: Transforming coordinates and reconstructing PDFs...")
        self._reconstruct_with_detections_parallel(images_folder, detdiagrams_folder, 
                                                  detection_results, diagrams_folder)
        
        print("\nPipeline completed successfully!")
        print(f"Results saved to: {detdiagrams_folder}")
        
        return detdiagrams_folder
    
    def _run_pdf_to_snippets(self, diagrams_folder, images_folder, snippet_size, overlap):
        """Run the PDF to snippets conversion (same as original)"""
        import sys
        src_path = str(self.project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from preprocessing.SnipPdfToPng import process_pdf_folder, find_poppler_path
        
        poppler_path = find_poppler_path()
        if poppler_path is None:
            print("Native poppler not found, will try WSL if available")
        
        process_pdf_folder(
            input_folder=diagrams_folder,
            output_folder=images_folder,
            snippet_size=snippet_size,
            overlap=overlap,
            poppler_path=poppler_path
        )
    
    def _run_pdf_to_snippets_parallel(self, diagrams_folder, images_folder, snippet_size, overlap):
        """Run PDF to snippets conversion using true parallel processing"""
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
        all_metadata = processor.process_pdf_folder(
            input_folder=diagrams_folder,
            output_folder=images_folder,
            show_progress=True
        )
        
        return all_metadata
    
    def _detect_on_snippets_batch(self, images_folder):
        """Run YOLO11 detection on all image snippets using batch processing"""
        images_folder = Path(images_folder)
        
        # Find all PNG files (excluding metadata)
        image_files = [f for f in images_folder.glob("*.png") if not f.name.endswith("_metadata.json")]
        
        if not image_files:
            print("No image snippets found for detection")
            return {}
        
        print(f"Found {len(image_files)} image snippets to process")
        
        all_detections = {}
        
        # Process in batches
        for i in tqdm(range(0, len(image_files), self.batch_size), desc="Detecting"):
            batch_files = image_files[i:i + self.batch_size]
            batch_paths = [str(f) for f in batch_files]
            
            try:
                # Run batch inference
                results = self.model(batch_paths, conf=self.confidence_threshold, 
                                   device=self.device, verbose=False)
                
                # Process results
                for idx, (image_file, result) in enumerate(zip(batch_files, results)):
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
                    
                    all_detections[image_file.name] = {
                        "image_path": str(image_file),
                        "detections": detections,
                        "detection_count": len(detections)
                    }
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Fallback to individual processing for this batch
                for image_file in batch_files:
                    all_detections[image_file.name] = {
                        "image_path": str(image_file),
                        "error": str(e)
                    }
        
        # Save detection results
        results_file = images_folder / "all_detections.json"
        with open(results_file, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        # Print summary
        total_detections = sum(d.get("detection_count", 0) for d in all_detections.values())
        print(f"Total detections: {total_detections}")
        print(f"Detection results saved to: {results_file}")
        
        return all_detections
    
    def _reconstruct_with_detections_parallel(self, images_folder, detdiagrams_folder, 
                                            detection_results, diagrams_folder):
        """Reconstruct PDFs with detection overlays using parallel processing"""
        images_folder = Path(images_folder)
        
        # Find all metadata files
        metadata_files = list(images_folder.glob("*_metadata.json"))
        metadata_files = [f for f in metadata_files if f.name != "all_detections.json"]
        
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


def main():
    parser = argparse.ArgumentParser(description='Parallel PLC Detection Pipeline with GPU Batch Processing')
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
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--no-parallel-pdf', action='store_true',
                       help='Disable parallel PDF processing')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = ParallelPLCDetectionPipeline(
            model_path=args.model,
            confidence_threshold=args.conf,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
        
        # Run complete pipeline
        result_folder = pipeline.process_pdf_folder(
            diagrams_folder=args.diagrams,
            output_folder=args.output,
            snippet_size=tuple(args.snippet_size),
            overlap=args.overlap,
            skip_pdf_conversion=args.skip_pdf_conversion,
            parallel_pdfs=not args.no_parallel_pdf
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
