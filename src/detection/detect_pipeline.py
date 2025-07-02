"""
Complete Detection Pipeline for PLC Diagrams
Orchestrates: PDF → Snippets → Detection → Coordinate Mapping → Reconstructed PDF
"""

import json
import argparse
from pathlib import Path
import sys
import subprocess
from ultralytics import YOLO

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.detection.yolo11_infer import load_model, predict_image
from src.detection.coordinate_transform import transform_detections_to_global
from src.detection.reconstruct_with_detections import reconstruct_pdf_with_detections

class PLCDetectionPipeline:
    def __init__(self, model_path=None, confidence_threshold=0.25):
        """
        Initialize the PLC detection pipeline
        
        Args:
            model_path: Path to YOLO11 model (None for auto-detect)
            confidence_threshold: Detection confidence threshold
        """
        self.model = load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
    def process_pdf_folder(self, diagrams_folder, output_folder=None, snippet_size=(1500, 1200), overlap=500, skip_pdf_conversion=False):
        """
        Complete pipeline: PDF → Snippets → Detection → Reconstruction
        
        Args:
            diagrams_folder: Folder containing PDF files
            output_folder: Output folder for results (auto-generated if None)
            snippet_size: Size of image snippets
            overlap: Overlap between snippets
            skip_pdf_conversion: Skip PDF to image conversion step (assumes images already exist)
        """
        diagrams_folder = Path(diagrams_folder)
        if not diagrams_folder.exists():
            raise FileNotFoundError(f"Diagrams folder not found: {diagrams_folder}")
        
        # Set up output folders using config system (like the working pipeline)
        if output_folder is None:
            # Use config to get the correct data root instead of assuming parent relationships
            from src.config import get_config
            config = get_config()
            data_root = Path(config.config["data_root"])
            images_folder = data_root / "processed" / "images"
            detdiagrams_folder = data_root / "processed" / "detdiagrams"
        else:
            # When output_folder is provided, it's already pointing to the final destination
            # Don't add /processed/ because the caller already provides the full path
            output_folder = Path(output_folder)
            if "processed" in str(output_folder):
                # The output_folder already includes processed path, use it directly
                images_folder = output_folder.parent / "images"
                detdiagrams_folder = output_folder
            else:
                # Legacy behavior for backward compatibility
                images_folder = output_folder / "processed" / "images"
                detdiagrams_folder = output_folder / "processed" / "detdiagrams"
        
        # Create output directories
        images_folder.mkdir(parents=True, exist_ok=True)
        detdiagrams_folder.mkdir(parents=True, exist_ok=True)
        
        print("Starting PLC Detection Pipeline")
        print("=" * 50)
        
        # Step 1: Convert PDFs to snippets (skip if requested)
        if skip_pdf_conversion:
            print("Step 1: Skipping PDF conversion (using existing images)")
            # Verify images exist
            image_count = len(list(images_folder.glob("*.png")))
            if image_count == 0:
                raise FileNotFoundError(f"No PNG images found in {images_folder}. Run preprocessing first.")
            print(f"Found {image_count} existing image snippets")
        else:
            print("Step 1: Converting PDFs to image snippets...")
            self._run_pdf_to_snippets(diagrams_folder, images_folder, snippet_size, overlap)
        
        # Step 2: Run detection on all snippets
        print("\nStep 2: Running YOLO11 detection on snippets...")
        detection_results = self._detect_on_snippets(images_folder)
        
        # Step 3: Transform coordinates and reconstruct PDFs
        print("\nStep 3: Transforming coordinates and reconstructing PDFs...")
        self._reconstruct_with_detections(images_folder, detdiagrams_folder, detection_results, diagrams_folder)
        
        print("\nPipeline completed successfully!")
        print(f"Results saved to: {detdiagrams_folder}")
        
        return detdiagrams_folder
    
    def _run_pdf_to_snippets(self, diagrams_folder, images_folder, snippet_size, overlap):
        """Run the PDF to snippets conversion"""
        # Add the src directory to sys.path to ensure proper imports
        import sys
        src_path = str(self.project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Import the preprocessing module properly
        from preprocessing.SnipPdfToPng import process_pdf_folder, find_poppler_path
        
        # Try to find poppler path
        poppler_path = find_poppler_path()
        if poppler_path is None:
            print("Native poppler not found, will try WSL if available")
        
        # Run the snipping process
        process_pdf_folder(
            input_folder=diagrams_folder,
            output_folder=images_folder,  # Output images to images_folder
            snippet_size=snippet_size,
            overlap=overlap,
            poppler_path=poppler_path
        )
    
    def _detect_on_snippets(self, images_folder):
        """Run YOLO11 detection on all image snippets"""
        images_folder = Path(images_folder)
        
        # Find all PNG files (excluding metadata)
        image_files = [f for f in images_folder.glob("*.png") if not f.name.endswith("_metadata.json")]
        
        if not image_files:
            print("No image snippets found for detection")
            return {}
        
        print(f"Found {len(image_files)} image snippets to process")
        
        all_detections = {}
        
        for image_file in image_files:
            try:
                # Run detection on this snippet
                detections, _ = predict_image(
                    self.model, 
                    image_file, 
                    self.confidence_threshold, 
                    save_results=False
                )
                
                all_detections[image_file.name] = {
                    "image_path": str(image_file),
                    "detections": detections,
                    "detection_count": len(detections)
                }
                
                print(f"Processed {image_file.name}: {len(detections)} detections")
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
                all_detections[image_file.name] = {
                    "image_path": str(image_file),
                    "error": str(e)
                }
        
        # Save detection results
        results_file = images_folder / "all_detections.json"
        with open(results_file, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        print(f"Detection results saved to: {results_file}")
        return all_detections
    
    def _reconstruct_with_detections(self, images_folder, detdiagrams_folder, detection_results, diagrams_folder):
        """Reconstruct PDFs with detection overlays and coordinate mapping"""
        images_folder = Path(images_folder)
        
        # Find all metadata files
        metadata_files = list(images_folder.glob("*_metadata.json"))
        metadata_files = [f for f in metadata_files if f.name != "all_detections.json"]
        
        if not metadata_files:
            print("No metadata files found for reconstruction")
            return
        
        for metadata_file in metadata_files:
            try:
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                pdf_name = metadata["original_pdf"]
                print(f"Processing {pdf_name}...")
                
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
                
                print(f"{pdf_name} completed: {len(output_files)} files generated")
                
            except Exception as e:
                print(f"Error reconstructing {metadata_file.name}: {e}")
    
    def _find_original_pdf(self, base_name, diagrams_folder):
        """Find the original PDF file"""
        diagrams_folder = Path(diagrams_folder)
        
        # Try exact match first
        exact_match = diagrams_folder / f"{base_name}.pdf"
        if exact_match.exists():
            return exact_match
        
        # Try pattern matching
        matches = list(diagrams_folder.glob(f"{base_name}*.pdf"))
        if matches:
            return matches[0]
        
        return None

def main():
    parser = argparse.ArgumentParser(description='Complete PLC Detection Pipeline')
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
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = PLCDetectionPipeline(
            model_path=args.model,
            confidence_threshold=args.conf
        )
        
        # Run complete pipeline
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
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
