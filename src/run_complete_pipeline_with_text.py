"""
Complete PLC Pipeline with Text Extraction
Runs the entire pipeline: Training → Detection → Text Extraction
"""

import sys
import time
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.detection.run_complete_pipeline import CompletePipelineRunner
from src.ocr.text_extraction_pipeline import TextExtractionPipeline
from src.utils.detection_text_extraction_pdf_creator import DetectionPDFCreator
from src.config import get_config

class CompleteTextPipelineRunner(CompletePipelineRunner):
    """Extended pipeline runner that includes text extraction"""
    
    def __init__(self, epochs=10, confidence_threshold=0.25, snippet_size=(1500, 1200), 
                 overlap=500, model_name=None, device=None, ocr_confidence=0.7, ocr_lang="en",
                 pdf_confidence_threshold=0.8, create_enhanced_pdf=False, enhanced_pdf_version='short', detection_folder=None):
        """
        Initialize the complete pipeline runner with text extraction
        
        Args:
            epochs: Number of training epochs
            confidence_threshold: Detection confidence threshold
            snippet_size: Size of image snippets for PDF processing
            overlap: Overlap between snippets
            model_name: YOLO model to use (None for auto-detection)
            device: Device to use for training ('auto', 'cpu', '0', '1', etc.)
            ocr_confidence: OCR confidence threshold
            ocr_lang: OCR language
            pdf_confidence_threshold: Confidence threshold for PDF enhancement
            create_enhanced_pdf: Whether to create enhanced PDFs
            enhanced_pdf_version: 'short' (1 page) or 'long' (4 pages) for troubleshooting
            detection_folder: Custom detection folder path (None for auto-detection)
        """
        super().__init__(epochs, confidence_threshold, snippet_size, overlap, model_name, device)
        
        self.ocr_confidence = ocr_confidence
        self.ocr_lang = ocr_lang
        self.pdf_confidence_threshold = pdf_confidence_threshold
        self.create_enhanced_pdf = create_enhanced_pdf
        self.enhanced_pdf_version = enhanced_pdf_version
        self.custom_detection_folder = Path(detection_folder) if detection_folder else None
        
        # Add text extraction results tracking
        self.results["text_extraction"] = {}
        self.results["pdf_enhancement"] = {}
    
    def _get_effective_detection_folder(self):
        """Get the effective detection folder (custom or default)"""
        if self.custom_detection_folder:
            if self.custom_detection_folder.exists():
                print(f"Using custom detection folder: {self.custom_detection_folder}")
                return self.custom_detection_folder
            else:
                print(f"Warning: Custom detection folder not found: {self.custom_detection_folder}")
                print(f"Falling back to default: {self.detdiagrams_folder}")
                return self.detdiagrams_folder
        else:
            return self.detdiagrams_folder
    
    def _filter_detections_by_confidence(self, detection_file: Path, min_confidence: float) -> dict:
        """Filter detection JSON file by confidence threshold"""
        try:
            with open(detection_file, 'r') as f:
                detection_data = json.load(f)
            
            original_count = 0
            filtered_count = 0
            
            # Filter detections in each page
            for page_data in detection_data.get("pages", []):
                original_detections = page_data.get("detections", [])
                original_count += len(original_detections)
                
                # Filter by confidence
                filtered_detections = [
                    det for det in original_detections 
                    if det.get("confidence", 0.0) >= min_confidence
                ]
                
                page_data["detections"] = filtered_detections
                filtered_count += len(filtered_detections)
            
            print(f"  Confidence filtering: {original_count} → {filtered_count} detections (≥{min_confidence:.1f})")
            
            return detection_data
            
        except Exception as e:
            print(f"  Error filtering detections: {e}")
            # Return original data if filtering fails
            with open(detection_file, 'r') as f:
                return json.load(f)
    
    def run_complete_pipeline_with_text(self):
        """
        Run the complete pipeline including text extraction
        """
        print("Starting Complete PLC Pipeline with Text Extraction")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1-3: Run the standard detection pipeline
            print("Phase 1: Detection Pipeline")
            print("-" * 30)
            
            detection_success = self.run_complete_pipeline()
            
            if not detection_success:
                print("Detection pipeline failed, aborting text extraction")
                return False
            
            # Step 4: Run text extraction
            print(f"\nPhase 2: Text Extraction Pipeline")
            print("-" * 30)
            text_results = self._run_text_extraction()
            
            # Step 5: Generate combined summary report
            print(f"\nPhase 3: Generating Combined Summary")
            print("-" * 30)
            self._generate_combined_summary_report()
            
            total_time = time.time() - start_time
            
            print(f"\nComplete pipeline with text extraction finished successfully!")
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Results saved to: {self.detdiagrams_folder}")
            
            return True
            
        except Exception as e:
            print(f"\nPipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_text_extraction(self):
        """Run text extraction on detection results with custom folder and confidence filtering"""
        
        print(f"Running text extraction with OCR confidence: {self.ocr_confidence}")
        print(f"Detection confidence threshold: {self.pdf_confidence_threshold}")
        
        # Get effective detection folder (custom or default)
        detection_folder = self._get_effective_detection_folder()
        
        # Initialize text extraction pipeline
        text_pipeline = TextExtractionPipeline(
            confidence_threshold=self.ocr_confidence,
            ocr_lang=self.ocr_lang
        )
        
        # Set up paths
        pdf_folder = self.diagrams_folder
        text_output_folder = detection_folder.parent / "text_extraction"
        
        # Create a temporary folder for filtered detection files
        filtered_detection_folder = detection_folder.parent / "filtered_detections_temp"
        filtered_detection_folder.mkdir(exist_ok=True)
        
        text_start = time.time()
        
        try:
            # Filter detection files by confidence threshold
            detection_files = list(detection_folder.glob("*_detections.json"))
            print(f"Processing {len(detection_files)} detection files...")
            
            for detection_file in detection_files:
                print(f"Processing {detection_file.name}...")
                
                # Filter detections by confidence
                filtered_data = self._filter_detections_by_confidence(
                    detection_file, self.pdf_confidence_threshold
                )
                
                # Save filtered data to temporary folder
                filtered_file = filtered_detection_folder / detection_file.name
                with open(filtered_file, 'w') as f:
                    json.dump(filtered_data, f, indent=2)
            
            # Run text extraction on filtered detection files
            text_summary = text_pipeline.process_detection_folder(
                filtered_detection_folder, pdf_folder, text_output_folder
            )
            
            text_time = time.time() - text_start
            
            # Store text extraction results
            self.results["text_extraction"] = {
                "extraction_time": text_time,
                "ocr_confidence": self.ocr_confidence,
                "ocr_language": self.ocr_lang,
                "detection_confidence_threshold": self.pdf_confidence_threshold,
                "custom_detection_folder": str(detection_folder) if self.custom_detection_folder else None,
                "processed_files": text_summary["processed_files"],
                "total_text_regions": text_summary["total_text_regions"],
                "output_folder": str(text_output_folder),
                "summary": text_summary
            }
            
            print(f"Text extraction completed in {text_time:.2f} seconds")
            print(f"Processed {text_summary['processed_files']} files")
            print(f"Extracted {text_summary['total_text_regions']} text regions")
            
        finally:
            # Clean up temporary filtered detection files
            try:
                import shutil
                shutil.rmtree(filtered_detection_folder)
                print(f"Cleaned up temporary folder: {filtered_detection_folder}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary folder: {e}")
        
        return text_summary
    
    def _run_enhanced_pdf_creation(self):
        """Create enhanced PDFs with detection boxes and text extraction using PNG-first approach"""
        
        if not self.create_enhanced_pdf:
            return
        
        print(f"Creating enhanced PDFs using PNG-first approach (version: {self.enhanced_pdf_version})...")
        
        # Initialize your new PDF creator
        pdf_creator = DetectionPDFCreator(font_size=10)
        
        # Get effective detection folder (custom or default)
        detection_folder = self._get_effective_detection_folder()
        
        # Set up paths
        text_extraction_folder = detection_folder.parent / "text_extraction"
        enhanced_pdf_folder = detection_folder.parent / "enhanced_pdfs"
        enhanced_pdf_folder.mkdir(parents=True, exist_ok=True)
        
        pdf_start = time.time()
        
        # Find PNG images created during detection pipeline
        png_files = list(detection_folder.glob("*_page_*_detected.png"))
        
        if not png_files:
            print("Warning: No PNG files found from detection pipeline")
            return {"processed_files": 0, "error": "No PNG files found"}
        
        print(f"Found {len(png_files)} PNG files to process")
        
        # Group PNG files by PDF name
        pdf_groups = {}
        for png_file in png_files:
            # Extract PDF name from PNG filename: {pdf_name}_page_{page_num}_detected.png
            pdf_name = png_file.name.split('_page_')[0]
            if pdf_name not in pdf_groups:
                pdf_groups[pdf_name] = []
            pdf_groups[pdf_name].append(png_file)
        
        processed_count = 0
        enhancement_results = []
        
        # Process each PDF
        for pdf_name, png_list in pdf_groups.items():
            print(f"Processing {pdf_name}...")
            
            # Find corresponding JSON files
            detection_file = detection_folder / f"{pdf_name}_detections.json"
            text_file = text_extraction_folder / f"{pdf_name}_text_extraction.json"
            
            if not detection_file.exists():
                print(f"  Warning: Detection file not found: {detection_file}")
                continue
            
            if not text_file.exists():
                print(f"  Warning: Text file not found: {text_file}")
                continue
            
            # Use the first PNG (typically one per PDF)
            image_file = png_list[0]
            
            # Create enhanced PDF
            output_file = enhanced_pdf_folder / f"{pdf_name}_enhanced.pdf"
            
            try:
                # Use configured version (default 'short', can be changed to 'long' for troubleshooting)
                result_pdf = pdf_creator.create_enhanced_pdf(
                    image_file=image_file,
                    detections_file=detection_file,
                    text_file=text_file,
                    output_file=output_file,
                    version=self.enhanced_pdf_version
                )
                
                enhancement_results.append({
                    "pdf_name": pdf_name,
                    "image_file": str(image_file),
                    "detection_file": str(detection_file),
                    "text_file": str(text_file),
                    "enhanced_pdf": str(result_pdf),
                    "version": self.enhanced_pdf_version,
                    "status": "success"
                })
                
                processed_count += 1
                print(f"  ✓ Created: {output_file.name} ({self.enhanced_pdf_version} version)")
                
            except Exception as e:
                print(f"  ✗ Failed to create {output_file.name}: {e}")
                enhancement_results.append({
                    "pdf_name": pdf_name,
                    "status": "failed",
                    "error": str(e)
                })
                continue
        
        pdf_time = time.time() - pdf_start
        
        # Store PDF enhancement results
        self.results["pdf_enhancement"] = {
            "enhancement_time": pdf_time,
            "method": "png_first",
            "version": self.enhanced_pdf_version,
            "processed_files": processed_count,
            "total_files": len(pdf_groups),
            "success_rate": (processed_count / len(pdf_groups) * 100) if pdf_groups else 0,
            "output_folder": str(enhanced_pdf_folder),
            "results": enhancement_results
        }
        
        print(f"Enhanced PDF creation completed in {pdf_time:.2f} seconds")
        print(f"Created {processed_count}/{len(pdf_groups)} enhanced PDFs")
        print(f"Success rate: {self.results['pdf_enhancement']['success_rate']:.1f}%")
        print(f"Enhanced PDFs saved to: {enhanced_pdf_folder}")
        
        return self.results["pdf_enhancement"]
    
    def run_text_extraction_only(self):
        """Run only text extraction and PDF enhancement (skip detection)"""
        print("Running Text Extraction Pipeline (Detection Skipped)")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Check if detection results exist
            if not self.detdiagrams_folder.exists():
                print(f"Error: Detection folder not found: {self.detdiagrams_folder}")
                print("Please run detection first or use a different mode.")
                return False
            
            detection_files = list(self.detdiagrams_folder.glob("*_detections.json"))
            if not detection_files:
                print(f"Error: No detection files found in {self.detdiagrams_folder}")
                return False
            
            print(f"Found {len(detection_files)} detection files")
            
            # Initialize results for text-only mode
            self.results["training"] = {"training_time": 0, "epochs": "skipped"}
            self.results["detection"] = {"detection_time": 0, "status": "skipped"}
            
            # Step 1: Run text extraction
            print(f"\nPhase 1: Text Extraction Pipeline")
            print("-" * 30)
            text_results = self._run_text_extraction()
            
            # Step 2: Create enhanced PDFs if requested
            if self.create_enhanced_pdf:
                print(f"\nPhase 2: Enhanced PDF Creation")
                print("-" * 30)
                self._run_enhanced_pdf_creation()
            
            # Step 3: Generate summary report
            print(f"\nPhase 3: Generating Summary")
            print("-" * 30)
            self._generate_text_only_summary_report()
            
            total_time = time.time() - start_time
            
            print(f"\nText extraction pipeline finished successfully!")
            print(f"Total execution time: {total_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"\nText extraction pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_text_only_summary_report(self):
        """Generate summary report for text-only mode"""
        
        # Create summary for text-only mode
        summary = {
            "pipeline_mode": "text_extraction_only",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text_extraction_results": self.results["text_extraction"],
            "performance_metrics": {
                "text_extraction_time": self.results["text_extraction"]["extraction_time"],
                "total_processing_time": self.results["text_extraction"]["extraction_time"]
            }
        }
        
        # Add PDF enhancement results if available
        if "pdf_enhancement" in self.results and self.results["pdf_enhancement"]:
            summary["pdf_enhancement_results"] = self.results["pdf_enhancement"]
            summary["performance_metrics"]["pdf_enhancement_time"] = self.results["pdf_enhancement"]["enhancement_time"]
            summary["performance_metrics"]["total_processing_time"] += self.results["pdf_enhancement"]["enhancement_time"]
        
        # Add text extraction specific metrics
        if self.results["text_extraction"]["processed_files"] > 0:
            avg_texts_per_file = (
                self.results["text_extraction"]["total_text_regions"] / 
                self.results["text_extraction"]["processed_files"]
            )
            summary["performance_metrics"]["average_text_regions_per_file"] = avg_texts_per_file
        
        # Save summary report
        summary_file = self.detdiagrams_folder / "text_extraction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\nText Extraction Pipeline Summary:")
        print("-" * 50)
        print(f"Text extraction time: {self.results['text_extraction']['extraction_time']:.2f}s")
        if "pdf_enhancement" in self.results and self.results["pdf_enhancement"]:
            print(f"PDF enhancement time: {self.results['pdf_enhancement']['enhancement_time']:.2f}s")
        print(f"Total time: {summary['performance_metrics']['total_processing_time']:.2f}s")
        print(f"Text regions extracted: {self.results['text_extraction']['total_text_regions']}")
        
        if self.results["text_extraction"]["processed_files"] > 0:
            avg_texts = self.results["text_extraction"]["total_text_regions"] / self.results["text_extraction"]["processed_files"]
            print(f"Avg text regions/PDF: {avg_texts:.1f}")
        
        if "pdf_enhancement" in self.results and self.results["pdf_enhancement"]:
            print(f"Enhanced PDFs created: {self.results['pdf_enhancement']['processed_files']}")
        
        print(f"Summary saved to: {summary_file}")
        
        self.results["summary"] = summary
    
    def _generate_combined_summary_report(self):
        """Generate a comprehensive summary report including text extraction"""
        
        # Get the base summary from parent class
        super()._generate_summary_report()
        
        # Enhance with text extraction data
        combined_summary = self.results["summary"].copy()
        combined_summary["text_extraction_results"] = self.results["text_extraction"]
        
        # Calculate combined metrics
        total_processing_time = (
            self.results["training"]["training_time"] + 
            self.results["detection"]["detection_time"] +
            self.results["text_extraction"]["extraction_time"]
        )
        
        combined_summary["performance_metrics"]["total_processing_time_with_text"] = total_processing_time
        
        # Add text extraction specific metrics
        if self.results["text_extraction"]["processed_files"] > 0:
            avg_texts_per_file = (
                self.results["text_extraction"]["total_text_regions"] / 
                self.results["text_extraction"]["processed_files"]
            )
            combined_summary["performance_metrics"]["average_text_regions_per_file"] = avg_texts_per_file
        
        # Save enhanced summary report
        summary_file = self.detdiagrams_folder / "complete_pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        
        # Print enhanced summary
        print("\nComplete Pipeline Summary:")
        print("-" * 50)
        print(f"Training time: {self.results['training']['training_time']:.2f}s")
        print(f"Detection time: {self.results['detection']['detection_time']:.2f}s")
        print(f"Text extraction time: {self.results['text_extraction']['extraction_time']:.2f}s")
        print(f"Total time: {total_processing_time:.2f}s")
        print(f"PDFs processed: {self.results['detection']['statistics']['total_pdfs_processed']}")
        print(f"Symbols detected: {self.results['detection']['statistics']['total_detections']}")
        print(f"Text regions extracted: {self.results['text_extraction']['total_text_regions']}")
        
        if self.results["text_extraction"]["processed_files"] > 0:
            avg_texts = self.results["text_extraction"]["total_text_regions"] / self.results["text_extraction"]["processed_files"]
            print(f"Avg text regions/PDF: {avg_texts:.1f}")
        
        print(f"Complete summary saved to: {summary_file}")
        
        self.results["summary"] = combined_summary

def main():
    parser = argparse.ArgumentParser(description='Run Complete PLC Pipeline with Text Extraction')
    
    # Detection arguments (inherited from parent)
    parser.add_argument('--model', '-m', default=None,
                       help='YOLO model to use (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models and exit')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                       help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--snippet-size', nargs=2, type=int, default=[1500, 1200],
                       help='Snippet size as width height (default: 1500 1200)')
    parser.add_argument('--overlap', type=int, default=500,
                       help='Overlap between snippets (default: 500)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and use existing best model')
    parser.add_argument('--device', '-d', default='auto',
                       help='Device to use: auto, cpu, cuda, 0, 1, etc. (default: auto)')
    
    # Text extraction arguments
    parser.add_argument('--ocr-confidence', type=float, default=0.7,
                       help='OCR confidence threshold (default: 0.7)')
    parser.add_argument('--ocr-lang', type=str, default='en',
                       help='OCR language (default: en)')
    parser.add_argument('--skip-text-extraction', action='store_true',
                       help='Skip text extraction and only run detection pipeline')
    parser.add_argument('--skip-detection', action='store_true',
                       help='Skip detection and only run text extraction (requires existing detection results)')
    
    # Enhanced PDF arguments
    parser.add_argument('--create-enhanced-pdf', action='store_true',
                       help='Create enhanced PDFs with detection boxes and text extraction')
    parser.add_argument('--pdf-confidence-threshold', type=float, default=0.8,
                       help='Confidence threshold for showing detection boxes in enhanced PDFs (default: 0.8)')
    parser.add_argument('--enhanced-pdf-version', type=str, choices=['short', 'long'], 
                       default='short', help='Enhanced PDF version: short (1 page) or long (4 pages) for troubleshooting')
    parser.add_argument('--detection-folder', type=str, default=None,
                       help='Custom detection folder path (default: auto-detected based on pipeline)')
    
    args = parser.parse_args()
    
    # Handle list-models command (delegate to parent)
    if args.list_models:
        from src.detection.run_complete_pipeline import main as parent_main
        return parent_main()
    
    try:
        # Initialize pipeline runner
        runner = CompleteTextPipelineRunner(
            epochs=args.epochs,
            confidence_threshold=args.conf,
            snippet_size=tuple(args.snippet_size),
            overlap=args.overlap,
            model_name=args.model,
            device=args.device,
            ocr_confidence=args.ocr_confidence,
            ocr_lang=args.ocr_lang,
            pdf_confidence_threshold=args.pdf_confidence_threshold,
            create_enhanced_pdf=args.create_enhanced_pdf,
            enhanced_pdf_version=args.enhanced_pdf_version,
            detection_folder=args.detection_folder
        )
        
        # Handle skip-detection mode
        if args.skip_detection:
            print("Detection will be skipped, running text extraction only...")
            success = runner.run_text_extraction_only()
            
            if success:
                print("\nText extraction pipeline completed successfully!")
                print("Results:")
                print(f"- Text extraction results: {runner.detdiagrams_folder.parent / 'text_extraction'}")
                if args.create_enhanced_pdf:
                    print(f"- Enhanced PDFs: {runner.detdiagrams_folder.parent / 'enhanced_pdfs'}")
                print(f"- Summary: {runner.detdiagrams_folder / 'text_extraction_summary.json'}")
            return 0 if success else 1
        
        if args.skip_training:
            print("Skipping training, using existing model...")
            # Find existing best model using config
            runs_dir = runner.config.get_run_path('train')
            if runs_dir.exists():
                train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "plc_symbol_detector" in d.name]
                if train_dirs:
                    latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
                    best_model = latest_dir / "weights" / "best.pt"
                    if best_model.exists():
                        print(f"Using existing model: {best_model}")
                        runner.results["training"] = {"model_path": str(best_model), "epochs": "skipped", "training_time": 0}
                        
                        # Run detection pipeline
                        runner._run_detection_pipeline(best_model)
                        
                        # Run text extraction if not skipped
                        if not args.skip_text_extraction:
                            runner._run_text_extraction()
                        
                        # Create enhanced PDFs if requested
                        if args.create_enhanced_pdf:
                            runner._run_enhanced_pdf_creation()
                        
                        runner._generate_combined_summary_report()
                        return 0
            
            print("No existing trained model found, running full pipeline...")
        
        # Run complete pipeline
        if args.skip_text_extraction:
            print("Text extraction will be skipped")
            success = runner.run_complete_pipeline()
        else:
            success = runner.run_complete_pipeline_with_text()
            
            # Create enhanced PDFs if requested and pipeline succeeded
            if success and args.create_enhanced_pdf:
                print(f"\nPhase 4: Enhanced PDF Creation")
                print("-" * 30)
                runner._run_enhanced_pdf_creation()
        
        if success:
            print("\nPipeline completed successfully!")
            print("Next steps:")
            print("1. Review detection results in processed/detdiagrams/")
            if not args.skip_text_extraction:
                print("2. Review text extraction results in processed/text_extraction/")
                if args.create_enhanced_pdf:
                    print("3. Review enhanced PDFs in processed/enhanced_pdfs/")
                    print("4. Check complete_pipeline_summary.json for detailed metrics")
                    print("5. Proceed to data structuring stage (LayoutLM)")
                else:
                    print("3. Check complete_pipeline_summary.json for detailed metrics")
                    print("4. Proceed to data structuring stage (LayoutLM)")
            else:
                print("2. Run text extraction: python src/run_complete_pipeline_with_text.py --skip-detection")
                print("3. Proceed to data structuring stage")
            return 0
        else:
            print("\nPipeline failed!")
            return 1
            
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
