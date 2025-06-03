"""
Complete PLC Detection Pipeline Runner
Runs the entire pipeline: Training → Detection → Reconstruction
"""

import sys
import time
import json
import argparse
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.detection.yolo11_train import train_yolo11, validate_dataset
from src.detection.detect_pipeline import PLCDetectionPipeline
from src.config import get_config

class CompletePipelineRunner:
    def __init__(self, epochs=10, confidence_threshold=0.25, snippet_size=(1500, 1200), overlap=500):
        """
        Initialize the complete pipeline runner
        
        Args:
            epochs: Number of training epochs
            confidence_threshold: Detection confidence threshold
            snippet_size: Size of image snippets for PDF processing
            overlap: Overlap between snippets
        """
        self.epochs = epochs
        self.confidence_threshold = confidence_threshold
        self.snippet_size = snippet_size
        self.overlap = overlap
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Get configuration
        self.config = get_config()
        
        # Set up paths using config
        dataset_path = self.config.get_dataset_path()
        self.test_folder = dataset_path.parent / "test"  # Assuming test data is alongside dataset
        self.diagrams_folder = self.test_folder / "diagrams"
        self.images_folder = self.test_folder / "images"
        self.detdiagrams_folder = self.test_folder / "detdiagrams"
        
        # Results tracking
        self.results = {
            "training": {},
            "detection": {},
            "reconstruction": {},
            "summary": {}
        }
    
    def run_complete_pipeline(self):
        """
        Run the complete pipeline from training to final output
        """
        print("Starting Complete PLC Detection Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Validate setup
            print("Step 1: Validating pipeline setup...")
            if not self._validate_setup():
                raise Exception("Pipeline setup validation failed")
            
            # Step 2: Train YOLO11 model
            print(f"\nStep 2: Training YOLO11 model ({self.epochs} epochs)...")
            trained_model_path = self._run_training()
            
            # Step 3: Run detection pipeline
            print(f"\nStep 3: Running detection pipeline...")
            detection_results = self._run_detection_pipeline(trained_model_path)
            
            # Step 4: Generate summary report
            print(f"\nStep 4: Generating summary report...")
            self._generate_summary_report()
            
            total_time = time.time() - start_time
            
            print(f"\nComplete pipeline finished successfully!")
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Results saved to: {self.detdiagrams_folder}")
            
            return True
            
        except Exception as e:
            print(f"\nPipeline failed: {e}")
            return False
    
    def _validate_setup(self):
        """Validate that all required components are available"""
        
        # Check folders
        required_folders = [self.diagrams_folder, self.images_folder, self.detdiagrams_folder]
        for folder in required_folders:
            if not folder.exists():
                print(f"Error: Required folder missing: {folder}")
                return False
        
        # Check for PDFs
        pdf_files = list(self.diagrams_folder.glob("*.pdf"))
        if not pdf_files:
            print(f"Error: No PDF files found in {self.diagrams_folder}")
            return False
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        # Check model files using config
        model_file = self.config.get_model_path('yolo11m.pt', 'pretrained')
        if not model_file.exists():
            print(f"Error: YOLO11m model not found: {model_file}")
            return False
        
        # Check configuration using config
        config_file = self.config.data_yaml_path
        if not config_file.exists():
            print(f"Error: Configuration file not found: {config_file}")
            return False
        
        # Validate dataset for training
        if not validate_dataset():
            print("Error: Dataset validation failed")
            return False
        
        print("Pipeline setup validation completed successfully")
        return True
    
    def _run_training(self):
        """Run YOLO11 training with specified epochs"""
        
        print(f"Training YOLO11 model with {self.epochs} epochs...")
        
        # Import and modify training function to use custom epochs
        import importlib.util
        from ultralytics import YOLO
        
        # Get paths using config
        model_path = self.config.get_model_path('yolo11m.pt', 'pretrained')
        config_path = self.config.data_yaml_path
        
        print(f"Loading YOLO11m model from: {model_path}")
        print(f"Using dataset config: {config_path}")
        
        # Load model
        model = YOLO(str(model_path))
        
        # Train with custom epochs
        training_start = time.time()
        
        results = model.train(
            data=str(config_path),
            epochs=self.epochs,
            imgsz=640,
            batch=16,
            name="plc_symbol_detector_yolo11m_pipeline",
            project=str(self.config.get_run_path('train')),
            save=True,
            save_period=max(1, self.epochs // 5),  # Save checkpoints
            patience=max(10, self.epochs // 2),    # Early stopping
            device='auto',
            workers=8,
            verbose=True
        )
        
        training_time = time.time() - training_start
        
        # Find the trained model using config
        runs_dir = self.config.get_run_path('train')
        latest_run = max([d for d in runs_dir.iterdir() if d.is_dir() and "plc_symbol_detector" in d.name], 
                        key=lambda x: x.stat().st_mtime)
        trained_model_path = latest_run / "weights" / "best.pt"
        
        # Store training results
        self.results["training"] = {
            "epochs": self.epochs,
            "training_time": training_time,
            "model_path": str(trained_model_path),
            "run_directory": str(latest_run),
            "final_metrics": str(results) if results else "No metrics available"
        }
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best model saved to: {trained_model_path}")
        
        return trained_model_path
    
    def _run_detection_pipeline(self, model_path):
        """Run the complete detection pipeline using the trained model"""
        
        print(f"Running detection pipeline with model: {model_path}")
        
        # Initialize pipeline with trained model
        pipeline = PLCDetectionPipeline(
            model_path=str(model_path),
            confidence_threshold=self.confidence_threshold
        )
        
        # Run pipeline
        detection_start = time.time()
        
        result_folder = pipeline.process_pdf_folder(
            diagrams_folder=self.diagrams_folder,
            output_folder=self.test_folder,
            snippet_size=self.snippet_size,
            overlap=self.overlap
        )
        
        detection_time = time.time() - detection_start
        
        # Collect detection statistics
        detection_stats = self._collect_detection_statistics()
        
        # Store detection results
        self.results["detection"] = {
            "detection_time": detection_time,
            "confidence_threshold": self.confidence_threshold,
            "snippet_size": self.snippet_size,
            "overlap": self.overlap,
            "statistics": detection_stats
        }
        
        print(f"Detection pipeline completed in {detection_time:.2f} seconds")
        
        return result_folder
    
    def _collect_detection_statistics(self):
        """Collect statistics from all detection results"""
        
        stats = {
            "total_pdfs_processed": 0,
            "total_detections": 0,
            "pdfs_with_detections": 0,
            "detection_files": []
        }
        
        # Find all detection result files
        detection_files = list(self.detdiagrams_folder.glob("*_detections.json"))
        
        for detection_file in detection_files:
            try:
                with open(detection_file, 'r') as f:
                    detection_data = json.load(f)
                
                pdf_name = detection_data["original_pdf"]
                pdf_detections = 0
                
                for page in detection_data["pages"]:
                    pdf_detections += len(page["detections"])
                
                stats["detection_files"].append({
                    "pdf_name": pdf_name,
                    "detections": pdf_detections,
                    "pages": len(detection_data["pages"])
                })
                
                stats["total_pdfs_processed"] += 1
                stats["total_detections"] += pdf_detections
                
                if pdf_detections > 0:
                    stats["pdfs_with_detections"] += 1
                    
            except Exception as e:
                print(f"Warning: Could not process {detection_file}: {e}")
        
        return stats
    
    def _generate_summary_report(self):
        """Generate a comprehensive summary report"""
        
        # Create summary
        summary = {
            "pipeline_configuration": {
                "training_epochs": self.epochs,
                "confidence_threshold": self.confidence_threshold,
                "snippet_size": self.snippet_size,
                "overlap": self.overlap
            },
            "training_results": self.results["training"],
            "detection_results": self.results["detection"],
            "output_files": {
                "detected_pdfs": len(list(self.detdiagrams_folder.glob("*_detected.pdf"))),
                "detection_jsons": len(list(self.detdiagrams_folder.glob("*_detections.json"))),
                "coordinate_files": len(list(self.detdiagrams_folder.glob("*_coordinates.txt"))),
                "statistics_files": len(list(self.detdiagrams_folder.glob("*_statistics.json")))
            }
        }
        
        # Calculate success metrics
        detection_stats = self.results["detection"]["statistics"]
        if detection_stats["total_pdfs_processed"] > 0:
            success_rate = (detection_stats["pdfs_with_detections"] / detection_stats["total_pdfs_processed"]) * 100
            avg_detections = detection_stats["total_detections"] / detection_stats["total_pdfs_processed"]
        else:
            success_rate = 0
            avg_detections = 0
        
        summary["performance_metrics"] = {
            "success_rate_percent": success_rate,
            "average_detections_per_pdf": avg_detections,
            "total_processing_time": (self.results["training"]["training_time"] + 
                                    self.results["detection"]["detection_time"])
        }
        
        # Save summary report
        summary_file = self.detdiagrams_folder / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\nPipeline Summary:")
        print("-" * 40)
        print(f"Training time: {self.results['training']['training_time']:.2f}s")
        print(f"Detection time: {self.results['detection']['detection_time']:.2f}s")
        print(f"PDFs processed: {detection_stats['total_pdfs_processed']}")
        print(f"Total detections: {detection_stats['total_detections']}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Avg detections/PDF: {avg_detections:.1f}")
        print(f"Summary saved to: {summary_file}")
        
        self.results["summary"] = summary

def main():
    parser = argparse.ArgumentParser(description='Run Complete PLC Detection Pipeline')
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
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline runner
        runner = CompletePipelineRunner(
            epochs=args.epochs,
            confidence_threshold=args.conf,
            snippet_size=tuple(args.snippet_size),
            overlap=args.overlap
        )
        
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
                        runner.results["training"] = {"model_path": str(best_model), "epochs": "skipped"}
                        runner._run_detection_pipeline(best_model)
                        runner._generate_summary_report()
                        return 0
            
            print("No existing trained model found, running full pipeline...")
        
        # Run complete pipeline
        success = runner.run_complete_pipeline()
        
        if success:
            print("\nPipeline completed successfully!")
            print("Next steps:")
            print("1. Review detection results in data/dataset/test/detdiagrams/")
            print("2. Check pipeline_summary.json for detailed metrics")
            print("3. Proceed to next stage of your processing pipeline")
            return 0
        else:
            print("\nPipeline failed!")
            return 1
            
    except Exception as e:
        print(f"Pipeline error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
