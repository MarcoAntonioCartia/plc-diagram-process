"""Lightweight Pipeline Runner - No heavy imports

This runner uses DetectionManager to coordinate detection tasks without
importing ultralytics or torch in the main process.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import get_config
from src.detection.detection_manager import DetectionManager


class LightweightPipelineRunner:
    """Pipeline runner that delegates heavy work to subprocesses/workers."""
    
    def __init__(self, epochs: int = 10, confidence_threshold: float = 0.25,
                 snippet_size=(1500, 1200), overlap: int = 500,
                 model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize lightweight runner.
        
        Args:
            epochs: Number of training epochs
            confidence_threshold: Detection confidence threshold
            snippet_size: Size of image snippets for PDF processing
            overlap: Overlap between snippets
            model_name: YOLO model to use (None for auto-detection)
            device: Device to use for training ('auto', 'cpu', '0', '1', etc.)
        """
        self.epochs = epochs
        self.confidence_threshold = confidence_threshold
        self.snippet_size = snippet_size
        self.overlap = overlap
        self.model_name = model_name
        self.device = device
        
        # Initialize config and paths
        self.config = get_config()
        data_root = Path(self.config.config['data_root'])
        self.diagrams_folder = data_root / "raw" / "pdfs"
        self.images_folder = data_root / "processed" / "images"
        self.detdiagrams_folder = data_root / "processed" / "detdiagrams"
        
        # Create output directories
        self.images_folder.mkdir(parents=True, exist_ok=True)
        self.detdiagrams_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize detection manager
        self.detection_manager = DetectionManager(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device
        )
        
        # Results tracking
        self.results = {
            "training": {},
            "detection": {},
            "reconstruction": {},
            "summary": {}
        }
    
    def validate_setup(self) -> bool:
        """Validate pipeline setup."""
        print("Validating pipeline setup...")
        
        success, error_msg = self.detection_manager.validate_setup()
        if not success:
            print(f"Validation failed: {error_msg}")
            return False
        
        print("Pipeline setup validation completed successfully")
        return True
    
    def run_training(self) -> Optional[Path]:
        """Run model training."""
        print(f"Training YOLO model with {self.epochs} epochs...")
        
        start_time = time.time()
        
        result = self.detection_manager.train_model(
            epochs=self.epochs,
            batch_size=16,
            project_name="plc_symbol_detector_lightweight"
        )
        
        training_time = time.time() - start_time
        
        if result["status"] == "success":
            model_path = result.get("model_path")
            if model_path:
                print(f"Training completed in {training_time:.2f} seconds")
                print(f"Best model saved to: {model_path}")
                
                self.results["training"] = {
                    "epochs": self.epochs,
                    "training_time": training_time,
                    "model_path": model_path,
                    "status": "success"
                }
                
                return Path(model_path)
            else:
                print("Training succeeded but model path not found in output")
                return None
        else:
            print(f"Training failed: {result.get('error', 'Unknown error')}")
            self.results["training"] = {
                "status": "failed",
                "error": result.get("error"),
                "training_time": training_time
            }
            return None
    
    def run_detection(self, model_path: Optional[Path] = None) -> bool:
        """Run detection on all PDFs."""
        print("Running detection pipeline...")
        
        if not model_path and self.detection_manager.resolved_model_path:
            model_path = self.detection_manager.resolved_model_path
        
        if not model_path:
            print("No model path available for detection")
            return False
        
        start_time = time.time()
        
        results = self.detection_manager.process_pdf_folder(
            pdf_folder=self.diagrams_folder,
            output_folder=self.detdiagrams_folder,
            snippet_size=self.snippet_size,
            overlap=self.overlap,
            model_path=model_path
        )
        
        detection_time = time.time() - start_time
        
        self.results["detection"] = {
            "detection_time": detection_time,
            "processed": results["processed"],
            "successful": results["successful"],
            "failed": results["failed"],
            "confidence_threshold": self.confidence_threshold,
            "snippet_size": self.snippet_size,
            "overlap": self.overlap
        }
        
        print(f"Detection completed in {detection_time:.2f} seconds")
        print(f"Processed: {results['processed']}, Successful: {results['successful']}, Failed: {results['failed']}")
        
        return results["failed"] == 0
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete pipeline."""
        print("Starting Lightweight PLC Detection Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Validate setup
            if not self.validate_setup():
                raise Exception("Pipeline setup validation failed")
            
            # Step 2: Train model
            trained_model_path = self.run_training()
            if not trained_model_path:
                raise Exception("Training failed")
            
            # Step 3: Run detection
            if not self.run_detection(trained_model_path):
                raise Exception("Detection failed")
            
            # Step 4: Generate summary
            self.generate_summary()
            
            total_time = time.time() - start_time
            
            print(f"\nPipeline completed successfully!")
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Results saved to: {self.detdiagrams_folder}")
            
            return True
            
        except Exception as e:
            print(f"\nPipeline failed: {e}")
            return False
    
    def generate_summary(self):
        """Generate summary report."""
        summary = {
            "pipeline_configuration": {
                "training_epochs": self.epochs,
                "confidence_threshold": self.confidence_threshold,
                "snippet_size": self.snippet_size,
                "overlap": self.overlap,
                "model_name": self.model_name,
                "device": self.device
            },
            "training_results": self.results.get("training", {}),
            "detection_results": self.results.get("detection", {}),
            "performance_metrics": {
                "total_processing_time": (
                    self.results.get("training", {}).get("training_time", 0) +
                    self.results.get("detection", {}).get("detection_time", 0)
                )
            }
        }
        
        # Save summary
        summary_file = self.detdiagrams_folder / "lightweight_pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
        self.results["summary"] = summary 