"""Detection Manager - Coordinates detection tasks without importing heavy dependencies.

This module provides a thin abstraction layer that can delegate detection work to:
1. A subprocess running in the same environment (single-env mode)
2. A worker in a dedicated detection_env (multi-env mode)
3. Direct function calls if dependencies are available (legacy mode)

The key design principle is that this module NEVER imports ultralytics, torch, or
any heavy detection dependencies directly.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.config import get_config
from src.utils import multi_env_active


class DetectionManager:
    """Manages detection operations without importing heavy dependencies."""
    
    def __init__(self, model_name: Optional[str] = None, 
                 confidence_threshold: float = 0.25,
                 device: Optional[str] = None):
        """Initialize detection manager.
        
        Args:
            model_name: YOLO model to use (None for auto-detection)
            confidence_threshold: Detection confidence threshold
            device: Device to use ('auto', 'cpu', '0', '1', etc.)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.config = get_config()
        
        # Resolve model path without importing YOLO
        self._resolve_model_info()
    
    def _resolve_model_info(self) -> None:
        """Resolve model information without importing YOLO."""
        try:
            model_path, model_name, model_type, was_fallback = self.config.get_model_path_with_fallback(self.model_name)
            self.resolved_model_path = model_path
            self.resolved_model_name = model_name
            self.resolved_model_type = model_type
            self.was_fallback = was_fallback
        except FileNotFoundError:
            # Model resolution failed - will be handled by validate()
            self.resolved_model_path = None
            self.resolved_model_name = None
            self.resolved_model_type = None
            self.was_fallback = False
    
    def validate_setup(self) -> Tuple[bool, Optional[str]]:
        """Validate that detection can run.
        
        Returns:
            (success, error_message)
        """
        # Check data directories
        data_root = Path(self.config.config['data_root'])
        pdf_folder = data_root / "raw" / "pdfs"
        
        if not pdf_folder.exists():
            return False, f"PDF folder not found: {pdf_folder}"
        
        pdf_files = list(pdf_folder.glob("*.pdf"))
        if not pdf_files:
            return False, f"No PDF files found in {pdf_folder}"
        
        # Check model availability
        if not self.resolved_model_path:
            available_models = []
            for model_type in ['pretrained', 'custom']:
                available_models.extend(self.config.discover_available_models(model_type))
            
            if available_models:
                model_list = ", ".join(available_models)
                return False, f"Model not found. Available models: {model_list}"
            else:
                return False, "No models found. Run: python setup/manage_models.py --interactive"
        
        # Check dataset configuration
        if not self.config.data_yaml_path.exists():
            return False, f"Dataset configuration not found: {self.config.data_yaml_path}"
        
        return True, None
    
    def train_model(self, epochs: int = 10, batch_size: int = 16,
                    project_name: str = "plc_symbol_detector") -> Dict[str, Any]:
        """Train a YOLO model using subprocess.
        
        Returns:
            Dictionary with training results
        """
        if multi_env_active():
            # In multi-env mode, this would be delegated to detection_env
            # For now, we'll use subprocess approach
            pass
        
        # Prepare training script call
        train_script = self.config.project_root / "src" / "detection" / "yolo11_train.py"
        
        cmd = [
            sys.executable,
            str(train_script),
            "--model", self.resolved_model_name,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--project-name", project_name,
        ]
        
        if self.device:
            cmd.extend(["--device", self.device])
        
        print(f"Starting training with command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse output to find model path
            output_lines = result.stdout.splitlines()
            model_path = None
            for line in output_lines:
                if "Best model saved to:" in line:
                    model_path = line.split("Best model saved to:")[-1].strip()
                    break
            
            return {
                "status": "success",
                "model_path": model_path,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
    
    def run_detection(self, pdf_path: Path, output_dir: Path,
                     snippet_size: Tuple[int, int] = (1500, 1200),
                     overlap: int = 500,
                     model_path: Optional[Path] = None) -> Dict[str, Any]:
        """Run detection on a single PDF.
        
        Returns:
            Dictionary with detection results
        """
        if not model_path:
            model_path = self.resolved_model_path
        
        # Prepare input for worker/subprocess
        input_data = {
            "model_path": str(model_path),
            "pdf_path": str(pdf_path),
            "output_dir": str(output_dir),
            "confidence_threshold": self.confidence_threshold,
            "snippet_size": list(snippet_size),
            "overlap": overlap
        }
        
        if multi_env_active():
            # Use worker pattern (would be handled by MultiEnvironmentManager)
            # For now, fall through to subprocess
            pass
        
        # Use subprocess to run detection
        detect_script = self.config.project_root / "src" / "detection" / "detect_pipeline_subprocess.py"
        
        # Write input to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name
        
        output_file = input_file.replace('.json', '_output.json')
        
        try:
            cmd = [
                sys.executable,
                str(detect_script),
                "--input", input_file,
                "--output", output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Read output
            with open(output_file, 'r') as f:
                output_data = json.load(f)
            
            return output_data
            
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
        finally:
            # Cleanup temp files
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)
    
    def process_pdf_folder(self, pdf_folder: Path, output_folder: Path,
                          snippet_size: Tuple[int, int] = (1500, 1200),
                          overlap: int = 500,
                          model_path: Optional[Path] = None) -> Dict[str, Any]:
        """Process all PDFs in a folder.
        
        Returns:
            Dictionary with processing results
        """
        pdf_files = list(pdf_folder.glob("*.pdf"))
        results = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "files": {}
        }
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            
            pdf_output_dir = output_folder / pdf_file.stem
            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            
            result = self.run_detection(
                pdf_path=pdf_file,
                output_dir=pdf_output_dir,
                snippet_size=snippet_size,
                overlap=overlap,
                model_path=model_path
            )
            
            results["processed"] += 1
            if result.get("status") == "success":
                results["successful"] += 1
            else:
                results["failed"] += 1
            
            results["files"][pdf_file.name] = result
        
        return results 