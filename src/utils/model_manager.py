"""
Model Manager for PLC Diagram Processor
Handles YOLO model downloads and management
"""

import os
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import time


class ModelManager:
    """Manages YOLO model downloads and verification"""
    
    def __init__(self, config: Dict):
        """
        Initialize model manager
        
        Args:
            config: Configuration dictionary from download_config.yaml
        """
        self.config = config
        
        # Set up paths relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        self.data_root = project_root.parent / 'plc-data'
        self.models_dir = self.data_root / 'models' / 'pretrained'
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # YOLO model download URLs (Ultralytics official)
        self.model_urls = {
            # YOLO11 models
            "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
            "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
            "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
            "yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
            "yolo11x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
            
            # YOLO10 models (if needed)
            "yolo10n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt",
            "yolo10s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt",
            "yolo10m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt",
            "yolo10l.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt",
            "yolo10x.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt",
        }
        
        print(f"Model manager initialized:")
        print(f"  Models directory: {self.models_dir}")
        print(f"  Available models: {len(self.model_urls)}")
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models for download"""
        available = {}
        
        for model_name in self.model_urls.keys():
            if model_name.startswith('yolo11'):
                version = 'yolo11'
            elif model_name.startswith('yolo10'):
                version = 'yolo10'
            else:
                version = 'other'
            
            if version not in available:
                available[version] = []
            available[version].append(model_name)
        
        return available
    
    def list_downloaded_models(self) -> List[str]:
        """List all downloaded models"""
        if not self.models_dir.exists():
            return []
        
        models = []
        for model_file in self.models_dir.glob("*.pt"):
            models.append(model_file.name)
        
        return sorted(models)
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded"""
        model_path = self.models_dir / model_name
        return model_path.exists() and model_path.stat().st_size > 0
    
    def download_model(self, model_name: str, force_redownload: bool = False) -> bool:
        """
        Download a YOLO model
        
        Args:
            model_name: Name of the model to download (e.g., 'yolo11m.pt')
            force_redownload: Whether to redownload if model already exists
            
        Returns:
            bool: True if download successful, False otherwise
        """
        if model_name not in self.model_urls:
            print(f"Error: Model '{model_name}' not available")
            print(f"Available models: {list(self.model_urls.keys())}")
            return False
        
        model_path = self.models_dir / model_name
        
        # Check if already downloaded
        if self.is_model_downloaded(model_name) and not force_redownload:
            print(f"Model '{model_name}' already downloaded at {model_path}")
            return True
        
        url = self.model_urls[model_name]
        print(f"Downloading {model_name} from {url}")
        
        try:
            # Download with progress
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Show progress
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\r  Progress: {progress:.1f}% ({downloaded_size // 1024 // 1024} MB / {total_size // 1024 // 1024} MB)", end='', flush=True)
            
            print(f"\nDownload completed: {model_path}")
            
            # Verify download
            if self.verify_model(model_path):
                print(f"Model verification successful")
                return True
            else:
                print(f"Model verification failed")
                model_path.unlink()  # Remove corrupted file
                return False
                
        except requests.RequestException as e:
            print(f"Download failed: {e}")
            if model_path.exists():
                model_path.unlink()  # Remove partial file
            return False
        except Exception as e:
            print(f"Unexpected error during download: {e}")
            if model_path.exists():
                model_path.unlink()  # Remove partial file
            return False
    
    def verify_model(self, model_path: Path) -> bool:
        """
        Verify that a downloaded model is valid
        
        Args:
            model_path: Path to the model file
            
        Returns:
            bool: True if model is valid, False otherwise
        """
        if not model_path.exists():
            return False
        
        # Check file size (YOLO models should be at least 1MB)
        file_size = model_path.stat().st_size
        if file_size < 1024 * 1024:  # Less than 1MB
            print(f"Warning: Model file seems too small ({file_size} bytes)")
            return False
        
        # For trusted YOLO models from official Ultralytics GitHub releases,
        # we use a simplified validation approach to avoid PyTorch 2.6+ compatibility issues
        try:
            import torch
            
            # Try to load just the checkpoint metadata without full deserialization
            # This avoids class compatibility issues while still validating the file format
            try:
                # Load with weights_only=True first (safest, works if no custom classes)
                checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=True)
                print(f"Model validation successful (weights_only): {model_path.name}")
                return True
            except Exception as e:
                error_msg = str(e)
                
                # If weights_only fails due to custom classes (expected for YOLO models)
                if "weights_only" in error_msg or "WeightsUnpickler" in error_msg or "C3k2" in error_msg:
                    print(f"Model contains custom classes (expected for YOLO), using basic validation...")
                    
                    # For trusted YOLO models, perform basic file validation without full deserialization
                    try:
                        # Just check if it's a valid PyTorch file by reading the header
                        with open(model_path, 'rb') as f:
                            # PyTorch files start with a specific magic number
                            magic = f.read(8)
                            
                        # Check for PyTorch pickle format magic numbers
                        if magic.startswith(b'PK') or magic.startswith(b'\x80\x02') or magic.startswith(b'\x80\x03'):
                            print(f"Model validation successful (basic format check): {model_path.name}")
                            return True
                        else:
                            print(f"Model validation failed: Invalid file format")
                            return False
                            
                    except Exception as e2:
                        print(f"Model validation failed (basic check): {e2}")
                        return False
                else:
                    # Re-raise unexpected errors
                    print(f"Model validation failed: {e}")
                    return False
                    
        except Exception as e:
            print(f"Model validation failed: {e}")
            return False
    
    def interactive_model_selection(self) -> List[str]:
        """
        Interactive CLI for model selection
        
        Returns:
            List[str]: List of selected model names
        """
        available_models = self.list_available_models()
        downloaded_models = self.list_downloaded_models()
        
        print("\nAvailable YOLO Models:")
        print("=" * 50)
        
        model_list = []
        index = 1
        
        for version, models in available_models.items():
            print(f"\n{version.upper()} Models:")
            for model in models:
                status = "Downloaded" if model in downloaded_models else "Not downloaded"
                print(f"  {index}. {model} ({status})")
                model_list.append(model)
                index += 1
        
        print(f"\nDefault model: {self.config['models']['default_model']}")
        print("\nOptions:")
        print("  - Enter numbers (e.g., '1,3,5' or '1-3')")
        print("  - Enter 'default' for default model only")
        print("  - Enter 'all' for all models")
        print("  - Enter 'none' to skip")
        
        while True:
            choice = input("\nSelect models to download: ").strip().lower()
            
            if choice == 'none':
                return []
            elif choice == 'default':
                return [self.config['models']['default_model']]
            elif choice == 'all':
                return model_list
            else:
                try:
                    selected_models = []
                    
                    # Parse selection
                    for part in choice.split(','):
                        part = part.strip()
                        if '-' in part:
                            # Range selection (e.g., '1-3')
                            start, end = map(int, part.split('-'))
                            for i in range(start, end + 1):
                                if 1 <= i <= len(model_list):
                                    selected_models.append(model_list[i - 1])
                        else:
                            # Single selection
                            i = int(part)
                            if 1 <= i <= len(model_list):
                                selected_models.append(model_list[i - 1])
                    
                    if selected_models:
                        return list(set(selected_models))  # Remove duplicates
                    else:
                        print("No valid selections made. Please try again.")
                        
                except ValueError:
                    print("Invalid input. Please use numbers, ranges, or keywords.")
    
    def download_multiple_models(self, model_names: List[str], force_redownload: bool = False) -> Dict[str, bool]:
        """
        Download multiple models
        
        Args:
            model_names: List of model names to download
            force_redownload: Whether to redownload existing models
            
        Returns:
            Dict[str, bool]: Results for each model (True = success, False = failure)
        """
        results = {}
        
        print(f"\nDownloading {len(model_names)} models...")
        print("=" * 50)
        
        for i, model_name in enumerate(model_names, 1):
            print(f"\n[{i}/{len(model_names)}] Processing {model_name}...")
            results[model_name] = self.download_model(model_name, force_redownload)
        
        # Summary
        print(f"\nDownload Summary:")
        print("=" * 30)
        successful = sum(1 for success in results.values() if success)
        print(f"Successful: {successful}/{len(model_names)}")
        
        for model_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {model_name}: {status}")
        
        return results
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get information about a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict with model information or None if not found
        """
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            return None
        
        info = {
            "name": model_name,
            "path": str(model_path),
            "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
            "downloaded": True,
            "url": self.model_urls.get(model_name, "Unknown")
        }
        
        # Use simplified validation approach to avoid PyTorch 2.6+ compatibility issues
        try:
            import torch
            
            # Try basic file format validation without full model loading
            try:
                # Try weights_only=True first (safest)
                checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=True)
                info["valid"] = True
            except Exception as e:
                error_msg = str(e)
                
                # If weights_only fails due to custom classes (expected for YOLO models)
                if "weights_only" in error_msg or "WeightsUnpickler" in error_msg or "C3k2" in error_msg:
                    # For trusted YOLO models, use basic file format validation
                    try:
                        with open(model_path, 'rb') as f:
                            magic = f.read(8)
                        # Check for PyTorch pickle format magic numbers
                        info["valid"] = magic.startswith(b'PK') or magic.startswith(b'\x80\x02') or magic.startswith(b'\x80\x03')
                    except Exception:
                        info["valid"] = False
                else:
                    info["valid"] = False
                    
        except Exception as e:
            info["valid"] = False
        
        return info
    
    def cleanup_models(self, keep_models: List[str] = None) -> int:
        """
        Clean up downloaded models, optionally keeping specified ones
        
        Args:
            keep_models: List of model names to keep (None = keep all)
            
        Returns:
            int: Number of models removed
        """
        if keep_models is None:
            keep_models = []
        
        removed_count = 0
        
        for model_file in self.models_dir.glob("*.pt"):
            if model_file.name not in keep_models:
                try:
                    model_file.unlink()
                    print(f"Removed: {model_file.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"Failed to remove {model_file.name}: {e}")
        
        return removed_count
