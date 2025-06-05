"""
Dataset Manager for PLC Diagram Processor
Handles dataset activation with symlinks and fallback to copying
"""

import os
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class DatasetManager:
    """Manages dataset activation and configuration for PLC training pipeline"""
    
    def __init__(self, config: Dict):
        """
        Initialize dataset manager
        
        Args:
            config: Configuration dictionary from download_config.yaml
        """
        self.config = config
        
        # Set up paths relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        self.data_root = project_root.parent / 'plc-data'
        self.datasets_dir = self.data_root / 'datasets'
        self.downloaded_dir = self.datasets_dir / 'downloaded'
        
        # Ensure directories exist
        self.downloaded_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Dataset manager initialized:")
        print(f"  Data root: {self.data_root}")
        print(f"  Datasets dir: {self.datasets_dir}")
        print(f"  Downloaded dir: {self.downloaded_dir}")
    
    def list_downloaded_datasets(self) -> List[str]:
        """List all downloaded datasets"""
        if not self.downloaded_dir.exists():
            return []
        
        datasets = []
        for item in self.downloaded_dir.iterdir():
            if item.is_dir():
                # Validate it's a proper YOLO dataset
                if self._validate_dataset_structure(item):
                    datasets.append(item.name)
                else:
                    print(f"Warning: {item.name} doesn't have valid YOLO structure")
        
        return sorted(datasets)
    
    def get_active_dataset(self) -> Optional[str]:
        """Get currently active dataset name"""
        yaml_path = self.datasets_dir / "plc_symbols.yaml"
        
        if not yaml_path.exists():
            return None
        
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            metadata = config.get('_metadata', {})
            return metadata.get('active_dataset')
        except Exception as e:
            print(f"Error reading plc_symbols.yaml: {e}")
            return None
    
    def activate_dataset(self, dataset_name: str, method: str = "auto") -> str:
        """
        Activate a dataset for training with symlinks + fallback to copy
        
        Args:
            dataset_name: Name of the dataset to activate
            method: "auto", "symlink", or "copy"
            
        Returns:
            str: Method used ("symlink" or "copy")
            
        Raises:
            Exception: If activation fails
        """
        dataset_path = self.downloaded_dir / dataset_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found in {self.downloaded_dir}")
        
        if not self._validate_dataset_structure(dataset_path):
            raise ValueError(f"Dataset '{dataset_name}' doesn't have valid YOLO structure")
        
        print(f"Activating dataset: {dataset_name}")
        
        # Try symlinks first (unless explicitly told to copy)
        if method == "auto" or method == "symlink":
            try:
                success = self._create_symlinks(dataset_name)
                if success:
                    print(f"Dataset '{dataset_name}' activated using symlinks")
                    self._update_plc_symbols_yaml(dataset_name, "symlink")
                    return "symlink"
            except (OSError, NotImplementedError, PermissionError) as e:
                print(f"Symlinks failed: {e}")
                if method == "symlink":
                    raise  # If user specifically requested symlinks, don't fallback
        
        # Fallback to copying
        print("Falling back to file copying...")
        success = self._copy_dataset(dataset_name)
        if success:
            print(f"Dataset '{dataset_name}' activated using file copy")
            self._update_plc_symbols_yaml(dataset_name, "copy")
            return "copy"
        else:
            raise Exception(f"Failed to activate dataset '{dataset_name}'")
    
    def _validate_dataset_structure(self, dataset_path: Path) -> bool:
        """Validate that dataset has proper YOLO structure"""
        required_dirs = ["train", "valid"]  # test is optional
        required_subdirs = ["images", "labels"]
        
        for split_dir in required_dirs:
            split_path = dataset_path / split_dir
            if not split_path.exists():
                return False
            
            for subdir in required_subdirs:
                if not (split_path / subdir).exists():
                    return False
        
        return True
    
    def _cleanup_existing_links(self):
        """Remove existing symlinks or directories for train/valid/test"""
        for split in ["train", "valid", "test"]:
            target_path = self.datasets_dir / split
            
            if target_path.exists() or target_path.is_symlink():
                if target_path.is_symlink():
                    target_path.unlink()
                    print(f"  Removed symlink: {target_path}")
                elif target_path.is_dir():
                    shutil.rmtree(target_path)
                    print(f"  Removed directory: {target_path}")
    
    def _create_symlinks(self, dataset_name: str) -> bool:
        """Create symlinks for train/valid/test directories"""
        source_dir = self.downloaded_dir / dataset_name
        
        # Remove existing symlinks/directories
        self._cleanup_existing_links()
        
        # Create new symlinks
        created_links = []
        for split in ["train", "valid", "test"]:
            source_path = source_dir / split
            target_path = self.datasets_dir / split
            
            if source_path.exists():
                try:
                    # Create symlink
                    target_path.symlink_to(source_path.resolve(), target_is_directory=True)
                    created_links.append(split)
                    print(f"  Created symlink: {target_path} -> {source_path}")
                except (OSError, NotImplementedError) as e:
                    # Clean up any partial symlinks
                    for created_split in created_links:
                        created_path = self.datasets_dir / created_split
                        if created_path.is_symlink():
                            created_path.unlink()
                    raise e
            else:
                if split in ["train", "valid"]:  # Required directories
                    raise FileNotFoundError(f"Required directory '{split}' not found in dataset")
                else:
                    print(f"  Optional directory '{split}' not found, skipping")
        
        return True
    
    def _copy_dataset(self, dataset_name: str) -> bool:
        """Copy dataset files as fallback"""
        source_dir = self.downloaded_dir / dataset_name
        
        # Remove existing directories
        self._cleanup_existing_links()
        
        # Copy directories
        for split in ["train", "valid", "test"]:
            source_path = source_dir / split
            target_path = self.datasets_dir / split
            
            if source_path.exists():
                try:
                    shutil.copytree(source_path, target_path)
                    print(f"  Copied: {source_path} -> {target_path}")
                except Exception as e:
                    print(f"  Error copying {split}: {e}")
                    return False
            else:
                if split in ["train", "valid"]:  # Required directories
                    print(f"  Error: Required directory '{split}' not found in dataset")
                    return False
                else:
                    print(f"  Optional directory '{split}' not found, skipping")
        
        return True
    
    def _update_plc_symbols_yaml(self, dataset_name: str, activation_method: str):
        """Update plc_symbols.yaml with new dataset info"""
        yaml_path = self.datasets_dir / "plc_symbols.yaml"
        
        # Load dataset info from downloaded dataset
        source_yaml = self.downloaded_dir / dataset_name / "data.yaml"
        if source_yaml.exists():
            try:
                with open(source_yaml, 'r') as f:
                    source_config = yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Could not read source data.yaml: {e}")
                source_config = {"names": {0: "PLC"}, "nc": 1}  # Default fallback
        else:
            print(f"Warning: No data.yaml found in dataset, using defaults")
            source_config = {"names": {0: "PLC"}, "nc": 1}  # Default fallback
        
        # Create updated config
        updated_config = {
            "path": str(self.datasets_dir),
            "train": "train/images",
            "val": "valid/images", 
            "test": "test/images",
            "names": source_config.get("names", {0: "PLC"}),
            "nc": source_config.get("nc", 1),
            # Metadata for tracking
            "_metadata": {
                "active_dataset": dataset_name,
                "activation_method": activation_method,
                "last_updated": datetime.now().isoformat(),
                "source_dataset": str(source_yaml) if source_yaml.exists() else None,
                "dataset_classes": source_config.get("nc", 1),
                "class_names": list(source_config.get("names", {0: "PLC"}).values())
            }
        }
        
        # Save updated config
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)
            
            print(f"Updated {yaml_path} for dataset '{dataset_name}'")
            print(f"   Classes: {updated_config['nc']}")
            print(f"   Names: {list(updated_config['names'].values())}")
            
        except Exception as e:
            print(f"Error updating plc_symbols.yaml: {e}")
            raise
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get information about a specific dataset"""
        dataset_path = self.downloaded_dir / dataset_name
        
        if not dataset_path.exists():
            return None
        
        info = {
            "name": dataset_name,
            "path": str(dataset_path),
            "valid_structure": self._validate_dataset_structure(dataset_path),
            "splits": {}
        }
        
        # Count files in each split
        for split in ["train", "valid", "test"]:
            split_path = dataset_path / split
            if split_path.exists():
                images_path = split_path / "images"
                labels_path = split_path / "labels"
                
                image_count = len(list(images_path.glob("*"))) if images_path.exists() else 0
                label_count = len(list(labels_path.glob("*"))) if labels_path.exists() else 0
                
                info["splits"][split] = {
                    "images": image_count,
                    "labels": label_count
                }
        
        # Try to get class information
        data_yaml = dataset_path / "data.yaml"
        if data_yaml.exists():
            try:
                with open(data_yaml, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                info["classes"] = yaml_config.get("nc", 0)
                info["class_names"] = list(yaml_config.get("names", {}).values())
            except Exception:
                info["classes"] = 0
                info["class_names"] = []
        
        return info
    
    def deactivate_dataset(self):
        """Deactivate current dataset (remove symlinks/copies)"""
        print("Deactivating current dataset...")
        self._cleanup_existing_links()
        
        # Remove plc_symbols.yaml or reset it
        yaml_path = self.datasets_dir / "plc_symbols.yaml"
        if yaml_path.exists():
            # Create a minimal default config
            default_config = {
                "path": str(self.datasets_dir),
                "train": "train/images",
                "val": "valid/images", 
                "test": "test/images",
                "names": {0: "PLC"},
                "nc": 1,
                "_metadata": {
                    "active_dataset": None,
                    "last_updated": datetime.now().isoformat(),
                    "status": "no_active_dataset"
                }
            }
            
            with open(yaml_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
            
            print("Reset plc_symbols.yaml to default state")
        
        print("Dataset deactivated")
