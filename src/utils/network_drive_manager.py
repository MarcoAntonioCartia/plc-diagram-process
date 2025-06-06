"""
Network Drive Manager for PLC Diagram Processor
Handles dataset retrieval from network storage locations
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re


class NetworkDriveManager:
    """Manages dataset downloads from network drive storage"""
    
    def __init__(self, config: Dict):
        """
        Initialize network drive manager
        
        Args:
            config: Configuration dictionary from download_config.yaml
        """
        self.config = config
        
        # Get network drive configuration
        network_config = config.get('network_drive', {})
        self.base_path = Path(network_config.get('base_path', ''))
        self.architecture = network_config.get('dataset', {}).get('architecture', 'YOLO')
        self.version = network_config.get('dataset', {}).get('version', 'v11')
        self.dataset_pattern = network_config.get('dataset', {}).get('dataset_pattern', 'plc_symbols_v11_*.zip')
        self.auto_select_latest = network_config.get('dataset', {}).get('auto_select_latest', True)
        
        # Set up local paths
        project_root = Path(__file__).resolve().parent.parent.parent
        self.data_root = project_root.parent / 'plc-data'
        self.datasets_dir = self.data_root / 'datasets'
        self.downloaded_dir = self.datasets_dir / 'downloaded'
        
        # Ensure directories exist
        self.downloaded_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Network Drive Manager initialized:")
        print(f"  Network path: {self.base_path}")
        print(f"  Architecture: {self.architecture}")
        print(f"  Version: {self.version}")
        print(f"  Pattern: {self.dataset_pattern}")
        print(f"  Local storage: {self.downloaded_dir}")
    
    def check_network_access(self) -> bool:
        """Check if network drive is accessible"""
        if not self.base_path:
            print("Error: Network drive path not configured")
            return False
        
        dataset_path = self.base_path / self.architecture / self.version
        
        if not dataset_path.exists():
            print(f"Error: Network path does not exist: {dataset_path}")
            return False
        
        if not os.access(dataset_path, os.R_OK):
            print(f"Error: No read access to network path: {dataset_path}")
            return False
        
        print(f"Network drive accessible: {dataset_path}")
        return True
    
    def list_available_datasets(self) -> List[Dict[str, str]]:
        """
        List available datasets on the network drive
        
        Returns:
            List of dataset info dictionaries
        """
        if not self.check_network_access():
            return []
        
        dataset_path = self.base_path / self.architecture / self.version
        datasets = []
        
        # Convert pattern to regex
        pattern_regex = self.dataset_pattern.replace('*', '(\\d{10})')
        pattern_regex = pattern_regex.replace('.', '\\.')
        
        try:
            for file_path in dataset_path.glob(self.dataset_pattern):
                if file_path.is_file() and file_path.suffix == '.zip':
                    # Parse filename to extract date and version
                    match = re.search(r'(\d{10})', file_path.name)
                    if match:
                        date_version = match.group(1)
                        date_str = date_version[:8]  # YYYYMMDD
                        version_str = date_version[8:]  # VV
                        
                        # Format date for display
                        try:
                            date_obj = datetime.strptime(date_str, '%Y%m%d')
                            formatted_date = date_obj.strftime('%Y-%m-%d')
                        except:
                            formatted_date = date_str
                        
                        datasets.append({
                            'name': file_path.name,
                            'path': str(file_path),
                            'date': formatted_date,
                            'version': version_str,
                            'date_version': date_version,
                            'size_mb': file_path.stat().st_size / (1024 * 1024)
                        })
            
            # Sort by date_version (newest first)
            datasets.sort(key=lambda x: x['date_version'], reverse=True)
            
            print(f"Found {len(datasets)} datasets on network drive")
            return datasets
            
        except Exception as e:
            print(f"Error listing datasets: {e}")
            return []
    
    def download_dataset(self, dataset_name: Optional[str] = None, use_latest: bool = False) -> Optional[Path]:
        """
        Download (copy) a dataset from network drive to local storage
        
        Args:
            dataset_name: Specific dataset filename to download
            use_latest: If True, download the latest dataset
            
        Returns:
            Path to the extracted dataset directory, or None if failed
        """
        # Get list of available datasets
        available_datasets = self.list_available_datasets()
        
        if not available_datasets:
            print("No datasets available on network drive")
            return None
        
        # Select dataset to download
        if use_latest or (dataset_name is None and self.auto_select_latest):
            selected_dataset = available_datasets[0]  # Already sorted by date
            print(f"Auto-selecting latest dataset: {selected_dataset['name']}")
        else:
            # Find specific dataset
            selected_dataset = None
            for dataset in available_datasets:
                if dataset['name'] == dataset_name:
                    selected_dataset = dataset
                    break
            
            if not selected_dataset:
                print(f"Dataset '{dataset_name}' not found on network drive")
                return None
        
        # Copy dataset from network to local
        source_path = Path(selected_dataset['path'])
        local_zip_path = self.downloaded_dir / selected_dataset['name']
        
        print(f"Copying dataset from network drive...")
        print(f"  Source: {source_path}")
        print(f"  Destination: {local_zip_path}")
        print(f"  Size: {selected_dataset['size_mb']:.1f} MB")
        
        try:
            # Copy file with progress indication
            shutil.copy2(source_path, local_zip_path)
            print("Dataset copied successfully")
            
            # Extract the dataset
            extracted_path = self._extract_dataset(local_zip_path)
            
            if extracted_path:
                # Remove the zip file after successful extraction
                local_zip_path.unlink()
                print(f"Removed temporary zip file: {local_zip_path}")
                
                return extracted_path
            else:
                # Keep zip file if extraction failed
                print("Extraction failed, keeping zip file for manual inspection")
                return None
                
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    def _extract_dataset(self, zip_path: Path) -> Optional[Path]:
        """
        Extract a dataset zip file
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            Path to extracted directory, or None if failed
        """
        # Create extraction directory name from zip filename
        # Remove .zip extension and use as directory name
        extract_dir_name = zip_path.stem
        extract_path = self.downloaded_dir / extract_dir_name
        
        # Check if already extracted
        if extract_path.exists():
            print(f"Dataset already extracted at: {extract_path}")
            return extract_path
        
        print(f"Extracting dataset to: {extract_path}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Create extraction directory
                extract_path.mkdir(exist_ok=True)
                
                # Extract all files
                zip_ref.extractall(extract_path)
                
                print(f"Successfully extracted {len(zip_ref.filelist)} files")
                
                # Verify YOLO structure
                if self._verify_yolo_structure(extract_path):
                    print("Dataset structure verified")
                    return extract_path
                else:
                    print("Warning: Dataset may not have proper YOLO structure")
                    return extract_path  # Return anyway, let user decide
                    
        except Exception as e:
            print(f"Error extracting dataset: {e}")
            # Clean up partial extraction
            if extract_path.exists():
                shutil.rmtree(extract_path)
            return None
    
    def _verify_yolo_structure(self, dataset_path: Path) -> bool:
        """Verify that extracted dataset has proper YOLO structure"""
        required_dirs = ["train", "valid"]  # test is optional
        required_subdirs = ["images", "labels"]
        
        for split_dir in required_dirs:
            split_path = dataset_path / split_dir
            if not split_path.exists():
                print(f"Missing required directory: {split_dir}")
                return False
            
            for subdir in required_subdirs:
                subdir_path = split_path / subdir
                if not subdir_path.exists():
                    print(f"Missing required subdirectory: {split_dir}/{subdir}")
                    return False
                
                # Check if directories have files
                file_count = len(list(subdir_path.glob("*")))
                print(f"  {split_dir}/{subdir}: {file_count} files")
        
        return True
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get information about a dataset on the network drive"""
        datasets = self.list_available_datasets()
        
        for dataset in datasets:
            if dataset['name'] == dataset_name:
                return dataset
        
        return None
    
    def cleanup_old_downloads(self, keep_count: int = 3):
        """
        Clean up old downloaded datasets, keeping only the most recent ones
        
        Args:
            keep_count: Number of recent datasets to keep
        """
        if not self.downloaded_dir.exists():
            return
        
        # Get all dataset directories
        dataset_dirs = []
        for item in self.downloaded_dir.iterdir():
            if item.is_dir():
                # Try to extract date from directory name
                match = re.search(r'(\d{10})', item.name)
                if match:
                    date_version = match.group(1)
                    dataset_dirs.append((item, date_version))
        
        # Sort by date (oldest first)
        dataset_dirs.sort(key=lambda x: x[1])
        
        # Remove old datasets
        if len(dataset_dirs) > keep_count:
            to_remove = dataset_dirs[:-keep_count]
            
            print(f"Cleaning up {len(to_remove)} old datasets...")
            for dir_path, date_version in to_remove:
                try:
                    shutil.rmtree(dir_path)
                    print(f"  Removed: {dir_path.name}")
                except Exception as e:
                    print(f"  Error removing {dir_path.name}: {e}")
