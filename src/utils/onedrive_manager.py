"""
OneDrive Manager for PLC Diagram Processor (LEGACY)
Handles dataset downloads from OneDrive/SharePoint

NOTE: This module is preserved for potential future cloud storage implementation.
Currently, the project uses NetworkDriveManager for dataset retrieval from network storage.
To reactivate OneDrive support, change storage_backend to "onedrive" in download_config.yaml.
"""

import os
import re
import zipfile
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import tempfile
import shutil
from datetime import datetime


class OneDriveManager:
    """Manages dataset downloads from OneDrive/SharePoint"""
    
    def __init__(self, config: Dict):
        """
        Initialize OneDrive manager
        
        Args:
            config: Configuration dictionary from download_config.yaml
        """
        self.config = config
        self.base_url = config['onedrive']['base_url']
        
        # Set up paths relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        self.data_root = project_root.parent / 'plc-data'
        self.datasets_dir = self.data_root / 'datasets'
        self.downloaded_dir = self.datasets_dir / 'downloaded'
        
        # Ensure directories exist
        self.downloaded_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"OneDrive manager initialized:")
        print(f"  Base URL: {self.base_url}")
        print(f"  Download directory: {self.downloaded_dir}")
    
    def parse_sharepoint_url(self) -> Optional[str]:
        """
        Parse SharePoint URL to get the direct access URL
        
        Returns:
            str: Direct access URL or None if parsing fails
        """
        try:
            # SharePoint URLs often need to be converted to direct access URLs
            # This is a simplified approach - may need adjustment based on actual URL structure
            
            if 'sharepoint.com' in self.base_url:
                # Try to convert to a direct download URL
                # This is a basic implementation and may need refinement
                return self.base_url
            
            return self.base_url
            
        except Exception as e:
            print(f"Error parsing SharePoint URL: {e}")
            return None
    
    def list_available_datasets(self) -> List[Dict[str, str]]:
        """
        List available datasets from SharePoint
        
        Returns:
            List[Dict]: List of available datasets with metadata
        """
        try:
            # Convert SharePoint sharing URL to a browsable format
            browse_url = self._convert_to_browse_url(self.base_url)
            if not browse_url:
                print("Error: Could not convert SharePoint URL to browsable format")
                return []
            
            print(f"Checking SharePoint folder: {browse_url}")
            
            # Try to get folder contents
            datasets = self._get_sharepoint_folder_contents(browse_url)
            
            if not datasets:
                print("No datasets found in SharePoint folder")
                return []
            
            print(f"Found {len(datasets)} datasets:")
            for dataset in datasets:
                print(f"  - {dataset['name']} ({dataset.get('date', 'Unknown date')}, {dataset.get('size', 'Unknown size')})")
            
            return datasets
            
        except Exception as e:
            print(f"Error listing datasets from SharePoint: {e}")
            print("Falling back to manual file specification...")
            
            # Fallback: Ask user to specify the file manually
            return self._manual_dataset_specification()
    
    def _convert_to_browse_url(self, sharing_url: str) -> Optional[str]:
        """
        Convert SharePoint sharing URL to a browsable URL
        
        Args:
            sharing_url: SharePoint sharing URL
            
        Returns:
            str: Browsable URL or None if conversion fails
        """
        try:
            # For now, return the original URL as SharePoint sharing URLs
            # can sometimes be accessed directly for public folders
            return sharing_url
        except Exception as e:
            print(f"Error converting URL: {e}")
            return None
    
    def _get_sharepoint_folder_contents(self, url: str) -> List[Dict[str, str]]:
        """
        Get contents of SharePoint folder
        
        Args:
            url: SharePoint folder URL
            
        Returns:
            List[Dict]: List of files in the folder
        """
        try:
            # Try to access the SharePoint folder
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 401:
                print("Authentication required for SharePoint access")
                return self._handle_authentication_required()
            elif response.status_code != 200:
                print(f"Failed to access SharePoint folder: HTTP {response.status_code}")
                print("SharePoint folder access failed. This is common with sharing URLs.")
                print("Will use manual dataset specification instead.")
                return []
            
            # Parse the response to extract file information
            # This is a simplified approach - SharePoint HTML parsing can be complex
            datasets = self._parse_sharepoint_response(response.text)
            
            if not datasets:
                print("No datasets found in HTML parsing.")
                print("SharePoint structure might be different than expected.")
                print("Will use manual dataset specification instead.")
                # Actually trigger the manual fallback
                return self._manual_dataset_specification()
            
            return datasets
            
        except requests.RequestException as e:
            print(f"Network error accessing SharePoint: {e}")
            print("This is common with SharePoint sharing URLs.")
            return []
        except Exception as e:
            print(f"Error getting folder contents: {e}")
            return []
    
    def _parse_sharepoint_response(self, html_content: str) -> List[Dict[str, str]]:
        """
        Parse SharePoint HTML response to extract file information
        
        Args:
            html_content: HTML content from SharePoint
            
        Returns:
            List[Dict]: List of datasets found
        """
        datasets = []
        pattern = self.config['onedrive']['dataset']['dataset_pattern'].replace('*', r'.*')
        
        try:
            # Look for file patterns in the HTML
            # This is a basic implementation - may need refinement
            import re
            
            # Pattern to match your new naming convention: plc_symbols_v11_YYYYMMDDVV.zip
            file_pattern = r'plc_symbols_v11_(\d{10})\.zip'
            
            matches = re.findall(file_pattern, html_content, re.IGNORECASE)
            
            for match in matches:
                timestamp = match  # YYYYMMDDVV format
                filename = f"plc_symbols_v11_{timestamp}.zip"
                
                # Extract date from timestamp (first 8 digits)
                date_str = timestamp[:8]  # YYYYMMDD
                try:
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                    formatted_date = date_obj.strftime('%Y-%m-%d')
                except:
                    formatted_date = "Unknown"
                
                dataset = {
                    'name': filename,
                    'date': formatted_date,
                    'timestamp': timestamp,
                    'size': 'Unknown',  # Size extraction would need more complex parsing
                    'url': self._get_direct_download_url(filename)
                }
                
                datasets.append(dataset)
            
            # Sort by timestamp (newest first)
            datasets.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return datasets
            
        except Exception as e:
            print(f"Error parsing SharePoint response: {e}")
            return []
    
    def _handle_authentication_required(self) -> List[Dict[str, str]]:
        """
        Handle case where SharePoint requires authentication
        
        Returns:
            List[Dict]: List of datasets (may be empty if auth fails)
        """
        print("\nSharePoint folder requires authentication.")
        print("Please provide your credentials:")
        
        try:
            username = input("Username (email): ").strip()
            import getpass
            password = getpass.getpass("Password: ")
            
            # Try to authenticate and get folder contents
            # This is a simplified approach - real SharePoint auth is more complex
            auth = (username, password)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(self.base_url, auth=auth, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return self._parse_sharepoint_response(response.text)
            else:
                print(f"Authentication failed: HTTP {response.status_code}")
                return []
                
        except KeyboardInterrupt:
            print("\nAuthentication cancelled by user")
            return []
        except Exception as e:
            print(f"Authentication error: {e}")
            return []
    
    def _manual_dataset_specification(self) -> List[Dict[str, str]]:
        """
        Fallback method to manually specify dataset
        
        Returns:
            List[Dict]: List with manually specified dataset
        """
        print("\nAutomatic dataset detection failed.")
        print("Please specify your dataset manually:")
        print("\nOption 1: Provide filename (if you know it)")
        print("Expected format: plc_symbols_v11_YYYYMMDDVV.zip")
        print("Example: plc_symbols_v11_2025060501.zip")
        print("\nOption 2: Provide direct file URL")
        print("Example: https://spieglteccloud-my.sharepoint.com/:u:/g/personal/...")
        
        try:
            user_input = input("\nEnter filename or direct URL: ").strip()
            
            if not user_input:
                return []
            
            # Check if input is a URL
            if user_input.startswith('http'):
                return self._handle_direct_url(user_input)
            else:
                # Treat as filename
                return self._handle_filename_input(user_input)
                
        except KeyboardInterrupt:
            print("\nManual specification cancelled")
            return []
        except Exception as e:
            print(f"Error in manual specification: {e}")
            return []
    
    def _handle_filename_input(self, filename: str) -> List[Dict[str, str]]:
        """Handle filename input"""
        # Validate filename format
        pattern = r'plc_symbols_v11_(\d{10})\.zip'
        match = re.match(pattern, filename)
        
        if not match:
            print("Invalid filename format. Expected: plc_symbols_v11_YYYYMMDDVV.zip")
            return []
        
        timestamp = match.group(1)
        date_str = timestamp[:8]
        
        try:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            formatted_date = date_obj.strftime('%Y-%m-%d')
        except:
            formatted_date = "Unknown"
        
        dataset = {
            'name': filename,
            'date': formatted_date,
            'timestamp': timestamp,
            'size': 'Unknown',
            'url': self._get_direct_download_url(filename)
        }
        
        return [dataset]
    
    def _handle_direct_url(self, url: str) -> List[Dict[str, str]]:
        """Handle direct URL input"""
        try:
            # Extract filename from URL or ask user
            print(f"Using direct URL: {url}")
            
            # Try to extract filename from URL
            filename = None
            if 'plc_symbols_v11_' in url:
                # Try to extract from URL
                match = re.search(r'(plc_symbols_v11_\d{10}\.zip)', url, re.IGNORECASE)
                if match:
                    filename = match.group(1)
            
            if not filename:
                print("Could not extract filename from URL.")
                filename = input("Please enter the filename (e.g., plc_symbols_v11_2025060501.zip): ").strip()
                
                if not filename:
                    return []
            
            # Validate filename format
            pattern = r'plc_symbols_v11_(\d{10})\.zip'
            match = re.match(pattern, filename, re.IGNORECASE)
            
            if not match:
                print("Invalid filename format. Expected: plc_symbols_v11_YYYYMMDDVV.zip")
                return []
            
            timestamp = match.group(1)
            date_str = timestamp[:8]
            
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                formatted_date = date_obj.strftime('%Y-%m-%d')
            except:
                formatted_date = "Unknown"
            
            dataset = {
                'name': filename,
                'date': formatted_date,
                'timestamp': timestamp,
                'size': 'Unknown',
                'url': url  # Use the direct URL provided
            }
            
            print(f"Dataset configured: {filename}")
            return [dataset]
            
        except Exception as e:
            print(f"Error handling direct URL: {e}")
            return []
    
    def _get_direct_download_url(self, filename: str) -> str:
        """
        Get direct download URL for a file
        
        Args:
            filename: Name of the file
            
        Returns:
            str: Direct download URL
        """
        # For SharePoint sharing URLs, we might need to construct the download URL differently
        # This is a simplified approach
        if 'sharepoint.com' in self.base_url:
            # Try to construct a direct download URL
            # This may need adjustment based on actual SharePoint structure
            return f"{self.base_url.split('?')[0]}/{filename}"
        else:
            return f"{self.base_url}/{filename}"
    
    def get_latest_dataset(self) -> Optional[Dict[str, str]]:
        """
        Get the latest dataset based on timestamp pattern
        
        Returns:
            Dict: Latest dataset info or None if none found
        """
        datasets = self.list_available_datasets()
        
        if not datasets:
            return None
        
        # Sort by timestamp (YYYYMMDDVV format)
        def extract_timestamp(dataset_name):
            # Extract timestamp from pattern like "plc_symbols_v11_YYYYMMDDVV.zip"
            match = re.search(r'plc_symbols_v11_(\d{10})\.zip', dataset_name)
            return match.group(1) if match else "0000000000"
        
        latest = max(datasets, key=lambda x: extract_timestamp(x['name']))
        return latest
    
    def download_dataset(self, dataset_name: Optional[str] = None, use_latest: bool = True) -> Optional[str]:
        """
        Download a dataset from OneDrive
        
        Args:
            dataset_name: Specific dataset name to download (None for latest)
            use_latest: Whether to download latest if dataset_name is None
            
        Returns:
            str: Path to extracted dataset directory or None if failed
        """
        # Determine which dataset to download
        if dataset_name is None and use_latest:
            latest_dataset = self.get_latest_dataset()
            if not latest_dataset:
                print("No datasets found")
                return None
            dataset_name = latest_dataset['name']
            download_url = latest_dataset['url']
        else:
            # Construct URL for specific dataset
            download_url = f"{self.base_url}/{dataset_name}"
        
        if not dataset_name:
            print("No dataset specified")
            return None
        
        print(f"Downloading dataset: {dataset_name}")
        print(f"From URL: {download_url}")
        
        # Check if already downloaded
        dataset_dir_name = dataset_name.replace('.zip', '')
        extracted_path = self.downloaded_dir / dataset_dir_name
        
        if extracted_path.exists():
            print(f"Dataset already exists at: {extracted_path}")
            response = input("Re-download? (y/n): ").strip().lower()
            if response != 'y':
                return str(extracted_path)
        
        try:
            # Download to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
                temp_path = Path(temp_file.name)
                
                print("Starting download...")
                response = requests.get(download_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Show progress
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\r  Progress: {progress:.1f}% ({downloaded_size // 1024 // 1024} MB / {total_size // 1024 // 1024} MB)", end='', flush=True)
                
                print(f"\nDownload completed: {temp_path}")
            
            # Extract and organize dataset
            extracted_path = self.extract_and_organize(temp_path, dataset_dir_name)
            
            # Clean up temporary file
            temp_path.unlink()
            
            if extracted_path:
                print(f"Dataset extracted to: {extracted_path}")
                return str(extracted_path)
            else:
                print("Failed to extract dataset")
                return None
                
        except requests.RequestException as e:
            print(f"Download failed: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during download: {e}")
            return None
    
    def extract_and_organize(self, zip_path: Path, dataset_name: str) -> Optional[Path]:
        """
        Extract dataset zip and organize in proper YOLO structure
        
        Args:
            zip_path: Path to the downloaded zip file
            dataset_name: Name for the extracted dataset directory
            
        Returns:
            Path: Path to extracted dataset directory or None if failed
        """
        extract_path = self.downloaded_dir / dataset_name
        
        try:
            # Remove existing directory if it exists
            if extract_path.exists():
                shutil.rmtree(extract_path)
            
            # Extract zip file
            print(f"Extracting {zip_path} to {extract_path}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Validate and organize structure
            if self._validate_and_organize_structure(extract_path):
                print(f"Dataset structure validated and organized")
                return extract_path
            else:
                print(f"Invalid dataset structure")
                return None
                
        except zipfile.BadZipFile:
            print(f"Error: Invalid zip file")
            return None
        except Exception as e:
            print(f"Error extracting dataset: {e}")
            return None
    
    def _validate_and_organize_structure(self, dataset_path: Path) -> bool:
        """
        Validate and organize dataset structure to match YOLO requirements
        
        Args:
            dataset_path: Path to the extracted dataset
            
        Returns:
            bool: True if structure is valid/organized, False otherwise
        """
        # Check if the dataset has the expected YOLO structure
        required_dirs = ["train", "valid"]  # test is optional
        required_subdirs = ["images", "labels"]
        
        # First, check if structure is already correct
        structure_valid = True
        for split_dir in required_dirs:
            split_path = dataset_path / split_dir
            if not split_path.exists():
                structure_valid = False
                break
            
            for subdir in required_subdirs:
                if not (split_path / subdir).exists():
                    structure_valid = False
                    break
        
        if structure_valid:
            print("Dataset structure is already valid")
            return True
        
        # If structure is not valid, try to organize it
        print("Attempting to organize dataset structure...")
        
        # Look for common alternative structures and reorganize
        # This is dataset-specific and may need adjustment
        
        # Check if there's a nested directory structure
        subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        if len(subdirs) == 1:
            # If there's only one subdirectory, the actual dataset might be inside it
            nested_path = subdirs[0]
            
            # Check if the nested directory has the correct structure
            nested_valid = True
            for split_dir in required_dirs:
                split_path = nested_path / split_dir
                if not split_path.exists():
                    nested_valid = False
                    break
                
                for subdir in required_subdirs:
                    if not (split_path / subdir).exists():
                        nested_valid = False
                        break
            
            if nested_valid:
                # Move contents up one level
                print(f"Moving contents from {nested_path} to {dataset_path}")
                
                temp_dir = dataset_path.parent / f"{dataset_path.name}_temp"
                nested_path.rename(temp_dir)
                
                for item in temp_dir.iterdir():
                    item.rename(dataset_path / item.name)
                
                temp_dir.rmdir()
                
                return True
        
        # Check for data.yaml file and validate it
        data_yaml = dataset_path / "data.yaml"
        if data_yaml.exists():
            try:
                import yaml
                with open(data_yaml, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                
                print(f"Found data.yaml with {yaml_config.get('nc', 0)} classes")
                
                # Validate that the paths in data.yaml exist
                for split in ['train', 'val', 'test']:
                    if split in yaml_config:
                        split_path = dataset_path / yaml_config[split]
                        if not split_path.exists():
                            print(f"Warning: Path {split_path} from data.yaml doesn't exist")
                
            except Exception as e:
                print(f"Warning: Could not read data.yaml: {e}")
        
        # Final validation
        final_valid = True
        for split_dir in required_dirs:
            split_path = dataset_path / split_dir
            if not split_path.exists():
                print(f"Error: Required directory {split_dir} not found")
                final_valid = False
            else:
                for subdir in required_subdirs:
                    subdir_path = split_path / subdir
                    if not subdir_path.exists():
                        print(f"Error: Required subdirectory {split_dir}/{subdir} not found")
                        final_valid = False
                    else:
                        # Count files
                        file_count = len(list(subdir_path.glob("*")))
                        print(f"  {split_dir}/{subdir}: {file_count} files")
        
        return final_valid
    
    def get_dataset_download_url(self, dataset_name: str) -> str:
        """
        Get direct download URL for a specific dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            str: Direct download URL
        """
        # This is a simplified implementation
        # In practice, you might need to use SharePoint API to get proper download URLs
        return f"{self.base_url}/{dataset_name}"
    
    def cleanup_downloads(self, keep_latest: int = 2) -> int:
        """
        Clean up old downloaded datasets, keeping only the latest N versions
        
        Args:
            keep_latest: Number of latest datasets to keep
            
        Returns:
            int: Number of datasets removed
        """
        if not self.downloaded_dir.exists():
            return 0
        
        # Get all dataset directories
        dataset_dirs = [d for d in self.downloaded_dir.iterdir() if d.is_dir()]
        
        if len(dataset_dirs) <= keep_latest:
            print(f"Only {len(dataset_dirs)} datasets found, nothing to clean up")
            return 0
        
        # Sort by modification time (newest first)
        dataset_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old datasets
        removed_count = 0
        for dataset_dir in dataset_dirs[keep_latest:]:
            try:
                shutil.rmtree(dataset_dir)
                print(f"Removed old dataset: {dataset_dir.name}")
                removed_count += 1
            except Exception as e:
                print(f"Failed to remove {dataset_dir.name}: {e}")
        
        return removed_count
