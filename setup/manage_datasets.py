#!/usr/bin/env python3
"""
Dataset Management Script for PLC Diagram Processor
Handles dataset downloads, activation, and management
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

try:
    import yaml
    from utils.dataset_manager import DatasetManager
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you have activated the virtual environment and installed dependencies")
    sys.exit(1)


def load_config():
    """Load download configuration"""
    config_path = project_root / 'setup' / 'config' / 'download_config.yaml'
    
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        print("Please run setup.py first to create the configuration")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def get_storage_manager(config):
    """Get the appropriate storage manager based on configuration"""
    storage_backend = config.get('storage_backend', 'network_drive')
    
    if storage_backend == 'network_drive':
        from utils.network_drive_manager import NetworkDriveManager
        return NetworkDriveManager(config), "Network Drive"
    else:  # Legacy OneDrive support
        from utils.onedrive_manager import OneDriveManager
        return OneDriveManager(config), "OneDrive"


def list_available_datasets(config):
    """List datasets available for download"""
    print("=== Available Datasets ===")
    
    try:
        storage_manager, backend_name = get_storage_manager(config)
        print(f"Using storage backend: {backend_name}")
        
        datasets = storage_manager.list_available_datasets()
        
        if not datasets:
            print(f"No datasets found on {backend_name}.")
            return []
        
        print(f"Found {len(datasets)} datasets:")
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset['name']}")
            print(f"     Date: {dataset.get('date', 'Unknown')}")
            size_info = dataset.get('size_mb')
            if size_info:
                print(f"     Size: {size_info:.1f} MB")
            else:
                print(f"     Size: {dataset.get('size', 'Unknown')}")
            print()
        
        return datasets
            
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return []


def list_downloaded_datasets(config):
    """List locally downloaded datasets"""
    print("=== Downloaded Datasets ===")
    
    try:
        dataset_manager = DatasetManager(config)
        datasets = dataset_manager.list_downloaded_datasets()
        active_dataset = dataset_manager.get_active_dataset()
        
        if not datasets:
            print("No datasets downloaded")
            return
        
        print(f"Found {len(datasets)} downloaded datasets:")
        for dataset in datasets:
            status = " (ACTIVE)" if dataset == active_dataset else ""
            print(f"  - {dataset}{status}")
            
            # Get dataset info
            info = dataset_manager.get_dataset_info(dataset)
            if info:
                print(f"    Classes: {info.get('classes', 0)}")
                print(f"    Valid structure: {info.get('valid_structure', False)}")
                for split, counts in info.get('splits', {}).items():
                    print(f"    {split}: {counts.get('images', 0)} images, {counts.get('labels', 0)} labels")
            print()
            
    except Exception as e:
        print(f"Error listing downloaded datasets: {e}")


def download_dataset(config, dataset_name=None, use_latest=False):
    """Download a specific dataset or latest"""
    print("=== Dataset Download ===")
    
    try:
        storage_manager, backend_name = get_storage_manager(config)
        dataset_manager = DatasetManager(config)
        
        print(f"Using storage backend: {backend_name}")
        
        if use_latest:
            print("Downloading latest dataset...")
            dataset_path = storage_manager.download_dataset(use_latest=True)
        elif dataset_name:
            print(f"Downloading dataset: {dataset_name}")
            dataset_path = storage_manager.download_dataset(dataset_name, use_latest=False)
        else:
            # Interactive selection
            datasets = storage_manager.list_available_datasets()
            if not datasets:
                print("No datasets available")
                return False
            
            print("Available datasets:")
            for i, dataset in enumerate(datasets, 1):
                print(f"  {i}. {dataset['name']} ({dataset.get('date', 'Unknown date')})")
            
            while True:
                try:
                    choice = input(f"\nSelect dataset (1-{len(datasets)}): ").strip()
                    choice_num = int(choice)
                    
                    if 1 <= choice_num <= len(datasets):
                        selected_dataset = datasets[choice_num - 1]
                        dataset_path = storage_manager.download_dataset(selected_dataset['name'], use_latest=False)
                        break
                    else:
                        print(f"Invalid choice. Please enter 1-{len(datasets)}")
                except ValueError:
                    print("Please enter a valid number")
                except KeyboardInterrupt:
                    print("\nDownload cancelled")
                    return False
        
        if dataset_path:
            print(f"Dataset downloaded successfully to: {dataset_path}")
            
            # Ask if user wants to activate it
            response = input("Activate this dataset? (y/n): ").strip().lower()
            if response == 'y':
                dataset_name = Path(dataset_path).name
                activation_method = dataset_manager.activate_dataset(dataset_name)
                print(f"Dataset activated using {activation_method}")
            
            return True
        else:
            print("Dataset download failed")
            return False
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def activate_dataset(config, dataset_name):
    """Activate a downloaded dataset"""
    print(f"=== Activating Dataset: {dataset_name} ===")
    
    try:
        dataset_manager = DatasetManager(config)
        
        # Check if dataset exists
        downloaded_datasets = dataset_manager.list_downloaded_datasets()
        if dataset_name not in downloaded_datasets:
            print(f"Error: Dataset '{dataset_name}' not found")
            print("Available datasets:")
            for dataset in downloaded_datasets:
                print(f"  - {dataset}")
            return False
        
        # Activate dataset
        activation_method = dataset_manager.activate_dataset(dataset_name)
        print(f"Dataset '{dataset_name}' activated using {activation_method}")
        return True
        
    except Exception as e:
        print(f"Error activating dataset: {e}")
        return False


def show_status(config):
    """Show current dataset status"""
    print("=== Dataset Status ===")
    
    try:
        dataset_manager = DatasetManager(config)
        
        # Show active dataset
        active_dataset = dataset_manager.get_active_dataset()
        if active_dataset:
            print(f"Active dataset: {active_dataset}")
            
            # Show dataset info
            info = dataset_manager.get_dataset_info(active_dataset)
            if info:
                print(f"Classes: {info.get('classes', 0)}")
                print(f"Class names: {info.get('class_names', [])}")
                print(f"Valid structure: {info.get('valid_structure', False)}")
                print("Dataset splits:")
                for split, counts in info.get('splits', {}).items():
                    print(f"  {split}: {counts.get('images', 0)} images, {counts.get('labels', 0)} labels")
        else:
            print("No active dataset")
        
        # Show all downloaded datasets
        print("\nDownloaded datasets:")
        downloaded_datasets = dataset_manager.list_downloaded_datasets()
        if downloaded_datasets:
            for dataset in downloaded_datasets:
                status = " (ACTIVE)" if dataset == active_dataset else ""
                print(f"  - {dataset}{status}")
        else:
            print("  No datasets downloaded")
            
    except Exception as e:
        print(f"Error getting status: {e}")


def deactivate_dataset(config):
    """Deactivate current dataset"""
    print("=== Deactivating Dataset ===")
    
    try:
        dataset_manager = DatasetManager(config)
        
        active_dataset = dataset_manager.get_active_dataset()
        if not active_dataset:
            print("No active dataset to deactivate")
            return True
        
        print(f"Deactivating dataset: {active_dataset}")
        dataset_manager.deactivate_dataset()
        print("Dataset deactivated successfully")
        return True
        
    except Exception as e:
        print(f"Error deactivating dataset: {e}")
        return False


def interactive_mode(config):
    """Interactive dataset management"""
    print("=== Interactive Dataset Management ===")
    
    # Get storage backend name for display
    storage_backend = config.get('storage_backend', 'network_drive')
    backend_name = "Network Drive" if storage_backend == 'network_drive' else "OneDrive"
    
    while True:
        print("\nOptions:")
        print(f"1. List available datasets ({backend_name})")
        print("2. List downloaded datasets")
        print("3. Download dataset")
        print("4. Activate dataset")
        print("5. Show status")
        print("6. Deactivate current dataset")
        print("7. Exit")
        
        try:
            choice = input("\nChoose option (1-7): ").strip()
            
            if choice == "1":
                list_available_datasets(config)
            elif choice == "2":
                list_downloaded_datasets(config)
            elif choice == "3":
                download_dataset(config)
            elif choice == "4":
                dataset_manager = DatasetManager(config)
                datasets = dataset_manager.list_downloaded_datasets()
                if not datasets:
                    print("No datasets available to activate")
                    continue
                
                print("Available datasets:")
                for i, dataset in enumerate(datasets, 1):
                    print(f"  {i}. {dataset}")
                
                try:
                    choice_num = int(input(f"Select dataset (1-{len(datasets)}): "))
                    if 1 <= choice_num <= len(datasets):
                        activate_dataset(config, datasets[choice_num - 1])
                    else:
                        print("Invalid choice")
                except ValueError:
                    print("Please enter a valid number")
                    
            elif choice == "5":
                show_status(config)
            elif choice == "6":
                deactivate_dataset(config)
            elif choice == "7":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='PLC Dataset Management')
    parser.add_argument('--list-available', action='store_true',
                       help='List datasets available for download')
    parser.add_argument('--list-downloaded', action='store_true',
                       help='List locally downloaded datasets')
    parser.add_argument('--download', type=str, metavar='DATASET_NAME',
                       help='Download specific dataset')
    parser.add_argument('--download-latest', action='store_true',
                       help='Download latest dataset')
    parser.add_argument('--activate', type=str, metavar='DATASET_NAME',
                       help='Activate a downloaded dataset')
    parser.add_argument('--status', action='store_true',
                       help='Show current dataset status')
    parser.add_argument('--deactivate', action='store_true',
                       help='Deactivate current dataset')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive dataset management')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Execute commands
    if args.list_available:
        list_available_datasets(config)
    elif args.list_downloaded:
        list_downloaded_datasets(config)
    elif args.download:
        download_dataset(config, args.download)
    elif args.download_latest:
        download_dataset(config, use_latest=True)
    elif args.activate:
        activate_dataset(config, args.activate)
    elif args.status:
        show_status(config)
    elif args.deactivate:
        deactivate_dataset(config)
    elif args.interactive:
        interactive_mode(config)
    else:
        # Default to interactive mode if no arguments
        interactive_mode(config)


if __name__ == "__main__":
    main()
