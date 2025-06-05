#!/usr/bin/env python3
"""
Model Management Script for PLC Diagram Processor
Handles YOLO model downloads and management
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

try:
    import yaml
    from utils.model_manager import ModelManager
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


def list_available_models(config):
    """List models available for download"""
    print("=== Available Models ===")
    
    try:
        model_manager = ModelManager(config)
        available_models = model_manager.list_available_models()
        
        if not available_models:
            print("No models available")
            return
        
        for version, models in available_models.items():
            print(f"\n{version.upper()} Models:")
            for model in models:
                print(f"  - {model}")
                
    except Exception as e:
        print(f"Error listing available models: {e}")


def list_downloaded_models(config):
    """List locally downloaded models"""
    print("=== Downloaded Models ===")
    
    try:
        model_manager = ModelManager(config)
        models = model_manager.list_downloaded_models()
        
        if not models:
            print("No models downloaded")
            return
        
        print(f"Found {len(models)} downloaded models:")
        for model in models:
            print(f"  - {model}")
            
            # Get model info
            info = model_manager.get_model_info(model)
            if info:
                print(f"    Size: {info.get('size_mb', 0)} MB")
                print(f"    Valid: {info.get('valid', False)}")
                print(f"    URL: {info.get('url', 'Unknown')}")
            print()
            
    except Exception as e:
        print(f"Error listing downloaded models: {e}")


def download_model(config, model_name=None, force_redownload=False):
    """Download a specific model"""
    print("=== Model Download ===")
    
    try:
        model_manager = ModelManager(config)
        
        if model_name:
            print(f"Downloading model: {model_name}")
            success = model_manager.download_model(model_name, force_redownload)
            
            if success:
                print(f"Model '{model_name}' downloaded successfully")
                return True
            else:
                print(f"Failed to download model '{model_name}'")
                return False
        else:
            # Interactive selection
            available_models = model_manager.list_available_models()
            if not available_models:
                print("No models available")
                return False
            
            # Flatten model list for selection
            model_list = []
            print("Available models:")
            index = 1
            
            for version, models in available_models.items():
                print(f"\n{version.upper()} Models:")
                for model in models:
                    downloaded = model_manager.is_model_downloaded(model)
                    status = " (Downloaded)" if downloaded else ""
                    print(f"  {index}. {model}{status}")
                    model_list.append(model)
                    index += 1
            
            while True:
                try:
                    choice = input(f"\nSelect model (1-{len(model_list)}): ").strip()
                    choice_num = int(choice)
                    
                    if 1 <= choice_num <= len(model_list):
                        selected_model = model_list[choice_num - 1]
                        
                        # Check if already downloaded
                        if model_manager.is_model_downloaded(selected_model) and not force_redownload:
                            response = input(f"Model '{selected_model}' already downloaded. Re-download? (y/n): ").strip().lower()
                            if response != 'y':
                                print("Download cancelled")
                                return True
                            force_redownload = True
                        
                        success = model_manager.download_model(selected_model, force_redownload)
                        
                        if success:
                            print(f"Model '{selected_model}' downloaded successfully")
                            return True
                        else:
                            print(f"Failed to download model '{selected_model}'")
                            return False
                    else:
                        print(f"Invalid choice. Please enter 1-{len(model_list)}")
                except ValueError:
                    print("Please enter a valid number")
                except KeyboardInterrupt:
                    print("\nDownload cancelled")
                    return False
                    
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


def download_multiple_models(config, model_names, force_redownload=False):
    """Download multiple models"""
    print("=== Multiple Model Download ===")
    
    try:
        model_manager = ModelManager(config)
        
        print(f"Downloading {len(model_names)} models...")
        results = model_manager.download_multiple_models(model_names, force_redownload)
        
        # Show results
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"\nDownload completed: {successful}/{total} successful")
        
        for model_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {model_name}: {status}")
        
        return successful == total
        
    except Exception as e:
        print(f"Error downloading models: {e}")
        return False


def get_model_info(config, model_name):
    """Get detailed information about a model"""
    print(f"=== Model Info: {model_name} ===")
    
    try:
        model_manager = ModelManager(config)
        info = model_manager.get_model_info(model_name)
        
        if not info:
            print(f"Model '{model_name}' not found")
            return
        
        print(f"Name: {info['name']}")
        print(f"Path: {info['path']}")
        print(f"Size: {info['size_mb']} MB")
        print(f"Downloaded: {info['downloaded']}")
        print(f"Valid: {info['valid']}")
        print(f"URL: {info['url']}")
        
    except Exception as e:
        print(f"Error getting model info: {e}")


def cleanup_models(config, keep_models=None):
    """Clean up downloaded models"""
    print("=== Model Cleanup ===")
    
    try:
        model_manager = ModelManager(config)
        
        if keep_models is None:
            # Interactive selection of models to keep
            downloaded_models = model_manager.list_downloaded_models()
            if not downloaded_models:
                print("No models to clean up")
                return
            
            print("Downloaded models:")
            for i, model in enumerate(downloaded_models, 1):
                print(f"  {i}. {model}")
            
            print("\nSelect models to KEEP (comma-separated numbers, or 'all' to keep all):")
            choice = input("Models to keep: ").strip().lower()
            
            if choice == 'all':
                print("Keeping all models")
                return
            elif choice == '':
                keep_models = []
            else:
                try:
                    indices = [int(x.strip()) for x in choice.split(',')]
                    keep_models = [downloaded_models[i-1] for i in indices if 1 <= i <= len(downloaded_models)]
                except (ValueError, IndexError):
                    print("Invalid selection")
                    return
        
        removed_count = model_manager.cleanup_models(keep_models)
        print(f"Removed {removed_count} models")
        
        if keep_models:
            print("Kept models:")
            for model in keep_models:
                print(f"  - {model}")
        
    except Exception as e:
        print(f"Error cleaning up models: {e}")


def interactive_mode(config):
    """Interactive model management"""
    print("=== Interactive Model Management ===")
    
    while True:
        print("\nOptions:")
        print("1. List available models")
        print("2. List downloaded models")
        print("3. Download model")
        print("4. Download multiple models")
        print("5. Get model info")
        print("6. Cleanup models")
        print("7. Exit")
        
        try:
            choice = input("\nChoose option (1-7): ").strip()
            
            if choice == "1":
                list_available_models(config)
            elif choice == "2":
                list_downloaded_models(config)
            elif choice == "3":
                download_model(config)
            elif choice == "4":
                model_manager = ModelManager(config)
                selected_models = model_manager.interactive_model_selection()
                if selected_models:
                    download_multiple_models(config, selected_models)
                else:
                    print("No models selected")
            elif choice == "5":
                model_manager = ModelManager(config)
                models = model_manager.list_downloaded_models()
                if not models:
                    print("No models available")
                    continue
                
                print("Downloaded models:")
                for i, model in enumerate(models, 1):
                    print(f"  {i}. {model}")
                
                try:
                    choice_num = int(input(f"Select model (1-{len(models)}): "))
                    if 1 <= choice_num <= len(models):
                        get_model_info(config, models[choice_num - 1])
                    else:
                        print("Invalid choice")
                except ValueError:
                    print("Please enter a valid number")
                    
            elif choice == "6":
                cleanup_models(config)
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
    parser = argparse.ArgumentParser(description='PLC Model Management')
    parser.add_argument('--list-available', action='store_true',
                       help='List models available for download')
    parser.add_argument('--list-downloaded', action='store_true',
                       help='List locally downloaded models')
    parser.add_argument('--download', type=str, metavar='MODEL_NAME',
                       help='Download specific model')
    parser.add_argument('--download-multiple', nargs='+', metavar='MODEL_NAME',
                       help='Download multiple models')
    parser.add_argument('--force-redownload', action='store_true',
                       help='Force redownload even if model exists')
    parser.add_argument('--info', type=str, metavar='MODEL_NAME',
                       help='Get information about a model')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up downloaded models')
    parser.add_argument('--keep', nargs='+', metavar='MODEL_NAME',
                       help='Models to keep during cleanup')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive model management')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Execute commands
    if args.list_available:
        list_available_models(config)
    elif args.list_downloaded:
        list_downloaded_models(config)
    elif args.download:
        download_model(config, args.download, args.force_redownload)
    elif args.download_multiple:
        download_multiple_models(config, args.download_multiple, args.force_redownload)
    elif args.info:
        get_model_info(config, args.info)
    elif args.cleanup:
        cleanup_models(config, args.keep)
    elif args.interactive:
        interactive_mode(config)
    else:
        # Default to interactive mode if no arguments
        interactive_mode(config)


if __name__ == "__main__":
    main()
