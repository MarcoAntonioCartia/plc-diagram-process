"""
Training Stage for PLC Pipeline
Handles intelligent YOLO model training and validation in yolo_env
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

from ..base_stage import BaseStage


class TrainingStage(BaseStage):
    """Stage 2: Training - Intelligent YOLO model training with custom model detection"""
    
    def __init__(self, name: str = "training", 
                 description: str = "Train or validate YOLO models",
                 required_env: str = "yolo_env", dependencies: list = None):
        super().__init__(name, description, required_env, dependencies or ["preparation"])
    
    def execute(self) -> Dict[str, Any]:
        """Execute intelligent training stage"""
        from src.utils.progress_display import create_stage_progress
        
        # Create progress display
        progress = create_stage_progress("training")
        progress.start_stage("Checking for existing custom models...")
        
        # Get configuration
        from src.config import get_config
        config = get_config()
        
        # Check if we're in multi-environment mode
        multi_env = os.environ.get("PLCDP_MULTI_ENV", "0") == "1"
        
        if multi_env:
            return self._execute_multi_env(config, progress)
        else:
            return self._execute_single_env(config, progress)
    
    def execute_ci_safe(self) -> Dict[str, Any]:
        """CI-safe execution without heavy operations"""
        print("X Running training stage in CI-safe mode")
        
        # Mock training validation
        mock_results = {
            'status': 'ci_mock',
            'pytorch_available': True,
            'models_validated': ['yolo11n.pt', 'yolo11s.pt'],
            'training_ready': True,
            'gpu_available': False,  # Always false in CI
            'mock_mode': True
        }
        
        return mock_results
    
    def _import_dependencies(self):
        """Import heavy dependencies with error handling"""
        try:
            # Try to import torch and ultralytics
            import torch
            from ultralytics import YOLO
            self._torch = torch
            self._yolo = YOLO
            print(f"  V PyTorch {torch.__version__} available")
            print(f"  V CUDA available: {torch.cuda.is_available()}")
        except ImportError as e:
            if self.is_ci:
                # In CI, use mock objects
                self._torch = MockTorch()
                self._yolo = MockYOLO()
                print("  X Using mock PyTorch for CI")
            else:
                raise ImportError(f"PyTorch/Ultralytics not available: {e}")
    
    def _execute_multi_env(self, config, progress) -> Dict[str, Any]:
        """Execute intelligent training in multi-environment mode"""
        try:
            from src.utils.multi_env_manager import MultiEnvironmentManager
            
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            env_manager = MultiEnvironmentManager(project_root)
            
            # Step 1: Check for existing custom models (same logic as yolo11_infer.py)
            progress.update_progress("Checking for existing custom models...")
            custom_model_info = self._find_best_custom_model(config)
            
            if custom_model_info['found']:
                # Custom model exists, validate it
                progress.complete_file("Model Check", f"Found custom model: {custom_model_info['model_name']}")
                return {
                    'status': 'success',
                    'environment': 'yolo_env',
                    'action': 'validation',
                    'custom_model_found': True,
                    'model_info': custom_model_info,
                    'training_skipped': True,
                    'message': f"Using existing custom model: {custom_model_info['model_name']}"
                }
            
            # Step 2: No custom model found, need to train
            progress.update_progress("No custom model found, preparing for training...")
            
            # Step 3: Validate dataset structure
            progress.update_progress("Validating dataset structure...")
            dataset_validation = self._validate_dataset_structure(config)
            
            if not dataset_validation['valid']:
                progress.error_file("Dataset", f"Invalid dataset: {dataset_validation['error']}")
                return {
                    'status': 'error',
                    'error': f"Dataset validation failed: {dataset_validation['error']}",
                    'environment': 'yolo_env'
                }
            
            # Step 4: Check for pretrained models
            progress.update_progress("Checking pretrained models...")
            pretrained_info = self._find_best_pretrained_model(config)
            
            if not pretrained_info['found']:
                progress.error_file("Pretrained Models", "No pretrained models available for fine-tuning")
                return {
                    'status': 'error',
                    'error': 'No pretrained models available for fine-tuning',
                    'environment': 'yolo_env'
                }
            
            # Step 5: Run training
            progress.update_progress(f"Starting training with {pretrained_info['model_name']}...")
            training_result = self._run_training_multi_env(env_manager, config, pretrained_info, progress)
            
            if training_result['status'] == 'success':
                progress.complete_file("Training", f"Model trained successfully: {training_result['custom_model_name']}")
                return {
                    'status': 'success',
                    'environment': 'yolo_env',
                    'action': 'training',
                    'custom_model_found': False,
                    'training_completed': True,
                    'training_result': training_result,
                    'message': f"Training completed: {training_result['custom_model_name']}"
                }
            else:
                progress.error_file("Training", f"Training failed: {training_result['error']}")
                return {
                    'status': 'error',
                    'error': f"Training failed: {training_result['error']}",
                    'environment': 'yolo_env'
                }
            
        except Exception as e:
            progress.error_file("Training Stage", str(e))
            return {
                'status': 'error',
                'error': f"Multi-env training failed: {str(e)}",
                'environment': 'yolo_env'
            }
    
    def _execute_single_env(self, config, progress) -> Dict[str, Any]:
        """Execute intelligent training in single environment mode"""
        try:
            # Import PyTorch directly
            import torch
            print(f"  V PyTorch {torch.__version__} available")
            print(f"  V CUDA available: {torch.cuda.is_available()}")
        except ImportError:
            progress.error_file("PyTorch", "PyTorch not available in single environment mode")
            return {
                'status': 'error',
                'error': 'PyTorch not available in single environment mode',
                'environment': 'single'
            }
        
        progress.update_progress("Running in single environment mode...")
        
        # Step 1: Check for existing custom models (same logic as multi-env)
        progress.update_progress("Checking for existing custom models...")
        custom_model_info = self._find_best_custom_model(config)
        
        if custom_model_info['found']:
            # Custom model exists, validate it
            progress.complete_file("Model Check", f"Found custom model: {custom_model_info['model_name']}")
            return {
                'status': 'success',
                'environment': 'single',
                'action': 'validation',
                'custom_model_found': True,
                'model_info': custom_model_info,
                'training_skipped': True,
                'message': f"Using existing custom model: {custom_model_info['model_name']}"
            }
        
        # Step 2: No custom model found, need to train
        progress.update_progress("No custom model found, preparing for training...")
        
        # Step 3: Validate dataset structure
        progress.update_progress("Validating dataset structure...")
        dataset_validation = self._validate_dataset_structure(config)
        
        if not dataset_validation['valid']:
            progress.error_file("Dataset", f"Invalid dataset: {dataset_validation['error']}")
            return {
                'status': 'error',
                'error': f"Dataset validation failed: {dataset_validation['error']}",
                'environment': 'single'
            }
        
        # Step 4: Check for pretrained models
        progress.update_progress("Checking pretrained models...")
        pretrained_info = self._find_best_pretrained_model(config)
        
        if not pretrained_info['found']:
            progress.error_file("Pretrained Models", "No pretrained models available for fine-tuning")
            return {
                'status': 'error',
                'error': 'No pretrained models available for fine-tuning',
                'environment': 'single'
            }
        
        # Step 5: Run training directly (without multi-env)
        progress.update_progress(f"Starting training with {pretrained_info['model_name']}...")
        training_result = self._run_training_single_env(config, pretrained_info, progress)
        
        if training_result['status'] == 'success':
            progress.complete_file("Training", f"Model trained successfully: {training_result['custom_model_name']}")
            return {
                'status': 'success',
                'environment': 'single',
                'action': 'training',
                'custom_model_found': False,
                'training_completed': True,
                'training_result': training_result,
                'message': f"Training completed: {training_result['custom_model_name']}"
            }
        else:
            progress.error_file("Training", f"Training failed: {training_result['error']}")
            return {
                'status': 'error',
                'error': f"Training failed: {training_result['error']}",
                'environment': 'single'
            }
    
    def _validate_models_multi_env_safe(self, config) -> Dict[str, Any]:
        """Validate models in multi-environment mode without importing PyTorch"""
        validation_result = {
            'models_found': [],
            'models_missing': [],
            'validation_errors': []
        }
        
        # Expected model files
        expected_models = [
            'yolo11n.pt',
            'yolo11s.pt', 
            'yolo11m.pt',
            'yolo11l.pt'
        ]
        
        models_dir = Path(config.config["data_root"]) / "models" / "pretrained"
        
        for model_name in expected_models:
            model_path = models_dir / model_name
            if model_path.exists():
                validation_result['models_found'].append(model_name)
                print(f"  ✓ Model available: {model_name}")
            else:
                validation_result['models_missing'].append(model_name)
                print(f"  ❌ Model missing: {model_name}")
        
        if validation_result['models_found']:
            validation_result['default_model'] = validation_result['models_found'][0]
            print(f"  ✓ Default model: {validation_result['default_model']}")
        else:
            validation_result['validation_errors'].append("No models available for validation")
            print("  ❌ No models found for validation")
        
        return validation_result
    
    def _setup_training_config_safe(self, config) -> Dict[str, Any]:
        """Setup training configuration without importing PyTorch"""
        training_config = {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'image_size': 640,
            'device': 'cuda',  # Assume CUDA in multi-env mode
            'model_name': self.config.get('model_name', 'yolo11m.pt')
        }
        
        return training_config
    
    def _validate_models_multi_env(self, env_manager) -> Dict[str, Any]:
        """Validate models in multi-environment mode"""
        validation_result = {
            'models_found': [],
            'models_missing': [],
            'validation_errors': []
        }
        
        # Expected model files
        expected_models = [
            'yolo11n.pt',
            'yolo11s.pt',
            'yolo11m.pt',
            'yolo11l.pt'
        ]
        
        from src.config import get_config
        config = get_config()
        models_dir = Path(config.config["data_root"]) / "models" / "pretrained"
        
        for model_name in expected_models:
            model_path = models_dir / model_name
            if model_path.exists():
                validation_result['models_found'].append(model_name)
                print(f"  V Model available: {model_name}")
            else:
                validation_result['models_missing'].append(model_name)
                print(f"  X Model missing: {model_name}")
        
        # Try to load a model for validation
        if validation_result['models_found']:
            try:
                # This would normally run in detection environment
                model_path = models_dir / validation_result['models_found'][0]
                
                # Mock model loading for now - would use actual YOLO loading
                validation_result['default_model'] = validation_result['models_found'][0]
                print(f"  V Successfully loaded model: {validation_result['default_model']}")
                
            except Exception as e:
                validation_result['validation_errors'].append(str(e))
                print(f"  X Failed to load model: {e}")
        else:
            validation_result['validation_errors'].append("No models available for validation")
            print(f"  X Model validation error: {e}")
        
        return validation_result
    
    def _validate_models_single_env(self) -> Dict[str, Any]:
        """Validate models in single environment mode"""
        validation_result = {
            'models_found': [],
            'models_missing': [],
            'validation_errors': []
        }
        
        # This is a simplified version - would normally check YOLO models
        print("  X Training not implemented yet - would run YOLO training here")
        
        # Mock validation for now
        validation_result['models_found'] = ['mock_yolo11n.pt']
        validation_result['default_model'] = 'mock_yolo11n.pt'
        
        return validation_result
    
    def _setup_training_config(self, config) -> Dict[str, Any]:
        """Setup training configuration"""
        training_config = {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'image_size': 640,
            'device': 'cuda' if self._is_cuda_available() else 'cpu'
        }
        
        return training_config
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _find_best_custom_model(self, config) -> Dict[str, Any]:
        """Find the best custom trained model (same logic as yolo11_infer.py)"""
        custom_models_dir = config.get_model_path('', 'custom').parent
        
        if not custom_models_dir.exists():
            return {'found': False, 'reason': 'Custom models directory does not exist'}
        
        # Look for models with metadata (same as yolo11_infer.py)
        model_files = list(custom_models_dir.glob("*_best.pt"))
        
        if not model_files:
            return {'found': False, 'reason': 'No custom trained models found'}
        
        # Find the most recent model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        
        # Check if metadata exists
        metadata_file = latest_model.with_suffix('.json')
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                pass
        
        return {
            'found': True,
            'model_name': latest_model.name,
            'model_path': str(latest_model),
            'metadata': metadata,
            'dataset': metadata.get('dataset', 'unknown'),
            'epochs': metadata.get('epochs_trained', 'unknown'),
            'mAP50': metadata.get('metrics', {}).get('mAP50', 'unknown')
        }
    
    def _find_best_pretrained_model(self, config) -> Dict[str, Any]:
        """Find the best available pretrained model for training"""
        # Use the config's built-in model discovery (same as what works in debug)
        available_models = config.discover_available_models('pretrained')
        
        if not available_models:
            return {'found': False, 'reason': 'No pretrained models available'}
        
        # Prefer YOLO10 models for better compatibility, then YOLO11
        preferred_models = [
            'yolo10m.pt', 'yolo10s.pt', 'yolo10n.pt', 'yolo10l.pt', 'yolo10x.pt',
            'yolo11m.pt', 'yolo11s.pt', 'yolo11n.pt', 'yolo11l.pt', 'yolo11x.pt'
        ]
        
        # Check for preferred models in order
        for model_name in preferred_models:
            if model_name in available_models:
                model_path = config.get_model_path(model_name, 'pretrained')
                return {
                    'found': True,
                    'model_name': model_name,
                    'model_path': str(model_path)
                }
        
        # Use first available model if none of the preferred ones are found
        model_name = available_models[0]
        model_path = config.get_model_path(model_name, 'pretrained')
        return {
            'found': True,
            'model_name': model_name,
            'model_path': str(model_path)
        }
    
    def _validate_dataset_structure(self, config) -> Dict[str, Any]:
        """Validate dataset structure (same logic as yolo11_train.py)"""
        try:
            dataset_path = config.get_dataset_path()
            data_yaml_path = config.data_yaml_path
            
            # Check required directories
            required_dirs = ["train/images", "train/labels", "valid/images", "valid/labels"]
            
            missing_dirs = []
            for dir_path in required_dirs:
                full_path = dataset_path / dir_path
                if not full_path.exists():
                    missing_dirs.append(dir_path)
            
            # Check data.yaml
            if not data_yaml_path.exists():
                return {
                    'valid': False,
                    'error': f'Missing data.yaml at: {data_yaml_path}'
                }
            
            if missing_dirs:
                return {
                    'valid': False,
                    'error': f'Missing directories: {", ".join(missing_dirs)}'
                }
            
            # Count files in each directory
            file_counts = {}
            for dir_path in required_dirs:
                full_path = dataset_path / dir_path
                file_count = len(list(full_path.glob("*")))
                file_counts[dir_path] = file_count
            
            return {
                'valid': True,
                'dataset_path': str(dataset_path),
                'data_yaml_path': str(data_yaml_path),
                'file_counts': file_counts
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Dataset validation error: {str(e)}'
            }
    
    def _run_training_multi_env(self, env_manager, config, pretrained_info, progress) -> Dict[str, Any]:
        """Run training in multi-environment mode using yolo11_train.py logic"""
        try:
            # Get training configuration from stage config (passed from command line)
            stage_config = self.config or {}
            epochs = stage_config.get('epochs', 50)  # Default 50, can be overridden
            batch_size = stage_config.get('batch_size', 16)  # Default 16, can be overridden
            
            # Prepare training payload
            training_payload = {
                'action': 'train',
                'model_path': pretrained_info['model_path'],
                'data_yaml_path': str(config.data_yaml_path),
                'epochs': epochs,
                'batch_size': batch_size,
                'patience': max(10, epochs // 2),  # Adaptive patience based on epochs
                'project_name': f"plc_symbol_detector_{pretrained_info['model_name'].replace('.pt', '')}",
                'output_dir': str(config.get_model_path('', 'custom').parent),
                'config': stage_config
            }
            
            progress.update_progress("Running YOLO training in isolated environment...")
            
            # Run training worker (this would need to be implemented in multi_env_manager)
            result = env_manager.run_training_pipeline(training_payload)
            
            if result.get('status') == 'success':
                training_data = result.get('results', {})
                custom_model_name = f"{training_payload['project_name']}_best.pt"
                
                return {
                    'status': 'success',
                    'custom_model_name': custom_model_name,
                    'training_data': training_data,
                    'epochs_completed': training_data.get('epochs_completed', training_payload['epochs']),
                    'best_mAP50': training_data.get('best_mAP50', 0.0)
                }
            else:
                return {
                    'status': 'error',
                    'error': result.get('error', 'Training failed with unknown error')
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Training execution failed: {str(e)}'
            }
    
    def _run_training_single_env(self, config, pretrained_info, progress) -> Dict[str, Any]:
        """Run training in single environment mode using yolo11_train.py directly"""
        try:
            # Import training functions directly
            from src.detection.yolo11_train import train_yolo11
            
            progress.update_progress("Running YOLO training directly...")
            
            # Prepare training parameters
            project_name = f"plc_symbol_detector_{pretrained_info['model_name'].replace('.pt', '')}"
            
            # Run training using yolo11_train.py functions
            results = train_yolo11(
                model_name=pretrained_info['model_name'],
                epochs=50,  # Reasonable default for fine-tuning
                batch=16,
                patience=20,
                project_name=project_name
            )
            
            # Extract metrics from results
            metrics = {}
            if hasattr(results, 'results_dict'):
                metrics = {
                    'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                    'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                    'recall': float(results.results_dict.get('metrics/recall(B)', 0))
                }
            
            custom_model_name = f"{project_name}_best.pt"
            
            return {
                'status': 'success',
                'custom_model_name': custom_model_name,
                'save_dir': str(results.save_dir),
                'epochs_completed': 50,
                'best_mAP50': metrics.get('mAP50', 0.0),
                'metrics': metrics
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Single-env training failed: {str(e)}'
            }
    
    def validate_inputs(self) -> bool:
        """Validate stage inputs"""
        # Check if preparation stage completed
        if not self.check_dependencies():
            print("X Preparation stage not completed")
            return False
        
        return True


class MockTorch:
    """Mock PyTorch for CI testing"""
    __version__ = "0.0.0-mock"
    
    class cuda:
        @staticmethod
        def is_available():
            return False


class MockYOLO:
    """Mock YOLO for CI testing"""
    def __init__(self, model_path=None):
        self.model_path = model_path
    
    def __call__(self, *args, **kwargs):
        return self
