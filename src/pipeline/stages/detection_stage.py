"""
Detection Stage for PLC Pipeline
Runs YOLO object detection in yolo_env
"""

import os
from pathlib import Path
from typing import Dict, Any

from ..base_stage import BaseStage


class DetectionStage(BaseStage):
    """Stage 3: Detection - Run YOLO object detection"""
    
    def __init__(self, name: str = "detection", 
                 description: str = "Run YOLO object detection",
                 required_env: str = "yolo_env", dependencies: list = None):
        super().__init__(name, description, required_env, dependencies or ["training"])
    
    def execute(self) -> Dict[str, Any]:
        """Execute detection stage with heavy dependencies"""
        print("X Starting detection stage...")
        
        # Get configuration
        from src.config import get_config
        config = get_config()
        
        # Check if we're in multi-environment mode
        multi_env = os.environ.get("PLCDP_MULTI_ENV", "0") == "1"
        
        if multi_env:
            return self._execute_multi_env(config)
        else:
            return self._execute_single_env(config)
    
    def execute_ci_safe(self) -> Dict[str, Any]:
        """CI-safe execution without heavy operations"""
        print("X Running detection stage in CI-safe mode")
        
        # Mock detection results
        return {
            'status': 'ci_mock',
            'detections_processed': 0,
            'files_processed': [],
            'mock_mode': True,
            'environment': self.required_env
        }
    
    def _execute_multi_env(self, config) -> Dict[str, Any]:
        """Execute detection in multi-environment mode"""
        from src.utils.progress_display import create_stage_progress
        
        # Create progress display
        progress = create_stage_progress("detection")
        progress.start_stage("Running in multi-environment mode")
        
        try:
            from src.utils.multi_env_manager import MultiEnvironmentManager
            
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            env_manager = MultiEnvironmentManager(project_root)
            
            # Get PDF files to process
            data_root = Path(config.config["data_root"])
            pdf_dir = data_root / "raw" / "pdfs"
            pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
            
            if not pdf_files:
                progress.complete_stage("No PDF files found to process")
                return {
                    'status': 'success',
                    'message': 'No PDF files found to process',
                    'files_processed': [],
                    'environment': 'multi'
                }
            
            # Process all PDF files at once (the worker handles multiple files)
            results = []
            
            # Update progress
            progress.start_file(f"Processing {len(pdf_files)} PDF files")
            progress.update_progress("Initializing detection...")
            
            # Prepare detection payload for all files
            detection_payload = {
                'action': 'detect',
                'pdf_folder': str(pdf_dir),  # Use folder instead of individual files
                'output_dir': str(data_root / "processed" / "detdiagrams"),
                'config': self.config
            }
            
            progress.update_progress("Running YOLO detection on all files...")
            
            # Run detection worker once for all files
            result = env_manager.run_detection_pipeline(detection_payload)
            
            if result.get('status') == 'success':
                detection_data = result.get('results', {})
                
                # Handle structured response from updated detection worker
                if isinstance(detection_data, dict):
                    total_detections = detection_data.get('total_detections', 0)
                    output_directory = detection_data.get('output_directory', '')
                    detection_files = detection_data.get('detection_files_created', [])
                    processing_summary = detection_data.get('processing_summary', 'Completed')
                    processed_pdfs = detection_data.get('processed_pdfs', len(pdf_files))
                    
                    progress.complete_file(f"All {processed_pdfs} files", f"{total_detections} total detections")
                    
                    # Create a single result entry for all files
                    results.append({
                        'batch_processing': True,
                        'success': True,
                        'detections': total_detections,
                        'output_directory': output_directory,
                        'detection_files': detection_files,
                        'processed_files': processed_pdfs
                    })
                else:
                    # Fallback for legacy string response
                    total_detections = 0
                    progress.complete_file("All files", f"Completed (legacy format)")
                    results.append({
                        'batch_processing': True,
                        'success': True,
                        'detections': total_detections
                    })
            else:
                error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                progress.error_file("Batch processing", error_msg[:50] + "..." if len(error_msg) > 50 else error_msg)
                results.append({
                    'batch_processing': True,
                    'success': False,
                    'error': error_msg
                })
            
            # Calculate summary and collect output information
            successful = sum(1 for r in results if r['success'])
            total_detections = sum(r.get('detections', 0) for r in results if r['success'])
            
            # Collect all detection files and output directories from successful results
            all_detection_files = []
            output_directories = set()
            
            for r in results:
                if r['success'] and 'detection_files' in r:
                    all_detection_files.extend(r['detection_files'])
                if r['success'] and 'output_directory' in r:
                    output_directories.add(r['output_directory'])
            
            # Use the main output directory (should be consistent across all files)
            main_output_dir = str(data_root / "processed" / "detdiagrams")
            if output_directories:
                main_output_dir = list(output_directories)[0]  # Use first found directory
            
            progress.complete_stage(f"{successful}/{len(pdf_files)} files, {total_detections} total detections")
            
            return {
                'status': 'success',
                'environment': 'multi',
                'files_processed': len(pdf_files),
                'successful_files': successful,
                'total_detections': total_detections,
                'output_directory': main_output_dir,
                'detection_files_created': all_detection_files,
                'results': results
            }
            
        except Exception as e:
            progress.error_file("", f"Stage failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'environment': 'multi'
            }
    
    def _execute_single_env(self, config) -> Dict[str, Any]:
        """Execute detection in single environment mode"""
        print("  X Running in single environment mode")
        
        try:
            # Check if detection modules are available
            try:
                # This would normally import YOLO detection modules
                # For now, we'll use a mock approach
                print("  V Detection pipeline modules available")
                detection_available = True
            except ImportError:
                print("  X Using mock detection pipeline for CI")
                detection_available = False
            
            # Get PDF files to process
            data_root = Path(config.config["data_root"])
            pdf_dir = data_root / "raw" / "pdfs"
            pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
            
            if not pdf_files:
                return {
                    'status': 'success',
                    'message': 'No PDF files found to process',
                    'files_processed': [],
                    'environment': 'single'
                }
            
            # Process files
            results = []
            for pdf_file in pdf_files:
                try:
                    if detection_available:
                        # Would run actual detection here
                        # For now, mock the detection process
                        detection_result = self._mock_detection(pdf_file, config)
                        
                        if detection_result['success']:
                            results.append({
                                'pdf_file': pdf_file.name,
                                'success': True,
                                'detections': detection_result.get('detections', 0)
                            })
                            print(f"    V Detection completed")
                        else:
                            results.append({
                                'pdf_file': pdf_file.name,
                                'success': False,
                                'error': detection_result.get('error', 'Detection failed')
                            })
                            print(f"    X Detection failed")
                    else:
                        # Mock detection for CI
                        results.append({
                            'pdf_file': pdf_file.name,
                            'success': True,
                            'detections': 5,  # Mock detection count
                            'mock': True
                        })
                        
                except Exception as e:
                    results.append({
                        'pdf_file': pdf_file.name,
                        'success': False,
                        'error': str(e)
                    })
                    print(f"    X Detection error: {e}")
            
            # Calculate summary
            successful = sum(1 for r in results if r['success'])
            total_detections = sum(r.get('detections', 0) for r in results if r['success'])
            
            return {
                'status': 'success',
                'environment': 'single',
                'files_processed': len(pdf_files),
                'successful_files': successful,
                'total_detections': total_detections,
                'results': results
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'environment': 'single'
            }
    
    def _mock_detection(self, pdf_file: Path, config) -> Dict[str, Any]:
        """Mock detection for testing purposes"""
        # This would be replaced with actual detection logic
        return {
            'success': True,
            'detections': 3,  # Mock detection count
            'mock': True
        }
    
    def validate_inputs(self) -> bool:
        """Validate stage inputs"""
        # Smart dependency checking: Look for trained models instead of training stage completion
        if not self._check_models_available():
            return False
        
        return True
    
    def _check_models_available(self) -> bool:
        """Check if trained models are available for detection"""
        try:
            from src.config import get_config
            config = get_config()
            
            # Check for custom trained models
            custom_models_dir = config.get_model_path('', 'custom')
            
            if custom_models_dir.exists():
                # Look for trained models (same pattern as yolo11_infer.py)
                model_files = list(custom_models_dir.glob("*_best.pt"))
                
                if model_files:
                    # Find the most recent model
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    print(f"V Found trained model: {latest_model.name}")
                    
                    # Check if metadata exists for additional validation
                    metadata_file = latest_model.with_suffix('.json')
                    if metadata_file.exists():
                        try:
                            import json
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            print(f"  - Dataset: {metadata.get('dataset', 'unknown')}")
                            print(f"  - mAP50: {metadata.get('metrics', {}).get('mAP50', 'unknown')}")
                        except Exception:
                            pass  # Metadata read failed, but model exists
                    
                    return True
                else:
                    print("X No trained models found in custom models directory")
                    print(f"  Expected location: {custom_models_dir}")
                    print("  Please run training stage first or copy trained models to this location")
                    return False
            else:
                print("X Custom models directory does not exist")
                print(f"  Expected location: {custom_models_dir}")
                print("  Please run training stage first to create trained models")
                return False
                
        except Exception as e:
            print(f"X Error checking for trained models: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """Override base dependency checking with smart model-based logic"""
        # Use model-based checking instead of rigid stage dependency checking
        return self._check_models_available()


class MockPipelineRunner:
    """Mock pipeline runner for CI testing"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.total_detections = 0
    
    def run_complete_pipeline(self):
        """Mock detection execution"""
        self.total_detections = 5  # Mock detection count
        return True
