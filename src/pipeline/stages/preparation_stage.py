"""
Preparation Stage for PLC Pipeline
Validates inputs, sets up directories, and prepares configuration
"""

import os
from pathlib import Path
from typing import Dict, Any

from ..base_stage import BaseStage


class PreparationStage(BaseStage):
    """Stage 1: Preparation - Validate inputs and setup directories"""
    
    def __init__(self, name: str = "preparation", 
                 description: str = "Validate inputs and setup directories",
                 required_env: str = "core", dependencies: list = None):
        super().__init__(name, description, required_env, dependencies or [])
    
    def execute(self) -> Dict[str, Any]:
        """Execute preparation stage"""
        print("X Starting preparation stage...")
        
        # Get configuration
        from src.config import get_config
        config = get_config()
        
        # Validate and create directories
        directories_created = self._setup_directories(config)
        
        # Validate input files
        input_validation = self._validate_inputs(config)
        
        # Check environment health
        env_health = self._check_environment_health()
        
        return {
            'status': 'success',
            'directories_created': directories_created,
            'input_validation': input_validation,
            'environment_health': env_health,
            'config_validated': True
        }
    
    def execute_ci_safe(self) -> Dict[str, Any]:
        """CI-safe execution without heavy operations"""
        print("X Running preparation stage in CI-safe mode")
        
        # Basic directory structure validation
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        
        # Check if basic directories exist
        src_dir = project_root / "src"
        setup_dir = project_root / "setup"
        
        directories_ok = src_dir.exists() and setup_dir.exists()
        
        return {
            'status': 'ci_mock',
            'directories_created': ['mock_data_dir', 'mock_output_dir'],
            'input_validation': {
                'pdfs_found': 0,
                'config_valid': True,
                'mock_mode': True
            },
            'environment_health': {
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                'directories_ok': directories_ok,
                'ci_mode': True
            },
            'config_validated': True
        }
    
    def _setup_directories(self, config) -> list:
        """Setup required directories"""
        directories_created = []
        
        try:
            data_root = Path(config.config["data_root"])
            
            # Required directories
            required_dirs = [
                data_root / "raw" / "pdfs",
                data_root / "processed" / "images", 
                data_root / "processed" / "detdiagrams",
                data_root / "processed" / "text_extraction",
                data_root / "processed" / "enhanced_pdfs",
                data_root / "processed" / "csv_output",
                data_root / "models" / "pretrained",
                data_root / "models" / "custom",
                data_root / "datasets",
                data_root / "runs"
            ]
            
            for dir_path in required_dirs:
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    directories_created.append(str(dir_path))
                    print(f"  V Created directory: {dir_path}")
                else:
                    print(f"  V Directory exists: {dir_path}")
            
        except Exception as e:
            print(f"  X Error setting up directories: {e}")
            raise
        
        return directories_created
    
    def _validate_inputs(self, config) -> Dict[str, Any]:
        """Validate input files and configuration"""
        validation_result = {
            'pdfs_found': 0,
            'config_valid': True,
            'errors': []
        }
        
        try:
            data_root = Path(config.config["data_root"])
            pdf_dir = data_root / "raw" / "pdfs"
            
            if pdf_dir.exists():
                pdf_files = list(pdf_dir.glob("*.pdf"))
                validation_result['pdfs_found'] = len(pdf_files)
                print(f"  V Found {len(pdf_files)} PDF files")
                
                # List PDF files for reference
                for pdf_file in pdf_files[:5]:  # Show first 5
                    print(f"    - {pdf_file.name}")
                if len(pdf_files) > 5:
                    print(f"    ... and {len(pdf_files) - 5} more")
            else:
                print(f"  X PDF directory not found: {pdf_dir}")
                validation_result['errors'].append(f"PDF directory not found: {pdf_dir}")
            
            # Validate configuration
            required_config_keys = ['data_root', 'paths']
            for key in required_config_keys:
                if key not in config.config:
                    validation_result['config_valid'] = False
                    validation_result['errors'].append(f"Missing config key: {key}")
            
        except Exception as e:
            validation_result['config_valid'] = False
            validation_result['errors'].append(str(e))
            print(f"  X Error validating inputs: {e}")
        
        return validation_result
    
    def _check_environment_health(self) -> Dict[str, Any]:
        """Check environment health and dependencies"""
        health_result = {
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'dependencies_available': {},
            'warnings': []
        }
        
        # Check core dependencies (always available)
        core_deps = ['json', 'pathlib', 'os', 'sys']
        for dep in core_deps:
            try:
                __import__(dep)
                health_result['dependencies_available'][dep] = True
            except ImportError:
                health_result['dependencies_available'][dep] = False
                health_result['warnings'].append(f"Core dependency missing: {dep}")
        
        # Check optional dependencies (only if not in CI)
        if not self.is_ci:
            optional_deps = ['fitz', 'cv2', 'numpy', 'PIL']
            for dep in optional_deps:
                try:
                    __import__(dep)
                    health_result['dependencies_available'][dep] = True
                    print(f"  V Optional dependency available: {dep}")
                except ImportError:
                    health_result['dependencies_available'][dep] = False
                    health_result['warnings'].append(f"Optional dependency missing: {dep}")
                    print(f"  X Optional dependency missing: {dep}")
        
        return health_result
    
    def validate_inputs(self) -> bool:
        """Validate stage inputs"""
        # Preparation stage doesn't have external inputs to validate
        return True
