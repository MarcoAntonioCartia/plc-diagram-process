# src/config.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Centralized configuration management for the PLC diagram processor."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
        self._validate_paths()
    
    def _find_config_file(self) -> str:
        """Find configuration file in order of precedence."""
        # Check environment variable first
        if os.getenv('PLC_CONFIG_PATH'):
            return os.getenv('PLC_CONFIG_PATH')
        
        # Check common locations
        locations = [
            Path.home() / '.plc-processor' / 'config.yaml',
            Path.cwd() / 'config.local.yaml',
            Path.cwd() / 'config.yaml'
        ]
        
        for loc in locations:
            if loc.exists():
                return str(loc)
        
        # Create default config if none exists
        return self._create_default_config()
    
    def _create_default_config(self) -> str:
        """Create a default configuration file."""
        config_dir = Path.home() / '.plc-processor'
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / 'config.yaml'
        
        # Try to find the project root by looking for setup_data.py or other project markers
        current_path = Path(__file__).parent.parent.absolute()  # src -> project root
        
        # Create data directory as sibling to project
        data_root = current_path.parent / 'plc-data'
        
        default_config = {
            'data_root': str(data_root),
            'paths': {
                'datasets': '${data_root}/datasets',
                'models': '${data_root}/models',
                'raw_data': '${data_root}/raw',
                'processed_data': '${data_root}/processed',
                'runs': '${data_root}/runs'
            },
            'model_names': {
                'yolo_pretrained': ['yolo11m.pt', 'yolo11n.pt'],
                'custom_models': []
            },
            'dataset': {
                'name': 'plc-symbols-v2',
                'yaml_file': 'data.yaml'
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default config at: {config_path}")
        print(f"Data directory will be at: {data_root}")
        
        return str(config_path)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Resolve environment variables and references
        config = self._resolve_variables(config)
        return config
    
    def _resolve_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ${var} references in configuration."""
        def resolve_value(value, context):
            if isinstance(value, str) and '${' in value:
                for key, val in context.items():
                    placeholder = f'${{{key}}}'
                    if placeholder in value:
                        value = value.replace(placeholder, str(val))
                # Also resolve environment variables
                for match in os.environ:
                    placeholder = f'${{{match}}}'
                    if placeholder in value:
                        value = value.replace(placeholder, os.environ[match])
            elif isinstance(value, dict):
                return {k: resolve_value(v, context) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item, context) for item in value]
            return value
        
        # First pass: resolve top-level variables
        resolved = {}
        for key, value in config.items():
            resolved[key] = value
        
        # Second pass: resolve nested references
        for key, value in resolved.items():
            resolved[key] = resolve_value(value, resolved)
        
        return resolved
    
    def _validate_paths(self):
        """Validate that required paths exist or create them."""
        # First check if data_root exists, if not, update to use sibling directory
        data_root_path = Path(self.config.get('data_root', ''))
        
        # If data root doesn't exist, it might be because project was moved
        if not data_root_path.exists():
            print(f"Warning: Data root {data_root_path} not found.")
            # Try to find it as sibling to project
            current_path = Path(__file__).parent.parent.absolute()
            sibling_data = current_path.parent / 'plc-data'
            
            if sibling_data.exists():
                print(f"Found data directory at: {sibling_data}")
                self.config['data_root'] = str(sibling_data)
                # Update all paths
                for key in self.config.get('paths', {}):
                    self.config['paths'][key] = self.config['paths'][key].replace(
                        str(data_root_path), str(sibling_data)
                    )
                # Save updated config
                with open(self.config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
                print("Updated configuration with new data location.")
        
        # Now create directories as needed
        for path_key, path_value in self.config.get('paths', {}).items():
            path = Path(path_value)
            if not path.exists():
                print(f"Creating directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_path(self, dataset_name: Optional[str] = None) -> Path:
        """Get path to a specific dataset."""
        dataset_name = dataset_name or self.config['dataset']['name']
        return Path(self.config['paths']['datasets']) / dataset_name
    
    def get_model_path(self, model_name: str, model_type: str = 'pretrained') -> Path:
        """Get path to a specific model."""
        if model_type == 'pretrained':
            return Path(self.config['paths']['models']) / 'pretrained' / model_name
        else:
            return Path(self.config['paths']['models']) / 'custom' / model_name
    
    def get_run_path(self, run_type: str) -> Path:
        """Get path for experiment runs."""
        return Path(self.config['paths']['runs']) / run_type
    
    @property
    def data_yaml_path(self) -> Path:
        """Get path to the data.yaml file for current dataset."""
        return self.get_dataset_path() / self.config['dataset']['yaml_file']

# Singleton instance
_config_instance = None

def get_config() -> Config:
    """Get or create the singleton config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance