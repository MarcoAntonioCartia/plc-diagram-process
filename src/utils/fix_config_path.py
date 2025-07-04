#!/usr/bin/env python3
"""Fix config path to point to correct version folder"""

from src.config import get_config
from pathlib import Path
import yaml

def main():
    config = get_config()
    print('Current data root:', config.config['data_root'])
    
    # Update to correct version folder (0.4 instead of 0.3)
    old_data_root = config.config['data_root']
    new_data_root = old_data_root.replace('0.3', '0.4')
    
    print('Should be:', new_data_root)
    
    # Check if the new path exists
    if not Path(new_data_root).exists():
        print(f"Warning: {new_data_root} does not exist!")
        return
    
    # Update config
    config.config['data_root'] = new_data_root
    
    # Update all paths
    for key in config.config.get('paths', {}):
        old_path = config.config['paths'][key]
        new_path = old_path.replace('0.3', '0.4')
        config.config['paths'][key] = new_path
        print(f"Updated {key}: {old_path} -> {new_path}")
    
    # Save updated config
    with open(config.config_path, 'w') as f:
        yaml.dump(config.config, f, default_flow_style=False)
    
    print('Updated config saved to:', config.config_path)
    print('New data root:', config.config['data_root'])
    
    # Verify models are now found
    print('\nVerifying models...')
    updated_config = get_config()
    models = updated_config.discover_available_models('pretrained')
    print('Available models:', models)

if __name__ == '__main__':
    main()
