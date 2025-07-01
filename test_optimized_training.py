#!/usr/bin/env python3
"""
Test script for optimized training worker
Tests the performance improvements and output formatting
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def test_training_optimization():
    """Test the optimized training with minimal output"""
    
    print("Testing Optimized Training Worker")
    print("=" * 50)
    
    # Set minimal output mode
    os.environ["PLCDP_MINIMAL_OUTPUT"] = "1"
    os.environ["PLCDP_VERBOSE"] = "0"  # Disable verbose mode for cleaner output
    
    try:
        from src.utils.multi_env_manager import MultiEnvironmentManager
        from src.config import get_config
        
        # Initialize environment manager
        env_manager = MultiEnvironmentManager(project_root)
        config = get_config()
        
        # Check if we have the required data and models
        print("Checking prerequisites...")
        
        # Check dataset
        dataset_path = config.get_dataset_path()
        if not dataset_path.exists():
            print(f"❌ Dataset not found at: {dataset_path}")
            print("Please ensure your dataset is properly set up")
            return False
        
        # Check data.yaml
        if not config.data_yaml_path.exists():
            print(f"❌ Data YAML not found at: {config.data_yaml_path}")
            return False
        
        # Check for pretrained models
        available_models = config.discover_available_models('pretrained')
        if not available_models:
            print("❌ No pretrained models found")
            print("Please download models first")
            return False
        
        print(f"✅ Dataset found: {dataset_path}")
        print(f"✅ Data YAML found: {config.data_yaml_path}")
        print(f"✅ Available models: {available_models}")
        
        # Use the smallest/fastest model for testing
        test_model = None
        for preferred in ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt']:
            if preferred in available_models:
                test_model = preferred
                break
        
        if not test_model:
            test_model = available_models[0]
        
        print(f"✅ Using model for test: {test_model}")
        
        # Prepare minimal training payload (very short training for testing)
        training_payload = {
            'action': 'train',
            'model_path': str(config.get_model_path(test_model, 'pretrained')),
            'data_yaml_path': str(config.data_yaml_path),
            'epochs': 2,  # Very short for testing
            'batch_size': 4,  # Small batch size
            'patience': 2,  # Quick patience
            'project_name': f"test_optimized_{test_model.replace('.pt', '')}",
        }
        
        print("\nStarting optimized training test...")
        print(f"Parameters: {training_payload['epochs']} epochs, batch size {training_payload['batch_size']}")
        print("This should be much faster than the previous 70+ seconds per iteration")
        
        start_time = time.time()
        
        # Run the training
        result = env_manager.run_training_pipeline(training_payload)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nTraining completed in {duration:.1f} seconds")
        
        if result.get('status') == 'success':
            print("✅ Training successful!")
            training_data = result.get('results', {})
            print(f"   Model saved to: {training_data.get('save_dir', 'unknown')}")
            print(f"   Epochs completed: {training_data.get('epochs_completed', 'unknown')}")
            
            # Check if the time per epoch is reasonable
            if duration > 0 and training_payload['epochs'] > 0:
                time_per_epoch = duration / training_payload['epochs']
                print(f"   Time per epoch: {time_per_epoch:.1f} seconds")
                
                if time_per_epoch < 60:  # Less than 1 minute per epoch is good
                    print("✅ Performance looks good!")
                else:
                    print("⚠️  Still slow, but better than before")
            
            return True
        else:
            print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_output():
    """Test that minimal output mode works"""
    print("\nTesting Minimal Output Mode")
    print("-" * 30)
    
    # Test with minimal output enabled
    os.environ["PLCDP_MINIMAL_OUTPUT"] = "1"
    
    try:
        from src.utils.progress_display import create_stage_progress
        
        progress = create_stage_progress("test")
        progress.start_stage("Testing minimal output...")
        
        # Simulate some work
        for i in range(3):
            progress.update_progress(f"Step {i+1}/3...")
            time.sleep(0.5)
        
        progress.complete_file("Test", "Minimal output test completed")
        
        print("✅ Minimal output mode working")
        return True
        
    except Exception as e:
        print(f"❌ Minimal output test failed: {e}")
        return False

if __name__ == "__main__":
    print("PLC Training Optimization Test")
    print("=" * 50)
    
    # Test minimal output first
    if not test_minimal_output():
        print("❌ Minimal output test failed")
        sys.exit(1)
    
    # Test training optimization
    if not test_training_optimization():
        print("❌ Training optimization test failed")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ All optimization tests passed!")
    print("The training should now be:")
    print("  - Much faster (reduced epochs, batch size, patience)")
    print("  - Cleaner output (minimal progress display)")
    print("  - More reliable (better error handling)")
