#!/usr/bin/env python3
"""
Direct training test - bypasses worker wrapper to isolate issues
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def test_direct_training():
    """Test training directly without worker wrapper"""
    print("=" * 60)
    print("DIRECT TRAINING TEST")
    print("=" * 60)
    
    # Set environment variables to reduce memory pressure
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    try:
        # Import training function
        from src.detection.yolo11_train import train_yolo11, validate_dataset
        
        print("1. Validating dataset...")
        if not validate_dataset():
            print("❌ Dataset validation failed!")
            return False
        print("✅ Dataset validation passed!")
        
        print("\n2. Starting training with optimized settings...")
        print("   - Model: yolo11m.pt")
        print("   - Epochs: 2")
        print("   - Batch: 8")
        print("   - Workers: 8 (testing multiprocessing with pickle fix)")
        
        # Run training with optimized settings
        results = train_yolo11(
            model_name='yolo11m.pt',
            epochs=2,
            batch=8,
            patience=10,
            workers=8,  # Test multiprocessing with pickle fix
            project_name="test_training_direct"
        )
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Results saved to: {results.save_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_training()
    sys.exit(0 if success else 1)
