#!/usr/bin/env python3
"""
Solutions for PyTorch + PaddlePaddle GPU conflicts
Handle CUDA DLL conflicts when both frameworks are installed
"""

import os
import sys
import gc
import importlib

class GPUFrameworkManager:
    """Manage GPU framework switching to avoid conflicts"""
    
    def __init__(self):
        self.current_framework = None
        self.torch_available = False
        self.paddle_available = False
        
        # Check what's available
        try:
            import torch
            self.torch_available = True
        except ImportError:
            pass
            
        try:
            import paddle
            self.paddle_available = True
        except ImportError:
            pass
    
    def set_environment_for_paddle(self):
        """Set environment variables for PaddlePaddle"""
        print("🔧 Configuring environment for PaddlePaddle...")
        
        # PaddlePaddle environment variables
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # Lazy loading
        os.environ['PADDLE_TRAINERS_NUM'] = '1'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        
        # Disable TensorFlow if it interferes
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Memory optimization
        os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.5'
        os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
        
        print("✅ PaddlePaddle environment configured")
    
    def set_environment_for_torch(self):
        """Set environment variables for PyTorch"""
        print("🔧 Configuring environment for PyTorch...")
        
        # PyTorch environment variables
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Memory management
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
        
        print("✅ PyTorch environment configured")
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory between framework switches"""
        print("🧹 Cleaning GPU memory...")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if self.torch_available:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"   Warning: Could not clear PyTorch cache: {e}")
        
        if self.paddle_available:
            try:
                import paddle
                # Clear PaddlePaddle memory (if methods available)
                if hasattr(paddle.device, 'cuda') and hasattr(paddle.device.cuda, 'empty_cache'):
                    paddle.device.cuda.empty_cache()
            except Exception as e:
                print(f"   Warning: Could not clear PaddlePaddle cache: {e}")
        
        print("✅ Memory cleanup completed")
    
    def switch_to_paddle(self):
        """Switch to PaddlePaddle mode"""
        if self.current_framework == 'paddle':
            print("Already using PaddlePaddle")
            return True
        
        print("\n🔄 Switching to PaddlePaddle mode...")
        
        # Cleanup first
        self.cleanup_gpu_memory()
        
        # Set environment
        self.set_environment_for_paddle()
        
        try:
            # Import and configure PaddlePaddle
            import paddle
            
            if paddle.device.is_compiled_with_cuda():
                paddle.device.set_device('gpu:0')
                print(f"✅ PaddlePaddle GPU mode enabled")
                self.current_framework = 'paddle'
                return True
            else:
                print("⚠️  PaddlePaddle using CPU (CUDA not available)")
                self.current_framework = 'paddle'
                return False
                
        except Exception as e:
            print(f"❌ Failed to switch to PaddlePaddle: {e}")
            return False
    
    def switch_to_torch(self):
        """Switch to PyTorch mode"""
        if self.current_framework == 'torch':
            print("Already using PyTorch")
            return True
        
        print("\n🔄 Switching to PyTorch mode...")
        
        # Cleanup first
        self.cleanup_gpu_memory()
        
        # Set environment
        self.set_environment_for_torch()
        
        try:
            # Import and configure PyTorch
            import torch
            
            if torch.cuda.is_available():
                # Set device
                device = torch.device('cuda:0')
                print(f"✅ PyTorch GPU mode enabled: {torch.cuda.get_device_name(0)}")
                self.current_framework = 'torch'
                return True
            else:
                print("⚠️  PyTorch using CPU (CUDA not available)")
                self.current_framework = 'torch'
                return False
                
        except Exception as e:
            print(f"❌ Failed to switch to PyTorch: {e}")
            return False

def test_paddle_ocr():
    """Test PaddleOCR functionality"""
    print("\n=== Testing PaddleOCR ===")
    
    try:
        from paddleocr import PaddleOCR
        import cv2
        
        # Your test image
        image_path = r"D:\MarMe\github\0.3\plc-diagram-processor\debug_rois\page1_classTag-ID_prob0.87_x2235_y1291_w175_h102.png"
        
        if not os.path.exists(image_path):
            print(f"❌ Test image not found: {image_path}")
            return False
        
        # Initialize OCR
        ocr = PaddleOCR(
            use_textline_orientation=True,
            device="gpu:0",
            lang="en"
        )
        
        # Run test
        result = ocr.predict(image_path)
        
        if result and len(result) > 0:
            page_data = result[0]
            texts = page_data.get('rec_texts', [])
            scores = page_data.get('rec_scores', [])
            
            if texts:
                combined = ' '.join(texts)
                avg_confidence = sum(scores) / len(scores)
                print(f"✅ PaddleOCR working: '{combined}' (confidence: {avg_confidence:.3f})")
                return True
            else:
                print("⚠️  No text detected")
                return False
        else:
            print("❌ No OCR result")
            return False
            
    except Exception as e:
        print(f"❌ PaddleOCR test failed: {e}")
        return False

def test_pytorch():
    """Test PyTorch functionality"""
    print("\n=== Testing PyTorch ===")
    
    try:
        import torch
        import torch.nn as nn
        
        # Simple test
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            
            # Create a simple tensor operation
            x = torch.randn(10, 10).to(device)
            y = torch.randn(10, 10).to(device)
            z = torch.mm(x, y)
            
            print(f"✅ PyTorch working: Matrix multiplication on {device}")
            print(f"   Result shape: {z.shape}")
            print(f"   CUDA memory allocated: {torch.cuda.memory_allocated()/1024/1024:.1f} MB")
            
            return True
        else:
            print("⚠️  PyTorch CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def compatibility_test():
    """Test framework compatibility and switching"""
    print("🧪 Framework Compatibility Test")
    print("=" * 50)
    
    manager = GPUFrameworkManager()
    
    print(f"PyTorch available: {manager.torch_available}")
    print(f"PaddlePaddle available: {manager.paddle_available}")
    
    if not (manager.torch_available and manager.paddle_available):
        print("❌ Both frameworks not available - install missing framework")
        return False
    
    # Test 1: PaddlePaddle first
    print(f"\n📝 Test 1: PaddlePaddle → PyTorch")
    
    success_paddle_1 = manager.switch_to_paddle()
    if success_paddle_1:
        paddle_result_1 = test_paddle_ocr()
    else:
        paddle_result_1 = False
    
    success_torch_1 = manager.switch_to_torch()
    if success_torch_1:
        torch_result_1 = test_pytorch()
    else:
        torch_result_1 = False
    
    # Test 2: PyTorch first  
    print(f"\n📝 Test 2: PyTorch → PaddlePaddle")
    
    success_torch_2 = manager.switch_to_torch()
    if success_torch_2:
        torch_result_2 = test_pytorch()
    else:
        torch_result_2 = False
    
    success_paddle_2 = manager.switch_to_paddle()
    if success_paddle_2:
        paddle_result_2 = test_paddle_ocr()
    else:
        paddle_result_2 = False
    
    # Results
    print(f"\n" + "=" * 50)
    print(f"📊 COMPATIBILITY RESULTS")
    print(f"=" * 50)
    print(f"PaddlePaddle (Test 1):  {'✅ PASS' if paddle_result_1 else '❌ FAIL'}")
    print(f"PyTorch (Test 1):       {'✅ PASS' if torch_result_1 else '❌ FAIL'}")
    print(f"PyTorch (Test 2):       {'✅ PASS' if torch_result_2 else '❌ FAIL'}")
    print(f"PaddlePaddle (Test 2):  {'✅ PASS' if paddle_result_2 else '❌ FAIL'}")
    
    all_passed = all([paddle_result_1, torch_result_1, torch_result_2, paddle_result_2])
    
    if all_passed:
        print(f"\n🎉 All tests passed! Frameworks can coexist with proper switching.")
    else:
        print(f"\n⚠️  Some conflicts detected. Consider using separate environments.")
    
    return all_passed

# Usage example
def main():
    """Main test function"""
    print("🚀 GPU Framework Conflict Resolution Test")
    print("Testing PyTorch + PaddlePaddle compatibility")
    print("=" * 60)
    
    # Run compatibility test
    success = compatibility_test()
    
    if success:
        print(f"\n💡 Recommended usage pattern:")
        print(f"")
        print(f"# Initialize manager")
        print(f"manager = GPUFrameworkManager()")
        print(f"")
        print(f"# Use PaddleOCR")
        print(f"manager.switch_to_paddle()")
        print(f"# ... run OCR tasks ...")
        print(f"")
        print(f"# Use PyTorch")
        print(f"manager.switch_to_torch()")
        print(f"# ... run PyTorch tasks ...")
    else:
        print(f"\n⚠️  Consider using separate conda environments:")
        print(f"conda create -n paddle_env python=3.9")
        print(f"conda create -n torch_env python=3.9")

if __name__ == "__main__":
    main()
