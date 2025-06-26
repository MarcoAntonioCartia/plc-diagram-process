# PyTorch + PaddlePaddle GPU Conflict Solutions

## Problem Description

When both PyTorch and PaddlePaddle are installed in the same environment with CUDA **12.1**, they compete for GPU control through different CUDA libraries and DLLs. Common issues:

- **DLL conflicts**: Both frameworks load their own CUDA libraries
- **Memory conflicts**: GPU memory allocation disputes  
- **Driver conflicts**: Different CUDA runtime versions
- **Initialization errors**: Framework fails to initialize GPU

## Solution Approaches

### 🥇 **Method 1: Sequential Framework Loading (Recommended)**

**Best for**: Single script using both frameworks

```python
# Set environment before importing
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

# Use framework manager
manager = GPUFrameworkManager()

# PaddleOCR task
manager.switch_to_paddle()
ocr_result = run_paddle_ocr()

# PyTorch task  
manager.switch_to_torch()
torch_result = run_pytorch_model()
```

**Pros**: ✅ Single environment, ✅ Automatic cleanup, ✅ Memory management  
**Cons**: ⚠️ Context switching overhead

### 🥈 **Method 2: Separate Conda Environments**

**Best for**: Production deployment, complex projects

```bash
# PaddlePaddle environment
conda create -n paddle_env python=3.9
conda activate paddle_env
pip install paddlepaddle-gpu==3.0.0b2
pip install paddleocr

# PyTorch environment
conda create -n torch_env python=3.9  
conda activate torch_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Pros**: ✅ No conflicts, ✅ Clean separation, ✅ Production ready  
**Cons**: ⚠️ Environment switching, ⚠️ Disk space

### 🥉 **Method 3: Environment Variables Only**

**Best for**: Quick testing, simple scripts

```python
# Before any imports
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.5'

# Then import frameworks
import torch
import paddle
```

**Pros**: ✅ Simple, ✅ No code changes  
**Cons**: ⚠️ Still potential conflicts, ⚠️ Less reliable

## CUDA 12.1 Specific Configuration

### PyTorch Installation (Your Command)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### PaddlePaddle Compatible Version
```bash
# Use development version for CUDA 12.1
pip install paddlepaddle-gpu==3.0.0b2

# Or nightly build if issues persist
pip install --pre paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

## Environment Variables Reference

| Variable | Purpose | PaddlePaddle | PyTorch |
|----------|---------|--------------|---------|
| `CUDA_MODULE_LOADING` | Lazy loading | `LAZY` | `LAZY` |
| `FLAGS_allocator_strategy` | Memory strategy | `auto_growth` | - |
| `FLAGS_fraction_of_gpu_memory_to_use` | Memory limit | `0.5` | - |
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch memory | - | `max_split_size_mb:128` |
| `CUDA_LAUNCH_BLOCKING` | Synchronous mode | - | `0` |

## Testing Your Installation

### 1. Basic Compatibility Test
```bash
python gpu_conflict_solutions.py
```

### 2. Simple OCR Test  
```bash
python simple_ocr_test.py
```

### 3. Check CUDA Versions
```python
import torch
import paddle

print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"PaddlePaddle CUDA: {paddle.version.cuda()}")
```

## Troubleshooting

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Both frameworks allocating | Use memory limits |
| `CUDA driver version insufficient` | Version mismatch | Update GPU drivers |
| `Cannot load cuDNN` | Library conflicts | Set `CUDA_MODULE_LOADING=LAZY` |
| `RuntimeError: No CUDA devices` | Driver conflicts | Restart + environment vars |

### Memory Management

```python
# Clear PyTorch cache
torch.cuda.empty_cache()
torch.cuda.synchronize()

# Clear PaddlePaddle cache  
paddle.device.cuda.empty_cache()  # If available

# Force garbage collection
import gc
gc.collect()
```

## Production Recommendations

### For OCR-Heavy Applications
1. **Primary**: Use separate environments
2. **Fallback**: Framework manager with sequential loading
3. **Memory**: Limit each framework to 50% GPU memory

### For Mixed AI Workloads
1. **Architecture**: Separate microservices per framework
2. **Deployment**: Docker containers with framework isolation
3. **Orchestration**: Kubernetes with GPU resource limits

### For Development
1. **IDE**: Use framework manager for testing
2. **Testing**: Automated tests in separate environments  
3. **CI/CD**: Build separate containers per framework

## Quick Start

**Immediate Solution**: Use the framework manager from `gpu_conflict_solutions.py`

```python
from gpu_conflict_solutions import GPUFrameworkManager

manager = GPUFrameworkManager()

# Your OCR workflow
manager.switch_to_paddle()
# ... PaddleOCR code here ...

# Your PyTorch workflow  
manager.switch_to_torch()
# ... PyTorch code here ...
```

This approach will handle the DLL conflicts automatically and provide clean switching between frameworks in your CUDA 12.1 environment.
