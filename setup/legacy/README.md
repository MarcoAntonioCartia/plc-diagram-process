# Legacy Setup Files

This directory contains the previous setup scripts that were merged to create the unified setup.

## Files

### `setup_original.py`
- **Source**: Original `setup.py` 
- **Strengths**: 
  - Proven WSL poppler installation logic
  - Robust package installation with parallel processing
  - Excellent error handling and recovery strategies
  - Comprehensive user guidance and interactive prompts
- **Used in Unified Setup**: WSL management, package installation strategies, error handling

### `enhanced_setup_original.py`
- **Source**: Original `enhanced_setup.py`
- **Strengths**:
  - Modular architecture with separate components
  - Advanced GPU detection and PyTorch optimization
  - Build tools management
  - Enhanced progress reporting
- **Issues Fixed in Unified Setup**:
  - WSL management complexity that could block setup
  - PyTorch installation failures in CPU-only environments
  - Overly complex GPU detection that could timeout
- **Used in Unified Setup**: Modular components, GPU detection (simplified), build tools installer

## Why These Were Replaced

The unified setup (`../setup.py`) was created to:

1. **Fix Critical Issues**:
   - Enhanced setup's WSL management could fail and block entire setup
   - PyTorch installation was too dependent on GPU detection
   - Package installation lacked the robustness of the original setup

2. **Combine Best Features**:
   - Original's proven WSL and package installation logic
   - Enhanced's modular architecture and GPU detection
   - Better error handling that doesn't block setup progress

3. **Improve Reliability**:
   - CPU-first PyTorch installation approach
   - Non-blocking error handling
   - Multiple fallback strategies for all components

## Reference

These files are kept for:
- **Historical reference** - understanding the evolution of the setup process
- **Debugging** - comparing behavior if issues arise
- **Feature extraction** - if specific functionality needs to be restored
- **Documentation** - showing what problems were solved

## Usage

These files should **not** be used directly. Use the unified setup instead:

```bash
# Use this (unified setup)
python setup/setup.py

# Not these (legacy)
# python setup/legacy/setup_original.py
# python setup/legacy/enhanced_setup_original.py
```

If you need to reference specific functionality from these files, check the unified setup first - it likely incorporates the best parts of both approaches.
