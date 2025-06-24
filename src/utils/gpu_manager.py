from __future__ import annotations

"""gpu_manager.py
Unified GPU framework switcher for the PLC-Diagram-Processor project.

Why we need this
================
PaddlePaddle and PyTorch each ship their own builds of CUDA/cuDNN.  When both
frameworks are imported in the *same* interpreter session the second one often
fails to resolve symbols and silently drops back to CPU.  This helper makes it
safe (or at least *safer*) to alternate between the two:

    from src.utils.gpu_manager import GPUManager

    gpu = GPUManager.global_instance()
    gpu.use_torch()   # prepare environment, clear caches, import torch
    ... heavy torch work ...
    gpu.use_paddle()  # same for paddle

It is also usable as a context-manager so caches are cleaned on block exit::

    with GPUManager() as gpu:
        gpu.use_paddle()
        ... OCR ...

Implementation notes
--------------------
• Only *mutates* the process-wide environment variables that matter.
• Performs a best-effort CUDA memory purge between switches.
• Import errors are swallowed – you can still call .use_paddle() on a
  PyTorch-only installation and it will just warn.
"""

from contextlib import AbstractContextManager
import gc
import os
import sys
from typing import Optional


class _LazyImports:
    """Helper that lazily checks framework availability without importing twice."""

    torch_checked = False
    paddle_checked = False
    torch_available = False
    paddle_available = False

    @classmethod
    def refresh(cls):
        if not cls.torch_checked:
            cls.torch_checked = True
            try:
                import importlib

                importlib.import_module("torch")
                cls.torch_available = True
            except ModuleNotFoundError:
                cls.torch_available = False
        if not cls.paddle_checked:
            cls.paddle_checked = True
            try:
                import importlib

                importlib.import_module("paddle")
                cls.paddle_available = True
            except ModuleNotFoundError:
                cls.paddle_available = False


class GPUManager(AbstractContextManager):
    """Singleton in-process manager for framework switching."""

    _instance: Optional["GPUManager"] = None

    def __init__(self) -> None:  # noqa: D401 – simple init
        _LazyImports.refresh()
        self.current_framework: str = "none"  # "torch" | "paddle" | "none"

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @classmethod
    def global_instance(cls) -> "GPUManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # Convenience aliases – import side-effects are handled internally
    def use_torch(self) -> bool:
        """Prepare environment and import torch.  Returns *False* if CUDA off."""
        if self.current_framework == "torch":
            return self._torch_cuda_state()
        self._cleanup_gpu_mem()
        self._set_env_for_torch()
        self.current_framework = "torch"
        return self._torch_cuda_state()

    def use_paddle(self) -> bool:
        """Prepare environment and import paddle.  Returns *False* if CUDA off."""
        if self.current_framework == "paddle":
            return self._paddle_cuda_state()
        self._cleanup_gpu_mem()
        self._set_env_for_paddle()
        self.current_framework = "paddle"
        return self._paddle_cuda_state()

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------
    def __enter__(self):  # noqa: D401 – comply with AbstractContextManager
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: D401
        self._cleanup_gpu_mem()
        # Do *not* reset env vars – keep last selection.
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _cleanup_gpu_mem(self) -> None:
        """Best-effort cache purge before switching frameworks."""
        gc.collect()
        if _LazyImports.torch_available:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
        if _LazyImports.paddle_available:
            try:
                import paddle

                if hasattr(paddle.device, "cuda") and hasattr(paddle.device.cuda, "empty_cache"):
                    paddle.device.cuda.empty_cache()  # type: ignore[attr-defined]
            except Exception:
                pass

    # --- environment tuning -------------------------------------------------
    @staticmethod
    def _set_env_for_torch() -> None:
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    @staticmethod
    def _set_env_for_paddle() -> None:
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        os.environ["FLAGS_allocator_strategy"] = "auto_growth"
        os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.5")

    # --- convenience state helpers -----------------------------------------
    @staticmethod
    def _torch_cuda_state() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _paddle_cuda_state() -> bool:
        try:
            import paddle

            return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        except Exception:
            return False 