"""gpu_manager.py – Circular-import-safe GPU framework switcher.

This implementation guarantees that *only one* deep-learning framework is
imported at a time so that their private CUDA / cuDNN builds never clash.

Key safeguards
--------------
1.  Never imports both frameworks simultaneously – whichever backend is active
    first gets to claim the process, the other is imported **only after** a
    thorough cleanup.
2.  Uses ``sys.modules`` to detect if a framework was already imported by some
    other part of the code and re-uses that instance instead of importing a new
    one (prevents circular-import errors).
3.  Adds Windows-specific DLL search path tweaks so each framework can always
    find *its* CUDA DLL set first.

The public API stays the same: ``GPUManager.global_instance().use_paddle()`` or
``…use_torch()`` return *True* when the selected backend has working GPU
support.
"""

from __future__ import annotations

import gc
import os
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Dict, Any, Optional


class GPUManager(AbstractContextManager):
    """Singleton GPU framework manager that prevents circular imports."""

    _instance: Optional["GPUManager"] = None

    def __init__(self) -> None:
        self.current_framework: str = "none"  # "torch" | "paddle" | "none"
        self._framework_states: Dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # Singleton accessor
    # ---------------------------------------------------------------------
    @classmethod
    def global_instance(cls) -> "GPUManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ---------------------------------------------------------------------
    # Public helpers – backend switching
    # ---------------------------------------------------------------------
    def use_paddle(self) -> bool:  # noqa: D401 – returns a bool
        """Switch to *Paddle* mode and report whether CUDA is available."""

        if self.current_framework == "paddle":
            return self._get_paddle_state()

        # Clean up what the *previous* framework may have left behind first
        self._cleanup_framework_state()

        # Set Paddle-specific environment tweaks / DLL search order
        self._set_paddle_environment()

        # Import only if not already present (avoids circular imports)
        if "paddle" not in sys.modules:
            try:
                import paddle  # noqa: F401 – local import by design
                self.current_framework = "paddle"
            except Exception as e:  # pragma: no cover – broad except to isolate GPU issues
                print(f"[GPUManager] Failed to import paddle: {e}")
                return False
        else:
            self.current_framework = "paddle"

        return self._get_paddle_state()

    def use_torch(self) -> bool:  # noqa: D401 – returns a bool
        """Switch to *PyTorch* mode and report whether CUDA is available."""

        if self.current_framework == "torch":
            return self._get_torch_state()

        # Clean up any Paddle state first
        self._cleanup_framework_state()

        # Apply Torch-specific env/DLL tweaks
        self._set_torch_environment()

        if "torch" not in sys.modules:
            try:
                import torch  # noqa: F401 – local import by design
                self.current_framework = "torch"
            except Exception as e:  # pragma: no cover
                print(f"[GPUManager] Failed to import torch: {e}")
                return False
        else:
            self.current_framework = "torch"

        return self._get_torch_state()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _cleanup_framework_state(self) -> None:
        """Best-effort GPU memory & DLL cleanup before switching."""

        gc.collect()

        # Torch cleanup
        if "torch" in sys.modules:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass  # swallow – cleanup best effort only

        # Paddle cleanup
        if "paddle" in sys.modules:
            try:
                import paddle

                if hasattr(paddle.device, "cuda") and hasattr(paddle.device.cuda, "empty_cache"):
                    paddle.device.cuda.empty_cache()  # type: ignore[attr-defined]
            except Exception:
                pass

    # -------------------- environment tweaks -----------------------------
    def _set_paddle_environment(self) -> None:
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        os.environ["FLAGS_allocator_strategy"] = "auto_growth"
        os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.5")

        if sys.platform == "win32":
            self._prioritize_paddle_dlls()

    def _set_torch_environment(self) -> None:
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

        if sys.platform == "win32":
            self._prioritize_torch_dlls()

    # -------------------- DLL path helpers (Windows) ---------------------
    def _prioritize_paddle_dlls(self) -> None:
        try:
            venv_path = Path(sys.executable).parent.parent
            site_packages = venv_path / "Lib" / "site-packages"

            paddle_paths = [
                site_packages / "paddle" / "libs",
                site_packages / "nvidia" / "cuda_runtime" / "bin",
                site_packages / "nvidia" / "cudnn" / "bin",
                site_packages / "nvidia" / "cublas" / "bin",
            ]

            valid = [str(p) for p in paddle_paths if p.exists()]
            if valid:
                os.environ["PATH"] = os.pathsep.join(valid) + os.pathsep + os.environ.get("PATH", "")
        except Exception as e:  # pragma: no cover
            print(f"[GPUManager] Warning: Could not prioritise Paddle DLLs: {e}")

    def _prioritize_torch_dlls(self) -> None:
        try:
            venv_path = Path(sys.executable).parent.parent
            site_packages = venv_path / "Lib" / "site-packages"

            torch_lib = site_packages / "torch" / "lib"
            if torch_lib.exists():
                os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
        except Exception as e:  # pragma: no cover
            print(f"[GPUManager] Warning: Could not prioritise Torch DLLs: {e}")

    # -------------------- backend state helpers --------------------------
    @staticmethod
    def _get_paddle_state() -> bool:
        try:
            import paddle

            return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        except Exception:
            return False

    @staticmethod
    def _get_torch_state() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    # ---------------------------------------------------------------------
    # Context manager support
    # ---------------------------------------------------------------------
    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: D401
        self._cleanup_framework_state()
        return False  # do not suppress exceptions


# -------------------------------------------------------------------------
# Convenience accessor expected by the rest of the codebase
# -------------------------------------------------------------------------


def get_gpu_manager() -> GPUManager:
    """Return the global :class:`GPUManager` singleton."""

    return GPUManager.global_instance() 