"""PLC-Diagram-Processor utilities package

This module used to *eagerly* import every helper sub-module which caused heavy
third-party libraries (e.g. **requests**, **torch**) to be pulled in as soon as
*anything* under :pycode:`src.utils` was touched.  That broke the unified
installer: during the moment we create the base virtual-environment those heavy
dependencies are not installed yet, so the import chain failed.

We now expose the same public symbols **lazily**.  The first time you access
``utils.DatasetManager`` we dynamically import ``utils.dataset_manager`` and
cache the attribute.  This keeps startup times low and removes hard runtime
dependencies from the installer.  Inspired by Python 3.7's importlib
lazy-loader recipe.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Dict, Tuple, TYPE_CHECKING

# Map public attribute → (submodule, attribute name)
_lazy_targets: Dict[str, Tuple[str, str]] = {
    "DatasetManager": ("dataset_manager", "DatasetManager"),
    "ModelManager": ("model_manager", "ModelManager"),
    "OneDriveManager": ("onedrive_manager", "OneDriveManager"),
    "GPUManager": ("gpu_manager", "GPUManager"),
    "MultiEnvironmentManager": ("multi_env_manager", "MultiEnvironmentManager"),
    # Runtime flag helpers (tiny, safe to import anywhere)
    "multi_env_active": ("runtime_flags", "multi_env_active"),
    "skip_detection_requested": ("runtime_flags", "skip_detection_requested"),
}

__all__ = list(_lazy_targets.keys())  # what 'from utils import *' exposes


def __getattr__(name: str):  # noqa: D401, E501 – simple gateway
    """Dynamically import requested symbol on first access.

    Called by Python when *name* is not found in the module globals.  If the
    name is in our *lazy_targets* table we import the sub-module, grab the
    attribute, cache it in *utils*'s namespace and return it.
    """

    if name not in _lazy_targets:
        raise AttributeError(f"module 'utils' has no attribute '{name}'")

    submodule_name, attr_name = _lazy_targets[name]

    # Absolute import path (this package is 'src.utils')
    full_module = f"{__name__}.{submodule_name}"

    module: ModuleType = importlib.import_module(full_module)
    attr = getattr(module, attr_name)

    # Cache on this module so future lookups are fast and 'is' comparisons work.
    setattr(sys.modules[__name__], name, attr)

    return attr


# Optional: pre-populate typing / IDEs without importing heavy deps at runtime.
if TYPE_CHECKING:  # pragma: no cover
    from .dataset_manager import DatasetManager  # noqa: F401
    from .model_manager import ModelManager  # noqa: F401
    from .onedrive_manager import OneDriveManager  # noqa: F401
    from .gpu_manager import GPUManager  # noqa: F401
    from .multi_env_manager import MultiEnvironmentManager  # noqa: F401
