"""Worker sub-package

This folder is **not** meant to be imported by application code – each worker
script is executed in its own interpreter that lives inside the corresponding
virtual environment (see :pymod:`src.utils.multi_env_manager`).

Still, having a proper ``__init__`` module lets static type-checkers and IDEs
discover the module tree and prevents accidental namespace pollution when the
package is imported indirectly during tests.
"""

from __future__ import annotations

__all__: list[str] = [
    "detection_worker",
    "ocr_worker",
] 