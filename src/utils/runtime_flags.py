from __future__ import annotations

"""Runtime feature flags for PLC Diagram Processor.

These helpers keep heavy optional dependencies (Ultralytics / Paddle) from
being imported in the wrong interpreter.  They rely on *environment
variables* so that subprocesses inherit the same behaviour regardless of the
`sys.argv` they receive.

Flags
-----
PLCDP_MULTI_ENV = "1"  → Split-environment mode (detection & OCR run in
                         dedicated virtual-envs via MultiEnvironmentManager).

PLCDP_SKIP_DETECTION = "1"  → Force skipping detection import logic even when
                              not in multi-env mode (developer override).
"""

import os
import sys
from functools import lru_cache


@lru_cache(maxsize=None)
def multi_env_active() -> bool:  # noqa: D401
    """Return *True* when the split-environment workflow is active."""
    return os.getenv("PLCDP_MULTI_ENV") == "1"


@lru_cache(maxsize=None)
def skip_detection_requested() -> bool:  # noqa: D401
    """Determine whether the current interpreter should avoid importing torch / Ultralytics.

    Detection is skipped when:
    • the user passed ``--skip-detection`` on the CLI, OR
    • the split-environment flag is active, OR
    • the developer set PLCDP_SKIP_DETECTION=1, OR
    • the user is just asking for help (--help).
    """

    return (
        "--skip-detection" in sys.argv
        or multi_env_active()
        or os.getenv("PLCDP_SKIP_DETECTION") == "1"
        or "--help" in sys.argv
    ) 