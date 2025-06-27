#!/usr/bin/env python3
"""run_split_pipeline.py – high-level orchestrator for the two-venv PLC pipeline.

This **meta CLI** ensures the dedicated *detection_env* (PyTorch) and *ocr_env*
(Paddle) virtual environments exist and are healthy, then runs detection and
OCR in their respective workers via :class:`utils.MultiEnvironmentManager`.

It purposely does *not* import any GPU frameworks itself, so the main process
stays light-weight and conflict-free.  Existing single-env scripts continue to
function unchanged – this file is the recommended entry-point for production
/ web back-ends.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root = two levels above this file (src/run_split_pipeline.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))  # safety for editable-mode absence

from utils.multi_env_manager import MultiEnvironmentManager  # noqa: E402 – after path tweak


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run detection + OCR with split virtual-environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("pdf", type=Path, nargs="?", help="PDF file to process")
    p.add_argument("--output", "-o", type=Path, default=Path("output"), help="Base output directory")

    p.add_argument("--det-conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--ocr-conf", type=float, default=0.7, help="OCR confidence threshold")
    p.add_argument("--lang", default="en", help="OCR language code")

    # Environment maintenance flags
    p.add_argument("--setup", action="store_true", help="Create venvs if missing then exit")
    p.add_argument("--health-check", action="store_true", help="Run GPU import tests and exit")
    p.add_argument("--force-recreate", action="store_true", help="Recreate venvs from scratch during setup")
    p.add_argument("--dry-run", action="store_true", help="Only print actions – no venv creation / installs")
    p.add_argument("--pip-check", action="store_true", help="Run a dependency resolver dry-run before installing packages")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – script entry-point
    args = _parse_args()

    mgr = MultiEnvironmentManager(PROJECT_ROOT, dry_run=args.dry_run)

    # ------------------------------------------------------------------
    # Maintenance modes (setup / health-check only)
    # ------------------------------------------------------------------
    if args.setup:
        ok = mgr.setup(force_recreate=args.force_recreate, pip_check=args.pip_check)
        sys.exit(0 if ok else 1)

    if args.health_check:
        ok = mgr.health_check()
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------------
    # Pipeline run
    # ------------------------------------------------------------------
    if args.pdf is None:
        sys.exit("Error: PDF path is required when not using --setup / --health-check")

    # 1) Ensure envs exist (but do not force recreation unless asked)
    if not mgr.setup(force_recreate=False, pip_check=args.pip_check):
        sys.exit("❌  Failed to create virtual-environments")

    # 2) Quick GPU import test – if it fails attempt a repair once
    if not mgr.health_check():
        print("[meta] Health check failed – attempting env repair…", file=sys.stderr)
        if not mgr.setup(force_recreate=True, pip_check=args.pip_check) or not mgr.health_check():
            sys.exit("❌  Split environments remain broken after repair attempt")

    # 3) Run detection + OCR
    result = mgr.run_complete_pipeline(
        pdf_path=args.pdf,
        output_dir=args.output,
        detection_conf=args.det_conf,
        ocr_conf=args.ocr_conf,
        lang=args.lang,
    )

    print(json.dumps(result, indent=2))
    if result.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main() 