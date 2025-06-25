"""multi_env_manager.py

Utility to keep detection (PyTorch) and OCR (PaddleOCR) in *separate* Python
virtual-environments so their CUDA stacks never clash.

Directory layout (relative to project root)::

    environments/
        detection_env/    <-- venv with torch + cu128
        ocr_env/          <-- venv with paddlepaddle-gpu 3.0.0 cu126 + paddleocr

The manager can:
• create / recreate those venvs (setup())
• verify they load GPU correctly (health_check())
• launch lightweight worker scripts in the right env and exchange data via
  temporary JSON files (run_detection_pipeline / run_ocr_pipeline)

The heavy lifting (YOLO detection, PaddleOCR) is done by the worker scripts in
src/workers/ so that the main process never imports either framework.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


class _VenvPaths:
    """Compute venv-specific paths in a cross-platform way."""

    def __init__(self, root: Path) -> None:  # root = path to venv folder
        self.root = root
        if os.name == "nt":
            self.python = root / "Scripts" / "python.exe"
            self.pip = root / "Scripts" / "pip.exe"
        else:
            self.python = root / "bin" / "python"
            self.pip = root / "bin" / "pip"


# ---------------------------------------------------------------------------
# Main manager class
# ---------------------------------------------------------------------------


class MultiEnvironmentManager:
    """Create, validate and use split detection / OCR environments."""

    DETECTION_ENV_NAME = "detection_env"
    OCR_ENV_NAME = "ocr_env"

    # Package sets for the two envs ------------------------------------------------
    _DETECTION_PKGS = [
        "--upgrade", "pip", "setuptools", "wheel",
        # fixed GPU framework (torch cu128)
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ]

    _OCR_PKGS = [
        "--upgrade", "pip", "setuptools", "wheel",
        "paddleocr",
        # Paddle GPU wheel from paddle.org.cn (cu126 build)
        "paddlepaddle-gpu==3.0.0",
        "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    ]

    # ---------------------------------------------------------------------
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.env_root = self.project_root / "environments"
        self.detection_env_path = self.env_root / self.DETECTION_ENV_NAME
        self.ocr_env_path = self.env_root / self.OCR_ENV_NAME

        self.detection = _VenvPaths(self.detection_env_path)
        self.ocr = _VenvPaths(self.ocr_env_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def setup(self, *, force_recreate: bool = False) -> bool:
        """Create venvs and install the dedicated package sets."""

        self.env_root.mkdir(parents=True, exist_ok=True)

        ok = True
        ok &= self._ensure_env(self.detection_env_path, self._DETECTION_PKGS, force_recreate)
        ok &= self._ensure_env(self.ocr_env_path, self._OCR_PKGS, force_recreate)
        return ok

    def health_check(self) -> bool:
        """Import torch/paddle inside each venv and check GPU availability."""

        torch_check = self._run_simple_python(
            self.detection.python,
            textwrap.dedent(
                """
                import json, torch, sys
                info = {
                    'torch_version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                }
                print(json.dumps(info))
                """,
            ),
        )
        paddle_check = self._run_simple_python(
            self.ocr.python,
            textwrap.dedent(
                """
                import json, paddle, sys
                info = {
                    'paddle_version': paddle.__version__,
                    'cuda_available': paddle.device.is_compiled_with_cuda(),
                }
                print(json.dumps(info))
                """,
            ),
        )

        if not (torch_check and paddle_check):
            print("[MultiEnv] Health check failed – see messages above")
            return False
        return True

    # High-level helpers --------------------------------------------------
    def run_detection_pipeline(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_worker("detection_worker.py", input_dict, self.detection)

    def run_ocr_pipeline(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_worker("ocr_worker.py", input_dict, self.ocr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_env(self, env_path: Path, package_cmd: list[str], force: bool) -> bool:
        if force and env_path.exists():
            shutil.rmtree(env_path, ignore_errors=True)

        if not env_path.exists():
            print(f"[MultiEnv] Creating venv {env_path.name} …")
            subprocess.check_call([sys.executable, "-m", "venv", str(env_path)])

        python_exe = _VenvPaths(env_path).python
        pip_exe = _VenvPaths(env_path).pip

        # Upgrade tools & install packages
        print(f"[MultiEnv] Installing packages into {env_path.name} …")
        cmd = [str(pip_exe), "install"] + package_cmd
        return subprocess.call(cmd) == 0

    def _run_simple_python(self, python: Path, code: str) -> bool:
        try:
            subprocess.check_call([str(python), "-c", code])
            return True
        except subprocess.CalledProcessError as exc:
            print(exc)
            return False

    def _run_worker(
        self,
        worker_script: str,
        input_payload: Dict[str, Any],
        env: _VenvPaths,
    ) -> Dict[str, Any]:
        """Run *worker_script* inside *env* and return its JSON result."""

        workers_dir = self.project_root / "src" / "workers"
        script_path = workers_dir / worker_script
        if not script_path.exists():
            raise FileNotFoundError(script_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_file = tmpdir_path / "in.json"
            output_file = tmpdir_path / "out.json"
            input_file.write_text(json.dumps(input_payload, indent=2))

            env_vars = os.environ.copy()
            env_vars["PYTHONPATH"] = str(self.project_root)

            subprocess.check_call(
                [str(env.python), str(script_path), "--input", str(input_file), "--output", str(output_file)],
                env=env_vars,
            )

            return json.loads(output_file.read_text())


# ---------------------------------------------------------------------------
# CLI entry-point for convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-environment manager CLI")
    parser.add_argument("--setup", action="store_true", help="create both environments if missing")
    parser.add_argument("--force-recreate", action="store_true", help="recreate venvs even if they exist")
    parser.add_argument("--health-check", action="store_true", help="run import + GPU tests for both envs")
    args = parser.parse_args()

    mgr = MultiEnvironmentManager(Path(__file__).resolve().parent.parent)

    if args.setup:
        if not mgr.setup(force_recreate=args.force_recreate):
            sys.exit(1)
    if args.health_check:
        if not mgr.health_check():
            sys.exit(1) 