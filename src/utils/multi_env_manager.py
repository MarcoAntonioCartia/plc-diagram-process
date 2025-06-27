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

    DETECTION_ENV_NAME = "yolo_env"  # Renamed from detection_env
    OCR_ENV_NAME = "ocr_env"

    # Default package stubs – will be finalised in __init__ once we know the
    # absolute path of the project root.  Keep them here only for type hints.
    _DETECTION_PKGS: list[str]
    _OCR_PKGS: list[str]

    # ---------------------------------------------------------------------
    def __init__(self, project_root: Path, *, dry_run: bool = False):
        """Initialise the manager.

        Parameters
        ----------
        project_root : Path
            Root of the PLC-Diagram-Processor repository.
        dry_run : bool, optional
            If *True* no venvs will actually be created and every destructive
            action will be replaced by a log message.  Useful when running the
            top-level setup script with the global --dry-run flag.
        """

        self.project_root = project_root.resolve()
        self.dry_run = dry_run
        self.env_root = self.project_root / "environments"
        self.detection_env_path = self.env_root / self.DETECTION_ENV_NAME
        self.ocr_env_path = self.env_root / self.OCR_ENV_NAME

        self.detection = _VenvPaths(self.detection_env_path)
        self.ocr = _VenvPaths(self.ocr_env_path)

        # ------------------------------------------------------------------
        # Build package installation commands referencing the split
        # requirements files.  Using absolute paths avoids "pip install -r"
        # lookup issues when the working directory is not the repo root.
        # ------------------------------------------------------------------
        det_req = self.project_root / "requirements-detection.txt"
        core_req = self.project_root / "requirements-core.txt"
        ocr_req = self.project_root / "requirements-ocr.txt"

        # Install PyPI packages first, then give framework-specific wheels via
        # an *additional* index URL so we can still resolve common
        # dependencies like ``wheel`` from the default repository.

        self._DETECTION_PKGS = [
            "--extra-index-url", "https://download.pytorch.org/whl/cu128",
            "-r", str(core_req),
            "-r", str(det_req),
        ]

        self._OCR_PKGS = [
            "--extra-index-url", "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
            "requests",  # needed by many helpers, keep it explicit
            "-r", str(core_req),
            "-r", str(ocr_req),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def setup(self, *, force_recreate: bool = False, pip_check: bool = False) -> bool:
        """Create venvs and install the dedicated package sets.

        Parameters
        ----------
        force_recreate : bool, optional
            If *True* existing venv folders will be deleted first.
        pip_check : bool, optional
            If *True* run a fast ``pip install --dry-run`` dependency
            resolver before creating the heavy venvs.  Skipping this check
            can save time in the common case but enables an early failure
            mode when debugging broken requirements lists.
        """

        if self.dry_run:
            print("[MultiEnv] DRY-RUN: would create environments in", self.env_root)
        else:
            self.env_root.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Optional fast-fail dependency resolution check *before* heavy venv
        # creation.  Enabled via the ``--pip-check`` CLI flag or when
        # ``pip_check`` is passed explicitly.
        # ------------------------------------------------------------------
        if pip_check:
            if not self._preflight_resolve("detection", self._DETECTION_PKGS):
                return False
            if not self._preflight_resolve("ocr", self._OCR_PKGS):
                return False

        ok = True
        ok &= self._ensure_env(self.detection_env_path, self._DETECTION_PKGS, force_recreate)
        ok &= self._ensure_env(self.ocr_env_path, self._OCR_PKGS, force_recreate)
        return ok

    def health_check(self) -> bool:
        """Import torch/paddle inside each venv and check GPU availability.

        Skipped entirely in dry-run mode because the envs do not exist.
        """

        if self.dry_run:
            print("[MultiEnv] DRY-RUN: skipping health check – envs not created")
            return True

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

    # Convenience wrapper --------------------------------------------------
    def run_complete_pipeline(
        self,
        pdf_path: Path,
        output_dir: Path,
        *,
        model_path: Optional[str] = None,
        detection_conf: float = 0.25,
        ocr_conf: float = 0.7,
        lang: str = "en",
    ) -> Dict[str, Any]:
        """Run detection → OCR for a single PDF and return combined result."""

        # Ensure output directories exist
        output_dir.mkdir(parents=True, exist_ok=True)

        det_payload = {
            "pdf_path": str(pdf_path),
            "output_dir": str(output_dir / "detection"),
            "model_path": model_path,
            "confidence_threshold": detection_conf,
        }
        det_out = self.run_detection_pipeline(det_payload)

        if det_out.get("status") != "success":
            return {"status": "error", "stage": "detection", **det_out}

        ocr_payload = {
            "pdf_path": str(pdf_path),
            "output_dir": str(output_dir / "ocr"),
            "detection_results": det_out["results"],
            "confidence_threshold": ocr_conf,
            "language": lang,
        }

        ocr_out = self.run_ocr_pipeline(ocr_payload)

        if ocr_out.get("status") != "success":
            return {"status": "error", "stage": "ocr", **ocr_out}

        return {
            "status": "success",
            "detection": det_out["results"],
            "ocr": ocr_out["results"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_env(self, env_path: Path, package_cmd: list[str], force: bool) -> bool:
        if force and env_path.exists():
            if self.dry_run:
                print(f"[MultiEnv] DRY-RUN: would remove existing {env_path}")
            else:
                shutil.rmtree(env_path, ignore_errors=True)

        if not env_path.exists():
            if self.dry_run:
                print(f"[MultiEnv] DRY-RUN: would create venv {env_path.name}")
            else:
                print(f"[MultiEnv] Creating venv {env_path.name} …")
                subprocess.check_call([sys.executable, "-m", "venv", str(env_path)])

        python_exe = _VenvPaths(env_path).python
        pip_exe = _VenvPaths(env_path).pip

        # Upgrade tools & install packages
        if self.dry_run:
            print(f"[MultiEnv] DRY-RUN: would install packages into {env_path.name}: {' '.join(package_cmd)}")
            return True

        print(f"[MultiEnv] Installing packages into {env_path.name} … (quiet mode)")

        # 1) Ensure build tools are present *from PyPI* so that subsequent
        #    installs (which may point to vendor wheels) never fail to find
        #    ``wheel``.
        subprocess.check_call([
            str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--quiet", "--no-input"
        ])

        # 2) Install the actual requirements (may reference vendor wheel index)
        cmd = [
            str(pip_exe),
            "install",
            "--progress-bar",
            "off",
            "--quiet",
            "--no-input",
        ] + package_cmd

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            print(f"[MultiEnv] V {env_path.name} ready")
            return True
        else:
            # Show last few lines of pip output for debugging
            tail = "\n".join(proc.stderr.decode().strip().split('\n')[-3:])
            print(f"[MultiEnv] X pip failed for {env_path.name}:\n{tail}")
            return False

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

            MAX_RETRIES = 2
            TIMEOUT_SEC = int(os.getenv("PLC_WORKER_TIMEOUT", "1800"))  # 30 min default

            attempt = 0
            while True:
                attempt += 1
                try:
                    completed = subprocess.run(
                        [str(env.python), str(script_path), "--input", str(input_file), "--output", str(output_file)],
                        env=env_vars,
                        timeout=TIMEOUT_SEC,
                        capture_output=True,
                        text=True,
                    )

                    if completed.returncode != 0:
                        raise subprocess.CalledProcessError(
                            completed.returncode, completed.args, output=completed.stdout, stderr=completed.stderr,
                        )

                    # Parse result JSON – may still include error status inside
                    result = json.loads(output_file.read_text())
                    return result

                except subprocess.TimeoutExpired:
                    err = f"Worker {worker_script} timed out after {TIMEOUT_SEC}s (attempt {attempt}/{MAX_RETRIES})"
                    print(f"[MultiEnv] {err}")
                    if attempt > MAX_RETRIES:
                        return {"status": "error", "error": err}
                    continue  # retry
                except subprocess.CalledProcessError as exc:
                    stderr_output = exc.stderr.strip() if exc.stderr else "No stderr"
                    stdout_output = exc.stdout.strip() if exc.stdout else "No stdout"
                    err_msg = (
                        f"Worker {worker_script} failed with code {exc.returncode}\n"
                        f"STDOUT: {stdout_output}\n"
                        f"STDERR: {stderr_output}"
                    )
                    print(f"[MultiEnv] {err_msg} (attempt {attempt}/{MAX_RETRIES})")
                    if attempt > MAX_RETRIES:
                        return {"status": "error", "error": err_msg}
                    continue

    # ------------------------------------------------------------------
    # Dependency pre-resolution – saves 30 min installs when impossible
    # ------------------------------------------------------------------
    def _preflight_resolve(self, name: str, package_cmd: list[str]) -> bool:
        """Run ``pip install --dry-run`` in the *current* interpreter to
        verify that the requirements can be solved.  Skips the actual wheel
        downloads/builds, finishes in seconds, and prints the full resolver
        error if it fails.
        """

        if self.dry_run:
            print(f"[MultiEnv] DRY-RUN: would dry-run dependency resolution for {name}_env")
            return True

        print(f"[MultiEnv] Pre-resolving dependencies for {name}_env …")

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--no-input",
            "--no-cache-dir",
        ] + package_cmd

        # Capture output to avoid log spam unless there is a problem
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            print(f"[MultiEnv] {name}_env dependencies OK")
            return True

        print(
            f"[MultiEnv] Dependency resolution failed for {name}_env, aborting setup.\n"
            f"Command: {' '.join(cmd)}\n"
            f"--- pip stderr ---\n{proc.stderr}\n--- end stderr ---"
        )
        return False


# ---------------------------------------------------------------------------
# CLI entry-point for convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-environment manager CLI")
    parser.add_argument("--setup", action="store_true", help="create both environments if missing")
    parser.add_argument("--force-recreate", action="store_true", help="recreate venvs even if they exist")
    parser.add_argument("--health-check", action="store_true", help="run import + GPU tests for both envs")
    parser.add_argument("--dry-run", action="store_true", help="show actions without executing them")
    parser.add_argument("--pip-check", action="store_true", help="run a 'pip install --dry-run' resolution before creating envs")
    args = parser.parse_args()

    mgr = MultiEnvironmentManager(Path(__file__).resolve().parent.parent, dry_run=args.dry_run)

    if args.setup:
        if not mgr.setup(force_recreate=args.force_recreate, pip_check=args.pip_check):
            sys.exit(1)
    if args.health_check:
        if not mgr.health_check():
            sys.exit(1)
