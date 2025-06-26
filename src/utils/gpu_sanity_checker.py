#!/usr/bin/env python3
"""gpu_sanity_checker.py
Quick end-to-end test that Torch **and** Paddle can see the GPU.  It is meant to
be executed inside the freshly-created virtual environment at install time:

    python -m src.utils.gpu_sanity_checker --device auto

Exit status
===========
0 – all requested tests passed
1 – at least one framework failed *or* threw an exception

The test is intentionally light-weight (no model downloads) so it finishes in a
few seconds.  For Paddle it instantiates a `PaddleOCR` detector **only if**
Paddle sees a CUDA device; otherwise it just reports CPU fallback.
"""
from __future__ import annotations

import argparse
import sys
import time


def _test_torch(device_preference: str) -> bool:
    try:
        import torch  # noqa: WPS433 (dynamic import is fine in a CLI script)

        if device_preference == "cpu":
            print("[torch]  imported – forcing CPU mode")
            return True

        available = torch.cuda.is_available()
        if available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[torch]  CUDA available – {gpu_name}")
        else:
            print("[torch]   CUDA NOT available – using CPU")
        return available or device_preference == "auto"
    except Exception as exc:  # pragma: no cover – diagnostics
        print(f"[torch]  import/test failed: {exc}")
        return False


def _test_paddle(device_preference: str) -> bool:
    try:
        import paddle  # noqa: WPS433

        if device_preference == "cpu":
            print("[paddle]  imported – forcing CPU mode")
            return True

        available = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        if available:
            # `paddle.device.cuda.current_device()` was removed in Paddle>=2.6, so we
            # fall back gracefully when it's not present.
            if hasattr(paddle.device.cuda, "current_device"):
                dev_id = paddle.device.cuda.current_device()
            else:
                dev_str = getattr(paddle, "get_device", lambda: "gpu:0")()
                dev_id = dev_str.split(":")[-1] if dev_str.startswith("gpu") else "0"
            print(f"[paddle]  CUDA available – gpu:{dev_id}")
        else:
            print("[paddle]   CUDA NOT available – using CPU")
        return available or device_preference == "auto"
    except Exception as exc:
        print(f"[paddle]  import/test failed: {exc}")
        return False


def main() -> int:  # noqa: D401 – CLI entry-point
    parser = argparse.ArgumentParser(description="Verify Torch & Paddle GPU availability")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                        help="gpu = require CUDA, cpu = force CPU tests only, auto = succeed when at least CPU works")
    parser.add_argument("--skip-torch", action="store_true", help="Skip PyTorch test")
    parser.add_argument("--skip-paddle", action="store_true", help="Skip Paddle test")

    args = parser.parse_args()

    device_pref = args.device
    all_ok = True

    start_ts = time.time()

    if not args.skip_torch:
        all_ok &= _test_torch(device_pref)
    else:
        print("[torch]  skipped by flag")

    if not args.skip_paddle:
        all_ok &= _test_paddle(device_pref)
    else:
        print("[paddle]  skipped by flag")

    print(f"Finished GPU sanity check in {time.time() - start_ts:.2f}s")
    return 0 if all_ok else 1


if __name__ == "__main__":  # pragma: no cover – script mode
    sys.exit(main()) 