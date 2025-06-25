# GPU-Based PaddleOCR Debug Report

*Last updated: 2025-06-25*

---

## 1.  Host & Environment Snapshot

| Item | Value |
|------|-------|
| OS | Windows 10 Pro (22H2) |
| GPU | NVIDIA Quadro RTX 4000 |
| Driver | 537.24 (R535 family – CUDA 12.x capable) |
| Python | 3.11 (venv: `plcdp`) |
| CUDA toolchains present | **None** (we rely on wheels only) |
| Key Python wheels | `paddlepaddle-gpu==3.0.0` **cu126 build**<br>`torch==2.3.0+cu121` / `torchvision==0.18.0+cu121`<br>*No* stand-alone `nvidia-*-cuXX` wheels after cleanup |

---

## 2.  Root Cause of the DLL Hell

1. **Paddle's GPU wheel already bundles a _complete_ CUDA runtime** under
   `paddle\libs`.  Those DLLs have internal RPATHs pointing at each other and
   work out-of-the-box.
2. We installed additional `nvidia-*-cu12` & `nvidia-*-cu11` wheels trying to
   satisfy missing symbols. This introduced **two disjoint copies** of the same
   DLL names (`cudart64_12.dll`, `cublas64_12.dll`, …) into the process space.
3. Windows' DLL loader always takes **the first match in the search path**. A
   mixed chain (e.g. `cusolver64_11.dll` from cu11 wheel ⇒ tries to call
   `cusparse64_12.dll` from cu12 wheel ⇒ calls a _different_ `cublas64_12.dll`)
   produced the cascading *WinError 127* failures we observed.
4. The quickest and most stable remedy on Windows is therefore to **trust the
   bundled CUDA runtime that ships with Paddle** and **remove all external
   NVIDIA wheels**.

---

## 3.  Clean-room Procedure (that finally worked)

```powershell
# 0) Activate the venv first
#    PS> & D:\MarMe\github\0.3\plc-diagram-processor\plcdp\Scripts\Activate.ps1

# 1) Remove every stray CUDA wheel
pip uninstall -y nvidia-*-cu11 nvidia-*-cu12

# 2) Verify that *only* paddle's DLL folder is needed
python - << 'PY'
import ctypes, os, pathlib, paddle
libs = pathlib.Path(os.environ['VIRTUAL_ENV'])/'Lib/site-packages/paddle/libs'
ctypes.WinDLL(str(libs/'cudnn64_9.dll'))  # probe one heavy DLL
print('✓ paddle.libs DLLs load')
print('CUDA runtime seen by Paddle:', paddle.version.cuda())
PY
# Expect: no exception + a CUDA 12.x runtime string

# 3) Simplify launch.py & _apply_gpu_path_fix()
#    dll_paths = [<venv> / 'Lib/site-packages/paddle/libs']

# 4) Run OCR-only stage
python launch.py `
      --skip-detection `
      --detection-folder D:\MarMe\github\0.3\plc-data\processed\detdiagrams `
      --ocr-confidence 0.7 `
      --create-enhanced-pdf --pdf-confidence-threshold 0.8
```

Result: PaddleOCR initialises on **gpu:0**, ~140 text regions extracted per
sample PDF, enhanced PDFs show only boxes ≥ 0.8 confidence.

---

## 4.  Diagnostic Tools & One-liners

| Purpose | Command |
|---------|---------|
| Check which CUDA DLLs a wheel ships | `Expand-Archive .\wheel.whl -Destination .\tmp && dir -Recurse .\tmp | findstr .dll` |
| Load a DLL & reveal next missing dep | `python -c "import ctypes, sys; ctypes.WinDLL(r'full\\path\\foo.dll')"` |
| Show GPU availability across frameworks | `python -c "import torch, paddle, json, torch.cuda as tc; print(json.dumps({'torch': tc.is_available(),'paddle': paddle.device.is_compiled_with_cuda()}))"` |

---

## 5.  Remaining Edge Cases

1. **Model-weights auto-download** – first run may hang behind a corporate
   proxy. Pre-download the PP-OCRv4 weights into `%USERPROFILE%\.paddleocr` if
   necessary.
2. **Mixed entry-points** – If YOLO (Torch) is executed _first_, its DLLs are
   already loaded and may conflict with Paddle.  Our `GPUManager` ensures the
   loader order is deterministic:
   ```python
   from src.utils.gpu_manager import GPUManager as gpu
   gpu.global_instance().use_paddle()  # before importing paddleocr
   ```
3. **Future library bumps** – Paddle 3.1 may upgrade to cu128; then repeat the
   clean-room checklist and ensure Torch is still on a _different_ self-bundled
   runtime (`+cu128` wheels are expected mid-2025).

---

## 6.  Recommendations Going Forward

* **One CUDA runtime per interpreter** – rely on the library's bundled wheels;
  do not mix with stand-alone NVIDIA wheels unless absolutely required.
* Keep the PATH-patch logic minimal: only `paddle\libs` (OCR phase) or
  `torch\lib` (detection phase) should be prepended, never the whole
  `%SystemRoot%\System32` CUDA directories.
* Automate a self-check in `tests/validate_setup.py` that imports both
  frameworks sequentially with the `GPUManager` swap to catch regressions.
* Document the clean-room reinstall steps in `setup/README.md` so new
developers can recover quickly.

---

*Prepared by: Project Assistant "o3-Debug"* 