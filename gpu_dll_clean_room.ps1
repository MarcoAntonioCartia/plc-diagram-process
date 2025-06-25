# GPU DLL clean-room for PLC-Diagram-Processor  (PowerShell)
# 1) removes nvidia-*-cu11/cu12 wheels
# 2) verifies paddle\libs loads
# 3) prints Torch / Paddle GPU summary
param([switch]$Force)

# venv check
if (-not $env:VIRTUAL_ENV -and -not $Force) {
    Write-Host 'ERROR: activate venv first or add -Force' -ForegroundColor Red
    exit 1
}
if ($env:VIRTUAL_ENV) { Write-Host "Using venv: $($env:VIRTUAL_ENV)" }

function py ($code)  { $code | & python - 2>&1; if ($LASTEXITCODE) { exit 1 } }

# remove stand-alone CUDA wheels
$pkgs = (pip list --format=json | ConvertFrom-Json) |
        Where-Object { $_.name -match '^nvidia-.*-cu(11|12)$' } |
        Select-Object -ExpandProperty name
if ($pkgs) { pip uninstall -y @pkgs | Out-Null; Write-Host "Uninstalled: $($pkgs -join ', ')" }
else       { Write-Host 'No standalone CUDA wheels found' }

# verify paddle runtime
$code = @'
import ctypes, os, pathlib, sys, json
import paddle
libs = pathlib.Path(os.environ["VIRTUAL_ENV"]) / "Lib/site-packages/paddle/libs"
if not libs.exists():
    print("ERROR: paddle/libs folder not found")
    sys.exit(1)
dlls = list(libs.glob("*.dll"))
if not dlls:
    print("ERROR: no DLLs found in paddle/libs")
    sys.exit(1)
core = dlls[0]
ctypes.WinDLL(str(core))
print(json.dumps({"cuda_runtime": paddle.version.cuda(), "probe_dll": core.name}))
'@
py $code

# GPU summary
py @'
import json, torch, paddle, sys
print(json.dumps({
  "torch":  {"gpu": torch.cuda.is_available(), "cuda": torch.version.cuda},
  "paddle": {"gpu": paddle.device.is_compiled_with_cuda(), "cuda": paddle.version.cuda()}
}, indent=2))
'@

Write-Host 'Clean-room complete.  You can now run launch.py.' 