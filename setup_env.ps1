$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[HTTM Setup] Creating virtual environment at $VenvDir ..."
    $created = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv $VenvDir 2>$null
        if ($LASTEXITCODE -eq 0 -and (Test-Path $VenvPython)) {
            $created = $true
        }
    }
    if (-not $created) {
        & python -m venv $VenvDir
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
            throw "Failed to create Python virtual environment. Ensure Python 3.10+ (3.11 recommended) is installed."
        }
    }
}

Write-Host "[HTTM Setup] Upgrading pip..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$HasNvidiaGpu = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi 1>$null 2>$null
    if ($LASTEXITCODE -eq 0) {
        $HasNvidiaGpu = $true
    }
}

if ($HasNvidiaGpu) {
    Write-Host "[HTTM Setup] NVIDIA GPU detected. Installing PyTorch CUDA (cu128)..."
    & $VenvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[HTTM Setup] Failed to install cu128 PyTorch wheel; continuing with default PyPI requirements."
    }
} else {
    Write-Host "[HTTM Setup] No NVIDIA GPU detected (nvidia-smi unavailable). Using default CPU PyTorch."
}

Write-Host "[HTTM Setup] Installing requirements.txt..."
& $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[HTTM Setup] Verifying environment..."
& $VenvPython -c "import torch, ultralytics, cv2; print('Python:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[HTTM Setup] Done! Run .\run_camera_live.bat or .venv\Scripts\python.exe ai_training\integration\camera_coordinator_live.py"
