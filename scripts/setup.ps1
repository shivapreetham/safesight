# SafeSight one-shot setup (Windows PowerShell).
# Creates the venv, installs dependencies, downloads weights from HF Hub,
# and runs the test suite. Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 [-Slim]
#
# -Slim installs the torch-free serving runtime only (fast, small); default
# installs the full dev environment (torch, pytest, ruff, onnx tooling).

param(
    [switch]$Slim
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

$py = ".\.venv\Scripts\python.exe"

Write-Host "Installing dependencies (this can take a few minutes)..."
& $py -m pip install --quiet --upgrade pip
if ($Slim) {
    & $py -m pip install --quiet --no-cache-dir -r requirements.txt
} else {
    & $py -m pip install --quiet --no-cache-dir -r requirements-dev.txt
}

Write-Host "Downloading model weights from Hugging Face Hub..."
& $py scripts/download_weights.py --all

if (-not $Slim) {
    Write-Host "Running test suite..."
    & $py -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Tests failed - environment is set up but broken; investigate before serving."
        exit 1
    }
}

Write-Host ""
Write-Host "Setup complete. Start the API with: powershell scripts/run.ps1"
