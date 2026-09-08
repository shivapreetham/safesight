# Start the SafeSight API (Windows PowerShell). Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts/run.ps1 [-Backend onnx|torch] [-Port 8000] [-Preset cascade|fast|accurate]

param(
    [string]$Backend = "onnx",
    [int]$Port = 8000,
    [string]$Preset = "cascade"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:SAFESIGHT_BACKEND = $Backend
$env:SAFESIGHT_PRESET = $Preset

Write-Host "SafeSight API starting: backend=$Backend preset=$Preset port=$Port"
Write-Host "Health check: http://127.0.0.1:$Port/healthz"
& .\.venv\Scripts\python.exe -m uvicorn service.app:app --host 127.0.0.1 --port $Port
