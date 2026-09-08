#!/usr/bin/env bash
# SafeSight one-shot setup (bash / Git Bash / WSL).
# Creates the venv, installs dependencies, downloads weights from HF Hub,
# and runs the test suite. Run from the repo root:
#   bash scripts/setup.sh [--slim]
#
# --slim installs the torch-free serving runtime only; default installs the
# full dev environment (torch, pytest, ruff, onnx tooling).

set -euo pipefail
cd "$(dirname "$0")/.."

SLIM=0
[ "${1:-}" = "--slim" ] && SLIM=1

if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Windows venvs put python under Scripts/, POSIX under bin/.
if [ -x .venv/Scripts/python.exe ]; then
    PY=.venv/Scripts/python.exe
else
    PY=.venv/bin/python
fi

echo "Installing dependencies (this can take a few minutes)..."
"$PY" -m pip install --quiet --upgrade pip
if [ "$SLIM" = "1" ]; then
    "$PY" -m pip install --quiet --no-cache-dir -r requirements.txt
else
    "$PY" -m pip install --quiet --no-cache-dir -r requirements-dev.txt
fi

echo "Downloading model weights from Hugging Face Hub..."
"$PY" scripts/download_weights.py --all

if [ "$SLIM" = "0" ]; then
    echo "Running test suite..."
    "$PY" -m pytest -q
fi

echo ""
echo "Setup complete. Start the API with: bash scripts/run.sh"
