#!/usr/bin/env bash
# Start the SafeSight API (bash / Git Bash / WSL). Run from the repo root:
#   bash scripts/run.sh [backend] [port] [preset]
# Defaults: onnx 8000 cascade

set -euo pipefail
cd "$(dirname "$0")/.."

export SAFESIGHT_BACKEND="${1:-onnx}"
PORT="${2:-8000}"
export SAFESIGHT_PRESET="${3:-cascade}"

if [ -x .venv/Scripts/python.exe ]; then
    PY=.venv/Scripts/python.exe
else
    PY=.venv/bin/python
fi

echo "SafeSight API starting: backend=$SAFESIGHT_BACKEND preset=$SAFESIGHT_PRESET port=$PORT"
echo "Health check: http://127.0.0.1:$PORT/healthz"
"$PY" -m uvicorn service.app:app --host 127.0.0.1 --port "$PORT"
