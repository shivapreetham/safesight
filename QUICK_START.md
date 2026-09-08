# SafeSight - Quick Start Guide

Get SafeSight running in 5 minutes for demos and presentations.

## Prerequisites

- Python 3.11+
- Git
- 5GB free disk space

## One-Command Setup

```bash
# Windows (PowerShell)
powershell scripts/setup.ps1

# Linux/Mac
bash scripts/setup.sh
```

This will:
1. Create virtual environment
2. Install dependencies
3. Download model weights from Hugging Face Hub
4. Run tests to verify everything works

## Running the Demo

### Option 1: Quick Demo (Recommended)

```bash
# Terminal 1: Start API
.venv/Scripts/uvicorn service.app:app --port 8001

# Terminal 2: Run demo script
.venv/Scripts/python demo.py
```

### Option 2: Web UI Demo

```bash
# Start API
.venv/Scripts/uvicorn service.app:app --port 8001

# Open demo.html in your browser
start demo.html  # Windows
open demo.html   # Mac
```

### Option 3: Chrome Extension Demo

```bash
# Start API
.venv/Scripts/uvicorn service.app:app --port 8001

# Then in Chrome:
# 1. Go to chrome://extensions
# 2. Enable "Developer mode"
# 3. Click "Load unpacked"
# 4. Select the extension/ folder
# 5. Click the extension icon and set API URL to http://127.0.0.1:8001
```

## Verify Everything Works

```bash
# Run full test suite
.venv/Scripts/python test_api.py

# Expected output:
# [PASS] Root endpoint
# [PASS] Health check
# [PASS] Score URLs
# [PASS] Score upload
# [PASS] Metrics
# Passed: 5/5
```

## Quick Test

```bash
# Score a single image
.venv/Scripts/python -m ml.score_cli --image path/to/image.jpg --preset cascade

# Health check
curl http://127.0.0.1:8001/healthz
```

## Production Stack (Optional)

Run the full MLOps stack with Docker:

```bash
cd mlops
docker-compose up --build

# Access:
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/safesight)
```

## Troubleshooting

### Port 8001 already in use

```bash
# Use a different port
.venv/Scripts/uvicorn service.app:app --port 8002

# Update test_api.py:
# Change API_BASE = "http://127.0.0.1:8002"
```

### Model weights not found

```bash
# Manually download weights
.venv/Scripts/python scripts/download_weights.py
```

### Out of memory

```bash
# Use ONNX backend (lighter)
$env:SAFESIGHT_BACKEND="onnx"
.venv/Scripts/uvicorn service.app:app --port 8001
```

### Tests failing

```bash
# Check server is running
curl http://127.0.0.1:8001/healthz

# Run pytest
.venv/Scripts/python -m pytest -v

# Check logs
# Server logs show in the terminal where uvicorn is running
```

## Files Overview

```
safesight/
├── demo.py              # Interactive CLI demo script
├── demo.html            # Web UI for live testing
├── test_api.py          # Automated API tests
├── DEMO_GUIDE.md        # Complete presentation guide
├── TESTING.md           # Testing documentation
├── service/app.py       # FastAPI application
├── ml/                  # Model inference code
├── extension/           # Chrome extension
├── weights/             # Model weights (downloaded)
├── mlops/              # Docker stack configs
└── docs/               # Architecture & interview docs
```

## Next Steps

1. Read [DEMO_GUIDE.md](DEMO_GUIDE.md) for presentation tips
2. Check [TESTING.md](TESTING.md) for API testing examples
3. Review [docs/INTERVIEW.md](docs/INTERVIEW.md) for technical talking points
4. Open http://127.0.0.1:8001/docs for interactive API documentation

## Quick Demo Flow

1. Start server: `.venv/Scripts/uvicorn service.app:app --port 8001`
2. Open `demo.html` in browser
3. Upload an image or paste a URL
4. See results: SAFE/NSFW label, confidence scores, strategy used
5. Open http://127.0.0.1:8001/docs to explore all endpoints

That's it! You're ready to demo SafeSight.
