# SafeSight API Testing Guide

## Quick Start

1. Start the API server (default port 8001):
   ```bash
   .venv/Scripts/uvicorn service.app:app --host 127.0.0.1 --port 8001
   ```

   Or use the scripts (default port 8000):
   ```bash
   powershell scripts/run.ps1
   ```

2. Run the test suite:
   ```bash
   .venv/Scripts/python test_api.py
   ```

## Manual Testing with curl

### 1. Root endpoint (API info)
```bash
curl http://127.0.0.1:8001/
```

### 2. Health check
```bash
curl http://127.0.0.1:8001/healthz
```

### 3. Score image URLs
```bash
curl -X POST http://127.0.0.1:8001/v1/score-urls \
  -H "Content-Type: application/json" \
  -d "{\"urls\": [\"https://picsum.photos/200/300\"], \"preset\": \"fast\"}"
```

### 4. Score uploaded image
```bash
curl -X POST http://127.0.0.1:8001/v1/score?preset=fast \
  -F "file=@path/to/image.jpg"
```

### 5. Get explanation heatmap
```bash
curl -X POST http://127.0.0.1:8001/v1/explain?preset=fast \
  -F "file=@path/to/image.jpg" \
  --output heatmap.png
```

### 6. View metrics
```bash
curl http://127.0.0.1:8001/metrics
```

### 7. View interactive API docs
Open in browser: http://127.0.0.1:8001/docs

## Test with Python

### Using httpx (already installed)
```python
import httpx

client = httpx.Client()

response = client.get("http://127.0.0.1:8001/healthz")
print(response.json())

response = client.post(
    "http://127.0.0.1:8001/v1/score-urls",
    json={"urls": ["https://picsum.photos/200/300"], "preset": "fast"}
)
print(response.json())

client.close()
```

### Score a local image
```python
import httpx

with open("image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = httpx.post(
        "http://127.0.0.1:8001/v1/score?preset=fast",
        files=files,
        timeout=30.0
    )
    print(response.json())
```

## Chrome Extension Testing

1. Load the extension in Chrome:
   - Open `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the `extension/` folder

2. Configure the extension:
   - Click the extension icon
   - Set API URL to: `http://127.0.0.1:8001`
   - Choose sensitivity level

3. Browse any website with images - NSFW images will be automatically blurred

## Expected Results

### Healthy API response:
```json
{
  "status": "ok",
  "device": "cpu",
  "backend": "onnx",
  "default_preset": "cascade",
  "cache_entries": 0
}
```

### Score response:
```json
{
  "label": 0,
  "nsfw": false,
  "score": 0.0234,
  "score_large": 0.0123,
  "preset": "fast",
  "stages": ["full_frame"],
  "n_patches": 24
}
```

### Score-URLs response:
```json
{
  "results": [
    {
      "url": "https://example.com/image.jpg",
      "ok": true,
      "label": 0,
      "score": 0.0234,
      "score_large": 0.0123,
      "stages": ["full_frame"],
      "cached": false
    }
  ]
}
```

## Performance Notes

- `fast` preset: ~1s per image, ~24 patches
- `accurate` preset: ~12s per image, ~340 patches
- `cascade` preset (default): adaptive, exits early on clear verdicts
- URL fetching is concurrent (max 10 simultaneous)
- Results are cached by URL + strategy

## Troubleshooting

### 404 Not Found at root
- You need the latest code with the root endpoint
- Restart the server after updating code

### Connection refused
- Make sure the server is running: `powershell scripts/run.ps1`
- Check the port is 8000: `netstat -ano | findstr :8000`

### Model loading errors
- Run `python scripts/download_weights.py` to get weights from HF Hub
- Check `weights/` folder exists

### Out of memory
- Use `preset=fast` instead of `accurate`
- The cascade preset already optimizes memory usage
