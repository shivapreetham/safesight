# SafeSight Demo Guide

Complete guide for presenting SafeSight in interviews, demos, or technical presentations.

## Pre-Demo Checklist

- [ ] API server running: `.venv/Scripts/uvicorn service.app:app --port 8001`
- [ ] Browser ready with tabs: `demo.html`, `http://127.0.0.1:8001/docs`
- [ ] Chrome with extension loaded (chrome://extensions)
- [ ] Have 2-3 sample images ready to test

## Demo Flow (15 minutes)

### 1. Introduction (2 min)

"I built SafeSight, an NSFW image moderation platform for my data science portfolio. It showcases the full ML lifecycle from research to production deployment."

Key points:
- Built on my research model RQSA-MIL (MobileNetV2 + IoU-conditioned attention)
- Paper metrics: 87.9% recall, 85.5% specificity, 93.0% AUC on dense evaluation
- Full-stack: FastAPI backend, Chrome extension, complete MLOps pipeline

### 2. Quick API Demo (3 min)

**Open: demo.html in browser**

"Let me show you the core API in action."

1. Drag/drop an image or use URL
2. Show the result: label, score, strategy stages, patches analyzed
3. Try another image to show caching (instant response)

**Key talking point:**
"The cascade strategy adapts - it exits early on clear cases (full_frame check) but falls through to detailed analysis (340 patches) when needed. This balances accuracy and speed."

### 3. Interactive API Docs (3 min)

**Open: http://127.0.0.1:8001/docs**

"FastAPI auto-generates interactive docs. Let me show the endpoints."

1. Show the 5 main endpoints
2. Try `/v1/score-urls` with sample URLs
3. Show response structure: label, scores, stages, metrics

**Endpoints to highlight:**
- `/v1/score` - Upload scoring
- `/v1/score-urls` - Batch URL scoring (concurrent fetching)
- `/v1/explain` - Heatmap generation (explainability)
- `/metrics` - Prometheus metrics

### 4. Chrome Extension (3 min)

**Open Chrome with extension loaded**

1. Click extension icon, show settings (API URL, sensitivity)
2. Browse to a site with images (e.g., image gallery, news site)
3. Show real-time blurring of NSFW content
4. Click a blurred image to reveal

**Key talking point:**
"The extension demonstrates production readiness - it handles real-world scenarios like CDN CORS, timeout fallbacks, and user privacy (local processing option)."

### 5. Architecture & Technical Deep-Dive (4 min)

**Show/discuss (can use code or diagrams):**

#### Model Architecture
- RQSA-MIL: Multiple Instance Learning with IoU-conditioned attention
- Input: Variable-sized images (normalized to 690px short side)
- Sliding window: 224x224 patches with adjustable density
- Output: Per-patch probabilities + MIL aggregation

#### Production Optimizations
- **ONNX Backend:** 2.5x faster than PyTorch on CPU (measured)
- **Cascade Strategy:** Asymmetric early exits
  - Full-frame check → Fast preset (24 patches) → Accurate preset (340 patches)
  - Measured: ~1s on safe images, ~12s on edge cases
- **Caching:** LRU cache keyed by SHA256(url) + strategy
- **Concurrency:** Async fetching (10 concurrent), serial inference (memory stability)

#### Security & Production Features
- SSRF protection: validates redirects, blocks private IPs
- 10MB size limit, streaming downloads
- Prometheus metrics: latency histograms, score distributions, cache hits
- Logging: JSONL for feedback loop and drift detection

### 6. MLOps Pipeline (if time allows, 2 min)

**Show docker-compose.yml structure:**

"I built a complete MLOps stack:"

1. **MLflow:** Experiment tracking, model versioning
   - Show: http://localhost:5000 (if running)
2. **Prometheus:** Scrapes /metrics endpoint
   - Show: http://localhost:9090 (if running)
3. **Grafana:** Dashboards for model performance monitoring
   - Show: http://localhost:3000 (admin/safesight) (if running)

**Key talking point:**
"The feedback endpoint logs user corrections. In production, this feeds a drift detection pipeline that monitors score distribution shifts and triggers retraining alerts."

### 7. Deployment (1 min)

"I prepared multiple deployment paths:"

- **Hugging Face Spaces:** Docker blueprint ready (weights on HF Hub)
- **Render:** Free tier compatible (torch-free build, <512MB RAM)
- **CI/CD:** GitHub Actions runs tests + builds Docker on every push
- **Model Card:** Published on HF Hub with benchmarks and usage guide

### 8. Q&A Prep

**Common questions:**

**Q: How do you handle false positives?**
A: Two strategies - (1) The `score_large` threshold focuses on large NSFW regions to reduce false positives on small irrelevant patches, (2) User feedback loop logs corrections for model retraining

**Q: What about adversarial attacks?**
A: Current model isn't adversarially hardened. In production, I'd add adversarial training and ensemble with a secondary model (like a hash-based known-image filter)

**Q: Inference speed bottleneck?**
A: CPU-bound. Cascade helps but for high-volume production I'd:
- Use ONNX with int8 quantization on GPU (TensorRT)
- Batch inference with queue system
- Add edge caching (Cloudflare Workers)

**Q: Training data sources?**
A: Research used public benchmarks (NPDI, NudeNet datasets). For production, I'd combine with proprietary data and active learning on edge cases

**Q: Why not use existing APIs (AWS Rekognition, etc.)?**
A: This demonstrates end-to-end ML skills - research, implementation, optimization, deployment. In production, I'd evaluate build vs. buy based on cost, latency, and data privacy needs

## Quick Commands Reference

```bash
# Start API
.venv/Scripts/uvicorn service.app:app --port 8001

# Run demo script
.venv/Scripts/python demo.py

# Run tests
.venv/Scripts/python -m pytest -q

# Benchmarks
.venv/Scripts/python -m ml.benchmark

# Score single image
.venv/Scripts/python -m ml.score_cli --image test.jpg --preset cascade

# Start full MLOps stack
cd mlops && docker-compose up

# Stop background server
# Find task ID with /tasks, then kill it
```

## Demo URLs

- Demo UI: file:///path/to/demo.html
- API Docs: http://127.0.0.1:8001/docs
- Health: http://127.0.0.1:8001/healthz
- Metrics: http://127.0.0.1:8001/metrics
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Files to Have Ready

1. `docs/INTERVIEW.md` - Technical interview playbook
2. `docs/DECISIONS.md` - Architecture decision records
3. `docs/benchmarks.md` - Performance measurements
4. Sample images (safe and NSFW) in `tests/fixtures/`

## Pro Tips

1. **Start with business value:** "Content moderation is a $X billion market, every UGC platform needs it"
2. **Show, don't tell:** Live demos > slides
3. **Admit limitations:** Shows technical maturity
4. **Connect to role:** "In your platform, I'd adapt this to [specific use case]"
5. **Have metrics ready:** All claims backed by benchmarks
6. **Emphasize learning:** "This taught me X about production ML"

## Troubleshooting During Demo

**API won't start:**
- Check port: `netstat -ano | findstr :8001`
- Check weights: `ls weights/` should show rqsa_v1_clean.pt or rqsa_v1.onnx
- Fallback: Use pre-recorded screen recording

**Extension not working:**
- CORS issue: Some CDNs block server-side fetches
- Fallback: "This is a known limitation with Wikimedia/CDNs - in production we'd use client-side fetching or proxy"

**Slow inference:**
- Expected on first run (model loading)
- Cascade should make subsequent calls faster
- Mention: "On GPU this would be 10x faster"

**Demo site down:**
- Have local test images ready
- Use demo.html with file upload instead of URLs
