# SafeSight

Production NSFW image moderation platform built on **RQSA-MIL** (Region-Quality
Supervised Attention for Multiple Instance Learning), a research model with
IoU-conditioned spatial attention over a MobileNetV2 backbone. Ships as a
FastAPI microservice, a Chrome extension that blurs NSFW images in real time,
and a full MLOps stack (MLflow, Prometheus, Grafana, drift detection, CI/CD).

## Model

RQSA-MIL conditions a spatial attention gate on region quality (IoU with
annotations during training, pseudo-IoU at inference), trained with weighted
multiple-instance learning. Dense sliding-window inference scans images at the
scale the model was trained on.

Dense evaluation on the DS1 test set:

| Model | Recall | Specificity | AUC | TPR@1%FPR |
|---|---|---|---|---|
| **RQSA-MIL (this service)** | **87.9%** | 85.5% | **93.0%** | 48.7% |
| SAG-WMIL | 80.0% | 86.7% | 90.2% | 32.4% |
| WMIL no attention | 69.9% | 92.5% | 88.1% | 27.7% |
| GoogLeNet BCE | 47.0% | 98.3% | 89.9% | 38.8% |
| MobileNetV2 BCE | 100.0% | 27.2% | 87.6% | 0.0% |

## Quick start

One command from a fresh clone (creates the venv, installs dependencies,
downloads weights from HF Hub, runs the tests):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1   # Windows
bash scripts/setup.sh                                        # bash / WSL
```

Then start the API:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1     # Windows
bash scripts/run.sh                                          # bash / WSL
```

Options: `run.ps1 -Backend torch -Port 8000 -Preset cascade` (bash:
`run.sh onnx 8000 cascade`). Use `setup.ps1 -Slim` / `setup.sh --slim` for
the torch-free serving runtime only.

Score an image:

```bash
curl -X POST http://127.0.0.1:8000/v1/score -F "file=@photo.jpg"
curl -X POST http://127.0.0.1:8000/v1/score-urls \
     -H "Content-Type: application/json" \
     -d '{"urls": ["https://example.com/image.jpg"]}'
```

## Chrome extension

1. Open `chrome://extensions`, enable Developer mode, click "Load unpacked",
   and select the `extension/` folder.
2. Click the SafeSight icon, set the API URL (local server or the deployed
   Space), and save.
3. Browse. Images are blurred until verified; NSFW images stay blurred with a
   click-to-reveal badge. Revealing an image records feedback for the
   retraining queue.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/demo` | GET | web console: score by URL/upload, evidence heatmaps, live stats |
| `/healthz` | GET | liveness + model/backend/device status |
| `/v1/stats` | GET | JSON aggregates (volume, NSFW rate, cache, latency, cascade exits) |
| `/v1/score` | POST | score one uploaded image (multipart) |
| `/v1/score-urls` | POST | fetch and score a batch of image URLs |
| `/v1/explain` | POST | region-score heatmap PNG showing why the verdict fired |
| `/v1/feedback` | POST | record a user correction |
| `/metrics` | GET | Prometheus metrics |

Auth: set `SAFESIGHT_API_KEY` to require an `X-API-Key` header on `/v1/*`
scoring endpoints (health, demo, and stats stay open). Empty key = auth off
for local demos. The extension popup and demo console both have key fields.

Feedback loop: user corrections land in `logs/feedback.jsonl`;
`python mlops/review_feedback.py` summarizes disagreements and exports
`mlops/retraining_queue.csv` for the next labeling/training round.

Responses include `score` (UI confidence, max of both window tiers),
`score_large` (large-tier score, the recommended single-knob threshold),
and `stages` (which cascade stages ran).

URL batches are fetched concurrently (bounded) through an SSRF-hardened
fetcher: public hosts only, every redirect hop re-validated, streamed
reads with a hard byte cap, image content types only - each defense unit
tested. Verdicts are cached per URL + strategy.

## Inference strategies

Every strategy is a different answer to "which regions do we score, and when
do we stop?" - full rationale in [ml/strategies.py](ml/strategies.py) and
[docs/DECISIONS.md](docs/DECISIONS.md):

| Strategy | Patches (800x600) | Use |
|---|---|---|
| `fast` | ~24 | interactive serving, extension |
| `accurate` | ~340 | evaluation; the published numbers |
| `cascade` (default) | 1 to ~365, adaptive | production: early-exits obvious images, escalates ambiguous ones |

The cascade runs full-frame -> fast -> accurate with asymmetric exits: an
extreme full-frame score exits NSFW immediately, but a low full-frame score
is trusted only on small images (where the full frame matches training
scale); the fast tier escalates only inside its 0.35-0.65 uncertainty band.
Measured latencies: [docs/benchmarks.md](docs/benchmarks.md).

Serving backends: `torch` (default, GPU-capable) or `onnx`
(`SAFESIGHT_BACKEND=onnx`) - measured ~3x faster than eager PyTorch on
CPU-only hosts. An int8 quantized variant exists (3.7x smaller) but
benchmarked slower than fp32 on CPUs without fast ConvInteger kernels,
so fp32 ONNX is the CPU default. See [docs/benchmarks.md](docs/benchmarks.md).

## MLOps stack

```bash
cd mlops
docker compose up --build
```

- API: http://localhost:8000
- MLflow: http://localhost:5000 (`python mlops/mlflow_backfill.py` registers
  the model lineage and paper metrics)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (provisioned SafeSight dashboard: RPS,
  latency percentiles, score-distribution heatmap, verdict rates)

Drift detection: `python mlops/drift_report.py` compares the live score
distribution in `logs/scores.jsonl` against a reference window (Evidently if
installed, built-in KS test otherwise) and writes an HTML report.

Release gate: `python -m ml.eval_golden --golden-dir <labeled set>` fails the
build if recall or specificity regress below thresholds.

## Deployment

Weights live in a public HF Hub model repo
([Shivapreetham17/rqsa-mil-v1](https://huggingface.co/Shivapreetham17/rqsa-mil-v1),
uploaded by `mlops/deploy_hf.py`); the container downloads them at startup,
so no weights ship in git or the image.

The serving image is **torch-free** (ONNX backend, numpy inference pipeline):
it fits 512MB-RAM free hosting with measured ~317MB RSS under load. Install
`requirements-torch.txt` on top only for GPU serving, checkpoint conversion,
ONNX export, and benchmarks.

**Render (free):** push this repo to GitHub, then on render.com choose
New > Blueprint and select the repo - `render.yaml` provisions the service
(Docker, free plan, health checks, env vars preconfigured). Free instances
sleep after 15 idle minutes; first request after sleep takes ~1 min.

**HF Spaces (requires PRO):** `python mlops/deploy_hf.py` creates the Space
and uploads the code; CI (`HF_TOKEN` secret + `HF_SPACE_ID` variable) keeps
it deployed on every push to main.

## Repository layout

```
ml/         model, inference strategies, ONNX export/backend, eval gate, benchmarks
service/    FastAPI app, settings, URL cache
extension/  Chrome extension (Manifest V3)
mlops/      docker-compose stack, MLflow backfill, drift reports, dashboards
tests/      pytest suite (model, inference, strategies, API contract)
docs/       architecture, decision records (ADRs), benchmarks, interview playbook
```
