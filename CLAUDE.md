# SafeSight - project memory for Claude Code

NSFW image moderation platform built on RQSA-MIL, the owner's research model
(MobileNetV2 + IoU-conditioned attention MIL; paper metrics: 87.9% recall,
85.5% specificity, 93.0% AUC on DS1 dense eval). Portfolio project for data
science interviews: FastAPI moderation API + Chrome extension that blurs
NSFW images + full MLOps (MLflow, Prometheus/Grafana, drift, CI/CD).

## Hard rules

- Never use emojis in code (owner's global rule).
- Never commit weights, logs, or `.env` (holds the HF write token; gitignored).
- NSFW training/eval images never enter this repo; the golden-set gate runs
  locally on private data only.
- Keep `docs/INTERVIEW.md` and `docs/DECISIONS.md` in sync with any change
  that alters numbers, trade-offs, or architecture - the owner presents
  these in interviews.
- Every performance claim in docs must come from `python -m ml.benchmark`
  output, not estimates.

## Layout

- `ml/` - numpy-only inference core (`inference.py`), strategies incl.
  cascade + explain heatmaps (`strategies.py`), backends (`onnx_backend.py`
  torch-free, `torch_backend.py` lazy torch), export/benchmark/eval tools,
  torch model definition (`model.py`).
- `service/` - FastAPI app (`app.py`), SSRF-hardened fetcher (`fetcher.py`),
  LRU cache, pydantic-settings (`SAFESIGHT_` env prefix).
- `extension/` - Chrome MV3 extension (blur until verdict, click-to-reveal,
  popup sets API URL + sensitivity).
- `mlops/` - docker-compose (api+mlflow+prometheus+grafana), MLflow backfill,
  drift reports, HF deploy script.
- `tests/` - 31 pytest tests; `docs/` - ADRs, interview playbook, benchmarks.
- `scripts/` - setup/run/download helpers (see below).

## Commands (Windows; venv at .venv)

- Full setup from scratch: `powershell scripts/setup.ps1` (bash: `scripts/setup.sh`)
- Run API: `powershell scripts/run.ps1` (defaults: onnx backend, port 8000)
- Tests: `.venv/Scripts/python -m pytest -q` - must stay green
- Lint: `.venv/Scripts/python -m ruff check .` - must stay clean
- Download weights only: `.venv/Scripts/python scripts/download_weights.py`
- Benchmarks (regenerates docs/benchmarks.md): `.venv/Scripts/python -m ml.benchmark`
- Score one image: `.venv/Scripts/python -m ml.score_cli --image pic.jpg --preset cascade`

## Key design facts

- Scoring strategies: `fast` (~24 patches, ~1s CPU), `accurate` (paper
  settings, ~340 patches, ~12s CPU), `cascade` (default; full_frame -> fast
  -> accurate with asymmetric early exits - see ml/strategies.py docstring).
- Backends implement `score_regions(np.ndarray (N,3,224,224)) -> (N,) probs`.
  Serving runtime is torch-free (requirements.txt); torch comes from
  requirements-torch.txt for dev/export only. ~317MB RSS serving.
- Measured: ONNX fp32 ~2.5x faster than eager torch on CPU; int8 is 3.7x
  smaller but SLOWER here (no fast ConvInteger kernels) - fp32 is the CPU
  default. Threshold knob for clients is `score_large`.
- Verdict cache key is `sha256(url)[:32] + ":" + strategy`.
- URL fetching: concurrent (bounded), every redirect hop revalidated against
  private/loopback IPs, streamed with 10MB cap. Known limitation: some CDNs
  (Wikimedia) 403 server-side fetches; those fail open in the extension.

## Deployment state (2026-07)

- Weights + model card live on HF Hub: `Shivapreetham17/rqsa-mil-v1`
  (public; the container downloads at startup via SAFESIGHT_HF_MODEL_REPO).
- HF Docker Spaces require PRO - not used. Chosen host: Render free tier
  via `render.yaml` blueprint. Pending owner steps: create GitHub repo,
  `git push`, then Render New > Blueprint.
- CI (`.github/workflows/ci.yml`): ruff + pytest + docker build on every
  push; optional HF Space deploy if `HF_TOKEN`/`HF_SPACE_ID` are set.

## Gotchas

- Owner's C: drive runs near-full; prefer `pip install --no-cache-dir` and
  check free space before large installs.
- The checkpoint format is `safesight-rqsa-v1` (tensors-only,
  weights_only=True). Original research checkpoints need
  `python -m ml.convert_checkpoint` first; research repo lives at D:\nude-MIL.
- RQSA needs a pseudo-IoU of 0.5 at inference; backends handle it - never
  call the raw model without it.
- Do not resize images before `predict()`; it normalizes the short side to
  690px itself.
