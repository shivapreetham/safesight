"""SafeSight moderation API.

Endpoints:
    GET  /healthz          liveness + model/backend status
    POST /v1/score         score one uploaded image (multipart)
    POST /v1/score-urls    fetch and score a batch of image URLs (concurrent)
    POST /v1/explain       region-score heatmap PNG for an uploaded image
    POST /v1/feedback      record a user correction for the feedback loop
    GET  /metrics          Prometheus metrics
"""

import asyncio
import io
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import anyio
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from ml.inference import PRESETS, Verdict, load_model, predict
from ml.strategies import explain, predict_cascade
from service.cache import LRUCache, url_key
from service.fetcher import USER_AGENT, FetchError, fetch_image
from service.settings import settings
from service.stats import ServiceStats

STRATEGIES = set(PRESETS) | {"cascade"}
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

logger = logging.getLogger("safesight")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SCORE_HISTOGRAM = Histogram(
    "safesight_image_score", "Distribution of model scores on served images",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
NSFW_COUNTER = Counter("safesight_nsfw_total", "Images flagged NSFW")
SAFE_COUNTER = Counter("safesight_safe_total", "Images scored safe")
FETCH_ERRORS = Counter("safesight_fetch_errors_total", "Image URL fetch failures")
CACHE_HITS = Counter("safesight_cache_hits_total", "URL cache hits")
INFER_SECONDS = Histogram("safesight_inference_seconds", "Per-image inference latency")

_state = {"model": None}
_model_lock = threading.Lock()
_log_lock = threading.Lock()
cache = LRUCache(settings.cache_size)
stats = ServiceStats()


def require_api_key(x_api_key: str | None = Header(default=None)):
    """Shared-secret gate for /v1/* endpoints; disabled when no key is set."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _ensure_checkpoint() -> str:
    path = settings.checkpoint_path
    if os.path.exists(path):
        return path
    if settings.hf_model_repo:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading checkpoint from HF Hub repo %s", settings.hf_model_repo)
        return hf_hub_download(repo_id=settings.hf_model_repo,
                               filename=settings.hf_model_filename)
    raise FileNotFoundError(
        f"Checkpoint not found at {path} and SAFESIGHT_HF_MODEL_REPO is not set")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.backend == "onnx":
        from ml.onnx_backend import OnnxScorer
        onnx_path = settings.onnx_model_path
        if not os.path.exists(onnx_path) and settings.hf_model_repo:
            from huggingface_hub import hf_hub_download
            logger.info("Downloading ONNX model from HF Hub repo %s", settings.hf_model_repo)
            onnx_path = hf_hub_download(repo_id=settings.hf_model_repo,
                                        filename=settings.hf_onnx_filename)
        logger.info("Loading ONNX model %s", onnx_path)
        _state["model"] = OnnxScorer(onnx_path)
    else:
        ckpt = _ensure_checkpoint()
        logger.info("Loading RQSA-MIL checkpoint %s", ckpt)
        _state["model"] = load_model(ckpt, settings.device)
    logger.info("Model ready (backend=%s, device=%s)",
                settings.backend, _state["model"].device)
    yield


app = FastAPI(title="SafeSight Moderation API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class ScoreUrlsRequest(BaseModel):
    urls: list[str] = Field(min_length=1)
    preset: str | None = None


class UrlResult(BaseModel):
    url: str
    ok: bool
    label: int | None = None
    score: float | None = None
    score_large: float | None = None
    stages: list[str] = []
    cached: bool = False
    error: str | None = None


class FeedbackRequest(BaseModel):
    url_hash: str
    model_score: float
    user_label: int = Field(ge=0, le=1)
    comment: str | None = None


def _append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _record_verdict(verdict: Verdict, source: str, key: str | None = None) -> None:
    SCORE_HISTOGRAM.observe(verdict.score)
    if verdict.label == 1:
        NSFW_COUNTER.inc()
    else:
        SAFE_COUNTER.inc()
    _append_jsonl(settings.score_log_path, {
        "ts": time.time(),
        "source": source,
        "url_hash": key,
        "label": verdict.label,
        "score": round(verdict.score, 4),
        "score_large": round(verdict.score_large, 4),
        "preset": verdict.preset,
        "n_patches": verdict.n_patches,
    })


def _score_image(img: Image.Image, strategy: str) -> Verdict:
    # One image at a time through the GPU/CPU; sliding windows already batch
    # internally, and serializing avoids memory spikes under load.
    with _model_lock:
        start = time.perf_counter()
        if strategy == "cascade":
            verdict = predict_cascade(_state["model"], img)
        else:
            verdict = predict(_state["model"], img, preset=strategy)
        elapsed = time.perf_counter() - start
        INFER_SECONDS.observe(elapsed)
    stats.record(verdict.label, elapsed, verdict.stages)
    return verdict


def _validate_strategy(preset: str | None) -> str:
    name = preset or settings.preset
    if name not in STRATEGIES:
        raise HTTPException(status_code=422, detail=f"unknown preset: {name}")
    return name


@app.get("/")
def root():
    return {
        "service": "SafeSight Moderation API",
        "version": "1.0.0",
        "demo": "/demo",
        "docs": "/docs",
        "health": "/healthz",
        "stats": "/v1/stats",
        "endpoints": {
            "POST /v1/score": "Score an uploaded image",
            "POST /v1/score-urls": "Score a batch of image URLs",
            "POST /v1/explain": "Get heatmap for an image",
            "POST /v1/feedback": "Submit user feedback",
            "GET /metrics": "Prometheus metrics",
        }
    }


@app.get("/healthz")
def healthz():
    model = _state["model"]
    return {
        "status": "ok" if model is not None else "loading",
        "device": str(model.device) if model else None,
        "backend": settings.backend,
        "default_preset": settings.preset,
        "cache_entries": len(cache),
    }


def _read_upload(data: bytes) -> Image.Image:
    if len(data) > settings.fetch_max_bytes:
        raise HTTPException(status_code=413, detail="file too large")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="not a decodable image") from exc


@app.get("/demo")
def demo_page():
    return FileResponse(os.path.join(STATIC_DIR, "demo.html"), media_type="text/html")


@app.get("/v1/stats")
def service_stats():
    return stats.snapshot(cache.hits, cache.misses)


@app.post("/v1/score", dependencies=[Depends(require_api_key)])
async def score_upload(file: UploadFile, preset: str | None = None):
    strategy = _validate_strategy(preset)
    img = _read_upload(await file.read())

    verdict = await anyio.to_thread.run_sync(_score_image, img, strategy)
    _record_verdict(verdict, source="upload")
    return {
        "label": verdict.label,
        "nsfw": bool(verdict.label),
        "score": round(verdict.score, 4),
        "score_large": round(verdict.score_large, 4),
        "preset": verdict.preset,
        "stages": verdict.stages,
        "n_patches": verdict.n_patches,
    }


@app.post("/v1/explain", dependencies=[Depends(require_api_key)])
async def explain_upload(file: UploadFile, preset: str | None = None):
    """Return the image with a region-score heatmap overlay (PNG).

    Shows which windows drove the verdict; the score is in response headers.
    """
    name = preset or "fast"
    if name not in PRESETS:
        raise HTTPException(status_code=422, detail=f"unknown preset: {name}")
    img = _read_upload(await file.read())

    def _run():
        with _model_lock:
            return explain(_state["model"], img, preset=name)

    result = await anyio.to_thread.run_sync(_run)
    return Response(
        content=result.heatmap_png,
        media_type="image/png",
        headers={
            "X-SafeSight-Label": str(result.verdict.label),
            "X-SafeSight-Score": f"{result.verdict.score:.4f}",
            "X-SafeSight-Max-Window": f"{result.max_window_score:.4f}",
        },
    )


@app.post("/v1/score-urls", dependencies=[Depends(require_api_key)])
async def score_urls(req: ScoreUrlsRequest):
    strategy = _validate_strategy(req.preset)
    urls = req.urls[:settings.max_urls_per_request]

    # Phase 1: resolve cache hits and fetch the misses concurrently
    # (bounded). Fetching is pure I/O and parallelizes; inference is
    # CPU-bound and stays serialized behind the model lock in phase 2.
    fetched: dict[str, Image.Image | FetchError] = {}
    cached_payloads: dict[str, dict] = {}
    semaphore = asyncio.Semaphore(settings.fetch_concurrency)

    async def fetch_one(client: httpx.AsyncClient, url: str) -> None:
        async with semaphore:
            try:
                fetched[url] = await fetch_image(
                    client, url,
                    timeout=settings.fetch_timeout_seconds,
                    max_bytes=settings.fetch_max_bytes,
                    min_side=settings.min_image_side)
            except FetchError as exc:
                fetched[url] = exc
            except Exception as exc:
                # One malformed URL must fail its own entry, never the batch.
                fetched[url] = FetchError(str(exc)[:200])

    to_fetch = []
    for url in dict.fromkeys(urls):  # de-duplicate, preserve order
        hit = cache.get(f"{url_key(url)}:{strategy}")
        if hit is not None:
            CACHE_HITS.inc()
            cached_payloads[url] = hit
        else:
            to_fetch.append(url)

    if to_fetch:
        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
            await asyncio.gather(*(fetch_one(client, u) for u in to_fetch))

    # Phase 2: score fetched images and assemble results in request order.
    results: list[UrlResult] = []
    scored: dict[str, UrlResult] = {}
    for url in urls:
        if url in scored:
            results.append(scored[url])
            continue

        if url in cached_payloads:
            result = UrlResult(url=url, ok=True, cached=True, **cached_payloads[url])
        else:
            outcome = fetched.get(url)
            if isinstance(outcome, FetchError):
                FETCH_ERRORS.inc()
                stats.record_error()
                result = UrlResult(url=url, ok=False, error=str(outcome)[:200])
            else:
                key = url_key(url)
                verdict = await anyio.to_thread.run_sync(_score_image, outcome, strategy)
                _record_verdict(verdict, source="url", key=key)
                payload = {
                    "label": verdict.label,
                    "score": round(verdict.score, 4),
                    "score_large": round(verdict.score_large, 4),
                    "stages": verdict.stages,
                }
                cache.put(f"{key}:{strategy}", payload)
                result = UrlResult(url=url, ok=True, cached=False, **payload)

        scored[url] = result
        results.append(result)

    return {"results": [r.model_dump() for r in results]}


@app.post("/v1/feedback", dependencies=[Depends(require_api_key)])
def feedback(req: FeedbackRequest):
    _append_jsonl(settings.feedback_log_path, {
        "ts": time.time(),
        "url_hash": req.url_hash,
        "model_score": req.model_score,
        "user_label": req.user_label,
        "comment": (req.comment or "")[:500],
    })
    return {"status": "recorded"}
