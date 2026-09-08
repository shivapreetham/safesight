"""Benchmark inference strategies and serving backends, write docs/benchmarks.md.

Two questions this answers with measured numbers:
1. What does each inference strategy cost, and what does cascade save?
2. What does ONNX (and int8 quantization) buy over eager PyTorch on CPU?

Usage:
    python -m ml.benchmark --checkpoint weights/rqsa_v1_clean.pt \
        --onnx weights/rqsa_v1.onnx --onnx-int8 weights/rqsa_v1_int8.onnx
"""

import argparse
import os
import platform
import statistics
import time

import numpy as np
import torch
from PIL import Image

from ml.inference import predict
from ml.strategies import predict_cascade, score_full_frame
from ml.torch_backend import TorchScorer


def synthetic_photo(width: int, height: int, seed: int) -> Image.Image:
    """Deterministic photo-like image: smooth low-frequency color fields."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(0, 255, size=(height // 32 + 2, width // 32 + 2, 3))
    img = Image.fromarray(small.astype(np.uint8)).resize((width, height), Image.BILINEAR)
    return img


def time_call(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def bench_strategies(model, img: Image.Image) -> list[dict]:
    rows = []

    ms = time_call(lambda: score_full_frame(model, img))
    rows.append({"strategy": "full_frame", "patches": 1, "ms": ms})

    for preset in ("fast", "accurate"):
        verdict = predict(model, img, preset=preset)
        ms = time_call(lambda p=preset: predict(model, img, preset=p))
        rows.append({"strategy": preset, "patches": verdict.n_patches, "ms": ms})

    verdict = predict_cascade(model, img)
    ms = time_call(lambda: predict_cascade(model, img))
    rows.append({"strategy": f"cascade ({'>'.join(verdict.stages)})",
                 "patches": verdict.n_patches, "ms": ms})
    return rows


def bench_backends(torch_scorer, onnx_paths: dict, batch: int = 24) -> list[dict]:
    rng = np.random.default_rng(0)
    regions = rng.standard_normal((batch, 3, 224, 224)).astype(np.float32)
    rows = []

    ms = time_call(lambda: torch_scorer.score_regions(regions), repeats=5)
    rows.append({"backend": "pytorch eager (fp32)", "ms": ms})

    from ml.onnx_backend import OnnxScorer
    for name, path in onnx_paths.items():
        if not path or not os.path.exists(path):
            continue
        scorer = OnnxScorer(path)
        ms = time_call(lambda s=scorer: s.score_regions(regions), repeats=5)
        size_mb = os.path.getsize(path) / 1e6
        rows.append({"backend": f"{name} ({size_mb:.0f} MB)", "ms": ms})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="weights/rqsa_v1_clean.pt")
    parser.add_argument("--onnx", default="weights/rqsa_v1.onnx")
    parser.add_argument("--onnx-int8", default="weights/rqsa_v1_int8.onnx")
    parser.add_argument("--out", default="docs/benchmarks.md")
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    model = TorchScorer(args.checkpoint, device="cpu")

    images = {
        "small 320x240": synthetic_photo(320, 240, seed=1),
        "medium 800x600": synthetic_photo(800, 600, seed=2),
        "large 1920x1440": synthetic_photo(1920, 1440, seed=3),
    }

    lines = [
        "# Measured benchmarks",
        "",
        f"CPU-only, single request at a time. Host: {platform.processor() or platform.machine()}, "
        f"{os.cpu_count()} logical cores, torch {torch.__version__}. "
        "Median of repeated runs on deterministic synthetic photos.",
        "",
        "## Inference strategies (PyTorch backend)",
        "",
    ]

    for name, img in images.items():
        lines += [f"### {name}", "", "| Strategy | Patches | Latency (ms) |", "|---|---|---|"]
        for row in bench_strategies(model, img):
            lines.append(f"| {row['strategy']} | {row['patches']} | {row['ms']:.0f} |")
        lines.append("")

    lines += [
        "## Serving backends (batch of 24 patches, the fast-preset workload)",
        "",
        "| Backend | Latency (ms) |",
        "|---|---|",
    ]
    onnx_paths = {"onnxruntime fp32": args.onnx, "onnxruntime int8": args.onnx_int8}
    for row in bench_backends(model, onnx_paths):
        lines.append(f"| {row['backend']} | {row['ms']:.0f} |")
    lines += [
        "",
        "Interpretation: the cascade keeps median latency near the fast preset",
        "while reserving paper-grade accuracy for ambiguous images. Compare the",
        "backend rows before choosing a CPU serving backend: quantized kernels",
        "are not universally faster (int8 ConvInteger paths lack fast",
        "implementations on some CPUs), so the measured number on the target",
        "hardware decides, not the textbook expectation.",
        "",
    ]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
