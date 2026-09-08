"""Inference strategies for RQSA-MIL, from cheapest to most thorough.

The model scores 224x224 regions; every strategy below is a different answer
to "which regions of the image do we score, and when do we stop?". The
options, and why each exists:

1. full_frame (1 patch)
   Resize the whole image to 224x224 and score once. This is what a plain
   classifier deployment does. It is nearly free, but on high-resolution
   images nudity occupying a small fraction of the frame shrinks below the
   scale the model was trained on (200-330px crops of a 690px image) and is
   missed. Kept as the cheap first look and as the honest baseline that
   motivates everything else.

2. fast (single-tier sliding window, ~24 patches on 800x600)
   Slide one 265px window (the middle of the training scale range) with 50%
   overlap on the 690px-normalized image. Recovers the scale-matching that
   full_frame loses, at interactive CPU latency. No TTA, no small tier.

3. accurate (paper settings, ~340 patches on 800x600)
   Three large tiers (200/265/330px) + a strict 140px small tier for
   thumbnails/grid cells, fixed pyramid crops, horizontal-flip TTA, and
   Gaussian soft-NMS so overlapping windows do not vote twice. This is the
   configuration behind the published numbers (87.9% recall / 93.0% AUC).
   Roughly 12x slower than fast on CPU.

4. cascade (adaptive: 1 to ~365 patches)
   Spend compute only where the image is ambiguous:

       full_frame --confident--> verdict
            | uncertain
            v
          fast    --confident--> verdict
            | borderline
            v
         accurate --------------> verdict

   Escalation rules are deliberately asymmetric:
   - full_frame may exit NSFW when its score is extreme (>= 0.85): if
     nudity dominates a downscaled frame the model is emphatic, and a
     denser scan would only agree.
   - full_frame may exit SAFE only for small images (short side <= 400px
     after normalization) with a near-zero score (<= 0.05). For small
     images the full frame IS the training scale, so the shortcut is
     sound; for large images a low full-frame score proves nothing (the
     exact failure mode full_frame has), so we never trust it and fall
     through to fast.
   - fast escalates to accurate only inside the uncertainty band around
     its 0.50 decision threshold (0.35..0.65). Outside the band, more
     windows have never been observed to flip the verdict; inside it, the
     small tier and TTA are exactly the extra evidence that resolves
     borderline crops.

   Result: obvious content (most of real traffic) costs 1-25 patches,
   and only genuinely ambiguous images pay the full paper-grade price.

Alternatives considered and rejected:
- Object detector (YOLO-style) on full frames: needs dense box labels at
  deployment scale; the training data has boxes only on one dataset, and
  MIL was the point of the research (weak bag-level supervision).
- Random crop sampling: cheaper than sliding windows but non-deterministic;
  verdicts must be reproducible for moderation audits and caching.
- Downscale-only multi-resolution pyramid on the full frame: still scores
  whole frames, so small regions stay sub-scale; sliding windows fix the
  region-to-input ratio directly, which is what the model expects.
"""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from .inference import (
    NORM_SHORT,
    PRESETS,
    Verdict,
    _score_patches,
    _slide_windows,
    predict,
    resize_keep_aspect,
    to_region,
)

# Cascade exit thresholds. Chosen against the calibrated fast-tier decision
# threshold (0.50): the band is symmetric around it, and the full-frame exits
# demand near-certainty because that stage sees the least evidence.
FULL_FRAME_NSFW_EXIT = 0.85
FULL_FRAME_SAFE_EXIT = 0.05
FULL_FRAME_SAFE_MAX_SIDE = 400
FAST_UNCERTAIN_LOW = 0.35
FAST_UNCERTAIN_HIGH = 0.65


@dataclass
class Explanation:
    verdict: Verdict
    heatmap_png: bytes      # original image with region-score overlay
    max_window_score: float


def score_full_frame(scorer, img: Image.Image) -> float:
    """Score the whole image as a single 224x224 region."""
    patch = to_region(img.convert("RGB"))[None, ...]
    probs = scorer.score_regions(patch)
    return float(probs[0])


def predict_cascade(scorer, img: Image.Image) -> Verdict:
    """Adaptive early-exit inference. See module docstring for the rules."""
    img = img.convert("RGB")
    normalized = resize_keep_aspect(img, NORM_SHORT)

    # Stage 1: full frame (1 patch).
    ff_score = score_full_frame(scorer, normalized)
    if ff_score >= FULL_FRAME_NSFW_EXIT:
        return Verdict(label=1, score=ff_score, score_large=ff_score,
                       n_patches=1, preset="cascade", stages=["full_frame"])
    if ff_score <= FULL_FRAME_SAFE_EXIT and min(normalized.size) <= FULL_FRAME_SAFE_MAX_SIDE:
        return Verdict(label=0, score=ff_score, score_large=ff_score,
                       n_patches=1, preset="cascade", stages=["full_frame"])

    # Stage 2: fast sliding window.
    fast_verdict = predict(scorer, img, preset="fast")
    total = 1 + fast_verdict.n_patches
    if not (FAST_UNCERTAIN_LOW < fast_verdict.score_large < FAST_UNCERTAIN_HIGH):
        return Verdict(label=fast_verdict.label, score=fast_verdict.score,
                       score_large=fast_verdict.score_large, n_patches=total,
                       preset="cascade", stages=["full_frame", "fast"])

    # Stage 3: paper-grade dense scan for the ambiguous remainder.
    acc = predict(scorer, img, preset="accurate")
    return Verdict(label=acc.label, score=acc.score, score_large=acc.score_large,
                   n_patches=total + acc.n_patches, preset="cascade",
                   stages=["full_frame", "fast", "accurate"])


def explain(scorer, img: Image.Image, preset: str = "fast") -> Explanation:
    """Render which regions drove the verdict as a heatmap overlay.

    Window probabilities are max-pooled onto a canvas at each window's
    location, blurred for continuity, and composited in red over the image.
    This visualizes the actual decision evidence (per-window scores), not a
    saliency approximation, so the overlay is faithful to what the score
    aggregation saw.
    """
    cfg = PRESETS[preset]
    img = resize_keep_aspect(img.convert("RGB"), NORM_SHORT)
    W, H = img.size

    windows = cfg.windows_large + cfg.windows_small
    patches, boxes = _slide_windows(img, windows, cfg.stride_ratio)
    if not patches:
        patches = [to_region(img)]
        boxes = [(0, 0, W, H)]
    probs = _score_patches(patches, scorer, boxes=None,
                           batch_size=cfg.batch_size)

    canvas = np.zeros((H, W), dtype=np.float32)
    for (x1, y1, x2, y2), p in zip(boxes, probs.tolist(), strict=True):
        canvas[y1:y2, x1:x2] = np.maximum(canvas[y1:y2, x1:x2], p)

    heat = Image.fromarray((canvas * 255).astype(np.uint8), mode="L")
    heat = heat.filter(ImageFilter.GaussianBlur(radius=16))
    heat_np = np.asarray(heat, dtype=np.float32) / 255.0

    base = np.asarray(img, dtype=np.float32)
    red = np.zeros_like(base)
    red[:, :, 0] = 255.0
    alpha = (heat_np * 0.65)[:, :, None]
    blended = (base * (1 - alpha) + red * alpha).astype(np.uint8)

    out = io.BytesIO()
    Image.fromarray(blended).save(out, format="PNG")

    verdict = predict(scorer, img, preset=preset)
    return Explanation(verdict=verdict, heatmap_png=out.getvalue(),
                       max_window_score=float(probs.max()) if len(probs) else 0.0)
