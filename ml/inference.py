"""Dense sliding-window inference for RQSA-MIL.

Vendored and refactored from the research repo's predict_dense.py. The model
was trained on 200-330px crops of images normalized to a 690px short side, so
inference slides windows at that scale and aggregates per-tier top-k means.

The pipeline is numpy-only by design: the serving container ships without
PyTorch (onnxruntime backend), which keeps the image small enough for
512MB-RAM free hosting. Backends implement one method,
score_regions(regions: np.ndarray (N,3,224,224)) -> np.ndarray (N,) of
probabilities - see ml/torch_backend.py and ml/onnx_backend.py.

Two presets:
    accurate  paper settings: 3 large tiers + small tier, fixed crops,
              horizontal-flip TTA, Gaussian soft-NMS. Use for evaluation.
    fast      single large tier, no TTA, no small tier. Roughly 8x fewer
              patches; use for the browser extension / CPU serving.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

INPUT_SIZE = 224
NORM_SHORT = 690
PSEUDO_IOU = 0.5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SOFT_NMS_SIGMA = 0.5
SOFT_NMS_IOU_THRESH = 0.30


@dataclass
class Preset:
    name: str
    windows_large: list[int] = field(default_factory=lambda: [200, 265, 330])
    windows_small: list[int] = field(default_factory=lambda: [140])
    stride_ratio: float = 0.5
    use_tta: bool = True
    use_fixed_crops: bool = True
    top_k: int = 3
    threshold_large: float = 0.50
    threshold_small: float = 0.78
    batch_size: int = 64


PRESETS = {
    "accurate": Preset(name="accurate"),
    "fast": Preset(
        name="fast",
        windows_large=[265],
        windows_small=[],
        use_tta=False,
        use_fixed_crops=False,
    ),
}


@dataclass
class Verdict:
    label: int              # 1 = nsfw, 0 = safe (calibrated dual-tier decision)
    score: float            # max(score_large, score_small); UI confidence
    score_large: float      # large-tier score; use for custom thresholding
    n_patches: int
    preset: str
    stages: list = field(default_factory=list)  # populated by cascade strategy


def to_region(img: Image.Image) -> np.ndarray:
    """PIL crop -> normalized (3, 224, 224) float32 region.

    Matches torchvision Resize((224, 224)) + ToTensor + Normalize exactly:
    PIL bilinear resize, /255, per-channel ImageNet normalization.
    """
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)


def load_model(checkpoint_path: str, device: str = "auto"):
    """Load the clean safesight checkpoint into a torch-backed scorer.

    Torch is imported lazily so onnx-only deployments never need it
    installed; use ml.onnx_backend.OnnxScorer for those.
    """
    from ml.torch_backend import TorchScorer
    return TorchScorer(checkpoint_path, device=device)


def resize_keep_aspect(img: Image.Image, short_side: int) -> Image.Image:
    w, h = img.size
    if min(w, h) <= short_side:
        return img
    if w < h:
        return img.resize((short_side, int(h * short_side / w)), Image.BILINEAR)
    return img.resize((int(w * short_side / h), short_side), Image.BILINEAR)


def _slide_windows(img: Image.Image, window_sizes: list[int],
                   stride_ratio: float) -> tuple[list, list]:
    """Extract sliding-window regions and their (x1, y1, x2, y2) boxes."""
    W, H = img.size
    patches, boxes = [], []
    for win in window_sizes:
        if win > W or win > H:
            continue
        stride = max(1, int(win * stride_ratio))
        y = 0
        while y + win <= H:
            x = 0
            while x + win <= W:
                patches.append(to_region(img.crop((x, y, x + win, y + win))))
                boxes.append((x, y, x + win, y + win))
                x += stride
            if x - stride + win < W:
                patches.append(to_region(img.crop((W - win, y, W, y + win))))
                boxes.append((W - win, y, W, y + win))
            y += stride
        if y - stride + win < H:
            x = 0
            while x + win <= W:
                patches.append(to_region(img.crop((x, H - win, x + win, H))))
                boxes.append((x, H - win, x + win, H))
                x += stride
            if x - stride + win < W:
                patches.append(to_region(img.crop((W - win, H - win, W, H))))
                boxes.append((W - win, H - win, W, H))
    return patches, boxes


def _add_fixed_crops(img: Image.Image, patches: list, boxes: list) -> None:
    """Half- and two-thirds-scale corner/center crops plus the full image."""
    W, H = img.size
    for scale in (0.5, 2.0 / 3.0):
        sw = max(int(W * scale), 10)
        sh = max(int(H * scale), 10)
        cx, cy = (W - sw) // 2, (H - sh) // 2
        for px, py in ((0, 0), (W - sw, 0), (0, H - sh), (W - sw, H - sh), (cx, cy)):
            px = max(0, min(px, W - sw))
            py = max(0, min(py, H - sh))
            patches.append(to_region(img.crop((px, py, px + sw, py + sh))))
            boxes.append((px, py, px + sw, py + sh))
    patches.append(to_region(img))
    boxes.append((0, 0, W, H))


def _iou_matrix(bx: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (N, 4) boxes, vectorized."""
    x1, y1, x2, y2 = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    iw = np.clip(np.minimum(x2[:, None], x2[None, :]) -
                 np.maximum(x1[:, None], x1[None, :]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None, :]) -
                 np.maximum(y1[:, None], y1[None, :]), 0, None)
    inter = iw * ih
    return inter / (areas[:, None] + areas[None, :] - inter + 1e-6)


def _soft_nms(probs: np.ndarray, boxes: list,
              sigma: float = SOFT_NMS_SIGMA,
              iou_thresh: float = SOFT_NMS_IOU_THRESH) -> np.ndarray:
    """Gaussian soft-NMS: decay scores of heavily overlapping windows so that
    multiple 50%-overlap windows do not vote repeatedly on the same region.

    The pairwise IoU matrix is computed once (vectorized), then each window
    in descending original-score order decays every lower-ranked overlapping
    window in a single vector op.
    """
    if len(probs) <= 1 or not boxes:
        return probs.astype(np.float32, copy=True)
    scores = probs.astype(np.float32, copy=True)
    iou = _iou_matrix(np.asarray(boxes, dtype=np.float32))
    order = np.argsort(-scores)
    for rank in range(len(order) - 1):
        i = order[rank]
        lower = order[rank + 1:]
        overlap = iou[i, lower]
        decay = np.where(overlap > iou_thresh,
                         np.exp(-overlap ** 2 / sigma), 1.0)
        scores[lower] *= decay.astype(np.float32)
    return scores


def _score_patches(patches_list: list, scorer, boxes: list | None,
                   batch_size: int) -> np.ndarray:
    if not patches_list:
        return np.zeros(0, dtype=np.float32)
    probs_all = []
    # Stack per chunk, not all at once: the accurate preset produces 300+
    # patches (~200 MB as one fp32 array); chunked stacking caps peak
    # memory at one batch regardless of patch count.
    for i in range(0, len(patches_list), batch_size):
        batch = np.stack(patches_list[i:i + batch_size])
        probs_all.append(scorer.score_regions(batch))
    probs = np.concatenate(probs_all).astype(np.float32)
    if boxes is not None and len(boxes) == len(probs):
        probs = _soft_nms(probs, boxes)
    return probs


def _topk_mean(probs: np.ndarray, k: int) -> float:
    if len(probs) == 0:
        return 0.0
    k = min(k, len(probs))
    return float(np.sort(probs)[-k:].mean())


def predict(scorer, img: Image.Image, preset: str = "accurate") -> Verdict:
    """Two-tier dense sliding-window prediction.

    Pass the full-resolution PIL image; the short side is normalized to 690px
    internally. Do not resize beforehand. `scorer` is any backend exposing
    score_regions(np.ndarray) -> np.ndarray.
    """
    cfg = PRESETS[preset]
    img = resize_keep_aspect(img.convert("RGB"), NORM_SHORT)

    views = [img]
    if cfg.use_tta:
        views.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    probs_large_all, probs_small_all = [], []
    n_patches = 0
    for view in views:
        large_patches, large_boxes = _slide_windows(
            view, cfg.windows_large, cfg.stride_ratio)
        if cfg.use_fixed_crops:
            _add_fixed_crops(view, large_patches, large_boxes)
        elif not large_patches:
            # Image smaller than every window: fall back to the full image.
            W, H = view.size
            large_patches.append(to_region(view))
            large_boxes.append((0, 0, W, H))
        small_patches, small_boxes = _slide_windows(
            view, cfg.windows_small, cfg.stride_ratio)

        probs_large_all.append(_score_patches(
            large_patches, scorer, large_boxes, cfg.batch_size))
        probs_small_all.append(_score_patches(
            small_patches, scorer, small_boxes, cfg.batch_size))
        n_patches += len(large_patches) + len(small_patches)

    probs_large = np.concatenate(probs_large_all) if probs_large_all else np.zeros(0)
    probs_small = np.concatenate(probs_small_all) if probs_small_all else np.zeros(0)

    score_large = _topk_mean(probs_large, cfg.top_k)
    score_small = _topk_mean(probs_small, cfg.top_k)

    fire = (score_large > cfg.threshold_large) or (score_small > cfg.threshold_small)
    return Verdict(
        label=int(fire),
        score=max(score_large, score_small),
        score_large=score_large,
        n_patches=n_patches,
        preset=cfg.name,
    )
