"""Golden-set regression gate.

Scores every image under a labeled golden-set directory and fails (exit 1)
if recall or specificity drop below the release thresholds. Layout:

    golden/
      nsfw/   positive samples (kept private, never committed)
      safe/   negative samples

Usage:
    python -m ml.eval_golden --checkpoint weights/rqsa_v1_clean.pt \
        --golden-dir path/to/golden --min-recall 0.85 --min-specificity 0.80
"""

import argparse
import os
import sys

from PIL import Image

from ml.inference import load_model, predict

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def score_dir(model, directory, preset):
    labels = []
    for name in sorted(os.listdir(directory)):
        if os.path.splitext(name)[1].lower() not in EXTS:
            continue
        img = Image.open(os.path.join(directory, name)).convert("RGB")
        verdict = predict(model, img, preset=preset)
        labels.append(verdict.label)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="weights/rqsa_v1_clean.pt")
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--preset", default="accurate")
    parser.add_argument("--min-recall", type=float, default=0.85)
    parser.add_argument("--min-specificity", type=float, default=0.80)
    args = parser.parse_args()

    model = load_model(args.checkpoint)

    nsfw_dir = os.path.join(args.golden_dir, "nsfw")
    safe_dir = os.path.join(args.golden_dir, "safe")

    pos = score_dir(model, nsfw_dir, args.preset) if os.path.isdir(nsfw_dir) else []
    neg = score_dir(model, safe_dir, args.preset) if os.path.isdir(safe_dir) else []

    if not pos and not neg:
        print("No golden images found; nothing to gate.")
        sys.exit(1)

    recall = sum(pos) / len(pos) if pos else None
    specificity = (len(neg) - sum(neg)) / len(neg) if neg else None

    print(f"Golden set: {len(pos)} nsfw, {len(neg)} safe (preset={args.preset})")
    if recall is not None:
        print(f"  recall      = {recall:.4f} (min {args.min_recall})")
    if specificity is not None:
        print(f"  specificity = {specificity:.4f} (min {args.min_specificity})")

    failed = False
    if recall is not None and recall < args.min_recall:
        print("FAIL: recall below threshold")
        failed = True
    if specificity is not None and specificity < args.min_specificity:
        print("FAIL: specificity below threshold")
        failed = True

    if failed:
        sys.exit(1)
    print("PASS: golden-set gate satisfied")


if __name__ == "__main__":
    main()
