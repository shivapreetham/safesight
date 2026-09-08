"""Score images from the command line.

Usage:
    python -m ml.score_cli --checkpoint weights/rqsa_v1_clean.pt --image pic.jpg
    python -m ml.score_cli --checkpoint weights/rqsa_v1_clean.pt --dir folder --preset fast
"""

import argparse
import os
import time

from PIL import Image

from ml.inference import load_model, predict
from ml.strategies import predict_cascade


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="weights/rqsa_v1_clean.pt")
    parser.add_argument("--image")
    parser.add_argument("--dir")
    parser.add_argument("--preset", default="accurate",
                        choices=["accurate", "fast", "cascade"])
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    print(f"Device: {model.device}  Preset: {args.preset}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if args.image:
        paths = [args.image]
    elif args.dir:
        paths = sorted(
            os.path.join(args.dir, f) for f in os.listdir(args.dir)
            if os.path.splitext(f)[1].lower() in exts)
    else:
        parser.error("Provide --image or --dir")

    for path in paths:
        img = Image.open(path).convert("RGB")
        start = time.perf_counter()
        if args.preset == "cascade":
            v = predict_cascade(model, img)
        else:
            v = predict(model, img, preset=args.preset)
        ms = (time.perf_counter() - start) * 1000
        tag = "NSFW" if v.label == 1 else "safe"
        print(f"  [{tag}]  score={v.score:.3f}  score_large={v.score_large:.3f}  "
              f"patches={v.n_patches:>4}  {ms:6.0f} ms  {os.path.basename(path)}")


if __name__ == "__main__":
    main()
