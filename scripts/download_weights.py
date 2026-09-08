"""Download SafeSight model weights from the public HF Hub repo.

Idempotent: files already present in weights/ are skipped. No token needed
(the repo is public).

Usage:
    python scripts/download_weights.py [--all]

By default downloads the serving essentials (clean checkpoint + ONNX fp32);
--all adds the int8 variant.
"""

import argparse
import os
import shutil

from huggingface_hub import hf_hub_download

REPO = "Shivapreetham17/rqsa-mil-v1"
CORE_FILES = ["rqsa_v1_clean.pt", "rqsa_v1.onnx"]
EXTRA_FILES = ["rqsa_v1_int8.onnx"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="also download the int8 ONNX variant")
    parser.add_argument("--dest", default="weights")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    files = CORE_FILES + (EXTRA_FILES if args.all else [])
    for name in files:
        target = os.path.join(args.dest, name)
        if os.path.exists(target):
            print(f"  present  {target}")
            continue
        print(f"  fetching {name} from {REPO}")
        cached = hf_hub_download(repo_id=REPO, filename=name)
        shutil.copyfile(cached, target)
        print(f"  saved    {target} ({os.path.getsize(target) / 1e6:.1f} MB)")
    print("Weights ready.")


if __name__ == "__main__":
    main()
