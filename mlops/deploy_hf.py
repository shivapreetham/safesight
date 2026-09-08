"""Deploy SafeSight to Hugging Face: model weights to a model repo, the
serving code to a Docker Space.

Reads HF_TOKEN from the repo-root .env (never committed). Idempotent: safe
to re-run on every release.

Usage (from repo root):
    python mlops/deploy_hf.py [--space-name safesight] [--model-name rqsa-mil-v1]
"""

import argparse
import io
import os

from huggingface_hub import HfApi

WEIGHT_FILES = ["weights/rqsa_v1_clean.pt", "weights/rqsa_v1.onnx",
                "weights/rqsa_v1_int8.onnx"]

SPACE_IGNORE = [".git*", ".venv*", ".env", "logs/*", "weights/*",
                "extension/*", "mlops/*", "tests/*", "docs/*",
                "__pycache__*", "*.pyc", ".pytest_cache*", ".ruff_cache*"]

SPACE_README = """---
title: SafeSight Moderation API
emoji: \U0001F6E1
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# SafeSight Moderation API

NSFW image moderation service built on RQSA-MIL (MobileNetV2 +
IoU-conditioned attention, 87.9% recall / 93.0% AUC). ONNX fp32 backend,
cascade inference strategy.

Endpoints: `/healthz`, `/v1/score` (multipart upload), `/v1/score-urls`
(batch), `/v1/explain` (evidence heatmap PNG), `/v1/feedback`, `/metrics`.

Source, benchmarks, and design decisions: see the project repository.
"""


def read_env_token(env_path: str) -> str:
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise SystemExit(f"HF_TOKEN not found in {env_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-name", default="safesight")
    parser.add_argument("--model-name", default="rqsa-mil-v1")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    token = read_env_token(args.env_file)
    api = HfApi(token=token)
    user = api.whoami()["name"]
    model_repo = f"{user}/{args.model_name}"
    space_repo = f"{user}/{args.space_name}"

    print(f"Deploying as {user}")

    # 1. Model repo with the weights.
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    for path in WEIGHT_FILES:
        if not os.path.exists(path):
            print(f"  skip missing {path}")
            continue
        print(f"  uploading {path} -> {model_repo}")
        api.upload_file(path_or_fileobj=path,
                        path_in_repo=os.path.basename(path),
                        repo_id=model_repo, repo_type="model")

    if os.path.exists("MODEL_CARD.md"):
        with open("MODEL_CARD.md", encoding="utf-8") as f:
            card = ("---\nlicense: other\ntags:\n- image-classification\n"
                    "- content-moderation\n- not-for-all-audiences\n---\n\n"
                    + f.read())
        api.upload_file(path_or_fileobj=io.BytesIO(card.encode("utf-8")),
                        path_in_repo="README.md",
                        repo_id=model_repo, repo_type="model",
                        commit_message="Model card")

    # 2. Docker Space with the serving code. Requires an HF PRO account
    # (free accounts get 402 here); the primary free deploy path is Render
    # via render.yaml, which pulls weights from the model repo above.
    try:
        api.create_repo(space_repo, repo_type="space", space_sdk="docker",
                        exist_ok=True)
    except Exception as exc:
        if "402" in str(exc):
            print("\nModel repo synced. Skipping Space: Docker Spaces need "
                  "an HF PRO subscription (https://huggingface.co/pro).")
            print("Free alternative: deploy to Render with render.yaml "
                  "(see README deployment section).")
            print(f"Model: https://huggingface.co/{model_repo}")
            return
        raise
    for key, value in {
        "SAFESIGHT_HF_MODEL_REPO": model_repo,
        "SAFESIGHT_BACKEND": "onnx",
        "SAFESIGHT_PRESET": "cascade",
    }.items():
        api.add_space_variable(space_repo, key, value)

    print(f"  uploading code -> {space_repo}")
    api.upload_folder(folder_path=".", repo_id=space_repo, repo_type="space",
                      ignore_patterns=SPACE_IGNORE,
                      commit_message="Deploy SafeSight")
    api.upload_file(path_or_fileobj=io.BytesIO(SPACE_README.encode("utf-8")),
                    path_in_repo="README.md",
                    repo_id=space_repo, repo_type="space",
                    commit_message="Space metadata")

    print(f"\nModel: https://huggingface.co/{model_repo}")
    print(f"Space: https://huggingface.co/spaces/{space_repo}")
    print(f"API:   https://{user}-{args.space_name}.hf.space")
    print("The Space is building now; first build takes a few minutes.")


if __name__ == "__main__":
    main()
