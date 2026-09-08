"""Service configuration via environment variables (prefix SAFESIGHT_)."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    checkpoint_path: str = "weights/rqsa_v1_clean.pt"
    # Optional Hugging Face Hub model repo to download the checkpoint from
    # when checkpoint_path does not exist (used on HF Spaces).
    hf_model_repo: str = ""
    hf_model_filename: str = "rqsa_v1_clean.pt"

    preset: str = "cascade"         # default strategy: fast | accurate | cascade
    device: str = "auto"            # auto | cuda | cpu
    backend: str = "torch"          # torch | onnx (onnx is CPU-only)
    # fp32 by default: measured 3x faster than eager torch on CPU, while
    # int8 was slower on hosts without fast ConvInteger kernels.
    onnx_model_path: str = "weights/rqsa_v1.onnx"
    hf_onnx_filename: str = "rqsa_v1.onnx"

    fetch_timeout_seconds: float = 8.0
    fetch_max_bytes: int = 10 * 1024 * 1024
    fetch_concurrency: int = 8      # parallel URL downloads per request
    max_urls_per_request: int = 24
    min_image_side: int = 64        # skip tiny icons

    # Optional shared-secret auth for /v1/* endpoints. Empty = auth disabled
    # (local demos). When set, clients must send X-API-Key.
    api_key: str = ""

    cache_size: int = 4096
    score_log_path: str = "logs/scores.jsonl"
    feedback_log_path: str = "logs/feedback.jsonl"

    model_config = {"env_prefix": "SAFESIGHT_"}


settings = Settings()
