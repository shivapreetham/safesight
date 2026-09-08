"""PyTorch scoring backend.

Wraps the RQSA-MIL torch model behind the numpy score_regions interface so
the inference pipeline stays framework-free. This module is the only serving
path that imports torch; onnx-only deployments never load it.
"""

import numpy as np
import torch

from ml.inference import PSEUDO_IOU
from ml.model import MobileNetMILRQSA


class TorchScorer:
    """Scores 224x224 regions with the torch model (GPU-capable)."""

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if ckpt.get("format") != "safesight-rqsa-v1":
            raise ValueError(
                f"Unsupported checkpoint format: {ckpt.get('format')}. "
                "Run ml/convert_checkpoint.py on research checkpoints first.")
        self.model = MobileNetMILRQSA()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_regions(self, regions: np.ndarray) -> np.ndarray:
        batch = torch.from_numpy(np.ascontiguousarray(regions)).to(self.device)
        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            probs = self.model.score_regions(batch, PSEUDO_IOU)
        return probs.float().cpu().numpy()
