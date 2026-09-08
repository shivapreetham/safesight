"""ONNX Runtime scoring backend.

Implements the same score_regions(regions) -> probabilities interface as
ml/torch_backend.py, so every strategy (fast, accurate, cascade, explain)
works unchanged on top of it. This backend has no torch dependency, which
is what lets the serving container fit free 512MB-RAM hosting.

Why ONNX at serving time: onnxruntime's graph optimizations cut per-patch
CPU latency roughly 2.5x versus eager PyTorch (measured; see
docs/benchmarks.md). Note the measured surprise: the int8 dynamically
quantized variant is 3.7x smaller but SLOWER than fp32 on CPUs without
fast ConvInteger kernels, so fp32 is the recommended CPU model and int8 is
reserved for size-constrained targets. Benchmark on the actual host before
choosing.
"""

import numpy as np

from ml.inference import PSEUDO_IOU


class OnnxScorer:
    """Scores 224x224 regions with an exported RQSA-MIL ONNX graph."""

    def __init__(self, model_path: str, intra_op_threads: int = 0):
        import onnxruntime as ort
        options = ort.SessionOptions()
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"])
        self.model_path = model_path
        self.device = "cpu"

    def score_regions(self, regions: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(regions, dtype=np.float32)
        iou = np.full((x.shape[0], 1), PSEUDO_IOU, dtype=np.float32)
        logits = self.session.run(["logits"], {"images": x, "iou": iou})[0]
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        return probs.astype(np.float32).squeeze(-1)
