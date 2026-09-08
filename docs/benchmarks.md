# Measured benchmarks

CPU-only, single request at a time. Host: Intel64 Family 6 Model 142 Stepping 12, GenuineIntel, 8 logical cores, torch 2.5.1+cpu. Median of repeated runs on deterministic synthetic photos.

## Inference strategies (PyTorch backend)

### small 320x240

| Strategy | Patches | Latency (ms) |
|---|---|---|
| full_frame | 1 | 76 |
| fast | 1 | 78 |
| accurate | 58 | 2314 |
| cascade (full_frame) | 1 | 72 |

### medium 800x600

| Strategy | Patches | Latency (ms) |
|---|---|---|
| full_frame | 1 | 81 |
| fast | 24 | 957 |
| accurate | 340 | 11960 |
| cascade (full_frame>fast) | 25 | 942 |

### large 1920x1440

| Strategy | Patches | Latency (ms) |
|---|---|---|
| full_frame | 1 | 83 |
| fast | 30 | 1120 |
| accurate | 464 | 16906 |
| cascade (full_frame>fast) | 31 | 1247 |

## Serving backends (batch of 24 patches, the fast-preset workload)

| Backend | Latency (ms) |
|---|---|
| pytorch eager (fp32) | 780 |
| onnxruntime fp32 (10 MB) | 315 |
| onnxruntime int8 (3 MB) | 1603 |

Interpretation: the cascade keeps median latency near the fast preset
while reserving paper-grade accuracy for ambiguous images. Compare the
backend rows before choosing a CPU serving backend: quantized kernels
are not universally faster (int8 ConvInteger paths lack fast
implementations on some CPUs), so the measured number on the target
hardware decides, not the textbook expectation.

Numerical parity (structured inputs, 8 regions): ONNX fp32 matches torch
to 1e-10; int8 max probability deviation 0.038.
**Unverified**: `python -m ml.benchmark` (this file's generator) does not
produce this paragraph, and the only parity check in the test suite
(`tests/test_strategies.py::test_onnx_backend_matches_torch_verdict`) only
asserts labels match and `|score diff| < 0.05` — a much looser bound than
1e-10. Treat the exact 1e-10 / 0.038 figures as illustrative, not
independently reproduced.

Optimization note: vectorizing Gaussian soft-NMS (pairwise IoU matrix via
torchvision box_iou instead of a Python double loop) cut the accurate
preset 39-44% versus the reference implementation (medium image: 19.6s
to 12.0s; large: 30.0s to 16.9s) with identical outputs.
**Partially verified**: the "after" numbers (12.0s medium, 16.9s large)
match the current accurate-preset rows above (11960ms, 16906ms). The "before"
numbers (19.6s, 30.0s) describe a pre-vectorization implementation that no
longer exists in the repo (the shipped `_soft_nms` in `ml/inference.py` is
already the vectorized version), so the improvement percentage cannot be
re-derived from current source — treat it as an unverified historical claim.
