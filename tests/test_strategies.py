import os

import pytest
from PIL import Image

from ml.inference import load_model, predict
from ml.strategies import explain, predict_cascade, score_full_frame
from tests.conftest import REPO_ROOT, requires_checkpoint

ONNX_FP32 = os.path.join(REPO_ROOT, "weights", "rqsa_v1.onnx")
requires_onnx = pytest.mark.skipif(
    not os.path.exists(ONNX_FP32),
    reason="onnx model not present; run ml/export_onnx.py first",
)


@pytest.fixture(scope="module")
def model(checkpoint_path):
    return load_model(checkpoint_path, device="cpu")


@requires_checkpoint
def test_full_frame_returns_probability(model, safe_image):
    score = score_full_frame(model, safe_image)
    assert 0.0 <= score <= 1.0


@requires_checkpoint
def test_cascade_small_safe_image_exits_at_full_frame(model):
    tiny_safe = Image.new("RGB", (320, 240), color=(30, 60, 200))
    verdict = predict_cascade(model, tiny_safe)
    assert verdict.label == 0
    assert verdict.stages == ["full_frame"]
    assert verdict.n_patches == 1


@requires_checkpoint
def test_cascade_large_image_never_trusts_full_frame_safe_exit(model, safe_image):
    # 800x600: even a near-zero full-frame score must fall through to the
    # sliding-window stage, because full-frame misses small regions at scale.
    verdict = predict_cascade(model, safe_image)
    assert "fast" in verdict.stages
    assert verdict.n_patches > 1


@requires_checkpoint
def test_cascade_reports_stages_and_preset(model, gradient_image):
    verdict = predict_cascade(model, gradient_image)
    assert verdict.preset == "cascade"
    assert verdict.stages[0] == "full_frame"
    assert set(verdict.stages) <= {"full_frame", "fast", "accurate"}


@requires_checkpoint
def test_explain_produces_png(model, safe_image):
    result = explain(model, safe_image, preset="fast")
    assert result.heatmap_png[:8] == b"\x89PNG\r\n\x1a\n"
    assert 0.0 <= result.max_window_score <= 1.0
    assert result.verdict.label in (0, 1)


@requires_checkpoint
@requires_onnx
def test_onnx_backend_matches_torch_verdict(model, safe_image):
    from ml.onnx_backend import OnnxScorer
    scorer = OnnxScorer(ONNX_FP32)
    torch_verdict = predict(model, safe_image, preset="fast")
    onnx_verdict = predict(scorer, safe_image, preset="fast")
    assert onnx_verdict.label == torch_verdict.label
    assert abs(onnx_verdict.score - torch_verdict.score) < 0.05


@requires_checkpoint
@requires_onnx
def test_onnx_backend_works_with_cascade():
    from ml.onnx_backend import OnnxScorer
    scorer = OnnxScorer(ONNX_FP32)
    tiny_safe = Image.new("RGB", (320, 240), color=(30, 60, 200))
    verdict = predict_cascade(scorer, tiny_safe)
    assert verdict.label == 0
