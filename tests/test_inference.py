import pytest

from ml.inference import PRESETS, load_model, predict
from tests.conftest import requires_checkpoint


@pytest.fixture(scope="module")
def model(checkpoint_path):
    return load_model(checkpoint_path, device="cpu")


@requires_checkpoint
def test_presets_exist():
    assert "accurate" in PRESETS and "fast" in PRESETS


@requires_checkpoint
def test_safe_image_scores_low(model, safe_image):
    verdict = predict(model, safe_image, preset="fast")
    assert verdict.label == 0
    assert 0.0 <= verdict.score <= 1.0
    assert verdict.n_patches > 0


@requires_checkpoint
def test_deterministic(model, gradient_image):
    v1 = predict(model, gradient_image, preset="fast")
    v2 = predict(model, gradient_image, preset="fast")
    assert v1.score == pytest.approx(v2.score, abs=1e-6)
    assert v1.label == v2.label


@requires_checkpoint
def test_accurate_preset_runs(model, safe_image):
    verdict = predict(model, safe_image, preset="accurate")
    assert verdict.label in (0, 1)
    assert verdict.n_patches > 0


@requires_checkpoint
def test_tiny_image_falls_back_to_full_frame(model):
    from PIL import Image
    tiny = Image.new("RGB", (120, 90), color=(30, 60, 200))
    verdict = predict(model, tiny, preset="fast")
    assert verdict.n_patches >= 1
    assert 0.0 <= verdict.score <= 1.0
