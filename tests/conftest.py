import os
import sys

import pytest
from PIL import Image

# Make the repo root importable regardless of pytest invocation directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CLEAN_CHECKPOINT = os.path.join(REPO_ROOT, "weights", "rqsa_v1_clean.pt")

requires_checkpoint = pytest.mark.skipif(
    not os.path.exists(CLEAN_CHECKPOINT),
    reason="clean checkpoint not present; run ml/convert_checkpoint.py first",
)


@pytest.fixture(scope="session")
def checkpoint_path():
    return CLEAN_CHECKPOINT


@pytest.fixture()
def safe_image():
    """Flat blue image: contains no skin, should always score safe."""
    return Image.new("RGB", (800, 600), color=(30, 60, 200))


@pytest.fixture()
def gradient_image():
    """Deterministic gradient image for stability checks."""
    img = Image.new("RGB", (700, 500))
    px = img.load()
    for y in range(500):
        for x in range(700):
            px[x, y] = (x % 256, y % 256, (x + y) % 256)
    return img
