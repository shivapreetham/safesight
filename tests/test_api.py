import io

import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_checkpoint


@pytest.fixture(scope="module")
def client(checkpoint_path):
    import service.settings as settings_module
    settings_module.settings.checkpoint_path = checkpoint_path
    settings_module.settings.device = "cpu"
    from service.app import app
    with TestClient(app) as c:
        yield c


def _png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@requires_checkpoint
def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@requires_checkpoint
def test_score_upload_safe(client, safe_image):
    resp = client.post(
        "/v1/score",
        files={"file": ("blue.png", _png_bytes(safe_image), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["nsfw"] is False
    assert 0.0 <= body["score"] <= 1.0


@requires_checkpoint
def test_score_rejects_garbage(client):
    resp = client.post(
        "/v1/score",
        files={"file": ("junk.png", b"this is not an image", "image/png")},
    )
    assert resp.status_code == 422


@requires_checkpoint
def test_unknown_preset_rejected(client, safe_image):
    resp = client.post(
        "/v1/score?preset=bogus",
        files={"file": ("blue.png", _png_bytes(safe_image), "image/png")},
    )
    assert resp.status_code == 422


@requires_checkpoint
def test_score_urls_rejects_bad_scheme(client):
    resp = client.post("/v1/score-urls", json={"urls": ["ftp://example.com/a.jpg"]})
    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["ok"] is False


@requires_checkpoint
def test_feedback_recorded(client):
    resp = client.post("/v1/feedback", json={
        "url_hash": "abc123",
        "model_score": 0.91,
        "user_label": 0,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"


@requires_checkpoint
def test_metrics_exposed(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "safesight_image_score" in resp.text


@requires_checkpoint
def test_score_cascade_strategy(client, safe_image):
    resp = client.post(
        "/v1/score?preset=cascade",
        files={"file": ("blue.png", _png_bytes(safe_image), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["preset"] == "cascade"
    assert body["stages"][0] == "full_frame"
    assert body["nsfw"] is False


@requires_checkpoint
def test_explain_returns_png_with_headers(client, safe_image):
    resp = client.post(
        "/v1/explain",
        files={"file": ("blue.png", _png_bytes(safe_image), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert "X-SafeSight-Score" in resp.headers
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


@requires_checkpoint
def test_demo_page_served(client):
    resp = client.get("/demo")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "SafeSight" in resp.text


@requires_checkpoint
def test_stats_endpoint_counts(client, safe_image):
    before = client.get("/v1/stats").json()
    client.post(
        "/v1/score",
        files={"file": ("blue.png", _png_bytes(safe_image), "image/png")},
    )
    after = client.get("/v1/stats").json()
    assert after["images_scored"] == before["images_scored"] + 1
    assert 0.0 <= after["nsfw_rate"] <= 1.0
    assert "cascade_exit_counts" in after


@requires_checkpoint
def test_api_key_enforced_when_set(client, safe_image):
    import service.settings as settings_module
    settings_module.settings.api_key = "secret123"
    try:
        resp = client.post(
            "/v1/score",
            files={"file": ("blue.png", _png_bytes(safe_image), "image/png")},
        )
        assert resp.status_code == 401

        resp = client.post(
            "/v1/score",
            headers={"X-API-Key": "secret123"},
            files={"file": ("blue.png", _png_bytes(safe_image), "image/png")},
        )
        assert resp.status_code == 200

        # Health and demo stay open for load balancers and the dashboard.
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/stats").status_code == 200
    finally:
        settings_module.settings.api_key = ""
