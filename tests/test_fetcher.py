import asyncio
import io

import httpx
import pytest
from PIL import Image

import service.fetcher as fetcher
from service.fetcher import FetchError, _host_is_public, fetch_image


def fetch(url, handler=None, monkeypatch=None, treat_public=None, **kwargs):
    """Run fetch_image against an optional mock transport."""
    async def go():
        transport = httpx.MockTransport(handler) if handler else None
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_image(
                client, url,
                timeout=kwargs.get("timeout", 2),
                max_bytes=kwargs.get("max_bytes", 1024),
                min_side=kwargs.get("min_side", 10))

    if treat_public is not None:
        monkeypatch.setattr(fetcher, "_host_is_public", treat_public)
    return asyncio.run(go())


def test_private_hosts_rejected():
    assert _host_is_public("localhost") is False
    assert _host_is_public("nonexistent.invalid.safesight.test") is False


def test_scheme_rejected():
    with pytest.raises(FetchError, match="scheme"):
        fetch("ftp://example.com/a.jpg")


def test_loopback_url_rejected():
    with pytest.raises(FetchError, match="host not allowed"):
        fetch("http://127.0.0.1/secret.png")


def test_redirect_to_private_host_rejected(monkeypatch):
    """A public host that redirects to a private address must be blocked at
    the hop, and the private hop must never be requested."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.example":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/x.png"})
        raise AssertionError("private hop must never be requested")

    with pytest.raises(FetchError, match="host not allowed"):
        fetch("http://public.example/img.png", handler, monkeypatch,
              treat_public=lambda h: h == "public.example")


def test_oversized_body_rejected(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/png"},
                              content=b"x" * 2048)

    with pytest.raises(FetchError, match="too large"):
        fetch("http://public.example/big.png", handler, monkeypatch,
              treat_public=lambda h: True)


def test_non_image_content_type_rejected(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"},
                              content=b"<html></html>")

    with pytest.raises(FetchError, match="not an image"):
        fetch("http://public.example/page", handler, monkeypatch,
              treat_public=lambda h: True)


def test_valid_image_accepted(monkeypatch):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(buf, format="PNG")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/png"},
                              content=buf.getvalue())

    img = fetch("http://public.example/ok.png", handler, monkeypatch,
                treat_public=lambda h: True, max_bytes=1024 * 1024)
    assert img.size == (64, 64)


def test_redirect_chain_followed_to_image(monkeypatch):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(1, 2, 3)).save(buf, format="PNG")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(301, headers={"location": "/final.png"})
        return httpx.Response(200, headers={"content-type": "image/png"},
                              content=buf.getvalue())

    img = fetch("http://public.example/start", handler, monkeypatch,
                treat_public=lambda h: True, max_bytes=1024 * 1024)
    assert img.size == (64, 64)
