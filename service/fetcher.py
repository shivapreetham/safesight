"""SSRF-hardened image fetching.

The API fetches arbitrary URLs supplied by browser extensions, which makes it
a server-side request proxy unless constrained. Defenses, in order:

1. Scheme allowlist (http/https only).
2. DNS resolution of every host must yield only public addresses - private,
   loopback, link-local, and reserved ranges are rejected. Resolution runs in
   a worker thread so the event loop is never blocked.
3. Redirects are never delegated to the HTTP client: each hop is followed
   manually and its host re-validated, closing the classic
   "public URL 302s to http://169.254.169.254" hole.
4. Responses stream with a hard byte cap; a Content-Length lie cannot force
   a large allocation.
5. Content-Type must be image/*, and decoded images below a minimum side
   length are rejected (icons are not worth model time).
"""

import io
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import anyio
import httpx
from PIL import Image

MAX_REDIRECTS = 5

# Browser-like UA: image CDNs (e.g. Wikimedia) reject generic bot agents.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 SafeSight/1.0")


class FetchError(ValueError):
    """Raised for any policy violation or transport failure during fetch."""


def _host_is_public(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            return False
    return True


async def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise FetchError("unsupported scheme")
    if not parsed.hostname:
        raise FetchError("no host")
    ok = await anyio.to_thread.run_sync(_host_is_public, parsed.hostname)
    if not ok:
        raise FetchError("host not allowed")


async def _read_capped(resp: httpx.Response, max_bytes: int) -> bytes:
    chunks, total = [], 0
    async for chunk in resp.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise FetchError("image too large")
        chunks.append(chunk)
    return b"".join(chunks)


async def fetch_image(client: httpx.AsyncClient, url: str, *,
                      timeout: float, max_bytes: int,
                      min_side: int) -> Image.Image:
    """Fetch and decode one image URL under the policy above."""
    for _ in range(MAX_REDIRECTS + 1):
        await _validate_url(url)
        try:
            async with client.stream("GET", url, timeout=timeout,
                                     follow_redirects=False) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchError("redirect without location")
                    url = urljoin(url, location)
                    continue
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    raise FetchError(f"not an image: {content_type[:60]}")
                data = await _read_capped(resp, max_bytes)
        except httpx.HTTPError as exc:
            raise FetchError(str(exc)[:200]) from exc

        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as exc:
            raise FetchError("undecodable image") from exc
        if min(img.size) < min_side:
            raise FetchError("image too small")
        return img

    raise FetchError("too many redirects")
