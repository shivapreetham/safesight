"""Quick API testing script for SafeSight.

Usage:
    python test_api.py
"""

import sys
from pathlib import Path

import httpx

API_BASE = "http://127.0.0.1:8001"


def test_root():
    print("Testing GET /")
    with httpx.Client() as client:
        response = client.get(f"{API_BASE}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")
        return response.status_code == 200


def test_healthz():
    print("Testing GET /healthz")
    with httpx.Client() as client:
        response = client.get(f"{API_BASE}/healthz")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")
        return response.status_code == 200


def test_score_urls():
    print("Testing POST /v1/score-urls")
    test_urls = [
        "https://picsum.photos/200/300",
        "https://picsum.photos/400/600",
    ]
    payload = {
        "urls": test_urls,
        "preset": "fast"
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(f"{API_BASE}/v1/score-urls", json=payload)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Scored {len(result['results'])} images:")
        for r in result['results']:
            print(f"  - {r['url'][:60]}...")
            print(f"    Label: {r.get('label')}, Score: {r.get('score')}, Cached: {r.get('cached')}")
        print()
        return response.status_code == 200


def test_score_upload():
    print("Testing POST /v1/score")

    test_images = list(Path("tests/fixtures").glob("*.jpg"))
    if not test_images:
        print("  No test images found in tests/fixtures/, skipping...\n")
        return True

    test_image = test_images[0]
    print(f"  Using image: {test_image}")

    with open(test_image, "rb") as f:
        files = {"file": (test_image.name, f, "image/jpeg")}
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{API_BASE}/v1/score?preset=fast", files=files)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Result: Label={result['label']}, Score={result['score']}, "
              f"NSFW={result['nsfw']}, Patches={result['n_patches']}")
    else:
        print(f"Error: {response.text}")
    print()
    return response.status_code == 200


def test_metrics():
    print("Testing GET /metrics")
    with httpx.Client() as client:
        response = client.get(f"{API_BASE}/metrics")
        print(f"Status: {response.status_code}")
        lines = response.text.split('\n')
        safesight_metrics = [ln for ln in lines if 'safesight' in ln and not ln.startswith('#')]
        print(f"Found {len(safesight_metrics)} SafeSight metrics")
        if safesight_metrics:
            print("Sample metrics:")
            for line in safesight_metrics[:5]:
                print(f"  {line}")
        print()
        return response.status_code == 200


def main():
    print("SafeSight API Test Suite")
    print("=" * 60)
    print(f"API Base: {API_BASE}\n")

    tests = [
        ("Root endpoint", test_root),
        ("Health check", test_healthz),
        ("Score URLs", test_score_urls),
        ("Score upload", test_score_upload),
        ("Metrics", test_metrics),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR in {name}: {e}\n")
            results.append((name, False))

    print("=" * 60)
    print("Test Results:")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\nPassed: {passed_count}/{len(results)}")

    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
