"""Interactive demo script for SafeSight API.

Shows live scoring of images with visual feedback.
Perfect for presentations and interviews.

Usage:
    python demo.py
"""

import sys
import time
from pathlib import Path

import httpx

API_BASE = "http://127.0.0.1:8001"


def print_banner():
    print("\n" + "=" * 70)
    print("  SafeSight NSFW Moderation Demo")
    print("  RQSA-MIL: MobileNetV2 + IoU-conditioned Attention")
    print("=" * 70 + "\n")


def check_api():
    print("[1/3] Checking API health...")
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{API_BASE}/healthz")
            if response.status_code == 200:
                data = response.json()
                print(f"      Status: {data['status']}")
                print(f"      Backend: {data['backend']}")
                print(f"      Device: {data['device']}")
                print(f"      Preset: {data['default_preset']}")
                return True
            else:
                print(f"      ERROR: API returned {response.status_code}")
                return False
    except Exception:
        print(f"      ERROR: Cannot connect to API at {API_BASE}")
        print("      Make sure the server is running:")
        print("      .venv/Scripts/uvicorn service.app:app --port 8001")
        return False


def demo_url_scoring():
    print("\n[2/3] Demo: Scoring images from URLs...")
    print("      Testing with safe stock photos from picsum.photos\n")

    test_urls = [
        "https://picsum.photos/seed/demo1/400/300",
        "https://picsum.photos/seed/demo2/500/400",
        "https://picsum.photos/seed/demo3/600/400",
    ]

    payload = {
        "urls": test_urls,
        "preset": "cascade"
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            start = time.time()
            response = client.post(f"{API_BASE}/v1/score-urls", json=payload)
            elapsed = time.time() - start

            if response.status_code == 200:
                result = response.json()
                print(f"      Scored {len(result['results'])} images in {elapsed:.2f}s\n")

                for i, r in enumerate(result['results'], 1):
                    status = "SAFE" if r.get('label') == 0 else "NSFW"
                    score = r.get('score', 0)
                    score_large = r.get('score_large', 0)
                    cached = " (cached)" if r.get('cached') else ""
                    stages = ", ".join(r.get('stages', []))

                    print(f"      Image {i}: {status} {cached}")
                    print(f"         Score: {score:.4f} | Large regions: {score_large:.4f}")
                    print(f"         Strategy: {stages}")
                    print(f"         URL: {r['url'][:60]}...")
                    print()
                return True
            else:
                print(f"      ERROR: {response.status_code} - {response.text}")
                return False
    except Exception as e:
        print(f"      ERROR: {e}")
        return False


def demo_file_upload():
    print("\n[3/3] Demo: Scoring uploaded files...")

    test_images = list(Path("tests/fixtures").glob("*.jpg"))
    if not test_images:
        print("      No test images in tests/fixtures/")
        print("      Tip: Add sample images to tests/fixtures/ for file upload demo")
        return True

    test_image = test_images[0]
    print(f"      Uploading: {test_image.name}")

    try:
        with open(test_image, "rb") as f:
            files = {"file": (test_image.name, f, "image/jpeg")}
            with httpx.Client(timeout=60.0) as client:
                start = time.time()
                response = client.post(
                    f"{API_BASE}/v1/score?preset=cascade",
                    files=files
                )
                elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            status = "SAFE" if result['label'] == 0 else "NSFW"

            print(f"\n      Result: {status}")
            print(f"         Score: {result['score']:.4f}")
            print(f"         Large regions: {result['score_large']:.4f}")
            print(f"         Patches analyzed: {result['n_patches']}")
            print(f"         Strategy stages: {', '.join(result['stages'])}")
            print(f"         Time: {elapsed:.2f}s")
            return True
        else:
            print(f"      ERROR: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"      ERROR: {e}")
        return False


def show_next_steps():
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70)
    print("\nNext steps for your presentation:\n")
    print("1. Interactive API Docs:")
    print(f"   Open: {API_BASE}/docs")
    print("   Try the endpoints directly in your browser\n")

    print("2. Chrome Extension Demo:")
    print("   - Load extension/folder in chrome://extensions")
    print(f"   - Set API URL to: {API_BASE}")
    print("   - Browse any site with images to see auto-blurring\n")

    print("3. Explain/Heatmap Feature:")
    print("   curl -X POST http://127.0.0.1:8001/v1/explain?preset=fast \\")
    print("        -F 'file=@image.jpg' --output heatmap.png\n")

    print("4. Monitoring (if running Docker stack):")
    print("   - Prometheus: http://localhost:9090")
    print("   - Grafana: http://localhost:3000 (admin/safesight)")
    print("   - MLflow: http://localhost:5000\n")

    print("5. Key talking points:")
    print("   - 87.9% recall, 85.5% specificity (paper metrics)")
    print("   - Cascade strategy: adaptive early exits for speed")
    print("   - ONNX backend: 2.5x faster than PyTorch on CPU")
    print("   - Production-ready: caching, metrics, SSRF protection")
    print("   - Full MLOps: MLflow tracking, Prometheus/Grafana monitoring")
    print("\n" + "=" * 70 + "\n")


def main():
    print_banner()

    if not check_api():
        sys.exit(1)

    if not demo_url_scoring():
        print("\nURL scoring demo failed. Continuing...")

    if not demo_file_upload():
        print("\nFile upload demo failed (this is OK if no test images exist)")

    show_next_steps()

    return 0


if __name__ == "__main__":
    sys.exit(main())
