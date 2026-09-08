"""Lightweight in-process service statistics for the demo dashboard.

Prometheus metrics remain the operational source of truth; this module keeps
a small thread-safe aggregate that the /v1/stats endpoint can return as JSON
without a Prometheus server in the loop (the dashboard polls it directly).
"""

import threading
from collections import Counter, deque


class ServiceStats:
    def __init__(self, latency_window: int = 500):
        self._lock = threading.Lock()
        self._latencies = deque(maxlen=latency_window)
        self._stage_exits: Counter = Counter()
        self.total = 0
        self.nsfw = 0
        self.errors = 0

    def record(self, label: int, seconds: float, stages: list) -> None:
        with self._lock:
            self.total += 1
            if label == 1:
                self.nsfw += 1
            self._latencies.append(seconds)
            exit_stage = stages[-1] if stages else "direct"
            self._stage_exits[exit_stage] += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self, cache_hits: int, cache_misses: int) -> dict:
        with self._lock:
            lat = sorted(self._latencies)
            n = len(lat)
            p50 = lat[n // 2] if n else 0.0
            p95 = lat[min(n - 1, int(n * 0.95))] if n else 0.0
            lookups = cache_hits + cache_misses
            return {
                "images_scored": self.total,
                "nsfw_flagged": self.nsfw,
                "nsfw_rate": round(self.nsfw / self.total, 4) if self.total else 0.0,
                "fetch_errors": self.errors,
                "cache_hit_rate": round(cache_hits / lookups, 4) if lookups else 0.0,
                "latency_p50_ms": round(p50 * 1000),
                "latency_p95_ms": round(p95 * 1000),
                "cascade_exit_counts": dict(self._stage_exits),
            }
