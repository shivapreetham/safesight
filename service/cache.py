"""Thread-safe LRU cache for URL scoring results, keyed by URL hash."""

import hashlib
import threading
from collections import OrderedDict


def url_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


class LRUCache:
    def __init__(self, max_size: int = 4096):
        self.max_size = max_size
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
