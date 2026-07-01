"""Disk cache of model responses, keyed by a hash of the full request.

Every API call is cached so reruns cost $0 — the plan's headline cost mitigation.
The key is a SHA-256 of the request payload (model + messages + tools + all
sampling params), so any change to the request is a cache miss and anything
identical is a hit. Values are stored as individual JSON files under a cache
directory; the cache is regenerable and gitignored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path(".cache") / "responses"


def request_key(payload: dict[str, Any]) -> str:
    """Stable SHA-256 hash of a request payload.

    ``sort_keys`` makes the hash independent of dict ordering, so semantically
    identical requests collide (a hit) regardless of how they were built.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """A content-addressed disk cache for provider responses."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Shard by first 2 hex chars to avoid one giant directory.
        sub = self.cache_dir / key[:2]
        sub.mkdir(exist_ok=True)
        return sub / f"{key}.json"

    def get(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        path = self._path(request_key(payload))
        if not path.exists():
            return None
        with path.open() as f:
            return json.load(f)

    def set(self, payload: dict[str, Any], response: dict[str, Any]) -> None:
        path = self._path(request_key(payload))
        # Write atomically: temp file + rename, so a crash never leaves a
        # half-written cache entry that later reads as valid.
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(response, f)
        tmp.replace(path)

    def __contains__(self, payload: dict[str, Any]) -> bool:
        return self._path(request_key(payload)).exists()
