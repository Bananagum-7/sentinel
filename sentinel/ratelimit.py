"""Token-bucket rate limiter, one bucket per app.

Each bucket holds up to `burst` tokens and refills at rpm/60 tokens per second.
A request takes one token; with no token left it is rejected with 429 and a
Retry-After telling the client how long until the next token.

Token bucket over fixed window: it allows short bursts but enforces the
average rate, and it has no "double burst" at window boundaries.

This is in-process. With several gateway replicas, move the bucket to Redis
(an atomic Lua script doing the same refill-and-take) so all replicas share it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    def __init__(self, clock=time.monotonic):
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def try_acquire(self, key: str, rpm: int, burst: int) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        rate = rpm / 60.0
        now = self._clock()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = _Bucket(tokens=float(burst), updated=now)
            b.tokens = min(float(burst), b.tokens + (now - b.updated) * rate)
            b.updated = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return True, 0.0
            return False, (1.0 - b.tokens) / rate if rate > 0 else float("inf")
