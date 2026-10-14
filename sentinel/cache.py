"""Exact + semantic response cache.

Lookup order:
1. Exact: SHA-256 of the normalised (redacted) messages + model. Always safe.
2. Semantic: embed the last user message and compare (cosine) against cached
   questions that share the same model and the same system prompt. A hit
   needs similarity >= threshold AND identical "critical tokens".

Critical tokens are the classic semantic-cache failure: "revenue in 2021" and
"revenue in 2022" embed almost identically but need different answers. So
numbers, quoted strings and code-like identifiers must match exactly between
the new question and the cached one, or it is a miss.

What is never cached:
- requests with temperature > 0 (the caller asked for variety)
- requests with tools (they drive actions)
- requests that contained PII (placeholders map to different people per request)
- apps whose policy disables the cache (e.g. an agent that issues refunds)
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np

CRITICAL_RE = re.compile(r"\d+(?:[.,]\d+)*|\"[^\"]+\"|'[^']+'|`[^`]+`|\b[A-Za-z]+_[A-Za-z0-9_]+\b|\b[a-z]+[A-Z]\w*\b|\b[A-Z][a-z]+[A-Z]\w*\b")


def critical_tokens(text: str) -> frozenset[str]:
    return frozenset(t.lower().strip("\"'`") for t in CRITICAL_RE.findall(text))


def last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if p.get("type") == "text")
    return ""


def context_key(messages: list[dict], model: str) -> str:
    """Everything except the last user message: model + system prompt + earlier turns."""
    idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=len(messages))
    prefix = json.dumps({"model": model, "messages": messages[:idx]}, sort_keys=True)
    return hashlib.sha256(prefix.encode()).hexdigest()


def exact_key(messages: list[dict], model: str) -> str:
    return hashlib.sha256(json.dumps({"model": model, "messages": messages}, sort_keys=True).encode()).hexdigest()


@dataclass
class _Entry:
    question: str
    critical: frozenset[str]
    vector: np.ndarray
    response: dict
    expires: float


@dataclass
class CacheResult:
    kind: str  # "exact" | "semantic" | "miss"
    response: dict | None = None
    similarity: float | None = None


@dataclass
class SemanticCache:
    embedder: object
    max_entries: int = 5000
    _exact: dict[str, tuple[dict, float]] = field(default_factory=dict)
    _buckets: dict[str, list[_Entry]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def lookup(self, scope: str, messages: list[dict], model: str, threshold: float, now: float | None = None) -> CacheResult:
        now = now or time.time()
        ekey = f"{scope}:{exact_key(messages, model)}"
        with self._lock:
            hit = self._exact.get(ekey)
            if hit and hit[1] > now:
                return CacheResult("exact", hit[0], 1.0)
            bucket = self._buckets.get(f"{scope}:{context_key(messages, model)}", [])
            live = [e for e in bucket if e.expires > now]
        question = last_user_text(messages)
        if not live or not question:
            return CacheResult("miss")
        q_vec = self.embedder.embed_query(question)
        sims = np.stack([e.vector for e in live]) @ q_vec
        crit = critical_tokens(question)
        for i in np.argsort(-sims):
            if sims[i] < threshold:
                break
            if live[i].critical == crit:
                return CacheResult("semantic", live[i].response, float(sims[i]))
        return CacheResult("miss", similarity=float(sims.max()))

    def store(self, scope: str, messages: list[dict], model: str, response: dict, ttl: int, now: float | None = None) -> None:
        now = now or time.time()
        question = last_user_text(messages)
        entry = _Entry(question, critical_tokens(question), self.embedder.embed_query(question) if question else None, response, now + ttl)
        with self._lock:
            self._exact[f"{scope}:{exact_key(messages, model)}"] = (response, now + ttl)
            if question:
                bucket = self._buckets.setdefault(f"{scope}:{context_key(messages, model)}", [])
                bucket.append(entry)
                if len(bucket) > self.max_entries:
                    del bucket[: len(bucket) - self.max_entries]
            if len(self._exact) > self.max_entries:
                for k in list(self._exact)[: len(self._exact) - self.max_entries]:
                    del self._exact[k]


class HashEmbedder:
    """Model-free embedder (hashed bag of words + bigrams) for tests and offline runs."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed_query(self, text: str) -> np.ndarray:
        words = re.findall(r"[a-z0-9]+", text.lower())
        feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for w in feats:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 9) % 2 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    def embed_query(self, text: str) -> np.ndarray:
        v = np.array(next(iter(self.model.embed([text]))), dtype=np.float32)
        return v / np.linalg.norm(v)
