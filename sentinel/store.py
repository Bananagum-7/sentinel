"""Request log in SQLite: one row per request, for the dashboard and debugging.

Deliberately NOT stored: prompts, completions, PII values, the redaction
mapping, API keys. Only metadata and counts. If you need prompts for
debugging, store the redacted version behind an explicit opt-in.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# USD per 1M tokens (input, output). Estimates at paid-tier list prices;
# check the provider's pricing page and update. Free-tier traffic costs $0.
PRICES = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000


@dataclass
class RequestLog:
    request_id: str
    app: str
    requested_model: str
    ts: float = field(default_factory=time.time)
    served_model: str = ""
    provider: str = ""
    status: int = 200
    outcome: str = "ok"  # ok | blocked_injection | blocked_pii | rate_limited | upstream_error | bad_request
    cache: str = "miss"  # exact | semantic | miss | bypass
    stream: bool = False
    latency_ms: float = 0.0
    guard_ms: float = 0.0
    upstream_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_cost_usd: float = 0.0
    pii_redacted: int = 0
    pii_in_output: int = 0
    injection_score: float = 0.0
    injection_rules: str = ""
    attempts: int = 0
    fallback_used: bool = False


SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY, app TEXT, requested_model TEXT, ts REAL, served_model TEXT, provider TEXT,
    status INTEGER, outcome TEXT, cache TEXT, stream INTEGER, latency_ms REAL, guard_ms REAL, upstream_ms REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER, est_cost_usd REAL, pii_redacted INTEGER, pii_in_output INTEGER,
    injection_score REAL, injection_rules TEXT, attempts INTEGER, fallback_used INTEGER
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
"""


class RequestStore:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def add(self, row: RequestLog) -> None:
        d = asdict(row)
        cols = ",".join(d)
        with self.lock:
            self.conn.execute(f"INSERT OR REPLACE INTO requests ({cols}) VALUES ({','.join('?' * len(d))})", list(d.values()))
            self.conn.commit()

    def summary(self, since_seconds: float = 3600) -> dict:
        since = time.time() - since_seconds
        with self.lock:
            cur = self.conn.execute(
                """SELECT COUNT(*), SUM(outcome='ok'), SUM(outcome LIKE 'blocked%'), SUM(outcome='rate_limited'),
                          SUM(outcome='upstream_error'), SUM(cache IN ('exact','semantic')), SUM(fallback_used),
                          SUM(prompt_tokens), SUM(completion_tokens), SUM(est_cost_usd), SUM(pii_redacted)
                   FROM requests WHERE ts >= ?""", (since,))
            row = cur.fetchone()
            lat = [r[0] for r in self.conn.execute("SELECT latency_ms FROM requests WHERE ts >= ? AND outcome='ok' ORDER BY latency_ms", (since,))]
        n = row[0] or 0

        def pct(p):
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else 0.0

        return {
            "window_s": since_seconds, "requests": n, "ok": row[1] or 0, "blocked": row[2] or 0,
            "rate_limited": row[3] or 0, "upstream_errors": row[4] or 0,
            "cache_hit_rate": round((row[5] or 0) / n, 3) if n else 0.0, "fallbacks": row[6] or 0,
            "prompt_tokens": row[7] or 0, "completion_tokens": row[8] or 0, "est_cost_usd": round(row[9] or 0, 6),
            "pii_redacted": row[10] or 0, "p50_ms": pct(0.5), "p95_ms": pct(0.95),
        }
