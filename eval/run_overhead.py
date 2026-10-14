"""How much latency does the gateway itself add?

Runs N requests through the full gateway (guards, cache lookup, logging) with
an upstream that answers after a fixed delay, and compares against calling
that upstream directly. Overhead = gateway latency - upstream latency.

    python eval/run_overhead.py --n 300 --upstream-ms 300 --concurrency 20

For HTTP-level load testing with real network, see eval/locustfile.py.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from common import save
from fastapi.testclient import TestClient  # noqa: F401  (keeps deps consistent with tests)

from sentinel.cache import HashEmbedder, SemanticCache
from sentinel.gateway import Gateway
from sentinel.guards.injection import HeuristicInjectionDetector
from sentinel.guards.pii import RegexPIIDetector
from sentinel.policy import AppPolicy, CacheCfg, RateLimit, Route
from sentinel.providers import FakeProvider
from sentinel.store import RequestStore

PROMPTS = [
    "My order {i} hasn't arrived, my email is user{i}@example.com",
    "How do I reset my password? Attempt {i}",
    "Summarize ticket {i}: customer reports login failure after update",
]


async def main_async(n: int, upstream_ms: float, concurrency: int) -> dict:
    provider = FakeProvider("fake", ["Thanks, I've looked into it and here is what to do next."], latency=upstream_ms / 1000)
    gw = Gateway({"fake": provider}, RegexPIIDetector(), HeuristicInjectionDetector(), SemanticCache(HashEmbedder()), RequestStore())
    policy = AppPolicy(name="bench", rate_limit=RateLimit(requests_per_minute=10**9, burst=10**9),
                       cache=CacheCfg(enabled=False), models={"default": [Route(provider="fake", model="m")]})
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []

    async def one(i: int):
        body = {"model": "m", "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)].format(i=i)}]}
        async with sem:
            t0 = time.perf_counter()
            await gw.complete(policy, body)
            lat.append((time.perf_counter() - t0) * 1000)

    t_all = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(n)))
    wall = time.perf_counter() - t_all
    lat.sort()
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]
    return {"n": n, "concurrency": concurrency, "upstream_ms": upstream_ms,
            "gateway_p50_ms": round(p(0.5), 2), "gateway_p95_ms": round(p(0.95), 2),
            "overhead_p50_ms": round(p(0.5) - upstream_ms, 2), "overhead_p95_ms": round(p(0.95) - upstream_ms, 2),
            "throughput_rps": round(n / wall, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--upstream-ms", type=float, default=300)
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    r = asyncio.run(main_async(args.n, args.upstream_ms, args.concurrency))
    for k, v in r.items():
        print(f"{k:18s} {v}")
    save("overhead", r)


if __name__ == "__main__":
    main()
