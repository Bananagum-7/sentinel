"""Semantic cache eval: pick the similarity threshold from data, not by feel.

    python eval/run_cache.py             # hash embedder (offline)
    python eval/run_cache.py --fastembed # real embeddings (bge-small)

For each threshold we measure:
    paraphrase hit rate - share of 30 paraphrase pairs served from cache (want high)
    wrong hits          - near-miss pairs served from cache (must be 0: a wrong answer)
and report the intent guard's effect by running with and without it.
"""

from __future__ import annotations

import argparse

from common import DATA, load_jsonl, save

from sentinel.cache import SemanticCache

SYSTEM = {"role": "system", "content": "You are the support assistant for an online store."}


def run(embedder, pairs, threshold: float, guard: bool) -> dict:
    hits = wrong = 0
    n_para = sum(p["should_hit"] for p in pairs)
    n_near = len(pairs) - n_para
    for p in pairs:
        c = SemanticCache(embedder)  # fresh cache per pair: isolates each judgement
        if not guard:
            c.intent_guard = lambda a, b: True
        c.store("app", [SYSTEM, {"role": "user", "content": p["cached"]}], "m", {"a": 1}, ttl=60)
        res = c.lookup("app", [SYSTEM, {"role": "user", "content": p["query"]}], "m", threshold)
        served = res.response is not None
        if p["should_hit"] and served:
            hits += 1
        if not p["should_hit"] and served:
            wrong += 1
    return {"threshold": threshold, "guard": guard, "paraphrase_hit_rate": round(hits / n_para, 3), "wrong_hits": wrong, "n_near_miss": n_near}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastembed", action="store_true")
    args = ap.parse_args()
    if args.fastembed:
        from sentinel.cache import FastEmbedEmbedder

        embedder = FastEmbedEmbedder()
    else:
        from sentinel.cache import HashEmbedder

        embedder = HashEmbedder()
    pairs = load_jsonl(DATA / "cache_pairs.jsonl")
    rows = []
    print(f"{'thresh':>6s} {'guard':>5s} {'para hit':>8s} {'wrong':>5s}")
    for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        for guard in (False, True):
            r = run(embedder, pairs, t, guard)
            rows.append(r)
            print(f"{t:>6.2f} {str(guard):>5s} {r['paraphrase_hit_rate']:>8} {r['wrong_hits']:>5}/{r['n_near_miss']}")
    safe = [r for r in rows if r["guard"] and r["wrong_hits"] == 0]
    best = max(safe, key=lambda r: r["paraphrase_hit_rate"]) if safe else None
    if best:
        print(f"\nBest safe setting: threshold {best['threshold']} with the guard -> {best['paraphrase_hit_rate']:.0%} paraphrase hits, 0 wrong hits")
    save("cache", {"embedder": "fastembed" if args.fastembed else "hash", "rows": rows, "best": best})


if __name__ == "__main__":
    main()
