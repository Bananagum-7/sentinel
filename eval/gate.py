"""CI gate: fail the build if a guard regresses.

Runs the offline evals and checks them against the thresholds below. A pull
request that, say, loosens an injection rule and lets two more attacks
through fails CI before it can be merged.

Thresholds are set just under the current measured values; raise them as
the guards improve, never lower them silently.
"""

from __future__ import annotations

import asyncio
import sys

from common import DATA, load_jsonl
from run_cache import run as run_cache
from run_injection import evaluate as eval_injection
from run_overhead import main_async as run_overhead
from run_pii import build as build_pii

from sentinel.cache import HashEmbedder
from sentinel.guards.injection import HeuristicInjectionDetector
from sentinel.guards.pii import RegexPIIDetector

GATES = {
    "injection_recall_min": 0.90,
    "injection_fpr_max": 0.05,
    "pii_recall_min": 0.95,
    "pii_false_positives_max": 0,
    "cache_wrong_hits_max": 0,
    "overhead_p95_ms_max": 25.0,
}


def main() -> int:
    failures = []

    inj = eval_injection(HeuristicInjectionDetector(), load_jsonl(DATA / "attacks_custom.jsonl") + load_jsonl(DATA / "benign.jsonl"), 0.5)
    if inj["recall"] < GATES["injection_recall_min"]:
        failures.append(f"injection recall {inj['recall']} < {GATES['injection_recall_min']}")
    if inj["false_positive_rate"] > GATES["injection_fpr_max"]:
        failures.append(f"injection FPR {inj['false_positive_rate']} > {GATES['injection_fpr_max']}")

    det, rows = RegexPIIDetector(), build_pii(30)
    pos = [r for r in rows if r["entity"]]
    found = sum((r["entity"], r["value"]) in {(s.entity, s.text) for s in det.detect(r["text"])} for r in pos)
    pii_recall = found / len(pos)
    fps = sum(len(det.detect(r["text"])) for r in rows if not r["entity"])
    if pii_recall < GATES["pii_recall_min"]:
        failures.append(f"PII recall {pii_recall:.3f} < {GATES['pii_recall_min']}")
    if fps > GATES["pii_false_positives_max"]:
        failures.append(f"PII false positives {fps} > {GATES['pii_false_positives_max']}")

    cache = run_cache(HashEmbedder(), load_jsonl(DATA / "cache_pairs.jsonl"), threshold=0.75, guard=True)
    if cache["wrong_hits"] > GATES["cache_wrong_hits_max"]:
        failures.append(f"cache wrong hits {cache['wrong_hits']} > 0")

    ovh = asyncio.run(run_overhead(n=200, upstream_ms=50, concurrency=20))
    if ovh["overhead_p95_ms"] > GATES["overhead_p95_ms_max"]:
        failures.append(f"gateway overhead p95 {ovh['overhead_p95_ms']} ms > {GATES['overhead_p95_ms_max']} ms")

    print(f"injection recall={inj['recall']} fpr={inj['false_positive_rate']} | pii recall={pii_recall:.3f} fps={fps} | "
          f"cache para_hits={cache['paraphrase_hit_rate']} wrong={cache['wrong_hits']} | overhead p95={ovh['overhead_p95_ms']} ms")
    if failures:
        print("GATE FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("All gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
