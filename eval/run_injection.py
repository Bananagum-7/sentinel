"""Injection guard eval: recall on attacks, false-positive rate on benign prompts.

    python eval/run_injection.py                 # hand-written sets (offline)
    python eval/run_injection.py --public        # + deepset/prompt-injections from Hugging Face
    python eval/run_injection.py --classifier    # + the DeBERTa classifier (needs transformers, torch)

False-positive rate matters as much as recall: a guard that blocks normal
users gets switched off. The benign set is deliberately full of trigger words
("ignore", "act as", "system prompt", "previous") used innocently.
"""

from __future__ import annotations

import argparse
import time

from common import DATA, load_jsonl, percentile, save

from sentinel.guards.injection import CombinedInjectionDetector, HeuristicInjectionDetector


def load_public() -> list[dict]:
    from datasets import load_dataset  # pip install datasets

    ds = load_dataset("deepset/prompt-injections", split="test")
    return [{"text": r["text"], "label": int(r["label"]), "source": "deepset"} for r in ds]


def evaluate(detector, rows: list[dict], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    lat, missed, false_alarms = [], [], []
    for r in rows:
        t0 = time.perf_counter()
        s = detector.score(r["text"]).score
        lat.append((time.perf_counter() - t0) * 1000)
        pred = s >= threshold
        if r["label"] and pred:
            tp += 1
        elif r["label"]:
            fn += 1
            missed.append(r["text"])
        elif pred:
            fp += 1
            false_alarms.append(r["text"])
        else:
            tn += 1
    return {
        "n_attacks": tp + fn, "n_benign": tn + fp,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else None,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "p50_ms": round(percentile(lat, 0.5), 3), "p95_ms": round(percentile(lat, 0.95), 3),
        "missed": missed, "false_alarms": false_alarms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--classifier", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    sets = {"custom": load_jsonl(DATA / "attacks_custom.jsonl") + load_jsonl(DATA / "benign.jsonl")}
    if args.public:
        sets["deepset"] = load_public()

    detectors = {"heuristic": HeuristicInjectionDetector()}
    if args.classifier:
        from sentinel.guards.injection import ClassifierInjectionDetector

        clf = ClassifierInjectionDetector()
        detectors["classifier"] = clf
        detectors["combined"] = CombinedInjectionDetector([detectors["heuristic"], clf])

    results = {}
    print(f"{'set':10s} {'detector':11s} {'recall':>7s} {'FPR':>6s} {'prec':>6s} {'p95 ms':>7s}")
    for set_name, rows in sets.items():
        for det_name, det in detectors.items():
            r = evaluate(det, rows, args.threshold)
            results[f"{set_name}/{det_name}"] = r
            print(f"{set_name:10s} {det_name:11s} {r['recall']:>7} {r['false_positive_rate']:>6} {str(r['precision']):>6s} {r['p95_ms']:>7}")

    # Threshold sweep for the heuristic on the custom set
    sweep = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        r = evaluate(detectors["heuristic"], sets["custom"], t)
        sweep.append({"threshold": t, "recall": r["recall"], "fpr": r["false_positive_rate"]})
    print("\nHeuristic threshold sweep (custom set):")
    for s in sweep:
        print(f"  t={s['threshold']:.1f} recall={s['recall']} fpr={s['fpr']}")

    save("injection", {"threshold": args.threshold, "results": results, "sweep": sweep})
    miss = results["custom/heuristic"]["missed"]
    if miss:
        print(f"\nMissed by heuristic ({len(miss)}), the cases a classifier should catch:")
        for m in miss:
            print("  -", m[:90])


if __name__ == "__main__":
    main()
