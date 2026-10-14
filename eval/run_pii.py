"""PII detector eval on synthetic sentences with exact labels.

    python eval/run_pii.py            # regex detector
    python eval/run_pii.py --presidio # + Presidio (names)

Faker generates realistic values (never real people). Each sentence records
the exact value it inserted, so labels are free and exact. 50 "hard negative"
sentences full of order numbers, prices, dates and versions measure false
positives, which is what makes a redactor unusable in practice.
"""

from __future__ import annotations

import argparse
import random
import string
from collections import defaultdict

from common import save
from faker import Faker

from sentinel.guards.pii import RegexPIIDetector, luhn_ok

TEMPLATES = {
    "EMAIL": ["Please reply to {v} when it's done.", "My email is {v}", "cc {v} on the ticket", "Contact: {v}."],
    "PHONE": ["Call me on {v} after 5pm.", "my number is {v}", "Reach the store at {v}.", "WhatsApp {v} for updates"],
    "CREDIT_CARD": ["Charge card {v} for the upgrade.", "card number {v}, expiry next year", "I paid with {v}"],
    "IN_PAN": ["My PAN is {v}.", "PAN: {v} for the invoice", "Use PAN {v} on the GST bill"],
    "IN_AADHAAR": ["Aadhaar number {v} for KYC.", "my aadhaar is {v}", "KYC doc: {v}"],
    "US_SSN": ["SSN {v} on the W-2 form", "my social is {v}"],
    "IP_ADDRESS": ["Login came from {v} at midnight.", "Server {v} is down", "whitelist {v} please"],
}

NEGATIVES = [
    "Order 4471 shipped on 2024-03-01 and should arrive in 5 days.",
    "Revenue in 2022 was 1,749 million, up 12% from 2021.",
    "Upgrade to version 3.10.4 or later.",
    "The invoice total is Rs 12,499.00 including 18% GST.",
    "Ticket #88213 was escalated to tier 2.",
    "Our office is open 9 to 6, Monday to Saturday.",
    "Tracking ID 4111 1111 1111 1112 is not valid.",
    "SKU 100245-BLK is back in stock, 250 units.",
    "Build 2024.11.3 fixed the crash on startup.",
    "The meeting is at 10:30 on 14/10/2026.",
    "We processed 1234567890 events last month.",
    "Use coupon SAVE2024 for 20% off.",
    "The model has 70 billion parameters and an 8192 token context.",
    "Pin code 600017 is in Chennai.",
    "The API returned status 429 after 60 requests.",
    "Room 302, floor 3, block B.",
    "Batch 7 of 12 completed in 45.2 seconds.",
    "Latency p95 was 812 ms at 50 rps.",
    "Flight 6E-2134 departs at 07:45.",
    "Account balance: 2,45,000 INR.",
]


def gen_value(fake: Faker, entity: str) -> str:
    if entity == "EMAIL":
        return fake.email()
    if entity == "PHONE":
        return random.choice([f"+91 9{random.randint(100000000, 999999999)}", f"9{random.randint(100000000, 999999999)}",
                              f"+1 {random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}",
                              f"(0{random.randint(20, 99)}) {random.randint(2000, 9999)} {random.randint(1000, 9999)}"])
    if entity == "CREDIT_CARD":
        while True:
            n = fake.credit_card_number(card_type=random.choice(["visa16", "mastercard"]))
            if luhn_ok(n):
                return " ".join(n[i : i + 4] for i in range(0, len(n), 4)) if random.random() < 0.5 else n
    if entity == "IN_PAN":
        return "".join(random.choices(string.ascii_uppercase, k=5)) + f"{random.randint(0, 9999):04d}" + random.choice(string.ascii_uppercase)
    if entity == "IN_AADHAAR":
        d = str(random.randint(2, 9)) + "".join(random.choices(string.digits, k=11))
        return f"{d[:4]} {d[4:8]} {d[8:]}" if random.random() < 0.7 else d
    if entity == "US_SSN":
        return fake.ssn()
    if entity == "IP_ADDRESS":
        return fake.ipv4()
    raise ValueError(entity)


def build(n_per_entity: int, seed: int = 7) -> list[dict]:
    random.seed(seed)
    fake = Faker()
    Faker.seed(seed)
    rows = []
    for entity, templates in TEMPLATES.items():
        for _ in range(n_per_entity):
            v = gen_value(fake, entity)
            rows.append({"text": random.choice(templates).format(v=v), "entity": entity, "value": v})
    rows += [{"text": t, "entity": None, "value": None} for t in NEGATIVES]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="sentences per entity type")
    ap.add_argument("--presidio", action="store_true")
    args = ap.parse_args()

    detector = RegexPIIDetector()
    if args.presidio:
        from sentinel.guards.pii import PresidioPIIDetector

        detector = PresidioPIIDetector()
    rows = build(args.n)
    tp, fn, fp = defaultdict(int), defaultdict(int), defaultdict(int)
    errors = []
    for r in rows:
        spans = detector.detect(r["text"])
        found = {(s.entity, s.text) for s in spans}
        if r["entity"]:
            if (r["entity"], r["value"]) in found:
                tp[r["entity"]] += 1
            else:
                fn[r["entity"]] += 1
                errors.append(("missed", r["text"], sorted(found)))
            for e, t in found:
                if (e, t) != (r["entity"], r["value"]):
                    fp[e] += 1
                    errors.append(("extra", r["text"], (e, t)))
        else:
            for e, t in found:
                fp[e] += 1
                errors.append(("false positive", r["text"], (e, t)))

    per = {}
    print(f"{'entity':12s} {'recall':>7s} {'precision':>9s}")
    for e in TEMPLATES:
        rec = tp[e] / (tp[e] + fn[e]) if tp[e] + fn[e] else None
        prec = tp[e] / (tp[e] + fp[e]) if tp[e] + fp[e] else None
        per[e] = {"recall": round(rec, 3) if rec is not None else None, "precision": round(prec, 3) if prec is not None else None}
        print(f"{e:12s} {per[e]['recall']!s:>7s} {per[e]['precision']!s:>9s}")
    total_tp, total_fn, total_fp = sum(tp.values()), sum(fn.values()), sum(fp.values())
    overall = {"recall": round(total_tp / (total_tp + total_fn), 3), "precision": round(total_tp / (total_tp + total_fp), 3) if total_tp + total_fp else None,
               "false_positives_on_negatives": sum(1 for kind, *_ in errors if kind == "false positive"), "n_sentences": len(rows)}
    print(f"overall      {overall['recall']:>7} {overall['precision']!s:>9s}   false positives on {len(NEGATIVES)} hard negatives: {overall['false_positives_on_negatives']}")
    for kind, text, detail in errors[:15]:
        print(f"  {kind}: {text!r} -> {detail}")
    save("pii", {"overall": overall, "per_entity": per, "errors": [list(map(str, e)) for e in errors]})


if __name__ == "__main__":
    main()
