"""PII detection.

Detectors:
- RegexPIIDetector (default): emails, phone numbers, card numbers (with a
  Luhn checksum so random 16-digit IDs are not flagged), Indian PAN and
  Aadhaar, US SSN, IPv4. Fast (~0.1 ms), no model, fully testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    entity: str
    text: str


class PIIDetector(Protocol):
    def detect(self, text: str) -> list[Span]: ...


def luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Order = priority when two patterns overlap (earlier wins).
PATTERNS: list[tuple[str, re.Pattern, callable]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), None),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ \-]?){12,18}\d\b"), lambda s: 13 <= len(re.sub(r"\D", "", s)) <= 19 and luhn_ok(re.sub(r"\D", "", s))),
    ("IN_AADHAAR", re.compile(r"(?<!\d[ \-])\b[2-9]\d{3}[ \-]?\d{4}[ \-]?\d{4}\b(?![ \-]?\d)"), None),
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), None),
    ("IN_PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None),
    ("PHONE", re.compile(r"(?<![\w\-])(?<!\d[ \-])(?:\+\d{1,3}[\s\-]?)?(?:\(\d{2,5}\)[\s\-]?)?\d{3,5}[\s\-]?\d{3,4}[\s\-]?\d{0,4}(?![\w\-])(?![ \-]\d)"),
     lambda s: 10 <= len(re.sub(r"\D", "", s)) <= 13),
    ("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), None),
]


class RegexPIIDetector:
    def __init__(self, entities: set[str] | None = None):
        self.patterns = [p for p in PATTERNS if entities is None or p[0] in entities]

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        taken: list[tuple[int, int]] = []
        for entity, pattern, check in self.patterns:
            for m in pattern.finditer(text):
                s, e = m.span()
                value = m.group().strip()
                e = s + len(m.group().rstrip())
                if check and not check(value):
                    continue
                if any(s < te and ts < e for ts, te in taken):
                    continue
                spans.append(Span(s, e, entity, text[s:e]))
                taken.append((s, e))
        return sorted(spans, key=lambda sp: sp.start)
