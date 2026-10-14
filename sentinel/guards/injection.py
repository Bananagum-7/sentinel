"""Prompt-injection detection.

Two detectors, combined by taking the higher score:

HeuristicInjectionDetector - weighted regex rules for well-known attack
    phrasing ("ignore previous instructions", "reveal your system prompt",
    chat-template tokens...). Scores combine as 1 - prod(1 - w): several weak
    signals add up, one strong signal is enough. ~0.1 ms, no model.

ClassifierInjectionDetector (optional) - a small fine-tuned DeBERTa model
    from Hugging Face. Catches paraphrased attacks the rules miss, ~20-60 ms
    on CPU. pip install transformers torch

What we scan: user messages and tool results (role "tool"), because
"indirect" injection arrives through retrieved documents, emails or web
pages that a tool returned. System prompts are not scanned: they are written
by the app developer and naturally contain phrases like "you are".

No detector stops injection completely. The real defence is architectural:
the model must never hold authority it shouldn't (see the Helpdesk agent,
where refunds need deterministic checks and human approval).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class InjectionResult:
    score: float
    matched: list[str] = field(default_factory=list)
    detector: str = ""


class InjectionDetector(Protocol):
    def score(self, text: str) -> InjectionResult: ...


RULES: list[tuple[str, float, str]] = [
    # (name, weight, pattern). Weights are tuned so one strong signal (>= 0.5) blocks
    # on its own, while weak signals (< 0.5) only block in combination.
    ("ignore_instructions", 0.85,
     r"\b(ignore|disregard|forget|override|bypass|set aside|put aside|pay no attention to|stop following|don'?t follow)\b"
     r".{0,40}\b(previous|prior|above|earlier|all|any|your|the|whatever)\b.{0,30}\b(instructions?|prompts?|rules|directions|guidelines|guidance|context|polic(y|ies))\b"),
    ("instructions_void", 0.75,
     r"\b(instructions?|directions|rules|guidance|guidelines)\b.{0,30}\b(no longer apply|are (now )?void|don'?t (apply|matter)|are cancelled)\b"),
    ("new_instructions", 0.6,
     r"(\b(new|updated|real|actual)\s+(instructions?|rules|system prompt)\b\s*[:\-])|(\b(system prompt|instructions?)\s+(update|change|override)\b)"),
    ("reveal_prompt", 0.8,
     r"\b(reveal|show|print|repeat|output|leak|tell me|summari[sz]e|what (is|are))\b.{0,30}\b(your|the|its)\s+"
     r"(?:(?:full|exact|hidden|system|secret|initial|original|confidential)\s+){1,3}(prompt|instructions?|message|rules)\b(?!\s+(for|of)\b)"),
    ("prompt_extraction", 0.7,
     r"\b(repeat|print|output|begin|start)\b.{0,30}\b(text|everything|words) (above|before)\b"
     r"|\b(instructions?|rules)\b.{0,20}\b(before my first message|you operate under|you were given)\b"
     r"|\b(begin|start) your (reply|response|answer) with\b.{0,30}\b(rules|instructions|prompt)\b"),
    ("role_override", 0.4, r"\b(you are now|from now on,? you|act as|pretend (to be|you are)|roleplay as|you will now)\b"),
    ("jailbreak_strong", 0.85, r"\b(DAN|do anything now|jailbroken)\b"),
    ("jailbreak_weak", 0.35, r"\b(developer mode|jailbreak|unfiltered|no (rules|restrictions|limits|content policy)|two personas)\b"),
    ("unrestricted_answer", 0.6,
     r"\b(answer|respond|reply|continue)\b.{0,30}\b(without|with no)\s+(any\s+)?(limits|restrictions|filters|rules)\b"
     r"|\b(policy|policies|restrictions|filters) (was|were|have been|has been) (removed|lifted|disabled)\b"),
    ("template_tokens", 0.9, r"(<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[INST\]|<<SYS>>|###\s*(system|instruction)\s*:)"),
    ("exfiltration", 0.6, r"\b(send|post|forward|upload|exfiltrate)\b.{0,40}\b(to|at)\b.{0,20}(https?://|\b[\w.-]+@[\w.-]+\b)"),
    ("authority_claim", 0.35, r"\b(i am|this is)\s+(your|the)\s+(developer|admin|administrator|creator|owner|openai|anthropic|system)\b"),
    ("disable_safety", 0.6,
     r"\b(disable|turn off|deactivate|bypass|circumvent|get around)\b.{0,30}\b(safety|guardrails?|moderation|content filters?|your (filters|restrictions))\b"),
    ("addressed_to_ai", 0.45,
     r"\b(assistant|ai|agent|model|bot|llm)\b\s*[,:\-]\s*.{0,20}\b(ignore|approve|issue|send|forward|reveal|call|disregard)\b"),
    ("skip_review", 0.5, r"\b(without|skip(ping)?|bypass(ing)?)\s+(any\s+)?(human|manual)?\s*(review|approval|confirmation)\b"),
    ("tool_hijack", 0.55, r"\b(call|invoke|run|execute|use)\b.{0,20}\b(tool|function)\b.{0,40}\b(without|skip|bypass)\b.{0,20}\b(approval|confirmation|check)"),
]

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
SPACED_RE = re.compile(r"\b(?:\w[\-\.\s_*]){3,}\w\b")


def normalise(text: str) -> str:
    """Undo cheap obfuscation: leetspeak (1gn0re) and spaced letters (i-g-n-o-r-e)."""
    t = text.translate(LEET)
    return SPACED_RE.sub(lambda m: re.sub(r"[\-\.\s_*]", "", m.group()) if all(len(p) == 1 for p in re.split(r"[\-\.\s_*]", m.group())) else m.group(), t)


class HeuristicInjectionDetector:
    def __init__(self, rules=RULES):
        self.rules = [(n, w, re.compile(p, re.IGNORECASE | re.DOTALL)) for n, w, p in rules]

    def score(self, text: str) -> InjectionResult:
        matched, keep = [], 1.0
        variants = {text, normalise(text)}
        for name, weight, pattern in self.rules:
            if any(pattern.search(v) for v in variants):
                matched.append(name)
                keep *= 1.0 - weight
        return InjectionResult(score=round(1.0 - keep, 4), matched=matched, detector="heuristic")


class ClassifierInjectionDetector:
    def __init__(self, model_name: str = "protectai/deberta-v3-base-prompt-injection-v2", max_chars: int = 2000):
        from transformers import pipeline

        self.clf = pipeline("text-classification", model=model_name, truncation=True, max_length=512)
        self.max_chars = max_chars

    def score(self, text: str) -> InjectionResult:
        out = self.clf(text[: self.max_chars])[0]
        p = out["score"] if out["label"].upper() == "INJECTION" else 1.0 - out["score"]
        return InjectionResult(score=round(float(p), 4), matched=["classifier"] if p >= 0.5 else [], detector="classifier")


class CombinedInjectionDetector:
    def __init__(self, detectors: list[InjectionDetector]):
        self.detectors = detectors

    def score(self, text: str) -> InjectionResult:
        results = [d.score(text) for d in self.detectors]
        best = max(results, key=lambda r: r.score)
        return InjectionResult(score=best.score, matched=sorted({m for r in results for m in r.matched}), detector=best.detector)


def scannable_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        if m.get("role") not in ("user", "tool"):
            continue
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(p.get("text", "") for p in content if p.get("type") == "text")
    return "\n".join(parts)
