"""The request pipeline, independent of the web framework.

    rate limit -> validate -> injection guard -> PII guard -> cache
      -> upstream (with fallback) -> restore PII -> output guard -> cache store -> log

Every step records into one RequestLog row and one trace.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from sentinel.cache import SemanticCache
from sentinel.guards.injection import InjectionDetector, scannable_text
from sentinel.guards.pii import PIIDetector, Redactor, redact_messages
from sentinel.policy import AppPolicy
from sentinel.providers import AllRoutesFailed, Provider, ProviderError, route_completion
from sentinel.ratelimit import TokenBucketLimiter
from sentinel.store import RequestLog, RequestStore, estimate_cost
from sentinel.tracing import span

PASSTHROUGH = {"temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "seed", "response_format", "tools", "tool_choice", "n"}


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}

    def body(self) -> dict:
        return {"error": {"type": self.code, "message": self.message}}


@dataclass
class GatewayResult:
    response: dict
    headers: dict


class Gateway:
    def __init__(self, providers: dict[str, Provider], pii_detector: PIIDetector, injection_detector: InjectionDetector,
                 cache: SemanticCache | None, store: RequestStore, limiter: TokenBucketLimiter | None = None):
        self.providers, self.pii, self.injection = providers, pii_detector, injection_detector
        self.cache, self.store = cache, store
        self.limiter = limiter or TokenBucketLimiter()

    # ---------- shared guard stages ----------

    def _admit(self, policy: AppPolicy, body: dict, log: RequestLog) -> tuple[list[dict], Redactor]:
        allowed, retry = self.limiter.try_acquire(policy.name, policy.rate_limit.requests_per_minute, policy.rate_limit.burst)
        if not allowed:
            log.status, log.outcome = 429, "rate_limited"
            raise GatewayError(429, "rate_limited", "Rate limit exceeded", {"Retry-After": str(max(1, round(retry)))})

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) and "role" in m for m in messages):
            log.status, log.outcome = 400, "bad_request"
            raise GatewayError(400, "invalid_request", "`messages` must be a non-empty list of {role, content} objects")

        t0 = time.perf_counter()
        if policy.injection != "off":
            with span("guard.injection") as s:
                res = self.injection.score(scannable_text(messages))
                s.set_attributes({"score": res.score, "rules": ",".join(res.matched), "detector": res.detector})
            log.injection_score, log.injection_rules = res.score, ",".join(res.matched)
            if policy.injection == "block" and res.score >= policy.injection_threshold:
                log.status, log.outcome = 403, "blocked_injection"
                log.guard_ms = (time.perf_counter() - t0) * 1000
                raise GatewayError(403, "prompt_injection", f"Request blocked by injection guard ({', '.join(res.matched) or res.detector})")

        redactor = Redactor(self.pii)
        if policy.pii != "off":
            with span("guard.pii") as s:
                redacted = redact_messages(messages, redactor)
                s.set_attribute("entities", sum(redactor.counts.values()))
            log.pii_redacted = sum(redactor.counts.values())
            if policy.pii == "block" and redactor.mapping:
                log.status, log.outcome = 400, "blocked_pii"
                log.guard_ms = (time.perf_counter() - t0) * 1000
                raise GatewayError(400, "pii_detected", f"Request contains PII: {', '.join(sorted(redactor.counts))}")
            messages = redacted
        log.guard_ms = (time.perf_counter() - t0) * 1000
        return messages, redactor

    def _output_guard(self, text: str, redactor: Redactor, log: RequestLog) -> str:
        """Redact PII in the model's output that the user did not supply (hallucinated or leaked)."""
        known = set(redactor.mapping.values())
        spans = [s for s in self.pii.detect(text) if s.text not in known]
        log.pii_in_output = len(spans)
        for s in sorted(spans, key=lambda s: -s.start):
            text = text[: s.start] + f"[REDACTED_{s.entity}]" + text[s.end :]
        return text

    def _cacheable(self, policy: AppPolicy, body: dict, redactor: Redactor) -> bool:
        return (self.cache is not None and policy.cache.enabled and not redactor.mapping
                and not body.get("tools") and (body.get("temperature") or 0) == 0 and (body.get("n") or 1) == 1)

    def _finish(self, log: RequestLog, t_start: float) -> None:
        log.latency_ms = (time.perf_counter() - t_start) * 1000
        self.store.add(log)

    # ---------- non-streaming ----------

    async def complete(self, policy: AppPolicy, body: dict) -> GatewayResult:
        t_start = time.perf_counter()
        requested = body.get("model") or "default"
        log = RequestLog(request_id=f"req_{uuid.uuid4().hex[:16]}", app=policy.name, requested_model=requested)
        with span("gateway.request", app=policy.name, model=requested, request_id=log.request_id) as root:
            try:
                messages, redactor = self._admit(policy, body, log)
                payload = {k: v for k, v in body.items() if k in PASSTHROUGH}
                payload["messages"] = messages
                cacheable = self._cacheable(policy, body, redactor)
                log.cache = "miss" if cacheable else "bypass"

                if cacheable:
                    with span("cache.lookup") as s:
                        hit = self.cache.lookup(policy.name, messages, requested, policy.cache.similarity)
                        s.set_attributes({"result": hit.kind, "similarity": hit.similarity or 0.0})
                    if hit.response is not None:
                        log.cache = hit.kind
                        log.served_model = hit.response.get("model", "")
                        resp = {**hit.response, "id": f"chatcmpl-{log.request_id}"}
                        return GatewayResult(resp, self._headers(log))

                t_up = time.perf_counter()
                try:
                    with span("upstream.call") as s:
                        result = await route_completion(self.providers, policy.routes_for(requested), payload,
                                                        policy.timeout_seconds, policy.max_retries)
                        s.set_attributes({"provider": result.provider, "model": result.model, "attempts": len(result.attempts)})
                except AllRoutesFailed as exc:
                    log.attempts = len(exc.attempts)
                    log.status, log.outcome = 502, "upstream_error"
                    status = 429 if exc.last and exc.last.rate_limited else 502
                    raise GatewayError(status, "upstream_unavailable", "All upstream models failed or are rate limited") from exc
                except ProviderError as exc:
                    log.status, log.outcome = 502, "upstream_error"
                    raise GatewayError(502, "upstream_error", exc.message) from exc
                log.upstream_ms = (time.perf_counter() - t_up) * 1000
                log.provider, log.served_model = result.provider, result.model
                log.attempts, log.fallback_used = len(result.attempts), result.fallback_used

                resp = result.response
                usage = resp.get("usage") or {}
                log.prompt_tokens, log.completion_tokens = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
                log.est_cost_usd = estimate_cost(result.model, log.prompt_tokens, log.completion_tokens)

                if cacheable:
                    self.cache.store(policy.name, messages, requested, resp, policy.cache.ttl_seconds)

                with span("guard.output"):
                    for choice in resp.get("choices", []):
                        msg = choice.get("message") or {}
                        if isinstance(msg.get("content"), str):
                            restored = redactor.restore(msg["content"])
                            msg["content"] = self._output_guard(restored, redactor, log) if policy.pii != "off" else restored
                resp = {**resp, "id": f"chatcmpl-{log.request_id}"}
                return GatewayResult(resp, self._headers(log))
            finally:
                root.set_attributes({"outcome": log.outcome, "cache": log.cache, "status": log.status,
                                     "prompt_tokens": log.prompt_tokens, "completion_tokens": log.completion_tokens})
                self._finish(log, t_start)

    @staticmethod
    def _headers(log: RequestLog) -> dict:
        return {"x-sentinel-request-id": log.request_id, "x-sentinel-cache": log.cache,
                "x-sentinel-model": log.served_model or "", "x-sentinel-fallback": str(log.fallback_used).lower(),
                "x-sentinel-pii-redacted": str(log.pii_redacted)}
