"""Upstream LLM providers and the fallback router.

Any OpenAI-compatible endpoint works (Groq, OpenRouter, Together, a local
vLLM/Ollama server), so adding a provider is one entry in PROVIDERS.

Router policy, per route in the fallback chain:
- 429 (rate limited): skip to the next route immediately. Waiting a few
  hundred ms won't clear a per-minute quota, and the user is waiting.
- 5xx / timeout / connection error: retry the same route with exponential
  backoff (0.25 s, 0.5 s ...) up to max_retries, then move on.
- 4xx other than 429 (bad request, auth): fail fast. Retrying won't help and
  another model would likely reject it too.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

import httpx

PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None},
}


class ProviderError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message

    @property
    def retryable(self) -> bool:
        return self.status >= 500 or self.status in (408, 0)

    @property
    def rate_limited(self) -> bool:
        return self.status == 429


class Provider(Protocol):
    name: str

    async def complete(self, payload: dict, timeout: float) -> dict: ...

    def stream(self, payload: dict, timeout: float) -> AsyncIterator[dict]: ...


class OpenAICompatibleProvider:
    def __init__(self, name: str, base_url: str, api_key: str | None, client: httpx.AsyncClient | None = None):
        self.name, self.base_url, self.api_key = name, base_url.rstrip("/"), api_key
        self.client = client or httpx.AsyncClient()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def complete(self, payload: dict, timeout: float) -> dict:
        try:
            r = await self.client.post(f"{self.base_url}/chat/completions", json={**payload, "stream": False},
                                       headers=self._headers(), timeout=timeout)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderError(0, f"{exc.__class__.__name__}") from exc
        if r.status_code >= 400:
            raise ProviderError(r.status_code, r.text[:300])
        return r.json()

    async def stream(self, payload: dict, timeout: float) -> AsyncIterator[dict]:
        try:
            async with self.client.stream("POST", f"{self.base_url}/chat/completions", json={**payload, "stream": True},
                                          headers=self._headers(), timeout=timeout) as r:
                if r.status_code >= 400:
                    raise ProviderError(r.status_code, (await r.aread()).decode()[:300])
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    yield json.loads(data)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderError(0, exc.__class__.__name__) from exc


def build_providers(env: dict | None = None) -> dict[str, Provider]:
    env = env if env is not None else os.environ
    client = httpx.AsyncClient(limits=httpx.Limits(max_connections=100, max_keepalive_connections=20))
    out = {}
    for name, cfg in PROVIDERS.items():
        key = env.get(cfg["key_env"]) if cfg["key_env"] else None
        if cfg["key_env"] and not key:
            continue
        out[name] = OpenAICompatibleProvider(name, cfg["base_url"], key, client)
    return out


@dataclass
class Attempt:
    provider: str
    model: str
    status: int | None = None  # None = success
    latency_ms: float = 0.0


@dataclass
class RouteResult:
    response: dict
    provider: str
    model: str
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def fallback_used(self) -> bool:
        return len({(a.provider, a.model) for a in self.attempts}) > 1


class AllRoutesFailed(Exception):
    def __init__(self, attempts: list[Attempt], last: ProviderError | None):
        super().__init__(f"all routes failed: {[(a.model, a.status) for a in attempts]}")
        self.attempts, self.last = attempts, last


async def route_completion(providers: dict[str, Provider], routes, payload: dict, timeout: float, max_retries: int,
                           backoff: float = 0.25) -> RouteResult:
    attempts: list[Attempt] = []
    last: ProviderError | None = None
    for route in routes:
        provider = providers.get(route.provider)
        if provider is None:
            continue
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = await provider.complete({**payload, "model": route.model}, timeout)
                attempts.append(Attempt(route.provider, route.model, None, (time.perf_counter() - t0) * 1000))
                return RouteResult(resp, route.provider, route.model, attempts)
            except ProviderError as err:
                attempts.append(Attempt(route.provider, route.model, err.status, (time.perf_counter() - t0) * 1000))
                last = err
                if err.rate_limited:
                    break  # next route
                if not err.retryable:
                    raise AllRoutesFailed(attempts, err) from err
                if attempt < max_retries:
                    await asyncio.sleep(backoff * (2**attempt))
    raise AllRoutesFailed(attempts, last)
