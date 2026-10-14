"""Upstream LLM providers.

Any OpenAI-compatible endpoint works (Groq, OpenRouter, Together, a local
vLLM/Ollama server), so adding a provider is one entry in PROVIDERS.
"""

from __future__ import annotations

import json
import os
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
