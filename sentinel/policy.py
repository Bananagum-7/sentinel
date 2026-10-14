"""Load per-app policies and resolve a gateway API key to its app.

Keys live in environment variables (named by `key_env`), so the YAML file
can be committed safely. Keys are compared with hmac.compare_digest to avoid
timing attacks.
"""

from __future__ import annotations

import copy
import hmac
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RateLimit(BaseModel):
    requests_per_minute: int = 60
    burst: int = 10


class CacheCfg(BaseModel):
    enabled: bool = True
    similarity: float = 0.92
    ttl_seconds: int = 86400


class Route(BaseModel):
    provider: str
    model: str


class AppPolicy(BaseModel):
    name: str
    key_env: str | None = None
    rate_limit: RateLimit = Field(default_factory=RateLimit)
    pii: Literal["redact", "block", "off"] = "redact"
    injection: Literal["block", "flag", "off"] = "block"
    injection_threshold: float = 0.5
    cache: CacheCfg = Field(default_factory=CacheCfg)
    models: dict[str, list[Route]] = Field(default_factory=dict)
    timeout_seconds: float = 30
    max_retries: int = 2

    def routes_for(self, requested: str) -> list[Route]:
        """The requested model goes first if it is a known model; then the default chain."""
        chain = list(self.models.get(requested) or [])
        default = self.models.get("default", [])
        if not chain:
            known = {r.model for routes in self.models.values() for r in routes}
            if requested in known:
                provider = next(r.provider for routes in self.models.values() for r in routes if r.model == requested)
                chain = [Route(provider=provider, model=requested)]
        for r in default:
            if r not in chain:
                chain.append(r)
        return chain


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class PolicyStore:
    def __init__(self, apps: dict[str, AppPolicy], keys: dict[str, str]):
        self.apps = apps
        self._keys = keys  # api key -> app name

    @classmethod
    def from_yaml(cls, path: str | Path, env: dict | None = None) -> "PolicyStore":
        env = env if env is not None else os.environ
        raw = yaml.safe_load(Path(path).read_text())
        defaults = raw.get("defaults", {})
        apps, keys = {}, {}
        for name, cfg in (raw.get("apps") or {}).items():
            policy = AppPolicy(name=name, **_merge(defaults, cfg or {}))
            apps[name] = policy
            if policy.key_env and env.get(policy.key_env):
                keys[env[policy.key_env]] = name
        return cls(apps, keys)

    def authenticate(self, api_key: str | None) -> AppPolicy | None:
        if not api_key:
            return None
        for key, app in self._keys.items():
            if hmac.compare_digest(key.encode(), api_key.encode()):
                return self.apps[app]
        return None
