"""Load per-app policies and resolve a gateway API key to its app.

Keys live in environment variables (named by `key_env`), so the YAML file
can be committed safely.
"""

from __future__ import annotations

import copy
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
