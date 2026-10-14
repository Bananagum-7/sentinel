"""FastAPI app exposing an OpenAI-compatible API.

Point any OpenAI SDK at it by changing two settings:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key=<your SENTINEL_KEY_...>)
    client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[...])
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sentinel.gateway import Gateway, GatewayError
from sentinel.policy import PolicyStore

log = logging.getLogger("sentinel")
state: dict = {}


def build_gateway() -> tuple[Gateway, PolicyStore]:
    """Production wiring. Heavier detectors are used when their packages are installed."""
    from sentinel.cache import HashEmbedder, SemanticCache
    from sentinel.guards.injection import CombinedInjectionDetector, HeuristicInjectionDetector
    from sentinel.guards.pii import RegexPIIDetector
    from sentinel.providers import build_providers
    from sentinel.store import RequestStore

    pii = RegexPIIDetector()
    if os.getenv("SENTINEL_PRESIDIO") == "1":
        from sentinel.guards.pii import PresidioPIIDetector

        pii = PresidioPIIDetector()
    detectors = [HeuristicInjectionDetector()]
    if os.getenv("SENTINEL_INJECTION_MODEL") == "1":
        from sentinel.guards.injection import ClassifierInjectionDetector

        detectors.append(ClassifierInjectionDetector())
    embedder = HashEmbedder()
    if os.getenv("SENTINEL_FASTEMBED") == "1":
        from sentinel.cache import FastEmbedEmbedder

        embedder = FastEmbedEmbedder()
    providers = build_providers()
    if os.getenv("SENTINEL_FAKE_UPSTREAM") == "1":  # load tests / demos without an API key
        from sentinel.providers import FakeProvider

        providers = {"groq": FakeProvider("groq", ["This is a canned answer from the fake upstream."], latency=0.3)}
    gateway = Gateway(
        providers=providers,
        pii_detector=pii,
        injection_detector=CombinedInjectionDetector(detectors),
        cache=SemanticCache(embedder),
        store=RequestStore(os.getenv("SENTINEL_DB", "data/requests.db")),
    )
    policies = PolicyStore.from_yaml(os.getenv("SENTINEL_POLICY_FILE", "config/policies.yaml"))
    return gateway, policies


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "gateway" not in state:  # tests inject their own
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        state["gateway"], state["policies"] = build_gateway()
    yield


app = FastAPI(title="Sentinel LLM Gateway", version="0.1.0", lifespan=lifespan)


@app.exception_handler(GatewayError)
async def gateway_error(_: Request, exc: GatewayError):
    return JSONResponse(exc.body(), status_code=exc.status, headers=exc.headers)


def _policy(authorization: str | None):
    key = authorization.removeprefix("Bearer ").strip() if authorization else None
    policy = state["policies"].authenticate(key)
    if policy is None:
        raise GatewayError(401, "invalid_api_key", "Missing or invalid gateway API key")
    return policy


@app.get("/health")
def health():
    return {"status": "ok", "providers": sorted(state["gateway"].providers)}


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)):
    policy = _policy(authorization)
    names = sorted({r.model for routes in policy.models.values() for r in routes})
    return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "sentinel"} for n in names]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    policy = _policy(authorization)
    try:
        body = await request.json()
    except Exception as exc:
        raise GatewayError(400, "invalid_json", "Request body must be JSON") from exc
    gw: Gateway = state["gateway"]
    if body.get("stream"):
        events, headers = await gw.stream(policy, body)
        return StreamingResponse(events, media_type="text/event-stream", headers=headers)
    result = await gw.complete(policy, body)
    return JSONResponse(result.response, headers=result.headers)


@app.get("/metrics")
def metrics(window_s: float = 3600, authorization: str | None = Header(default=None)):
    _policy(authorization)
    return state["gateway"].store.summary(window_s)
