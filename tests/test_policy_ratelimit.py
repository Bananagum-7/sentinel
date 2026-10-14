from pathlib import Path

from sentinel.policy import PolicyStore
from sentinel.ratelimit import TokenBucketLimiter

POLICY = Path(__file__).resolve().parent.parent / "config" / "policies.yaml"
ENV = {"SENTINEL_KEY_DOCSIGHT": "k-docs", "SENTINEL_KEY_HELPDESK": "k-help", "SENTINEL_KEY_DEMO": "k-demo"}


def test_policies_merge_defaults_and_overrides():
    store = PolicyStore.from_yaml(POLICY, ENV)
    help_app = store.authenticate("k-help")
    assert help_app.name == "helpdesk"
    assert help_app.cache.enabled is False  # override
    assert help_app.injection == "block"  # inherited default
    demo = store.authenticate("k-demo")
    assert demo.rate_limit.requests_per_minute == 20 and demo.rate_limit.burst == 5


def test_unknown_or_missing_key_rejected():
    store = PolicyStore.from_yaml(POLICY, ENV)
    assert store.authenticate("nope") is None
    assert store.authenticate(None) is None


def test_routes_put_requested_model_first():
    app = PolicyStore.from_yaml(POLICY, ENV).apps["docsight"]
    routes = [r.model for r in app.routes_for("llama-3.1-8b-instant")]
    assert routes == ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    assert [r.model for r in app.routes_for("anything")] == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def test_token_bucket_burst_then_refill():
    now = [0.0]
    lim = TokenBucketLimiter(clock=lambda: now[0])
    results = [lim.try_acquire("a", rpm=60, burst=3)[0] for _ in range(4)]
    assert results == [True, True, True, False]
    allowed, retry = lim.try_acquire("a", rpm=60, burst=3)
    assert not allowed and 0.9 < retry <= 1.0
    now[0] += 1.0  # one token refills per second at 60 rpm
    assert lim.try_acquire("a", rpm=60, burst=3)[0]
    assert lim.try_acquire("b", rpm=60, burst=3)[0]  # buckets are per key
