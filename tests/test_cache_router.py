import asyncio

import pytest

from sentinel.cache import HashEmbedder, SemanticCache, critical_tokens
from sentinel.policy import Route
from sentinel.providers import AllRoutesFailed, FakeProvider, route_completion

SYS = {"role": "system", "content": "You answer questions about our product."}


def msgs(q):
    return [SYS, {"role": "user", "content": q}]


def test_critical_tokens():
    assert critical_tokens("What was revenue in 2021?") == {"2021"}
    assert critical_tokens("How do I use `add_task` with BackgroundTasks?") == {"add_task", "backgroundtasks"}


def test_exact_then_semantic_hit_and_near_miss():
    cache = SemanticCache(HashEmbedder())
    cache.store("app", msgs("How do I reset my password on the mobile app?"), "m", {"answer": "A"}, ttl=60)
    assert cache.lookup("app", msgs("How do I reset my password on the mobile app?"), "m", 0.8).kind == "exact"
    hit = cache.lookup("app", msgs("how do I reset my password on the mobile app"), "m", 0.8)
    assert hit.kind == "semantic" and hit.response == {"answer": "A"}

    cache.store("app", msgs("What was the refund limit in 2021?"), "m", {"answer": "2021"}, ttl=60)
    # Nearly identical text, different year -> must miss
    assert cache.lookup("app", msgs("What was the refund limit in 2022?"), "m", 0.5).kind == "miss"


def test_cache_scoped_by_app_model_system_prompt_and_ttl():
    cache = SemanticCache(HashEmbedder())
    cache.store("a", msgs("hello there friend"), "m", {"x": 1}, ttl=60, now=1000)
    assert cache.lookup("b", msgs("hello there friend"), "m", 0.5, now=1001).kind == "miss"
    assert cache.lookup("a", msgs("hello there friend"), "other", 0.5, now=1001).kind == "miss"
    other_sys = [{"role": "system", "content": "different"}, {"role": "user", "content": "hello there friend!"}]
    assert cache.lookup("a", other_sys, "m", 0.5, now=1001).kind == "miss"
    assert cache.lookup("a", msgs("hello there friend"), "m", 0.5, now=2000).kind == "miss"  # expired


ROUTES = [Route(provider="p1", model="big"), Route(provider="p2", model="small")]


def run(coro):
    return asyncio.run(coro)


def test_router_falls_back_on_429_without_retrying_same_route():
    p1, p2 = FakeProvider("p1", [429]), FakeProvider("p2", ["hi"])
    res = run(route_completion({"p1": p1, "p2": p2}, ROUTES, {"messages": []}, 5, max_retries=2, backoff=0))
    assert res.model == "small" and res.fallback_used
    assert len(p1.calls) == 1  # 429 -> move on immediately


def test_router_retries_5xx_then_succeeds():
    p1 = FakeProvider("p1", [503, 503, "ok"])
    res = run(route_completion({"p1": p1}, ROUTES[:1], {"messages": []}, 5, max_retries=2, backoff=0))
    assert res.model == "big" and len(res.attempts) == 3 and not res.fallback_used


def test_router_fails_fast_on_400():
    p1, p2 = FakeProvider("p1", [400]), FakeProvider("p2", ["hi"])
    with pytest.raises(AllRoutesFailed):
        run(route_completion({"p1": p1, "p2": p2}, ROUTES, {"messages": []}, 5, max_retries=2, backoff=0))
    assert p2.calls == []


def test_router_all_fail():
    with pytest.raises(AllRoutesFailed) as exc:
        run(route_completion({"p1": FakeProvider("p1", [429]), "p2": FakeProvider("p2", [500])}, ROUTES, {"messages": []}, 5, 1, backoff=0))
    assert [a.status for a in exc.value.attempts] == [429, 500, 500]
