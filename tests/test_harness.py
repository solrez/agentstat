"""Harness tests that need no network and no API keys.

The cache is pure. The provider is tested via its cache-hit path (no HTTP) and
with a monkeypatched ``_post`` so we never make a real call.
"""

import pytest

from agentstat.harness.cache import ResponseCache, request_key


def test_request_key_is_order_independent():
    a = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}
    b = {"temperature": 0, "messages": [{"role": "user", "content": "hi"}], "model": "m"}
    assert request_key(a) == request_key(b)


def test_request_key_changes_with_content():
    a = {"model": "m", "temperature": 0.0}
    b = {"model": "m", "temperature": 0.5}
    assert request_key(a) != request_key(b)


def test_cache_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path / "c")
    payload = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
    assert cache.get(payload) is None
    assert payload not in cache
    cache.set(payload, {"answer": 42})
    assert payload in cache
    assert cache.get(payload) == {"answer": 42}


def test_cache_distinguishes_payloads(tmp_path):
    cache = ResponseCache(tmp_path / "c")
    cache.set({"model": "a"}, {"r": 1})
    cache.set({"model": "b"}, {"r": 2})
    assert cache.get({"model": "a"}) == {"r": 1}
    assert cache.get({"model": "b"}) == {"r": 2}


def test_provider_unknown_name_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    from agentstat.harness.providers import ChatProvider

    with pytest.raises(ValueError, match="unknown provider"):
        ChatProvider(provider="nope", cache=ResponseCache(tmp_path / "c"))


def test_provider_missing_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    from agentstat.harness.providers import ChatProvider, ProviderError

    with pytest.raises(ProviderError, match="missing API key"):
        ChatProvider(provider="deepinfra", cache=ResponseCache(tmp_path / "c"))


def test_provider_cache_hit_skips_http(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from agentstat.harness.providers import ChatProvider

    cache = ResponseCache(tmp_path / "c")
    prov = ChatProvider(provider="openrouter", cache=cache)

    # Fail loudly if HTTP is attempted — the point is the cache short-circuits it.
    def boom(payload):
        raise AssertionError("_post should not be called on a cache hit")

    messages = [{"role": "user", "content": "hi"}]
    # Pre-seed the cache with the exact payload the provider will build.
    cache_payload = {"__provider__": "openrouter", **prov._payload(
        "model-x", messages, None, 0.0, None, None
    )}
    cache.set(cache_payload, {"choices": [{"message": {"content": "cached"}}]})

    monkeypatch.setattr(prov, "_post", boom)
    out = prov.chat(model="model-x", messages=messages)
    assert out["choices"][0]["message"]["content"] == "cached"


def test_provider_writes_cache_after_call(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from agentstat.harness.providers import ChatProvider

    cache = ResponseCache(tmp_path / "c")
    prov = ChatProvider(provider="openrouter", cache=cache)

    fake_response = {"choices": [{"message": {"content": "fresh"}}]}
    monkeypatch.setattr(prov, "_post", lambda payload: fake_response)

    messages = [{"role": "user", "content": "hello"}]
    out1 = prov.chat(model="m", messages=messages)
    assert out1 == fake_response

    # Second call must hit cache — swap _post to a bomb to prove it.
    monkeypatch.setattr(
        prov, "_post", lambda p: (_ for _ in ()).throw(AssertionError("should be cached"))
    )
    out2 = prov.chat(model="m", messages=messages)
    assert out2 == fake_response
