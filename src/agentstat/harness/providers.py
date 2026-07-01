"""Thin provider abstraction over OpenAI-compatible chat APIs.

Both OpenRouter and DeepInfra speak the OpenAI ``/chat/completions`` protocol, so
one client covers both — we just switch the base URL and API key by provider
name (read from ``.env``). Every call goes through the disk cache, so a repeated
request costs nothing and reruns are free and reproducible.

We deliberately do NOT depend on the ``openai`` SDK: a small httpx client keeps
the surface minimal and the request payload explicit (which matters, because the
exact payload is what the cache hashes).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

from agentstat.harness.cache import ResponseCache

load_dotenv()

# provider name -> (env var for key, env var for base URL, default base URL)
_PROVIDERS = {
    "openrouter": (
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    ),
    "deepinfra": (
        "DEEPINFRA_API_KEY",
        "DEEPINFRA_BASE_URL",
        "https://api.deepinfra.com/v1/openai",
    ),
}


class ProviderError(RuntimeError):
    pass


@dataclass
class ChatProvider:
    """An OpenAI-compatible chat client for one provider, with disk caching."""

    provider: str
    cache: ResponseCache = field(default_factory=ResponseCache)
    timeout: float = 120.0
    max_retries: int = 3

    def __post_init__(self):
        if self.provider not in _PROVIDERS:
            raise ValueError(
                f"unknown provider {self.provider!r}; known: {list(_PROVIDERS)}"
            )
        key_env, url_env, default_url = _PROVIDERS[self.provider]
        self.api_key = os.getenv(key_env)
        self.base_url = os.getenv(url_env, default_url).rstrip("/")
        if not self.api_key:
            raise ProviderError(
                f"missing API key: set {key_env} in your .env for provider "
                f"{self.provider!r}"
            )

    def _payload(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict] | None,
        temperature: float,
        seed: int | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # This dict is BOTH the request body and the cache key, so it must
        # contain everything that could change the response.
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if seed is not None:
            payload["seed"] = seed
        if extra:
            payload.update(extra)
        return payload

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        extra: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """One chat completion. Returns the raw OpenAI-style response dict.

        The provider name is folded into the cache key so identical requests to
        different providers don't collide.
        """
        payload = self._payload(model, messages, tools, temperature, seed, extra)
        cache_payload = {"__provider__": self.provider, **payload}

        if use_cache:
            hit = self.cache.get(cache_payload)
            if hit is not None:
                return hit

        response = self._post(payload)

        if use_cache:
            self.cache.set(cache_payload, response)
        return response

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = httpx.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                if resp.status_code == 200:
                    return resp.json()
                # Retry on transient server / rate-limit errors; fail fast on 4xx.
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = ProviderError(
                        f"{self.provider} returned {resp.status_code}: {resp.text[:200]}"
                    )
                    continue
                raise ProviderError(
                    f"{self.provider} returned {resp.status_code}: {resp.text[:300]}"
                )
            except httpx.HTTPError as e:  # network-level errors are retryable
                last_exc = e
        raise ProviderError(
            f"{self.provider} request failed after {self.max_retries} attempts"
        ) from last_exc
