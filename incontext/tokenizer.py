"""Exact prompt token counting through vLLM's ``/tokenize`` endpoint."""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from .settings import Settings

DEFAULT_CACHE_ENTRIES = 64


class TokenizationError(RuntimeError):
    """Raised when the tokenizer response violates its safety contract."""


def build_tokenize_payload(request: dict[str, Any]) -> dict[str, Any]:
    """Build the vLLM request using the provider-visible prompt shape."""

    payload: dict[str, Any] = {
        "model": request.get("model"),
        "messages": request.get("messages"),
        "add_generation_prompt": True,
    }
    tools = request.get("tools")
    if isinstance(tools, list) and tools:
        payload["tools"] = tools
    extra_body = request.get("extra_body")
    if isinstance(extra_body, dict):
        template_kwargs = extra_body.get("chat_template_kwargs")
        if isinstance(template_kwargs, dict):
            payload["chat_template_kwargs"] = template_kwargs
    return payload


def _response_positive_int(response: dict[str, Any], key: str) -> int:
    value = response.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TokenizationError(f"vLLM /tokenize returned invalid {key}")
    return value


class VllmTokenizer:
    """Thread-safe, bounded and exact vLLM token counter."""

    def __init__(
        self,
        settings: Settings,
        *,
        cache_entries: int = DEFAULT_CACHE_ENTRIES,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if isinstance(cache_entries, bool) or not isinstance(cache_entries, int):
            raise TypeError("cache_entries must be a positive integer")
        if cache_entries <= 0:
            raise ValueError("cache_entries must be a positive integer")
        self._settings = settings
        self._cache_entries = cache_entries
        self._opener = urllib.request.urlopen if opener is None else opener
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._cache_lock = threading.Lock()

    def count(self, request: dict[str, Any]) -> int:
        """Return the exact provider-visible prompt token count."""

        encoded = json.dumps(
            build_tokenize_payload(request),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        cache_key = hashlib.sha256(encoded).hexdigest()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached

        tokenizer_request = urllib.request.Request(
            self._settings.tokenizer_url,
            data=encoded,
            headers={
                "Content-Type": "application/json",
                "User-Agent": self._settings.tokenizer_user_agent,
            },
            method="POST",
        )
        with self._opener(
            tokenizer_request,
            timeout=self._settings.tokenizer_timeout_seconds,
        ) as http_response:
            response = json.load(http_response)
        if not isinstance(response, dict):
            raise TokenizationError("vLLM /tokenize returned a non-object response")
        count = _response_positive_int(response, "count")
        reported_context = _response_positive_int(response, "max_model_len")
        if reported_context != self._settings.context_length:
            raise TokenizationError(
                "vLLM max_model_len does not match Hermes model.context_length: "
                f"{reported_context} != {self._settings.context_length}",
            )

        with self._cache_lock:
            self._cache[cache_key] = count
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._cache_entries:
                self._cache.popitem(last=False)
        return count

    def clear_cache(self) -> None:
        """Discard cached counts without disturbing an in-flight request."""

        with self._cache_lock:
            self._cache.clear()
