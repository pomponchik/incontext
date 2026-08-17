"""The bundled vLLM implementation of the inference-backend contract."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, cast
from urllib.parse import urlsplit

from skelet import EnvSource, Field, Storage

from .backend import Backend


class VllmEnvironment(
    Storage,
    sources=cast(
        Any,
        [
            *EnvSource.for_library("incontext"),
            *EnvSource.for_library("hermes_vllm"),
        ],
    ),
):
    """vLLM-only settings resolved by the bundled backend plugin."""

    tokenizer_url: str = Field(
        "",
        conversion=lambda value: value.strip(),
        validation={
            "tokenizer_url must not be blank": lambda value: bool(value),
        },
        validate_default=False,
        read_only=True,
    )
    tokenizer_user_agent: str = Field(
        "incontext/0.0.1",
        conversion=lambda value: value.strip(),
        validation={
            "tokenizer_user_agent must not be blank": lambda value: bool(value),
        },
        read_only=True,
    )
    tokenizer_timeout_seconds: float = Field(
        30.0,
        validation={
            "tokenizer_timeout_seconds must be greater than 0.0": (
                lambda value: math.isfinite(value) and value > 0.0
            ),
        },
        read_only=True,
    )


class VllmBackend(Backend):
    """Thread-safe exact token counting through vLLM's ``/tokenize`` API."""

    class VllmBackendError(RuntimeError):
        """Raised when vLLM configuration or output violates the contract."""

    def __init__(
        self,
        *,
        cache_entries: int = 64,
        opener: Callable[..., Any] | None = None,
        environment: VllmEnvironment | None = None,
    ) -> None:
        if isinstance(cache_entries, bool) or not isinstance(cache_entries, int):
            raise TypeError("cache_entries must be a positive integer")
        if cache_entries <= 0:
            raise ValueError("cache_entries must be a positive integer")
        try:
            self._environment = (
                VllmEnvironment() if environment is None else environment
            )
        except (TypeError, ValueError) as exc:
            raise self.VllmBackendError(str(exc)) from exc
        self._validate_url(self._environment.tokenizer_url)
        self._cache_entries = cache_entries
        self._opener = urllib.request.urlopen if opener is None else opener
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._cache_lock = threading.Lock()

    @property
    def source(self) -> str:
        """Identify successful counts without exposing endpoint details."""

        return "vllm-tokenize"

    def count(
        self,
        request: dict[str, Any],
        *,
        context_length: int,
    ) -> int:
        """Return the exact provider-visible prompt token count."""

        encoded = json.dumps(
            self._build_payload(request),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        cache_key = hashlib.sha256(
            encoded + b":" + str(context_length).encode("ascii"),
        ).hexdigest()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached

        tokenizer_request = urllib.request.Request(
            self._environment.tokenizer_url,
            data=encoded,
            headers={
                "Content-Type": "application/json",
                "User-Agent": self._environment.tokenizer_user_agent,
            },
            method="POST",
        )
        with self._opener(
            tokenizer_request,
            timeout=self._environment.tokenizer_timeout_seconds,
        ) as http_response:
            response = json.load(http_response)
        if not isinstance(response, dict):
            raise self.VllmBackendError(
                "vLLM /tokenize returned a non-object response",
            )
        count = self._positive_response_integer(response, "count")
        reported_context = self._positive_response_integer(
            response,
            "max_model_len",
        )
        if reported_context != context_length:
            raise self.VllmBackendError(
                "vLLM max_model_len does not match Hermes model.context_length: "
                f"{reported_context} != {context_length}",
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

    @staticmethod
    def _build_payload(request: dict[str, Any]) -> dict[str, Any]:
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

    @classmethod
    def _positive_response_integer(
        cls,
        response: dict[str, Any],
        key: str,
    ) -> int:
        value = response.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise cls.VllmBackendError(
                f"vLLM /tokenize returned invalid {key}",
            )
        return value

    @classmethod
    def _validate_url(cls, value: str) -> None:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise cls.VllmBackendError(
                "tokenizer_url must contain an HTTP(S) URL",
            )
        if parsed.username is not None or parsed.password is not None:
            raise cls.VllmBackendError(
                "tokenizer_url must not embed credentials",
            )
        if parsed.fragment:
            raise cls.VllmBackendError(
                "tokenizer_url must not contain a URL fragment",
            )
