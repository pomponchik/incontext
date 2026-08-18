"""The bundled vLLM implementation of the inference-backend contract."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import urllib.request
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, cast
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
        opener: Optional[Callable[..., Any]] = None,
        environment: Optional[VllmEnvironment] = None,
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
        request: Dict[str, Any],
        *,
        context_length: int,
    ) -> int:
        """Return the exact provider-visible prompt token count."""

        truncation_limit = self._prompt_truncation_limit(request)
        encoded = json.dumps(
            self._build_payload(request),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        cache_key = hashlib.sha256(
            encoded + b":" + str(context_length).encode("ascii"),
        ).hexdigest()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return (
                    min(cached, truncation_limit)
                    if truncation_limit is not None
                    else cached
                )

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
        return min(count, truncation_limit) if truncation_limit is not None else count

    def clear_cache(self) -> None:
        """Discard cached counts without disturbing an in-flight request."""

        with self._cache_lock:
            self._cache.clear()

    @staticmethod
    def _build_payload(
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        extra_body = request.get("extra_body")
        extra_body = extra_body if isinstance(extra_body, dict) else {}
        payload: Dict[str, Any] = {
            "model": extra_body.get("model", request.get("model")),
            "messages": extra_body.get("messages", request.get("messages")),
        }
        tools = extra_body.get("tools", request.get("tools"))
        if isinstance(tools, list):
            payload["tools"] = tools
        prompt_options = (
            "add_generation_prompt",
            "continue_final_message",
            "add_special_tokens",
            "chat_template",
            "chat_template_kwargs",
            "media_io_kwargs",
            "mm_processor_kwargs",
        )
        for option in prompt_options:
            if option in request:
                payload[option] = request[option]
            if option in extra_body:
                # The OpenAI client merges extra_body over its generated JSON;
                # mirror that precedence for the tokenizer request.
                payload[option] = extra_body[option]
        template_kwargs = payload.get("chat_template_kwargs")
        if template_kwargs is None or isinstance(template_kwargs, dict):
            merged_template_kwargs = dict(template_kwargs or {})
            documents = extra_body.get("documents", request.get("documents"))
            reasoning_effort = extra_body.get(
                "reasoning_effort",
                request.get("reasoning_effort"),
            )
            if documents is not None:
                merged_template_kwargs["documents"] = documents
            if reasoning_effort is not None:
                merged_template_kwargs["reasoning_effort"] = reasoning_effort
                if "enable_thinking" not in merged_template_kwargs:
                    merged_template_kwargs["enable_thinking"] = (
                        reasoning_effort != "none"
                    )
            if merged_template_kwargs or isinstance(template_kwargs, dict):
                payload["chat_template_kwargs"] = merged_template_kwargs
        payload.setdefault("add_generation_prompt", True)
        return payload

    @classmethod
    def _positive_response_integer(
        cls,
        response: Dict[str, Any],
        key: str,
    ) -> int:
        value = response.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise cls.VllmBackendError(
                f"vLLM /tokenize returned invalid {key}",
            )
        return value

    @classmethod
    def _prompt_truncation_limit(
        cls,
        request: Dict[str, Any],
    ) -> Optional[int]:
        """Resolve vLLM's positive prompt truncation from the wire request."""

        extra_body = request.get("extra_body")
        value = (
            extra_body.get(
                "truncate_prompt_tokens",
                request.get("truncate_prompt_tokens"),
            )
            if isinstance(extra_body, dict)
            else request.get("truncate_prompt_tokens")
        )
        if value == -1:
            raise cls.VllmBackendError(
                "truncate_prompt_tokens=-1 depends on the provider output budget",
            )
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
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
