from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from incontext.settings import Settings
from incontext.tokenizer import (
    TokenizationError,
    VllmTokenizer,
    _response_positive_int,
    build_tokenize_payload,
)


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class RecordingOpener:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[urllib.request.Request, float]] = []

    def __call__(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> FakeResponse:
        self.calls.append((request, timeout))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return FakeResponse(json.dumps(response).encode())


def response(count: Any = 123, context: Any = 65_536) -> dict[str, Any]:
    return {"count": count, "max_model_len": context}


def test_build_tokenize_payload_minimal_shape() -> None:
    request = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    assert build_tokenize_payload(request) == {
        "model": "qwen",
        "messages": request["messages"],
        "add_generation_prompt": True,
    }


def test_build_tokenize_payload_includes_tools_and_template_kwargs() -> None:
    request = {
        "model": "qwen",
        "messages": [],
        "tools": [{"type": "function"}],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    payload = build_tokenize_payload(request)
    assert payload["tools"] is request["tools"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize(
    ("tools", "extra_body"),
    [([], None), ("invalid", []), (None, {"chat_template_kwargs": []})],
)
def test_build_tokenize_payload_ignores_non_effective_optional_fields(
    tools: Any,
    extra_body: Any,
) -> None:
    payload = build_tokenize_payload(
        {"model": "qwen", "messages": [], "tools": tools, "extra_body": extra_body},
    )
    assert "tools" not in payload
    assert "chat_template_kwargs" not in payload


@pytest.mark.parametrize("value", [None, True, False, 0, -1, "1", 1.5])
def test_response_positive_int_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(TokenizationError, match="invalid count"):
        _response_positive_int({"count": value}, "count")


def test_response_positive_int_accepts_positive_integer() -> None:
    assert _response_positive_int({"count": 1}, "count") == 1


@pytest.mark.parametrize("cache_entries", [True, "1"])
def test_tokenizer_rejects_invalid_cache_capacity_type(
    runtime_settings: Settings,
    cache_entries: Any,
) -> None:
    with pytest.raises(TypeError, match="positive integer"):
        VllmTokenizer(runtime_settings, cache_entries=cache_entries)


@pytest.mark.parametrize("cache_entries", [0, -1])
def test_tokenizer_rejects_non_positive_cache_capacity(
    runtime_settings: Settings,
    cache_entries: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        VllmTokenizer(runtime_settings, cache_entries=cache_entries)


def test_tokenizer_sends_exact_request_and_caches_result(
    runtime_settings: Settings,
) -> None:
    opener = RecordingOpener([response(321)])
    tokenizer = VllmTokenizer(runtime_settings, opener=opener)
    request = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "Привет"}],
        "tools": [{"type": "function", "function": {"name": "test"}}],
    }
    assert tokenizer.count(request) == 321
    assert tokenizer.count(request) == 321
    assert len(opener.calls) == 1
    http_request, timeout = opener.calls[0]
    assert http_request.full_url == runtime_settings.tokenizer_url
    assert http_request.method == "POST"
    assert http_request.get_header("Content-type") == "application/json"
    assert http_request.get_header("User-agent") == "incontext-tests"
    assert timeout == 3.5
    assert json.loads(http_request.data or b"") == build_tokenize_payload(request)


def test_tokenizer_default_opener_is_resolved_at_construction(
    runtime_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = RecordingOpener([response()])
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    tokenizer = VllmTokenizer(runtime_settings)
    assert tokenizer.count({"model": "qwen", "messages": []}) == 123


def test_tokenizer_lru_evicts_oldest_entry(runtime_settings: Settings) -> None:
    opener = RecordingOpener([response(1), response(2), response(3), response(4)])
    tokenizer = VllmTokenizer(runtime_settings, cache_entries=2, opener=opener)
    first = {"model": "qwen", "messages": [{"content": "first"}]}
    second = {"model": "qwen", "messages": [{"content": "second"}]}
    third = {"model": "qwen", "messages": [{"content": "third"}]}
    assert tokenizer.count(first) == 1
    assert tokenizer.count(second) == 2
    assert tokenizer.count(first) == 1
    assert tokenizer.count(third) == 3
    assert tokenizer.count(second) == 4
    assert len(opener.calls) == 4


def test_tokenizer_clear_cache_forces_new_request(runtime_settings: Settings) -> None:
    opener = RecordingOpener([response(1), response(2)])
    tokenizer = VllmTokenizer(runtime_settings, opener=opener)
    request = {"model": "qwen", "messages": []}
    assert tokenizer.count(request) == 1
    tokenizer.clear_cache()
    assert tokenizer.count(request) == 2


def test_tokenizer_does_not_cache_transport_failure(runtime_settings: Settings) -> None:
    opener = RecordingOpener(
        [urllib.error.URLError("offline"), response(5)],
    )
    tokenizer = VllmTokenizer(runtime_settings, opener=opener)
    request = {"model": "qwen", "messages": []}
    with pytest.raises(urllib.error.URLError):
        tokenizer.count(request)
    assert tokenizer.count(request) == 5


def test_tokenizer_rejects_non_serializable_payload(runtime_settings: Settings) -> None:
    tokenizer = VllmTokenizer(runtime_settings, opener=RecordingOpener([]))
    with pytest.raises(TypeError):
        tokenizer.count({"model": "qwen", "messages": [object()]})


def test_tokenizer_rejects_non_object_response(runtime_settings: Settings) -> None:
    opener = RecordingOpener([[1, 2, 3]])
    tokenizer = VllmTokenizer(runtime_settings, opener=opener)
    with pytest.raises(TokenizationError, match="non-object"):
        tokenizer.count({"model": "qwen", "messages": []})


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"max_model_len": 65_536}, "invalid count"),
        ({"count": 1}, "invalid max_model_len"),
        (response(1, 32_000), "does not match"),
    ],
)
def test_tokenizer_validates_response_contract(
    runtime_settings: Settings,
    payload: dict[str, Any],
    message: str,
) -> None:
    tokenizer = VllmTokenizer(
        runtime_settings,
        opener=RecordingOpener([payload]),
    )
    with pytest.raises(TokenizationError, match=message):
        tokenizer.count({"model": "qwen", "messages": []})
