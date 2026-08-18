from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from incontext.backend import Backend
from incontext.budget import DynamicOutputBudget
from incontext.preflight import install
from incontext.settings import Settings


class Counter(Backend):
    def __init__(self, result: int | BaseException) -> None:
        self.result = result
        self.requests: list[dict[str, Any]] = []

    @property
    def source(self) -> str:
        return "test-counter"

    def count(self, request: dict[str, Any], *, context_length: int) -> int:
        del context_length
        self.requests.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    def clear_cache(self) -> None:
        self.requests.clear()


def runtime(counter: Counter) -> DynamicOutputBudget:
    return DynamicOutputBudget(
        Settings(
            model_name="qwen-test",
            context_length=65_536,
            compression_window=64_000,
            fallback_margin_tokens=1024,
        ),
        counter,
    )


def install_fake_hermes(
    monkeypatch: pytest.MonkeyPatch,
    original: Any,
) -> types.ModuleType:
    agent = types.ModuleType("agent")
    agent.__path__ = []  # type: ignore[attr-defined]
    loop = types.ModuleType("agent.conversation_loop")
    loop.estimate_request_tokens_rough = original  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.conversation_loop", loop)
    return loop


def test_install_replaces_rough_preflight_with_exact_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rough(messages: Any, tools: Any = None) -> int:
        raise AssertionError("exact backend should be used")

    loop = install_fake_hermes(monkeypatch, rough)
    counter = Counter(64_000)
    original = install(runtime(counter))

    assert original is rough
    assert (
        loop.estimate_request_tokens_rough(
            [{"role": "user", "content": "large"}],
            tools=[{"type": "function"}],
        )
        == 64_000
    )
    assert counter.requests == [
        {
            "model": "qwen-test",
            "messages": [{"role": "user", "content": "large"}],
            "tools": [{"type": "function"}],
        },
    ]


def test_preflight_preserves_empty_tool_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = install_fake_hermes(monkeypatch, lambda messages, tools=None: 42)
    counter = Counter(123)
    install(runtime(counter))

    assert loop.estimate_request_tokens_rough([]) == 123
    assert counter.requests == [{"model": "qwen-test", "messages": []}]


def test_preflight_falls_back_when_backend_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def rough(messages: Any, tools: Any = None) -> int:
        assert messages == [{"role": "user", "content": "fallback"}]
        assert tools == []
        return 321

    loop = install_fake_hermes(monkeypatch, rough)
    install(runtime(Counter(TimeoutError("secret"))))

    assert (
        loop.estimate_request_tokens_rough(
            [{"role": "user", "content": "fallback"}], tools=[]
        )
        == 321
    )
    assert "TimeoutError" in caplog.text
    assert "secret" not in caplog.text


def test_preflight_falls_back_for_unsupported_message_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rough(messages: Any, tools: Any = None) -> int:
        assert messages == "not-a-list"
        assert tools == "also-not-a-list"
        return 7

    loop = install_fake_hermes(monkeypatch, rough)
    install(runtime(Counter(1)))
    assert loop.estimate_request_tokens_rough("not-a-list", "also-not-a-list") == 7


def test_install_is_idempotent_and_retains_the_initial_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rough(messages: Any, tools: Any = None) -> int:
        return 5

    loop = install_fake_hermes(monkeypatch, rough)
    first = Counter(100)
    second = Counter(200)
    assert install(runtime(first)) is rough
    assert install(runtime(second)) is rough
    assert loop.estimate_request_tokens_rough([]) == 200
    assert first.requests == []
    assert second.requests == [{"model": "qwen-test", "messages": []}]


def test_install_reports_absent_hermes(caplog: pytest.LogCaptureFixture) -> None:
    original = sys.modules.pop("agent", None)
    conversation_loop = sys.modules.pop("agent.conversation_loop", None)
    try:
        assert install(runtime(Counter(1))) is None
    finally:
        if original is not None:
            sys.modules["agent"] = original
        if conversation_loop is not None:
            sys.modules["agent.conversation_loop"] = conversation_loop
    assert "Hermes is not installed" in caplog.text
