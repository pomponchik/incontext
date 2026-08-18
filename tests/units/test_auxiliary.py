from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from incontext.auxiliary import _AuxiliaryBudget, install
from incontext.backend import Backend
from incontext.budget import DynamicOutputBudget
from incontext.settings import Settings


class Counter(Backend):
    def __init__(self, result: int) -> None:
        self.result = result
        self.requests: list[dict[str, Any]] = []

    @property
    def source(self) -> str:
        return "test-counter"

    def count(self, request: dict[str, Any], *, context_length: int) -> int:
        assert context_length == 65_536
        self.requests.append(request)
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
) -> tuple[types.ModuleType, Any]:
    agent = types.ModuleType("agent")
    agent.__path__ = []  # type: ignore[attr-defined]
    auxiliary = types.ModuleType("agent.auxiliary_client")

    def build(
        provider: str,
        model: str,
        messages: list[Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[Any] | None = None,
        timeout: float = 30.0,
        extra_body: dict[str, Any] | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        del provider, max_tokens, base_url
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "timeout": timeout,
            "extra_body": extra_body,
        }

    auxiliary._build_call_kwargs = build  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary)
    return auxiliary, build


def test_install_preserves_a_smaller_auxiliary_output_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auxiliary, original = install_fake_hermes(monkeypatch)
    counter = Counter(12_345)

    assert install(runtime(counter)) is original
    result = auxiliary._build_call_kwargs(  # type: ignore[attr-defined]
        "custom",
        "qwen-test",
        [{"role": "user", "content": "summarize"}],
        temperature=0.25,
        max_tokens=2048,
        tools=[{"type": "function"}],
        timeout=3600,
        extra_body={"answer": 42},
        base_url="https://inference.invalid/v1",
    )

    assert result == {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "summarize"}],
        "temperature": 0.25,
        "tools": [{"type": "function"}],
        "timeout": 3600,
        "extra_body": {"answer": 42},
        "max_tokens": 2048,
    }
    assert counter.requests == [result]


def test_auxiliary_without_a_caller_bound_uses_the_free_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auxiliary, _ = install_fake_hermes(monkeypatch)
    install(runtime(Counter(12_345)))

    result = auxiliary._build_call_kwargs(  # type: ignore[attr-defined]
        "custom",
        "qwen-test",
        [{"role": "user", "content": "title"}],
    )

    assert result["max_tokens"] == 64_000 - 12_345


@given(
    prompt_tokens=st.integers(min_value=1, max_value=63_999),
    caller_cap=st.integers(min_value=1, max_value=100_000),
)
def test_auxiliary_budget_never_increases_the_caller_cap(
    prompt_tokens: int,
    caller_cap: int,
) -> None:
    def build(
        provider: str,
        model: str,
        messages: list[Any],
        **options: Any,
    ) -> dict[str, Any]:
        del provider, options
        return {"model": model, "messages": messages}

    result = _AuxiliaryBudget(runtime(Counter(prompt_tokens)), build)(
        "custom",
        "qwen-test",
        [{"role": "user", "content": "property"}],
        max_tokens=caller_cap,
    )

    assert result["max_tokens"] == min(caller_cap, 64_000 - prompt_tokens)


def test_auxiliary_fails_open_when_compression_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auxiliary, _ = install_fake_hermes(monkeypatch)
    install(runtime(Counter(64_000)))

    result = auxiliary._build_call_kwargs(  # type: ignore[attr-defined]
        "custom",
        "qwen-test",
        [{"role": "user", "content": "oversized"}],
        max_tokens=2048,
    )

    assert "max_tokens" not in result


def test_install_is_idempotent_and_updates_the_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auxiliary, original = install_fake_hermes(monkeypatch)
    first = Counter(10_000)
    second = Counter(20_000)

    assert install(runtime(first)) is original
    assert install(runtime(second)) is original
    result = auxiliary._build_call_kwargs(  # type: ignore[attr-defined]
        "custom",
        "qwen-test",
        [{"role": "user", "content": "latest"}],
    )

    assert result["max_tokens"] == 44_000
    assert first.requests == []
    assert second.requests == [
        {
            "model": "qwen-test",
            "messages": [{"role": "user", "content": "latest"}],
            "temperature": None,
            "tools": None,
            "timeout": 30.0,
            "extra_body": None,
        }
    ]


def test_install_reports_absent_hermes(caplog: pytest.LogCaptureFixture) -> None:
    original = sys.modules.pop("agent", None)
    auxiliary = sys.modules.pop("agent.auxiliary_client", None)
    try:
        assert install(runtime(Counter(1))) is None
    finally:
        if original is not None:
            sys.modules["agent"] = original
        if auxiliary is not None:
            sys.modules["agent.auxiliary_client"] = auxiliary
    assert "Hermes is not installed" in caplog.text
