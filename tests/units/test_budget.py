from __future__ import annotations

import logging
import sys
import types
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from incontext import budget
from incontext.backend import Backend
from incontext.settings import Settings


class Counter(Backend):
    def __init__(
        self,
        result: int | BaseException,
        *,
        source: str = "unit-backend",
    ) -> None:
        self.result = result
        self._source = source
        self.requests: list[tuple[dict[str, Any], int]] = []

    @property
    def source(self) -> str:
        return self._source

    def count(
        self,
        request: dict[str, Any],
        *,
        context_length: int,
    ) -> int:
        self.requests.append((request, context_length))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    def clear_cache(self) -> None:
        self.requests.clear()


@pytest.mark.parametrize(
    ("window", "prompt", "margin", "expected"),
    [
        (64_000, 1_000, 0, 63_000),
        (64_000, 1_000, 1024, 61_976),
        (64_000, 64_000, 0, None),
        (64_000, 70_000, 0, None),
    ],
)
def test_compute_max_tokens_boundaries(
    window: int,
    prompt: int,
    margin: int,
    expected: int | None,
) -> None:
    assert budget.compute_max_tokens(window, prompt, safety_margin=margin) == expected


@pytest.mark.parametrize(
    ("window", "prompt", "margin", "message"),
    [
        (0, 1, 0, "compression_window"),
        (1, 0, 0, "prompt_tokens"),
        (1, 1, -1, "safety_margin"),
    ],
)
def test_compute_max_tokens_rejects_invalid_inputs(
    window: int,
    prompt: int,
    margin: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        budget.compute_max_tokens(window, prompt, safety_margin=margin)


@given(
    window=st.integers(min_value=1, max_value=1_000_000),
    prompt=st.integers(min_value=1, max_value=2_000_000),
    margin=st.integers(min_value=0, max_value=100_000),
)
def test_compute_max_tokens_invariants(window: int, prompt: int, margin: int) -> None:
    result = budget.compute_max_tokens(window, prompt, safety_margin=margin)
    assert result == (window - prompt - margin if window > prompt + margin else None)


def install_estimator(monkeypatch: pytest.MonkeyPatch, value: Any) -> None:
    agent_package = types.ModuleType("agent")
    agent_package.__path__ = []  # type: ignore[attr-defined]
    metadata_module = types.ModuleType("agent.model_metadata")

    def estimator(messages: Any, *, tools: Any) -> Any:
        assert isinstance(messages, list)
        assert tools == [{"type": "function"}]
        return value

    metadata_module.estimate_request_tokens_rough = estimator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent", agent_package)
    monkeypatch.setitem(sys.modules, "agent.model_metadata", metadata_module)


@pytest.mark.parametrize(("value", "expected"), [(12.9, 12), (0, 1), (True, 1)])
def test_estimate_request_tokens_rough_normalizes_result(
    monkeypatch: pytest.MonkeyPatch,
    value: Any,
    expected: int,
) -> None:
    install_estimator(monkeypatch, value)
    request = {"messages": [], "tools": [{"type": "function"}]}
    assert budget.estimate_request_tokens_rough(request) == expected


@pytest.mark.parametrize("value", [None, "invalid"])
def test_estimate_request_tokens_rough_handles_invalid_result(
    monkeypatch: pytest.MonkeyPatch,
    value: Any,
) -> None:
    install_estimator(monkeypatch, value)
    request = {"messages": [], "tools": [{"type": "function"}]}
    assert budget.estimate_request_tokens_rough(request) == 1


def test_runtime_keeps_injected_backend_and_default_estimator(
    runtime_settings: Settings,
) -> None:
    backend = Counter(1)
    runtime = budget.DynamicOutputBudget(runtime_settings, backend)
    assert runtime.backend is backend
    assert runtime.rough_estimator is budget.estimate_request_tokens_rough


def test_runtime_exact_count_rewrites_all_output_aliases(
    runtime_settings: Settings,
) -> None:
    counter = Counter(10_000)
    runtime = budget.DynamicOutputBudget(
        runtime_settings,
        counter,
        rough_estimator=lambda request: 999,
    )
    request = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "secret prompt"}],
        "max_tokens": 10,
        "max_completion_tokens": 20,
        "max_output_tokens": 30,
        "temperature": 0.5,
    }
    result = runtime(request=request, session_id="ignored")
    assert result is not None
    rewritten = result["request"]
    assert rewritten["max_tokens"] == 45_705
    assert "max_completion_tokens" not in rewritten
    assert "max_output_tokens" not in rewritten
    assert rewritten["temperature"] == 0.5
    assert request["max_tokens"] == 10
    assert counter.requests == [(request, 65_536)]
    assert result["source"] == "incontext"
    assert "unit-backend" in result["reason"]


def test_runtime_fallback_reserves_safety_margin(
    runtime_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = budget.DynamicOutputBudget(
        runtime_settings,
        Counter(TimeoutError("secret failure")),
        rough_estimator=lambda request: 12_000,
    )
    request = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "private material"}],
    }
    with caplog.at_level(logging.INFO):
        result = runtime(request=request)
    assert result is not None
    assert result["request"]["max_tokens"] == 42_681
    assert "rough-fallback" in result["reason"]
    assert "private material" not in caplog.text
    assert "secret failure" not in caplog.text


@pytest.mark.parametrize("prompt_tokens", [55_705, 65_536])
def test_runtime_does_not_inject_an_invalid_sentinel_for_full_window(
    runtime_settings: Settings,
    prompt_tokens: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = budget.DynamicOutputBudget(runtime_settings, Counter(prompt_tokens))
    request = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "large prompt"}],
        "max_tokens": 8192,
    }
    with caplog.at_level(logging.WARNING):
        assert runtime(request=request) is None
    assert request["max_tokens"] == 8192
    assert "action=requires_compression" in caplog.text


def test_runtime_normalizes_non_positive_fallback_estimate(
    runtime_settings: Settings,
) -> None:
    runtime = budget.DynamicOutputBudget(
        runtime_settings,
        Counter(RuntimeError()),
        rough_estimator=lambda request: 0,
    )
    result = runtime(request={"model": "qwen", "messages": []})
    assert result is not None
    assert result["request"]["max_tokens"] == 54_680


def test_runtime_leaves_request_unchanged_when_both_estimators_fail(
    runtime_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken_fallback(request: dict[str, Any]) -> int:
        raise ValueError("fallback secret")

    runtime = budget.DynamicOutputBudget(
        runtime_settings,
        Counter(RuntimeError("exact secret")),
        rough_estimator=broken_fallback,
    )
    request = {"model": "qwen", "messages": []}
    with caplog.at_level(logging.ERROR):
        assert runtime(request=request) is None
    assert "exact secret" not in caplog.text
    assert "fallback secret" not in caplog.text


@pytest.mark.parametrize(
    "payload",
    [None, {}, {"messages": None}, {"messages": "text"}],
)
def test_runtime_ignores_unsupported_request_shapes(
    runtime_settings: Settings,
    payload: Any,
) -> None:
    runtime = budget.DynamicOutputBudget(runtime_settings, Counter(1))
    assert runtime(request=payload) is None


@given(
    small=st.integers(min_value=1, max_value=40_000),
    growth=st.integers(min_value=1, max_value=40_000),
)
def test_budget_never_increases_as_prompt_grows(small: int, growth: int) -> None:
    window = 64_000
    small_cap = budget.compute_max_tokens(window, small)
    large_cap = budget.compute_max_tokens(window, small + growth)
    assert small_cap is None or large_cap is None or large_cap <= small_cap
