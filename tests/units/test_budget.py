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


def test_rough_fallback_counts_extra_body_prompt_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count the provider-visible prompt when exact tokenization fails.

    OpenAI clients shallow-merge ``extra_body`` over generated request fields,
    and the exact backend mirrors that rule for messages and tools.  Hermes'
    rough estimator must receive the same effective values during a tokenizer
    outage; budgeting from superseded top-level content can otherwise allocate
    output beyond the compression boundary by an unbounded amount.
    """

    top_messages = [{"role": "user", "content": "superseded"}]
    wire_messages = [{"role": "user", "content": f"wire-{index}"} for index in range(7)]
    wire_tools = [{"type": "function", "function": {"name": "wire"}}]
    agent_package = types.ModuleType("agent")
    agent_package.__path__ = []  # type: ignore[attr-defined]
    metadata_module = types.ModuleType("agent.model_metadata")

    def estimator(messages: Any, *, tools: Any) -> int:
        assert messages is wire_messages
        assert tools is wire_tools
        return 700

    metadata_module.estimate_request_tokens_rough = estimator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent", agent_package)
    monkeypatch.setitem(sys.modules, "agent.model_metadata", metadata_module)

    assert (
        budget.estimate_request_tokens_rough(
            {
                "messages": top_messages,
                "tools": [{"type": "function"}],
                "extra_body": {
                    "messages": wire_messages,
                    "tools": wire_tools,
                },
            },
        )
        == 700
    )


def test_runtime_keeps_injected_backend_and_default_estimator(
    runtime_settings: Settings,
) -> None:
    backend = Counter(1)
    runtime = budget.DynamicOutputBudget(runtime_settings, backend)
    assert runtime.backend is backend
    assert runtime.rough_estimator is budget.estimate_request_tokens_rough


def test_runtime_exact_count_preserves_smallest_existing_output_cap(
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
    assert rewritten["max_tokens"] == 10
    assert "max_completion_tokens" not in rewritten
    assert "max_output_tokens" not in rewritten
    assert rewritten["temperature"] == 0.5
    assert request["max_tokens"] == 10
    assert counter.requests == [(request, 65_536)]
    assert result["source"] == "incontext"
    assert "unit-backend" in result["reason"]


def test_runtime_uses_all_free_space_without_an_existing_output_cap(
    runtime_settings: Settings,
) -> None:
    runtime = budget.DynamicOutputBudget(runtime_settings, Counter(10_000))
    result = runtime(request={"model": "qwen", "messages": []})
    assert result is not None
    assert result["request"]["max_tokens"] == 45_705


@pytest.mark.parametrize(
    "field",
    ["max_tokens", "max_completion_tokens", "max_output_tokens"],
)
def test_runtime_preserves_the_provider_selected_output_field(field: str) -> None:
    """Keep Hermes' provider-specific wire parameter while lowering its cap.

    Hermes chooses the output field before invoking request middleware.  Newer
    OpenAI-family models reject legacy ``max_tokens``, while other transports
    use their own alias; replacing the chosen name can turn a valid request into
    HTTP 400 even when the numeric dynamic budget is correct.
    """

    runtime_settings = Settings(
        model_name="qwen",
        context_length=65_536,
        compression_window=55_705,
        fallback_margin_tokens=1024,
        provider="",
        base_url="",
    )
    runtime = budget.DynamicOutputBudget(runtime_settings, Counter(50_000))

    result = runtime(
        request={
            "model": "qwen",
            "messages": [],
            field: 8192,
        },
    )

    assert result is not None
    assert result["request"][field] == 5705
    assert set(result["request"]).isdisjoint(
        set(budget.OUTPUT_BUDGET_FIELDS) - {field},
    )


def test_runtime_removes_extra_body_output_cap_override() -> None:
    """Prevent OpenAI's ``extra_body`` merge from undoing the dynamic budget.

    The OpenAI client shallow-merges ``extra_body`` after normal parameters, so
    an output alias left there wins on the HTTP wire.  Incontext must include it
    when selecting the smallest caller cap, move the safe result to the same
    top-level field, and remove every nested alias without mutating the input.
    """

    runtime_settings = Settings(
        model_name="qwen",
        context_length=1000,
        compression_window=800,
        fallback_margin_tokens=10,
        provider="",
        base_url="",
    )
    runtime = budget.DynamicOutputBudget(runtime_settings, Counter(100))
    request = {
        "model": "qwen",
        "messages": [],
        "extra_body": {
            "max_completion_tokens": 600,
            "max_tokens": 100_000,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }

    result = runtime(request=request)

    assert result is not None
    assert result["request"]["max_completion_tokens"] == 600
    assert result["request"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert request["extra_body"] == {
        "max_completion_tokens": 600,
        "max_tokens": 100_000,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_nested_smaller_cap_preserves_top_level_provider_field() -> None:
    """Honor a nested bound without changing Hermes' selected wire parameter.

    Hermes may choose ``max_completion_tokens`` for a model that rejects the
    legacy alias.  A smaller ``max_tokens`` left in ``extra_body`` still limits
    the numeric budget because OpenAI clients merge it onto the wire request,
    but removing that override must emit the minimum through the existing
    top-level provider field rather than reintroducing the rejected alias.
    """

    runtime_settings = Settings(
        model_name="gpt-5-test",
        context_length=1000,
        compression_window=800,
        fallback_margin_tokens=10,
        provider="",
        base_url="",
    )
    runtime = budget.DynamicOutputBudget(runtime_settings, Counter(100))

    result = runtime(
        request={
            "model": "gpt-5-test",
            "messages": [],
            "max_completion_tokens": 600,
            "extra_body": {"max_tokens": 200},
        },
    )

    assert result is not None
    assert result["request"]["max_completion_tokens"] == 200
    assert "max_tokens" not in result["request"]
    assert result["request"]["extra_body"] == {}


def test_runtime_rejects_extra_body_model_override(
    runtime_settings: Settings,
) -> None:
    """Never budget a wire-level model override with the primary window.

    OpenAI-compatible clients shallow-merge ``extra_body`` after their normal
    request fields.  Consequently its model value is the provider-visible
    route and must take precedence during scoping, just as it does in the
    bundled backend; otherwise a fallback model receives the primary model's
    token count and compression-window arithmetic.
    """

    counter = Counter(100)
    runtime = budget.DynamicOutputBudget(runtime_settings, counter)
    request = {
        "model": runtime_settings.model_name,
        "messages": [],
        "extra_body": {"model": "fallback-model"},
    }

    assert runtime(request=request) is None
    assert counter.requests == []


@pytest.mark.parametrize(
    "invalid_cap",
    [None, True, False, 0, -1, 1.5, "2048"],
)
def test_runtime_ignores_invalid_existing_output_caps(
    runtime_settings: Settings,
    invalid_cap: Any,
) -> None:
    runtime = budget.DynamicOutputBudget(runtime_settings, Counter(10_000))
    result = runtime(
        request={
            "model": "qwen",
            "messages": [],
            "max_tokens": invalid_cap,
        },
    )
    assert result is not None
    assert result["request"]["max_tokens"] == 45_705


@given(
    free_space=st.integers(min_value=1, max_value=50_000),
    requested_cap=st.integers(min_value=1, max_value=1_000_000),
)
def test_existing_output_cap_is_never_increased(
    free_space: int,
    requested_cap: int,
) -> None:
    runtime_settings = Settings(
        model_name="qwen",
        context_length=65_536,
        compression_window=55_705,
        fallback_margin_tokens=1024,
        provider="",
        base_url="",
    )
    prompt_tokens = runtime_settings.compression_window - free_space
    runtime = budget.DynamicOutputBudget(
        runtime_settings,
        Counter(prompt_tokens),
    )
    result = runtime(
        request={
            "model": "qwen",
            "messages": [],
            "max_tokens": requested_cap,
        },
    )
    assert result is not None
    assert result["request"]["max_tokens"] == min(free_space, requested_cap)


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


def test_runtime_ignores_request_for_a_different_model(
    runtime_settings: Settings,
) -> None:
    """Never apply the primary tokenizer and window after a model switch.

    Hermes can route a turn to a session override or fallback model while the
    process remains alive.  Without a separately configured backend and
    compression boundary for that model, fail-open is safer than computing an
    apparently exact cap from the primary model's tokenizer.
    """

    counter = Counter(100)
    runtime = budget.DynamicOutputBudget(runtime_settings, counter)

    result = runtime(
        request={
            "model": "fallback-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert result is None
    assert counter.requests == []


def test_runtime_ignores_same_model_on_a_different_provider_route() -> None:
    """Fail open when middleware context identifies a different endpoint.

    A session fallback may retain the same model alias while changing provider
    or base URL.  Hermes supplies both values to ``llm_request`` middleware, so
    incontext must not apply the primary vLLM tokenizer merely because the JSON
    model string still matches.
    """

    counter = Counter(100)
    runtime = budget.DynamicOutputBudget(
        Settings(
            model_name="shared-model",
            context_length=1000,
            compression_window=800,
            fallback_margin_tokens=10,
            provider="custom",
            base_url="https://primary.invalid/v1",
        ),
        counter,
    )
    request = {"model": "shared-model", "messages": []}

    assert (
        runtime(
            request=request,
            provider="openai",
            base_url="https://fallback.invalid/v1",
        )
        is None
    )
    assert counter.requests == []


@given(
    small=st.integers(min_value=1, max_value=40_000),
    growth=st.integers(min_value=1, max_value=40_000),
)
def test_budget_never_increases_as_prompt_grows(small: int, growth: int) -> None:
    window = 64_000
    small_cap = budget.compute_max_tokens(window, small)
    large_cap = budget.compute_max_tokens(window, small + growth)
    assert small_cap is None or large_cap is None or large_cap <= small_cap
