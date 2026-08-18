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
            provider="",
            base_url="",
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
        reasoning_config: dict[str, Any] | None = None,
        base_url: str | None = None,
        task: str | None = None,
    ) -> dict[str, Any]:
        del provider, max_tokens, base_url
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "timeout": timeout,
            "extra_body": extra_body,
            "reasoning_config": reasoning_config,
            "task": task,
        }

    auxiliary._build_call_kwargs = build  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary)
    return auxiliary, build


def test_install_preserves_a_smaller_auxiliary_output_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auxiliary, _ = install_fake_hermes(monkeypatch)
    counter = Counter(12_345)

    cleanup = install(runtime(counter))
    assert callable(cleanup)
    result = auxiliary._build_call_kwargs(  # type: ignore[attr-defined]
        "custom",
        "qwen-test",
        [{"role": "user", "content": "summarize"}],
        temperature=0.25,
        max_tokens=2048,
        tools=[{"type": "function"}],
        timeout=3600,
        extra_body={"answer": 42},
        reasoning_config={"effort": "high"},
        base_url="https://inference.invalid/v1",
        task="compression",
    )

    assert result == {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "summarize"}],
        "temperature": 0.25,
        "tools": [{"type": "function"}],
        "timeout": 3600,
        "extra_body": {"answer": 42},
        "reasoning_config": {"effort": "high"},
        "task": "compression",
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


@pytest.mark.parametrize("caller_cap", [None, 2048])
def test_auxiliary_uses_hermes_provider_output_alias(
    monkeypatch: pytest.MonkeyPatch,
    caller_cap: int | None,
) -> None:
    """Use Hermes' own model-aware selector for omitted output fields.

    Hermes deliberately omits an auxiliary cap before choosing a provider's
    accepted wire alias.  New OpenAI models reject the generic ``max_tokens``
    field, so both bounded and unbounded calls must seed dynamic budgeting with
    ``max_completion_tokens`` when Hermes selects it.
    """

    auxiliary, _ = install_fake_hermes(monkeypatch)
    selections: list[tuple[int, str]] = []

    def select(value: int, *, model: str) -> dict[str, int]:
        selections.append((value, model))
        return {"max_completion_tokens": value}

    auxiliary.auxiliary_max_tokens_param = select  # type: ignore[attr-defined]
    install(runtime(Counter(12_345)))

    result = auxiliary._build_call_kwargs(  # type: ignore[attr-defined]
        "custom",
        "qwen-test",
        [{"role": "user", "content": "title"}],
        max_tokens=caller_cap,
    )

    seed = 64_000 if caller_cap is None else caller_cap
    assert selections == [(seed, "qwen-test")]
    assert result["max_completion_tokens"] == min(seed, 64_000 - 12_345)
    assert "max_tokens" not in result


@pytest.mark.parametrize(
    "selector",
    [
        lambda value, **context: {"max_tokens": False},
        lambda value, **context: (_ for _ in ()).throw(RuntimeError("boom")),
    ],
)
def test_auxiliary_falls_back_from_an_invalid_hermes_output_selector(
    selector: Any,
) -> None:
    """Keep auxiliary requests usable when a Hermes selector is incompatible.

    The helper is a private Hermes API and may be absent, raise, or return an
    invalid cap during a version transition.  Such failures must degrade to
    the long-supported ``max_tokens`` field rather than disabling all
    auxiliary calls before they reach the provider.
    """

    def build(provider: str, model: str, messages: list[Any]) -> dict[str, Any]:
        del provider
        return {"model": model, "messages": messages}

    result = _AuxiliaryBudget(
        runtime(Counter(12_345)),
        build,
        selector,
    )("custom", "qwen-test", [{"role": "user", "content": "title"}])

    assert result["max_tokens"] == 64_000 - 12_345


def test_auxiliary_wrapper_accepts_additive_hermes_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward newer Hermes builder arguments without signature drift.

    Hermes 2026.8 added ``reasoning_config`` and ``task`` to the private
    auxiliary builder used by every sync and async call path.  A wrapper that
    duplicates the older signature raises ``TypeError`` before any provider
    request, so incontext must transparently forward additive parameters.
    """

    auxiliary, _ = install_fake_hermes(monkeypatch)
    install(runtime(Counter(12_345)))

    result = auxiliary._build_call_kwargs(  # type: ignore[attr-defined]
        "custom",
        "qwen-test",
        [{"role": "user", "content": "reason"}],
        reasoning_config={"effort": "high"},
        task="compression",
    )

    assert result["reasoning_config"] == {"effort": "high"}
    assert result["task"] == "compression"


def test_auxiliary_variadic_builder_without_optional_output_cap() -> None:
    """Resolve omitted names safely from a generic ``**kwargs`` signature.

    Decorators and future Hermes adapters may expose the request builder as a
    variadic callable.  When no caller cap is present, argument discovery must
    fall through cleanly and still apply the full dynamic remainder rather than
    mistaking an empty kwargs mapping for an incompatible signature.
    """

    def build(
        provider: str,
        model: str,
        messages: list[Any],
        **options: Any,
    ) -> dict[str, Any]:
        del provider, options
        return {"model": model, "messages": messages}

    result = _AuxiliaryBudget(runtime(Counter(12_345)), build)(
        "custom",
        "qwen-test",
        [{"role": "user", "content": "title"}],
    )

    assert result["max_tokens"] == 64_000 - 12_345


def test_auxiliary_wrapper_preserves_hermes_output_field() -> None:
    """Respect a provider-specific cap already emitted by Hermes.

    On newer OpenAI-family models Hermes emits ``max_completion_tokens``.
    Reintroducing the caller's generic ``max_tokens`` would overwrite that
    validated provider choice and cause HTTP 400 on the auxiliary request.
    """

    def build(
        provider: str,
        model: str,
        messages: list[Any],
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        del provider, max_tokens
        return {
            "model": model,
            "messages": messages,
            "max_completion_tokens": 2048,
        }

    result = _AuxiliaryBudget(runtime(Counter(12_345)), build)(
        "custom",
        "qwen-test",
        [{"role": "user", "content": "summarize"}],
        max_tokens=4096,
    )

    assert result["max_completion_tokens"] == 2048
    assert "max_tokens" not in result


def test_auxiliary_wrapper_ignores_a_different_fallback_model() -> None:
    """Never tokenize an auxiliary fallback with the primary model backend.

    Hermes can retry an auxiliary task on a different provider and model while
    the process-wide incontext runtime still points at the primary vLLM model.
    Applying that tokenizer and context window to the fallback would corrupt
    its request; the original provider kwargs must pass through untouched.
    """

    counter = Counter(12_345)

    def build(
        provider: str,
        model: str,
        messages: list[Any],
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        del provider, max_tokens
        return {"model": model, "messages": messages}

    result = _AuxiliaryBudget(runtime(counter), build)(
        "openai",
        "fallback-model",
        [{"role": "user", "content": "retry"}],
        max_tokens=4096,
    )

    assert result == {
        "model": "fallback-model",
        "messages": [{"role": "user", "content": "retry"}],
    }
    assert counter.requests == []


def test_auxiliary_wrapper_ignores_same_model_on_another_route() -> None:
    """Distinguish a fallback endpoint even when it reuses the model alias.

    Hermes fallback destinations carry provider and base URL independently of
    the model name.  Tokenizing ``shared-model`` on the primary local vLLM is
    still wrong when the request is headed to an external provider that happens
    to expose the same alias.
    """

    counter = Counter(12_345)
    scoped_runtime = DynamicOutputBudget(
        Settings(
            model_name="qwen-test",
            context_length=65_536,
            compression_window=64_000,
            fallback_margin_tokens=1024,
            provider="custom",
            base_url="https://primary.invalid/v1",
        ),
        counter,
    )

    def build(
        provider: str,
        model: str,
        messages: list[Any],
        base_url: str | None = None,
    ) -> dict[str, Any]:
        del provider, base_url
        return {"model": model, "messages": messages}

    result = _AuxiliaryBudget(scoped_runtime, build)(
        "custom",
        "qwen-test",
        [{"role": "user", "content": "retry"}],
        base_url="https://fallback.invalid/v1",
    )

    assert result == {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "retry"}],
    }
    assert counter.requests == []


def test_auxiliary_budgets_synthetic_main_agent_fallback_label() -> None:
    """Treat Hermes' main-agent fallback label as the configured primary route.

    When an auxiliary provider fails, Hermes' final safety net resolves the
    real main-model client but calls its builder with the diagnostic label
    ``main-agent(custom)``.  Its model and base URL still identify the primary
    deployment exactly; rejecting only the synthetic label loses both exact
    tokenization and the bounded summary cap that Hermes omitted upstream.
    """

    counter = Counter(12_345)
    scoped_runtime = DynamicOutputBudget(
        Settings(
            model_name="qwen-test",
            context_length=65_536,
            compression_window=64_000,
            fallback_margin_tokens=1024,
            provider="custom",
            base_url="https://primary.invalid/v1",
        ),
        counter,
    )

    def build(
        provider: str,
        model: str,
        messages: list[Any],
        max_tokens: int | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        del provider, max_tokens, base_url
        return {"model": model, "messages": messages}

    result = _AuxiliaryBudget(scoped_runtime, build)(
        "main-agent(custom)",
        "qwen-test",
        [{"role": "user", "content": "compression summary"}],
        max_tokens=2048,
        base_url="https://primary.invalid/v1",
    )

    assert result["max_tokens"] == 2048
    assert len(counter.requests) == 1


def test_auxiliary_wrapper_ignores_same_endpoint_on_another_provider() -> None:
    """Use provider identity as well as the normalized endpoint URL.

    Two Hermes providers may share a gateway URL but apply different wire
    contracts and model routing.  Matching only the URL and model would still
    let the primary backend rewrite a fallback request owned by another route.
    """

    counter = Counter(12_345)
    scoped_runtime = DynamicOutputBudget(
        Settings(
            model_name="qwen-test",
            context_length=65_536,
            compression_window=64_000,
            fallback_margin_tokens=1024,
            provider="custom",
            base_url="https://shared.invalid/v1",
        ),
        counter,
    )

    def build(
        provider: str,
        model: str,
        messages: list[Any],
        base_url: str | None = None,
    ) -> dict[str, Any]:
        del provider, base_url
        return {"model": model, "messages": messages}

    result = _AuxiliaryBudget(scoped_runtime, build)(
        "openai",
        "qwen-test",
        [{"role": "user", "content": "retry"}],
        base_url="https://shared.invalid/v1",
    )

    assert "max_tokens" not in result
    assert counter.requests == []


def test_auxiliary_accepts_canonical_equivalent_route() -> None:
    """Match the configured endpoint after HTTP-client canonicalization.

    Hermes passes its auxiliary client's rendered URL and normalized provider.
    Lowercasing DNS hosts, removing the default HTTPS port, or appending a slash
    does not change route identity, so those transformations must not silently
    bypass exact budgeting for the configured primary endpoint.
    """

    counter = Counter(12_345)
    scoped_runtime = DynamicOutputBudget(
        Settings(
            model_name="qwen-test",
            context_length=65_536,
            compression_window=64_000,
            fallback_margin_tokens=1024,
            provider="custom",
            base_url="https://primary.invalid/v1",
        ),
        counter,
    )

    def build(
        provider: str,
        model: str,
        messages: list[Any],
        base_url: str | None = None,
    ) -> dict[str, Any]:
        del provider, base_url
        return {"model": model, "messages": messages}

    result = _AuxiliaryBudget(scoped_runtime, build)(
        " Custom ",
        "qwen-test",
        [{"role": "user", "content": "primary"}],
        base_url="https://PRIMARY.INVALID:443/v1/",
    )

    assert result["max_tokens"] == 64_000 - 12_345
    assert len(counter.requests) == 1


def test_auxiliary_runtime_resolver_tracks_the_active_profile() -> None:
    """Resolve the profile-scoped runtime at call time, not installation time.

    A single process-global Hermes builder wrapper serves multiple active homes
    in 2026.8.  The resolver must therefore be invoked for every request so a
    profile switch cannot retain the previous profile's backend and window.
    """

    active = runtime(Counter(12_345))
    calls: list[None] = []

    def resolve() -> DynamicOutputBudget:
        calls.append(None)
        return active

    def build(
        provider: str,
        model: str,
        messages: list[Any],
    ) -> dict[str, Any]:
        del provider
        return {"model": model, "messages": messages}

    result = _AuxiliaryBudget(resolve, build)(
        "custom",
        "qwen-test",
        [{"role": "user", "content": "profile"}],
    )

    assert result["max_tokens"] == 64_000 - 12_345
    assert calls == [None]


def test_auxiliary_skips_profiles_without_an_active_plugin_owner() -> None:
    """Preserve Hermes' request for a profile where incontext is disabled.

    One imported auxiliary builder is shared by all profile managers in the
    process.  If the active home's resolver returns no registered runtime, the
    wrapper must return the original provider kwargs unchanged and avoid any
    tokenizer request owned by another profile.
    """

    def build(
        provider: str,
        model: str,
        messages: list[Any],
        **options: Any,
    ) -> dict[str, Any]:
        return {
            "provider": provider,
            "model": model,
            "messages": messages,
            **options,
        }

    result = _AuxiliaryBudget(lambda: None, build)(
        "custom",
        "qwen-test",
        [{"role": "user", "content": "private"}],
        max_tokens=2048,
    )

    assert result == {
        "provider": "custom",
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "private"}],
        "max_tokens": 2048,
    }


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
    auxiliary, _ = install_fake_hermes(monkeypatch)
    first = Counter(10_000)
    second = Counter(20_000)

    first_cleanup = install(runtime(first))
    second_cleanup = install(runtime(second))
    assert callable(first_cleanup)
    assert callable(second_cleanup)
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
            "reasoning_config": None,
            "task": None,
            "max_tokens": 64_000,
        }
    ]


def test_cleanup_restores_builder_after_the_last_plugin_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the global wrapper until every profile unloads, then restore it.

    Hermes 2026.8 can load one entry-point plugin for multiple profile-scoped
    managers in a single process.  Unloading either owner must not disable the
    other, while the final callback must conditionally restore the exact
    original builder and remain safe if invoked twice.
    """

    auxiliary, original = install_fake_hermes(monkeypatch)
    first_cleanup = install(runtime(Counter(100)))
    second_cleanup = install(runtime(Counter(200)))
    assert callable(first_cleanup)
    assert callable(second_cleanup)
    wrapper = auxiliary._build_call_kwargs  # type: ignore[attr-defined]

    first_cleanup()
    assert auxiliary._build_call_kwargs is wrapper  # type: ignore[attr-defined]
    first_cleanup()
    assert auxiliary._build_call_kwargs is wrapper  # type: ignore[attr-defined]

    second_cleanup()
    assert auxiliary._build_call_kwargs is original  # type: ignore[attr-defined]


def test_cleanup_never_overwrites_a_later_auxiliary_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Condition restoration on ownership of the current Hermes binding.

    Another plugin may replace the builder after incontext registers.  Its
    callable must survive incontext unload; cleanup only releases internal
    ownership and must never put an older function back over newer state.
    """

    auxiliary, _ = install_fake_hermes(monkeypatch)
    cleanup = install(runtime(Counter(100)))
    assert callable(cleanup)

    replacement = lambda *args, **kwargs: {}  # noqa: E731
    auxiliary._build_call_kwargs = replacement  # type: ignore[attr-defined]
    cleanup()

    assert auxiliary._build_call_kwargs is replacement  # type: ignore[attr-defined]


def test_stale_auxiliary_cleanup_cannot_remove_a_new_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore an unload callback whose wrapper is no longer globally active.

    Force rediscovery can replace the module binding and establish a new owner
    before an older manager disposes its ledger.  The stale callback must not
    decrement or restore the new installation's reference count.
    """

    _, _ = install_fake_hermes(monkeypatch)
    stale_cleanup = install(runtime(Counter(100)))
    assert callable(stale_cleanup)

    second_auxiliary, _ = install_fake_hermes(monkeypatch)
    active_cleanup = install(runtime(Counter(200)))
    assert callable(active_cleanup)
    active_wrapper = second_auxiliary._build_call_kwargs  # type: ignore[attr-defined]

    stale_cleanup()
    assert second_auxiliary._build_call_kwargs is active_wrapper  # type: ignore[attr-defined]


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
