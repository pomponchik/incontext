from __future__ import annotations

import sys
import threading
import types
from typing import Any

import pytest

from incontext import preflight
from incontext.backend import Backend
from incontext.budget import DynamicOutputBudget
from incontext.preflight import _ExactPreflight, _ExactPreflightGate, install
from incontext.settings import Settings

MINIMUM_OUTPUT_TOKENS = 4096


def pressure(prompt_tokens: int) -> int:
    return prompt_tokens + MINIMUM_OUTPUT_TOKENS - 1


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
            min_output_tokens=4096,
            provider="",
            base_url="",
        ),
        counter,
    )


def install_fake_hermes(
    monkeypatch: pytest.MonkeyPatch,
    original: Any,
) -> tuple[types.ModuleType, types.ModuleType]:
    agent = types.ModuleType("agent")
    agent.__path__ = []  # type: ignore[attr-defined]
    loop = types.ModuleType("agent.conversation_loop")
    turn_context = types.ModuleType("agent.turn_context")
    loop.estimate_request_tokens_rough = original  # type: ignore[attr-defined]
    turn_context.estimate_request_tokens_rough = original  # type: ignore[attr-defined]
    turn_context._should_run_preflight_estimate = (  # type: ignore[attr-defined]
        lambda messages, protect_first_n, protect_last_n, threshold_tokens: False  # noqa: ARG005
    )
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.conversation_loop", loop)
    monkeypatch.setitem(sys.modules, "agent.turn_context", turn_context)
    return loop, turn_context


def test_install_replaces_rough_preflight_with_exact_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch both proactive checks and count the complete provider prompt.

    Hermes checks once at turn start and again before every model call because
    tool results can make a previously small turn exceed the context boundary.
    Both modules import the estimator into independent local bindings, so
    leaving the in-turn binding rough lets an oversized request reach request
    middleware after the final opportunity to compress.  The exact wrapper
    must cover both bindings and still include the separately supplied system
    prompt and tools.
    """

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        raise AssertionError("exact backend should be used")

    loop, turn_context = install_fake_hermes(monkeypatch, rough)
    counter = Counter(64_000)
    cleanup = install(runtime(counter))

    assert callable(cleanup)
    expected_request = {
        "model": "qwen-test",
        "messages": [
            {"role": "system", "content": "Follow the policy"},
            {"role": "user", "content": "large"},
        ],
        "tools": [{"type": "function"}],
    }
    for module in (turn_context, loop):
        assert module.estimate_request_tokens_rough(
            [{"role": "user", "content": "large"}],
            system_prompt="Follow the policy",
            tools=[{"type": "function"}],
        ) == pressure(64_000)
    assert counter.requests == [expected_request, expected_request]


def test_in_turn_exact_pressure_is_never_deferred_as_a_noisy_rough_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compress tool-loop growth even after Hermes recently compacted history.

    Hermes may defer a high *rough* estimate immediately after compression so
    repeated schema over-counting does not cause a compaction loop.  An exact
    tokenizer result is authoritative and must bypass that heuristic: deferring
    it sends the request to middleware too late to compress, where incontext
    fails open and Hermes' full-window provider default survives on the wire.
    Ordinary rough values must retain Hermes' original defer behaviour.
    """

    class ContextCompressor:
        def should_defer_preflight_to_real_usage(self, tokens: int) -> bool:
            return tokens >= 64_000

    compressor_module = types.ModuleType("agent.context_compressor")
    compressor_module.ContextCompressor = ContextCompressor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)
    loop, _ = install_fake_hermes(
        monkeypatch,
        lambda messages, *, system_prompt="", tools=None: 30_508,  # noqa: ARG005
    )
    first_cleanup = install(runtime(Counter(61_052)))
    second_cleanup = install(runtime(Counter(61_052)))
    assert callable(first_cleanup)
    assert callable(second_cleanup)
    assert callable(ContextCompressor.should_defer_preflight_to_real_usage)

    exact_pressure = loop.estimate_request_tokens_rough(
        [{"role": "tool", "content": "large result"}],
        tools=[{"type": "function"}],
    )
    compressor = ContextCompressor()

    assert exact_pressure == pressure(61_052)
    assert compressor.should_defer_preflight_to_real_usage(exact_pressure) is False
    assert compressor.should_defer_preflight_to_real_usage(64_000) is True

    first_cleanup()
    assert compressor.should_defer_preflight_to_real_usage(exact_pressure) is False
    second_cleanup()
    assert compressor.should_defer_preflight_to_real_usage(exact_pressure) is True


@pytest.mark.parametrize("loop_api", [None, "renamed-private-api"])
def test_install_keeps_turn_preflight_when_in_turn_hook_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    loop_api: str | None,
) -> None:
    """Remain compatible with Hermes releases lacking the newer loop hook.

    The in-turn pressure check was added after the original turn-prologue
    preflight.  Older Hermes releases may omit its module, while a future one
    may temporarily expose no callable under the private binding.  Neither API
    shape should disable exact compression at the still-supported turn start.
    """
    _, turn_context = install_fake_hermes(
        monkeypatch,
        lambda messages, *, system_prompt="", tools=None: 7,  # noqa: ARG005
    )
    if loop_api is None:
        monkeypatch.delitem(sys.modules, "agent.conversation_loop")
    else:
        sys.modules["agent.conversation_loop"].estimate_request_tokens_rough = loop_api  # type: ignore[attr-defined]

    cleanup = install(runtime(Counter(123)))

    assert callable(cleanup)
    assert turn_context.estimate_request_tokens_rough([]) == pressure(123)


def test_install_tolerates_hermes_without_the_rough_defer_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not require a version-specific optimization to install preflight.

    Hermes versions predating the noisy-rough-estimate heuristic have nothing
    to bypass.  Absence of its compressor method must leave both exact pressure
    bindings active instead of turning a compatibility optimization into a
    mandatory private API dependency.
    """
    compressor_module = types.ModuleType("agent.context_compressor")
    compressor_module.ContextCompressor = None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)
    loop, turn_context = install_fake_hermes(
        monkeypatch,
        lambda messages, *, system_prompt="", tools=None: 7,  # noqa: ARG005
    )

    cleanup = install(runtime(Counter(123)))

    assert callable(cleanup)
    assert loop.estimate_request_tokens_rough([]) == pressure(123)
    assert turn_context.estimate_request_tokens_rough([]) == pressure(123)


def test_install_patches_the_turn_context_binding_used_for_compression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent rough preflight from surviving in Hermes ``turn_context``.

    Importing a function copies its binding into the consumer module.  Patching
    only ``conversation_loop`` therefore leaves the real proactive compression
    check untouched; this regression test exercises the independently imported
    ``turn_context`` name directly.
    """

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        return 7

    _, turn_context = install_fake_hermes(monkeypatch, rough)
    install(runtime(Counter(123)))

    assert turn_context.estimate_request_tokens_rough(
        [{"role": "user", "content": "large"}],
        tools=[{"type": "function"}],
    ) == pressure(123)


def test_preflight_counts_provider_visible_api_content_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count Hermes' API-bound sidecar rather than its clean stored content.

    Hermes persists clean conversation text beside an exact ``api_content``
    sidecar and substitutes that sidecar only while building the provider
    request.  vLLM ignores the private field, so forwarding both values would
    undercount injected memory or plugin context and skip compression.  The
    preflight copy must mirror substitution while preserving retry-owned input.
    """
    messages: list[Any] = [
        {
            "role": "user",
            "content": "clean",
            "api_content": "provider-visible context",
        },
        {
            "role": "system",
            "content": "policy",
            "api_content": "must not replace a system message",
        },
        "provider-invalid",
    ]
    _, turn_context = install_fake_hermes(
        monkeypatch,
        lambda messages, *, system_prompt="", tools=None: 1,  # noqa: ARG005
    )
    counter = Counter(321)
    install(runtime(counter))

    assert turn_context.estimate_request_tokens_rough(messages) == pressure(321)
    assert counter.requests == [
        {
            "model": "qwen-test",
            "messages": [
                {"role": "user", "content": "provider-visible context"},
                {"role": "system", "content": "policy"},
                "provider-invalid",
            ],
        },
    ]
    assert messages[0]["content"] == "clean"
    assert messages[0]["api_content"] == "provider-visible context"


def test_install_forces_exact_preflight_before_message_only_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run exact preflight for short histories with large hidden prompt parts.

    Hermes' cheap gate estimates only message content and can return false
    while a system prompt or tool schemas already cross the compression
    boundary.  If incontext leaves that gate unchanged, its exact estimator is
    never called and later middleware can only fail open after compression was
    skipped.  An active matching runtime must force the authoritative estimate
    before request construction.
    """

    def rough(messages: Any, *, system_prompt: str = "", tools: Any = None) -> int:
        del messages, system_prompt, tools
        return 7

    _, turn_context = install_fake_hermes(monkeypatch, rough)
    counter = Counter(60_000)
    cleanup = install(runtime(counter))
    assert callable(cleanup)

    assert not turn_context._should_run_preflight_estimate.original([], 3, 20, 64_000)  # type: ignore[attr-defined]
    assert turn_context._should_run_preflight_estimate([], 3, 20, 64_000)  # type: ignore[attr-defined]
    assert turn_context.estimate_request_tokens_rough(
        [],
        system_prompt="large policy",
        tools=({"type": "function"},),
    ) == pressure(60_000)
    assert counter.requests == [
        {
            "model": "qwen-test",
            "messages": [{"role": "system", "content": "large policy"}],
            "tools": [{"type": "function"}],
        },
    ]


def test_preflight_uses_rough_estimate_after_live_model_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never reuse the startup tokenizer after Hermes switches live route.

    Hermes' ``/model`` path changes model, provider, endpoint and compressor
    in-place without rediscovering plugins.  Public and auxiliary route guards
    already abstain, so preflight must likewise use Hermes' native estimator
    rather than render the new conversation with the stale startup template.
    """

    def rough(messages: Any, *, system_prompt: str = "", tools: Any = None) -> int:
        del messages, system_prompt, tools
        return 17

    loop, turn_context = install_fake_hermes(monkeypatch, rough)
    auxiliary = types.ModuleType("agent.auxiliary_client")
    live_route = {
        "provider": "custom",
        "model": "switched-model",
        "base_url": "https://switched.invalid/v1",
    }
    auxiliary._runtime_main_value = live_route.get  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary)
    counter = Counter(999)
    install(runtime(counter))

    assert not turn_context._should_run_preflight_estimate([], 3, 20, 64_000)  # type: ignore[attr-defined]
    assert loop.estimate_request_tokens_rough([]) == 17
    assert counter.requests == []


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_pressure", "requires_compression"),
    [
        (64_000 - MINIMUM_OUTPUT_TOKENS, 63_999, False),
        (64_000 - MINIMUM_OUTPUT_TOKENS + 1, 64_000, True),
    ],
)
def test_preflight_pressure_matches_the_viable_output_boundary(
    monkeypatch: pytest.MonkeyPatch,
    prompt_tokens: int,
    expected_pressure: int,
    requires_compression: bool,
) -> None:
    _, turn_context = install_fake_hermes(
        monkeypatch,
        lambda messages, *, system_prompt="", tools=None: 1,  # noqa: ARG005
    )
    install(runtime(Counter(prompt_tokens)))

    result = turn_context.estimate_request_tokens_rough([])

    assert result == expected_pressure
    assert (result >= 64_000) is requires_compression


@pytest.mark.parametrize("minimum_output_tokens", [1, 17, 99])
@pytest.mark.parametrize("shortfall", [0, 1])
def test_configured_output_reserve_drives_preflight_and_middleware_together(
    monkeypatch: pytest.MonkeyPatch,
    minimum_output_tokens: int,
    shortfall: int,
) -> None:
    """Keep arbitrary configured reserves wired through both decision sites.

    Pure arithmetic tests cannot detect one integration accidentally hardcoding
    the default 4096-token reserve.  Exercise the smallest legal reserve, a
    custom ordinary value, and the largest reserve below this test window.
    """
    window = 100
    prompt_tokens = window - minimum_output_tokens + shortfall
    configured = Settings(
        model_name="qwen-test",
        context_length=100,
        compression_window=window,
        fallback_margin_tokens=0,
        min_output_tokens=minimum_output_tokens,
        provider="",
        base_url="",
    )
    counter = Counter(prompt_tokens)
    active = DynamicOutputBudget(configured, counter)
    _, turn_context = install_fake_hermes(
        monkeypatch,
        lambda messages, *, system_prompt="", tools=None: 1,  # noqa: ARG005
    )
    install(active)

    pressure_result = turn_context.estimate_request_tokens_rough([])
    budget_result = active(request={"model": "qwen-test", "messages": []})

    assert pressure_result == window - 1 + shortfall
    if shortfall:
        assert budget_result is None
    else:
        assert budget_result is not None
        assert budget_result["request"]["max_tokens"] == minimum_output_tokens


@pytest.mark.parametrize(
    ("minimum_output_tokens", "fallback_margin_tokens"),
    [(1, 0), (17, 7), (80, 19)],
)
@pytest.mark.parametrize("shortfall", [0, 1])
def test_fallback_preflight_and_middleware_share_the_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
    minimum_output_tokens: int,
    fallback_margin_tokens: int,
    shortfall: int,
) -> None:
    """Prove ``W - P - F == R`` is viable and one token less compresses."""
    window = 100
    rough_prompt_tokens = (
        window - fallback_margin_tokens - minimum_output_tokens + shortfall
    )

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        return rough_prompt_tokens

    configured = Settings(
        model_name="qwen-test",
        context_length=100,
        compression_window=window,
        fallback_margin_tokens=fallback_margin_tokens,
        min_output_tokens=minimum_output_tokens,
        provider="",
        base_url="",
    )
    active = DynamicOutputBudget(
        configured,
        Counter(TimeoutError("exact tokenizer unavailable")),
        rough_estimator=lambda _request: rough_prompt_tokens,
    )
    _, turn_context = install_fake_hermes(monkeypatch, rough)
    install(active)

    pressure_result = turn_context.estimate_request_tokens_rough([])
    budget_result = active(request={"model": "qwen-test", "messages": []})

    assert pressure_result == window - 1 + shortfall
    if shortfall:
        assert budget_result is None
    else:
        assert budget_result is not None
        assert budget_result["request"]["max_tokens"] == minimum_output_tokens


def test_live_route_reader_supports_legacy_hermes_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read turn-local route mirrors from Hermes releases before ContextVar API.

    Hermes v2026.7 stores the primary route in three module attributes, while
    newer releases expose a private ContextVar reader.  Supporting both keeps
    live-switch protection active across the package's declared compatibility
    range instead of silently trusting a stale tokenizer on the older release.
    """
    auxiliary = types.ModuleType("agent.auxiliary_client")
    auxiliary._RUNTIME_MAIN_PROVIDER = "custom"  # type: ignore[attr-defined]
    auxiliary._RUNTIME_MAIN_MODEL = "qwen-test"  # type: ignore[attr-defined]
    auxiliary._RUNTIME_MAIN_BASE_URL = "https://primary.invalid/v1"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary)

    assert preflight._live_main_route() == (
        "custom",
        "qwen-test",
        "https://primary.invalid/v1",
    )


def test_live_route_reader_marks_a_broken_private_api_as_unmatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail to Hermes' rough estimator when private route lookup breaks."""
    auxiliary = types.ModuleType("agent.auxiliary_client")

    def fail(field: str) -> str:
        del field
        raise RuntimeError("private API changed")

    auxiliary._runtime_main_value = fail  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary)

    assert preflight._live_main_route() == ("", "\0route-unavailable", "")


@pytest.mark.parametrize(
    ("provider", "base_url", "expected"),
    [
        ("openai", "https://primary.invalid/v1", False),
        ("custom", "https://fallback.invalid/v1", False),
        (" Custom ", "https://PRIMARY.INVALID:443/v1/", True),
    ],
)
def test_preflight_route_guard_checks_provider_and_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    base_url: str,
    expected: bool,
) -> None:
    """Accept only the configured route after harmless normalization.

    A fallback can expose the same model alias through another provider or URL.
    Model equality alone does not prove tokenizer/template compatibility, so
    exact preflight must require all configured route dimensions to match.  DNS
    case, a default HTTPS port, and a trailing slash remain the same endpoint.
    """
    auxiliary = types.ModuleType("agent.auxiliary_client")
    live_route = {
        "provider": provider,
        "model": "qwen-test",
        "base_url": base_url,
    }
    auxiliary._runtime_main_value = live_route.get  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary)
    scoped = DynamicOutputBudget(
        Settings(
            model_name="qwen-test",
            context_length=65_536,
            compression_window=64_000,
            fallback_margin_tokens=1024,
            min_output_tokens=4096,
            provider="custom",
            base_url="https://primary.invalid/v1",
        ),
        Counter(1),
    )

    assert preflight._matches_live_route(scoped) is expected


def test_exact_gate_resolves_profile_runtime_at_call_time() -> None:
    """Use the active profile's resolver for every cheap-gate decision."""
    active = runtime(Counter(1))
    calls: list[None] = []

    def resolve() -> DynamicOutputBudget:
        calls.append(None)
        return active

    gate = _ExactPreflightGate(resolve, lambda *_args, **_kwargs: False)

    assert gate([], 3, 20, 64_000) is True
    assert calls == [None]


def test_preflight_preserves_empty_tool_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, turn_context = install_fake_hermes(
        monkeypatch,
        lambda messages, *, system_prompt="", tools=None: 42,  # noqa: ARG005
    )
    counter = Counter(123)
    install(runtime(counter))

    assert turn_context.estimate_request_tokens_rough([]) == pressure(123)
    assert counter.requests == [{"model": "qwen-test", "messages": []}]


def test_preflight_falls_back_when_backend_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Call Hermes' keyword-only fallback without leaking backend details.

    The real Hermes estimator declares ``system_prompt`` and ``tools`` after a
    ``*``.  A positional fallback call raises ``TypeError`` precisely when the
    tokenizer is unavailable.  The conservative fallback margin must also be
    added here so preflight compresses requests for which the middleware would
    otherwise find no positive safe budget.
    """

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        assert messages == [{"role": "user", "content": "fallback"}]
        assert system_prompt == "system fallback"
        assert tools == []
        return 321

    _, turn_context = install_fake_hermes(monkeypatch, rough)
    install(runtime(Counter(TimeoutError("secret"))))

    assert turn_context.estimate_request_tokens_rough(
        [{"role": "user", "content": "fallback"}],
        system_prompt="system fallback",
        tools=[],
    ) == pressure(321 + 1024)
    assert "TimeoutError" in caplog.text
    assert "secret" not in caplog.text


def test_preflight_falls_back_for_unsupported_message_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        assert messages == "not-a-list"
        assert system_prompt == ""
        assert tools == "also-not-a-list"
        return 7

    _, turn_context = install_fake_hermes(monkeypatch, rough)
    install(runtime(Counter(1)))
    assert (
        turn_context.estimate_request_tokens_rough(
            "not-a-list",
            tools="also-not-a-list",
        )
        == 7
    )


def test_preflight_fails_open_for_additive_hermes_estimator_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward unknown private-estimator context to Hermes' rough fallback.

    A later Hermes release may add keyword-only prompt buckets such as
    ``documents`` to its private estimator.  The reduced exact request cannot
    account for an unknown bucket safely, so the wrapper must accept and pass
    the complete call to the original estimator instead of raising before the
    request path can proceed.
    """
    calls: list[tuple[Any, str, Any, Any]] = []

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
        documents: Any = None,
    ) -> int:
        calls.append((messages, system_prompt, tools, documents))
        return 17

    loop, turn_context = install_fake_hermes(monkeypatch, rough)
    counter = Counter(123)
    install(runtime(counter))
    arguments = {
        "system_prompt": "policy",
        "tools": [],
        "documents": [{"text": "context"}],
    }

    assert loop.estimate_request_tokens_rough([], **arguments) == 17
    assert turn_context.estimate_request_tokens_rough([], **arguments) == 17
    assert calls == [
        ([], "policy", [], [{"text": "context"}]),
        ([], "policy", [], [{"text": "context"}]),
    ]
    assert counter.requests == []


def test_install_is_idempotent_and_retains_the_initial_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        return 5

    _, turn_context = install_fake_hermes(monkeypatch, rough)
    first = Counter(100)
    second = Counter(200)
    first_cleanup = install(runtime(first))
    second_cleanup = install(runtime(second))
    assert callable(first_cleanup)
    assert callable(second_cleanup)
    assert turn_context.estimate_request_tokens_rough([]) == pressure(200)
    assert first.requests == []
    assert second.requests == [{"model": "qwen-test", "messages": []}]

    second_cleanup()
    assert turn_context.estimate_request_tokens_rough([]) == pressure(100)
    assert first.requests == [{"model": "qwen-test", "messages": []}]
    assert second.requests == [{"model": "qwen-test", "messages": []}]
    first_cleanup()


def test_cleanup_restores_all_preflight_bindings_after_final_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference-count profile owners and restore the proactive binding.

    With two active profile managers, the first unload must leave the shared
    proactive wrapper intact; the final unload restores the original and a
    duplicate callback remains a no-op.
    """

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        return 5

    _, turn_context = install_fake_hermes(monkeypatch, rough)
    original_gate = turn_context.__dict__["_should_run_preflight_estimate"]
    first_cleanup = install(runtime(Counter(100)))
    second_cleanup = install(runtime(Counter(200)))
    assert callable(first_cleanup)
    assert callable(second_cleanup)
    turn_wrapper = turn_context.estimate_request_tokens_rough
    gate_wrapper = turn_context.__dict__["_should_run_preflight_estimate"]

    first_cleanup()
    assert turn_context.estimate_request_tokens_rough is turn_wrapper
    assert turn_context.__dict__["_should_run_preflight_estimate"] is gate_wrapper

    second_cleanup()
    second_cleanup()
    assert turn_context.estimate_request_tokens_rough is rough
    assert turn_context.__dict__["_should_run_preflight_estimate"] is original_gate


def test_install_snapshot_is_serialized_with_final_owner_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never wrap an ownerless estimator released by a concurrent unload.

    A new profile can begin preflight installation while the previous plugin
    manager disposes its unload ledger on another thread.  If installation
    snapshots the module bindings before taking ``_install_lock``, final-owner
    cleanup can restore Hermes' rough estimator before the installer consumes
    that stale snapshot.  Backend failure then traverses two exact wrappers,
    adds the fallback margin twice, and restores the stale wrapper on cleanup.
    Binding acquisition must therefore share cleanup's critical section.
    """

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        return 5

    loop, turn_context = install_fake_hermes(monkeypatch, rough)
    active = runtime(Counter(OSError("tokenizer unavailable")))

    def resolve() -> DynamicOutputBudget:
        return active

    old_cleanup = install(resolve)
    assert callable(old_cleanup)

    snapshot_requested = threading.Event()
    old_cleaned = threading.Event()
    mutex = threading.Lock()

    class OrderedLock:
        def __enter__(self) -> None:
            if threading.current_thread().name == "new-install":
                snapshot_requested.set()
                assert old_cleaned.wait(timeout=2)
            mutex.acquire()

        def __exit__(self, *exc: Any) -> None:
            mutex.release()
            if threading.current_thread().name == "old-cleanup":
                old_cleaned.set()

    monkeypatch.setattr(preflight, "_install_lock", OrderedLock())
    new_cleanups: list[Any] = []

    installer = threading.Thread(
        target=lambda: new_cleanups.append(install(resolve)),
        name="new-install",
    )
    installer.start()
    assert snapshot_requested.wait(timeout=2)

    unloader = threading.Thread(target=old_cleanup, name="old-cleanup")
    unloader.start()
    unloader.join(timeout=2)
    installer.join(timeout=2)

    assert not unloader.is_alive()
    assert not installer.is_alive()
    assert len(new_cleanups) == 1
    new_cleanup = new_cleanups[0]
    assert callable(new_cleanup)
    assert turn_context.estimate_request_tokens_rough([]) == pressure(5 + 1024)

    new_cleanup()
    assert turn_context.estimate_request_tokens_rough is rough
    assert loop.estimate_request_tokens_rough is rough


def test_preflight_runtime_resolver_tracks_the_active_profile() -> None:
    """Obtain the profile-specific backend for every preflight estimate.

    Hermes keeps the imported proactive estimator process-wide while its active
    home is ContextVar-scoped.  Resolving lazily ensures the wrapper uses
    the runtime belonging to the profile that initiated this particular turn.
    """
    active = runtime(Counter(456))
    calls: list[None] = []

    def resolve() -> DynamicOutputBudget:
        calls.append(None)
        return active

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        return 5

    wrapper = _ExactPreflight(resolve, rough)

    assert wrapper([]) == pressure(456)
    assert calls == [None]


def test_preflight_skips_profiles_without_an_active_plugin_owner() -> None:
    """Use Hermes' estimator when the active profile did not load incontext.

    The patched estimator binding is process-wide, but Hermes plugin managers
    are profile-scoped.  Returning no runtime for the current home must avoid
    both exact tokenization and any cross-profile prompt disclosure while a
    different profile keeps the shared wrapper installed.
    """
    counter = Counter(999)

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        assert messages == [{"role": "user", "content": "private"}]
        assert system_prompt == "profile system"
        assert tools == [{"type": "function"}]
        return 17

    wrapper = _ExactPreflight(lambda: None, rough)

    assert (
        wrapper(
            [{"role": "user", "content": "private"}],
            system_prompt="profile system",
            tools=[{"type": "function"}],
        )
        == 17
    )
    assert counter.requests == []


@pytest.mark.parametrize("is_gate", [False, True])
def test_final_owner_release_cannot_race_active_preflight(is_gate: bool) -> None:
    """Keep the final unload from invalidating an in-flight no-GIL lookup.

    The estimator and its cheap-gate wrapper are process-global, so plugin
    cleanup can release their last owner while another thread enters preflight.
    Reading either owner list twice permits cleanup to replace it with an empty
    list between truthiness and indexing on free-threaded CPython.  One snapshot
    must instead remain valid for the complete lookup in both wrappers.
    """
    checked = threading.Event()
    resume = threading.Event()

    class BlockingOwners(list):
        def __bool__(self) -> bool:
            checked.set()
            assert resume.wait(timeout=2)
            return super().__len__() != 0

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        return 7

    resolver = lambda: None  # noqa: E731
    wrapper = (
        _ExactPreflightGate(resolver, lambda *_args, **_kwargs: False)
        if is_gate
        else _ExactPreflight(resolver, rough)
    )
    owner = object()
    wrapper.acquire(owner, resolver)
    wrapper._owners = BlockingOwners(wrapper._owners)
    results: list[Any] = []
    errors: list[BaseException] = []

    def estimate() -> None:
        try:
            results.append(wrapper([]))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=estimate)
    worker.start()
    assert checked.wait(timeout=2)
    wrapper.release(owner)
    resume.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert errors == []
    assert results == [False if is_gate else 7]


def test_cleanup_never_overwrites_later_preflight_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve foreign bindings without chaining a restored stale wrapper.

    Cleanup owns only the exact wrapper object it installed.  If another plugin
    replaces that Hermes binding later, unloading incontext must leave the
    foreign callable intact.  That plugin can subsequently restore the wrapper
    it originally found; reinstalling incontext must unwrap the now-ownerless
    estimator and gate so fallback and final cleanup reach Hermes' originals.
    """

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        return 5

    loop, turn_context = install_fake_hermes(monkeypatch, rough)
    original_gate = turn_context.__dict__["_should_run_preflight_estimate"]
    cleanup = install(runtime(Counter(100)))
    assert callable(cleanup)
    stale_wrapper = turn_context.estimate_request_tokens_rough
    stale_gate = turn_context.__dict__["_should_run_preflight_estimate"]

    def replacement(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        return 99

    def replacement_gate(*args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return False

    turn_context.estimate_request_tokens_rough = replacement  # type: ignore[attr-defined]
    turn_context.__dict__["_should_run_preflight_estimate"] = replacement_gate
    cleanup()

    assert loop.estimate_request_tokens_rough is rough
    assert turn_context.estimate_request_tokens_rough is replacement
    assert turn_context.__dict__["_should_run_preflight_estimate"] is replacement_gate

    turn_context.estimate_request_tokens_rough = stale_wrapper  # type: ignore[attr-defined]
    turn_context.__dict__["_should_run_preflight_estimate"] = stale_gate
    second_cleanup = install(runtime(Counter(TimeoutError("offline"))))
    assert callable(second_cleanup)
    assert turn_context.estimate_request_tokens_rough([]) == pressure(5 + 1024)

    second_cleanup()
    assert turn_context.estimate_request_tokens_rough is rough
    assert turn_context.__dict__["_should_run_preflight_estimate"] is original_gate


def test_stale_preflight_cleanup_cannot_remove_new_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore cleanup from a manager superseded by force rediscovery.

    A new module binding can become active before an older profile manager runs
    its unload ledger.  The old callback must recognize that its wrapper is
    stale and leave the replacement installation untouched.
    """

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        return 5

    install_fake_hermes(monkeypatch, rough)
    stale_cleanup = install(runtime(Counter(100)))
    assert callable(stale_cleanup)

    _, second_turn_context = install_fake_hermes(monkeypatch, rough)
    active_cleanup = install(runtime(Counter(200)))
    assert callable(active_cleanup)
    active_wrapper = second_turn_context.estimate_request_tokens_rough

    stale_cleanup()
    assert second_turn_context.estimate_request_tokens_rough is active_wrapper


def test_install_reports_absent_hermes(caplog: pytest.LogCaptureFixture) -> None:
    original = sys.modules.pop("agent", None)
    conversation_loop = sys.modules.pop("agent.conversation_loop", None)
    turn_context = sys.modules.pop("agent.turn_context", None)
    try:
        assert install(runtime(Counter(1))) is None
    finally:
        if original is not None:
            sys.modules["agent"] = original
        if conversation_loop is not None:
            sys.modules["agent.conversation_loop"] = conversation_loop
        if turn_context is not None:
            sys.modules["agent.turn_context"] = turn_context
    assert "Hermes is not installed" in caplog.text


@pytest.mark.parametrize(
    ("missing_binding", "remaining_binding"),
    [
        ("estimate_request_tokens_rough", "_should_run_preflight_estimate"),
        ("_should_run_preflight_estimate", "estimate_request_tokens_rough"),
    ],
)
def test_install_leaves_no_wrapper_when_a_private_binding_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    missing_binding: str,
    remaining_binding: str,
) -> None:
    """Leave Hermes unchanged when its private preflight API has moved.

    A release may remove or rename the proactive estimator before incontext is
    updated.  Registration must fail open before changing either binding or
    acquiring an owner because it cannot clean up a partial installation.
    """

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        return 5

    _, turn_context = install_fake_hermes(monkeypatch, rough)
    original_remaining = turn_context.__dict__[remaining_binding]
    del turn_context.__dict__[missing_binding]

    assert install(runtime(Counter(100))) is None
    assert missing_binding not in turn_context.__dict__
    assert turn_context.__dict__[remaining_binding] is original_remaining
    assert "estimator API changed" in caplog.text
