from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from incontext.backend import Backend
from incontext.budget import DynamicOutputBudget
from incontext.preflight import _ExactPreflight, install
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
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.conversation_loop", loop)
    monkeypatch.setitem(sys.modules, "agent.turn_context", turn_context)
    return loop, turn_context


def test_install_replaces_rough_preflight_with_exact_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch every Hermes binding and count the complete provider prompt.

    Hermes imports the rough estimator into both ``turn_context`` (the actual
    proactive compression gate) and ``conversation_loop`` (later recovery and
    accounting paths).  The wrapper must replace both copies and include the
    separately supplied system prompt in the exact vLLM chat payload.
    """

    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        raise AssertionError("exact backend should be used")

    loop, turn_context = install_fake_hermes(monkeypatch, rough)
    counter = Counter(64_000)
    cleanup = install(runtime(counter))

    assert callable(cleanup)
    for module in (loop, turn_context):
        assert (
            module.estimate_request_tokens_rough(
                [{"role": "user", "content": "large"}],
                system_prompt="Follow the policy",
                tools=[{"type": "function"}],
            )
            == 64_000
        )
    assert counter.requests == [
        {
            "model": "qwen-test",
            "messages": [
                {"role": "system", "content": "Follow the policy"},
                {"role": "user", "content": "large"},
            ],
            "tools": [{"type": "function"}],
        },
        {
            "model": "qwen-test",
            "messages": [
                {"role": "system", "content": "Follow the policy"},
                {"role": "user", "content": "large"},
            ],
            "tools": [{"type": "function"}],
        },
    ]


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

    assert (
        turn_context.estimate_request_tokens_rough(
            [{"role": "user", "content": "large"}],
            tools=[{"type": "function"}],
        )
        == 123
    )


def test_preflight_preserves_empty_tool_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _ = install_fake_hermes(
        monkeypatch,
        lambda messages, *, system_prompt="", tools=None: 42,
    )
    counter = Counter(123)
    install(runtime(counter))

    assert loop.estimate_request_tokens_rough([]) == 123
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

    loop, _ = install_fake_hermes(monkeypatch, rough)
    install(runtime(Counter(TimeoutError("secret"))))

    assert (
        loop.estimate_request_tokens_rough(
            [{"role": "user", "content": "fallback"}],
            system_prompt="system fallback",
            tools=[],
        )
        == 321 + 1024
    )
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

    loop, _ = install_fake_hermes(monkeypatch, rough)
    install(runtime(Counter(1)))
    assert (
        loop.estimate_request_tokens_rough(
            "not-a-list",
            tools="also-not-a-list",
        )
        == 7
    )


def test_install_is_idempotent_and_retains_the_initial_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rough(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        return 5

    loop, _ = install_fake_hermes(monkeypatch, rough)
    first = Counter(100)
    second = Counter(200)
    first_cleanup = install(runtime(first))
    second_cleanup = install(runtime(second))
    assert callable(first_cleanup)
    assert callable(second_cleanup)
    assert loop.estimate_request_tokens_rough([]) == 200
    assert first.requests == []
    assert second.requests == [{"model": "qwen-test", "messages": []}]


def test_cleanup_restores_all_preflight_bindings_after_final_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference-count profile owners and restore only unchanged bindings.

    Both Hermes modules hold independent imported estimator names.  With two
    active profile managers, the first unload must leave both wrappers intact;
    the final unload restores each original and a duplicate callback is a no-op.
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
    first_cleanup = install(runtime(Counter(100)))
    second_cleanup = install(runtime(Counter(200)))
    assert callable(first_cleanup)
    assert callable(second_cleanup)
    loop_wrapper = loop.estimate_request_tokens_rough
    turn_wrapper = turn_context.estimate_request_tokens_rough

    first_cleanup()
    assert loop.estimate_request_tokens_rough is loop_wrapper
    assert turn_context.estimate_request_tokens_rough is turn_wrapper

    second_cleanup()
    second_cleanup()
    assert loop.estimate_request_tokens_rough is rough
    assert turn_context.estimate_request_tokens_rough is rough


def test_preflight_runtime_resolver_tracks_the_active_profile() -> None:
    """Obtain the profile-specific backend for every preflight estimate.

    Hermes keeps one pair of imported estimator bindings process-wide while its
    active home is ContextVar-scoped.  Resolving lazily ensures the wrapper uses
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

    assert wrapper([]) == 456
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


def test_cleanup_never_overwrites_later_preflight_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve another plugin's estimator replacement during unload.

    Cleanup owns only the exact wrapper objects it installed.  If either Hermes
    module receives a later replacement, unloading incontext must leave that
    callable intact while restoring only bindings that still belong to it.
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
    cleanup = install(runtime(Counter(100)))
    assert callable(cleanup)

    def replacement(
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        del messages, system_prompt, tools
        return 99

    turn_context.estimate_request_tokens_rough = replacement  # type: ignore[attr-defined]
    cleanup()

    assert loop.estimate_request_tokens_rough is rough
    assert turn_context.estimate_request_tokens_rough is replacement


def test_stale_preflight_cleanup_cannot_remove_new_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore cleanup from a manager superseded by force rediscovery.

    A new pair of module bindings can become active before an older profile
    manager runs its unload ledger.  The old callback must recognize that its
    wrapper tuple is stale and leave the replacement installation untouched.
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

    second_loop, _ = install_fake_hermes(monkeypatch, rough)
    active_cleanup = install(runtime(Counter(200)))
    assert callable(active_cleanup)
    active_wrapper = second_loop.estimate_request_tokens_rough

    stale_cleanup()
    assert second_loop.estimate_request_tokens_rough is active_wrapper


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
