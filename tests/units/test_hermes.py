from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from incontext import hermes
from incontext.backend import Backend
from incontext.budget import DynamicOutputBudget
from incontext.settings import Environment, Settings


class Context:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.unload_callbacks: list[Any] = []

    def register_middleware(self, kind: str, callback: Any) -> None:
        self.calls.append((kind, callback))

    def on_unload(self, callback: Any) -> None:
        self.unload_callbacks.append(callback)


@pytest.fixture(autouse=True)
def reset_runtime() -> None:
    hermes._reset_runtime_for_tests()
    yield
    hermes._reset_runtime_for_tests()


def test_build_runtime_uses_validated_settings(runtime_settings: Settings) -> None:
    environment = mock.create_autospec(Environment, instance=True)
    environment.backend = "selected_backend"
    backend = mock.create_autospec(Backend, instance=True)
    selection = mock.Mock()
    selection.one.return_value = backend
    backend_slot = mock.MagicMock()
    backend_slot.__getitem__.return_value = selection
    with mock.patch.object(
        hermes,
        "Environment",
        return_value=environment,
    ), mock.patch.object(
        hermes,
        "load_settings",
        return_value=runtime_settings,
    ) as settings_loader, mock.patch.object(
        hermes,
        "backends",
        backend_slot,
    ), mock.patch.object(hermes, "DynamicOutputBudget") as runtime_class:
        result = hermes.build_runtime()
    settings_loader.assert_called_once_with(environment=environment)
    backend_slot.__getitem__.assert_called_once_with("selected_backend")
    selection.one.assert_called_once_with()
    runtime_class.assert_called_once_with(runtime_settings, backend)
    assert result is runtime_class.return_value


def test_get_runtime_builds_once() -> None:
    runtime = mock.create_autospec(DynamicOutputBudget, instance=True)
    with mock.patch.object(
        hermes,
        "_runtime_key",
        return_value="profile-a",
    ), mock.patch.object(hermes, "build_runtime", return_value=runtime) as builder:
        assert hermes.get_runtime() is runtime
        assert hermes.get_runtime() is runtime
    builder.assert_called_once_with()


def test_get_runtime_isolated_by_active_hermes_home() -> None:
    """Cache independent runtimes for profiles sharing one Python process.

    Hermes 2026.8 selects a profile through a ContextVar-aware home override.
    Returning to profile A must reuse A's backend, while profile B receives a
    separately constructed settings/backend pair instead of inheriting A.
    """

    first = mock.create_autospec(DynamicOutputBudget, instance=True)
    second = mock.create_autospec(DynamicOutputBudget, instance=True)
    keys = iter(["profile-a", "profile-b", "profile-a"])
    with mock.patch.object(hermes, "_runtime_key", side_effect=keys), mock.patch.object(
        hermes,
        "build_runtime",
        side_effect=[first, second],
    ) as builder:
        assert hermes.get_runtime() is first
        assert hermes.get_runtime() is second
        assert hermes.get_runtime() is first
    assert builder.call_count == 2


def test_runtime_key_uses_context_aware_hermes_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Key runtimes with Hermes' ContextVar-aware resolved home path.

    Environment variables alone do not change when the profile multiplexer
    enters another home.  Importing ``get_hermes_home`` lazily and resolving its
    path mirrors Hermes' own profile-scoped PluginManager cache.
    """

    constants = types.ModuleType("hermes_constants")
    constants.get_hermes_home = lambda: tmp_path / "nested" / ".."  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)

    assert hermes._runtime_key() == str(tmp_path.resolve())


def test_runtime_key_falls_back_without_hermes_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep package-level imports usable when optional Hermes is absent.

    Incontext can be imported for backend tests and package inspection outside a
    Hermes environment.  Failure to import ``hermes_constants`` must map to the
    single default runtime key instead of making ``get_runtime`` unimportable.
    """

    original_import = builtins.__import__

    def rejecting_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "hermes_constants":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "hermes_constants", raising=False)
    monkeypatch.setattr(builtins, "__import__", rejecting_import)

    assert hermes._runtime_key() == ""


def test_runtime_key_preserves_unresolvable_home_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a stable fallback key when path normalization itself fails.

    Custom embedders can provide path-like home objects with unavailable or
    failing filesystem resolution.  The key resolver must still isolate their
    textual home rather than abort plugin registration.
    """

    class BrokenHome:
        def expanduser(self) -> BrokenHome:
            raise OSError("unavailable")

        def __str__(self) -> str:
            return "/virtual/hermes"

    constants = types.ModuleType("hermes_constants")
    constants.get_hermes_home = lambda: BrokenHome()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)

    assert hermes._runtime_key() == "/virtual/hermes"


def test_apply_incontext_delegates_context() -> None:
    runtime = mock.Mock(return_value={"request": {"max_tokens": 1}})
    with mock.patch.object(hermes, "get_runtime", return_value=runtime):
        result = hermes.apply_incontext(
            request={"messages": []},
            session_id="session",
        )
    assert result == {"request": {"max_tokens": 1}}
    runtime.assert_called_once_with(request={"messages": []}, session_id="session")


def test_register_validates_and_registers_middleware(
    runtime_settings: Settings,
) -> None:
    runtime = mock.Mock()
    runtime.settings = runtime_settings
    context = Context()
    preflight_cleanup = mock.Mock()
    auxiliary_cleanup = mock.Mock()
    with mock.patch.object(
        hermes, "get_runtime", return_value=runtime
    ), mock.patch.object(
        hermes,
        "install_exact_preflight",
    ) as install_preflight, mock.patch.object(
        hermes,
        "install_auxiliary_budget",
        return_value=auxiliary_cleanup,
    ) as install_auxiliary:
        install_preflight.return_value = preflight_cleanup
        runtime_getter = hermes.get_runtime
        hermes.register(context)
    install_preflight.assert_called_once_with(runtime_getter)
    install_auxiliary.assert_called_once_with(runtime_getter)
    assert context.calls == [("llm_request", hermes.apply_incontext)]
    assert context.unload_callbacks == [preflight_cleanup, auxiliary_cleanup]


def test_register_tolerates_legacy_context_without_unload_hook(
    runtime_settings: Settings,
) -> None:
    """Retain compatibility with Hermes releases predating ``on_unload``.

    Version 2026.7 exposes middleware registration but no unload callback API.
    The plugin must still install and register normally there; lifecycle cleanup
    is attached only when the newer context advertises the hook.
    """

    class LegacyContext:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        def register_middleware(self, kind: str, callback: Any) -> None:
            self.calls.append((kind, callback))

    runtime = mock.Mock()
    runtime.settings = runtime_settings
    context = LegacyContext()
    with mock.patch.object(
        hermes,
        "get_runtime",
        return_value=runtime,
    ), mock.patch.object(
        hermes,
        "install_exact_preflight",
        return_value=None,
    ), mock.patch.object(
        hermes,
        "install_auxiliary_budget",
        return_value=None,
    ):
        hermes.register(context)

    assert context.calls == [("llm_request", hermes.apply_incontext)]


def test_register_skips_missing_cleanup_callbacks(
    runtime_settings: Settings,
) -> None:
    """Do not register unload entries for integrations Hermes could not import.

    Optional/private Hermes modules may be absent in a constrained embedder.
    Their installers return ``None``; the public middleware should still load
    without handing invalid callbacks to the ownership ledger.
    """

    runtime = mock.Mock()
    runtime.settings = runtime_settings
    context = Context()
    with mock.patch.object(
        hermes,
        "get_runtime",
        return_value=runtime,
    ), mock.patch.object(
        hermes,
        "install_exact_preflight",
        return_value=None,
    ), mock.patch.object(
        hermes,
        "install_auxiliary_budget",
        return_value=None,
    ):
        hermes.register(context)

    assert context.unload_callbacks == []


def test_reset_runtime_removes_cached_value() -> None:
    hermes._runtimes["profile"] = mock.create_autospec(
        DynamicOutputBudget,
        instance=True,
    )
    hermes._reset_runtime_for_tests()
    assert hermes._runtimes == {}
