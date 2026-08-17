from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from incontext import plugin
from incontext.backend import Backend
from incontext.budget import DynamicOutputBudget
from incontext.settings import Environment, Settings


class Context:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def register_middleware(self, kind: str, callback: Any) -> None:
        self.calls.append((kind, callback))


@pytest.fixture(autouse=True)
def reset_runtime() -> None:
    plugin._reset_runtime_for_tests()
    yield
    plugin._reset_runtime_for_tests()


def test_build_runtime_uses_validated_settings(runtime_settings: Settings) -> None:
    environment = mock.create_autospec(Environment, instance=True)
    environment.backend = "selected_backend"
    backend = mock.create_autospec(Backend, instance=True)
    selection = mock.Mock()
    selection.one.return_value = backend
    backend_slot = mock.MagicMock()
    backend_slot.__getitem__.return_value = selection
    with mock.patch.object(
        plugin,
        "Environment",
        return_value=environment,
    ), mock.patch.object(
        plugin,
        "load_settings",
        return_value=runtime_settings,
    ) as settings_loader, mock.patch.object(
        plugin,
        "backends",
        backend_slot,
    ), mock.patch.object(plugin, "DynamicOutputBudget") as runtime_class:
        result = plugin.build_runtime()
    settings_loader.assert_called_once_with(environment=environment)
    backend_slot.__getitem__.assert_called_once_with("selected_backend")
    selection.one.assert_called_once_with()
    runtime_class.assert_called_once_with(runtime_settings, backend)
    assert result is runtime_class.return_value


def test_get_runtime_builds_once() -> None:
    runtime = mock.create_autospec(DynamicOutputBudget, instance=True)
    with mock.patch.object(plugin, "build_runtime", return_value=runtime) as builder:
        assert plugin.get_runtime() is runtime
        assert plugin.get_runtime() is runtime
    builder.assert_called_once_with()


def test_get_runtime_observes_value_created_while_waiting_for_lock() -> None:
    runtime = mock.create_autospec(DynamicOutputBudget, instance=True)

    class Lock:
        def __enter__(self) -> None:
            plugin._runtime = runtime

        def __exit__(self, *args: Any) -> None:
            return None

    with mock.patch.object(plugin, "_runtime_lock", Lock()), mock.patch.object(
        plugin,
        "build_runtime",
    ) as builder:
        assert plugin.get_runtime() is runtime
    builder.assert_not_called()


def test_apply_incontext_delegates_context() -> None:
    runtime = mock.Mock(return_value={"request": {"max_tokens": 1}})
    with mock.patch.object(plugin, "get_runtime", return_value=runtime):
        result = plugin.apply_incontext(
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
    with mock.patch.object(plugin, "get_runtime", return_value=runtime):
        plugin.register(context)
    assert context.calls == [("llm_request", plugin.apply_incontext)]


def test_reset_runtime_removes_cached_value() -> None:
    plugin._runtime = mock.create_autospec(DynamicOutputBudget, instance=True)
    plugin._reset_runtime_for_tests()
    assert plugin._runtime is None
