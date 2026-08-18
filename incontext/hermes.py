"""Hermes integration and process-wide runtime lifecycle."""

from __future__ import annotations

import logging
import threading
from typing import Any

from .auxiliary import install as install_auxiliary_budget
from .backend import backends
from .budget import DynamicOutputBudget
from .preflight import install as install_exact_preflight
from .settings import Environment, load_settings

LOGGER = logging.getLogger(__name__)
_runtimes: dict[str, DynamicOutputBudget] = {}
_runtime_lock = threading.Lock()


def build_runtime() -> DynamicOutputBudget:
    """Construct a fully validated runtime."""

    environment = Environment()
    settings = load_settings(environment=environment)
    backend = backends[environment.backend].one()
    return DynamicOutputBudget(settings, backend)


def get_runtime() -> DynamicOutputBudget:
    """Return the runtime scoped to the active Hermes profile/home."""

    key = _runtime_key()
    with _runtime_lock:
        runtime = _runtimes.get(key)
        if runtime is None:
            runtime = build_runtime()
            _runtimes[key] = runtime
        return runtime


def _runtime_key() -> str:
    """Resolve Hermes' ContextVar-aware home without making it a dependency."""

    try:
        from hermes_constants import (  # type: ignore[import-not-found]  # noqa: PLC0415
            get_hermes_home,
        )
    except ImportError:
        return ""
    try:
        return str(get_hermes_home().expanduser().resolve())
    except Exception:  # noqa: BLE001
        return str(get_hermes_home())


def apply_incontext(
    *,
    request: dict[str, Any],
    **context: Any,
) -> dict[str, Any] | None:
    """Stable function entry point used by Hermes middleware."""

    return get_runtime()(request=request, **context)


def register(ctx: Any) -> None:
    """Register the plugin with a Hermes ``PluginContext``."""

    runtime = get_runtime()
    preflight_cleanup = install_exact_preflight(get_runtime)
    auxiliary_cleanup = install_auxiliary_budget(get_runtime)
    on_unload = getattr(ctx, "on_unload", None)
    if callable(on_unload):
        if preflight_cleanup is not None:
            on_unload(preflight_cleanup)
        if auxiliary_cleanup is not None:
            on_unload(auxiliary_cleanup)
    ctx.register_middleware("llm_request", apply_incontext)
    LOGGER.info(
        "incontext registered context=%d compression_window=%d",
        runtime.settings.context_length,
        runtime.settings.compression_window,
    )


def _reset_runtime_for_tests() -> None:
    with _runtime_lock:
        _runtimes.clear()
