"""Hermes integration and process-wide runtime lifecycle."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from .auxiliary import install as install_auxiliary_budget
from .backend import backends
from .budget import DynamicOutputBudget
from .preflight import install as install_exact_preflight
from .settings import Environment, load_settings

LOGGER = logging.getLogger(__name__)
_runtimes: Dict[str, DynamicOutputBudget] = {}
_active_profiles: Dict[str, int] = {}
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


def get_active_runtime() -> Optional[DynamicOutputBudget]:
    """Return a runtime only where an active profile loaded the plugin."""

    key = _runtime_key()
    with _runtime_lock:
        if _active_profiles.get(key, 0) == 0:
            return None
        runtime = _runtimes.get(key)
        if runtime is None:
            runtime = build_runtime()
            _runtimes[key] = runtime
        return runtime


def _activate_profile(key: str) -> None:
    """Record one plugin-manager owner for a Hermes profile."""

    with _runtime_lock:
        _active_profiles[key] = _active_profiles.get(key, 0) + 1


def _deactivate_profile(key: str) -> None:
    """Release a profile owner and invalidate its runtime after the last one."""

    with _runtime_lock:
        owners = _active_profiles.get(key, 0)
        if owners <= 1:
            _active_profiles.pop(key, None)
            _runtimes.pop(key, None)
        else:
            _active_profiles[key] = owners - 1


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
    request: Dict[str, Any],
    **context: Any,
) -> Optional[Dict[str, Any]]:
    """Stable function entry point used by Hermes middleware."""

    return get_runtime()(request=request, **context)


def register(ctx: Any) -> None:
    """Register the plugin with a Hermes ``PluginContext``."""

    key = _runtime_key()
    runtime = get_runtime()
    _activate_profile(key)
    preflight_cleanup = install_exact_preflight(get_active_runtime)
    auxiliary_cleanup = install_auxiliary_budget(get_active_runtime)
    on_unload = getattr(ctx, "on_unload", None)
    if callable(on_unload):
        if preflight_cleanup is not None:
            on_unload(preflight_cleanup)
        if auxiliary_cleanup is not None:
            on_unload(auxiliary_cleanup)
        on_unload(lambda: _deactivate_profile(key))
    ctx.register_middleware("llm_request", apply_incontext)
    LOGGER.info(
        "incontext registered context=%d compression_window=%d",
        runtime.settings.context_length,
        runtime.settings.compression_window,
    )


def _reset_runtime_for_tests() -> None:
    with _runtime_lock:
        _runtimes.clear()
        _active_profiles.clear()
