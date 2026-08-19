"""Hermes integration and process-wide runtime lifecycle."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .auxiliary import install as install_auxiliary_budget
from .backend import backends
from .budget import DynamicOutputBudget
from .preflight import install as install_exact_preflight
from .settings import Environment, load_settings

LOGGER = logging.getLogger(__name__)
_runtimes: Dict[str, DynamicOutputBudget] = {}
_active_profiles: Dict[str, int] = {}
_runtime_lock = threading.Lock()
_registration_lock = threading.Lock()
_legacy_cleanups: Dict[str, Tuple[Callable[[], None], ...]] = {}


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
    if runtime is not None:
        return runtime
    candidate = build_runtime()
    with _runtime_lock:
        runtime = _runtimes.get(key)
        if runtime is None:
            runtime = candidate
            _runtimes[key] = runtime
        return runtime


def _profile_cleanup(key: str) -> Callable[[], None]:
    """Create an idempotent callback for one already-acquired profile owner."""

    closed = False
    cleanup_lock = threading.Lock()

    def cleanup() -> None:
        nonlocal closed
        with cleanup_lock:
            if closed:
                return
            closed = True
        _deactivate_profile(key)

    return cleanup


def _acquire_profile_runtime(
    key: str,
) -> Tuple[DynamicOutputBudget, Callable[[], None]]:
    """Acquire a validated runtime and its profile owner atomically."""

    with _runtime_lock:
        runtime = _runtimes.get(key)
        if runtime is not None:
            _active_profiles[key] = _active_profiles.get(key, 0) + 1
            return runtime, _profile_cleanup(key)
    candidate = build_runtime()
    with _runtime_lock:
        runtime = _runtimes.get(key)
        if runtime is None:
            runtime = candidate
            _runtimes[key] = runtime
        _active_profiles[key] = _active_profiles.get(key, 0) + 1
    return runtime, _profile_cleanup(key)


def get_active_runtime() -> Optional[DynamicOutputBudget]:
    """Return a runtime only where an active profile loaded the plugin."""

    key = _runtime_key()
    with _runtime_lock:
        if _active_profiles.get(key, 0) == 0:
            return None
        runtime = _runtimes.get(key)
    if runtime is not None:
        return runtime
    candidate = build_runtime()
    with _runtime_lock:
        if _active_profiles.get(key, 0) == 0:
            return None
        runtime = _runtimes.get(key)
        if runtime is None:
            runtime = candidate
            _runtimes[key] = runtime
        return runtime


def _activate_profile(key: str) -> Callable[[], None]:
    """Record one profile owner and return its idempotent release callback."""

    with _runtime_lock:
        _active_profiles[key] = _active_profiles.get(key, 0) + 1
    return _profile_cleanup(key)


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

    runtime = get_active_runtime()
    if runtime is None:
        return None
    return runtime(request=request, **context)


def register(ctx: Any) -> None:
    """Register the plugin with a Hermes ``PluginContext``."""

    key = _runtime_key()
    on_unload = getattr(ctx, "on_unload", None)
    with _registration_lock:
        if not callable(on_unload) and key in _legacy_cleanups:
            cleanups = _legacy_cleanups[key]
            try:
                runtime = build_runtime()
                ctx.register_middleware("llm_request", apply_incontext)
            except Exception:
                for cleanup in reversed(cleanups):
                    cleanup()
                _legacy_cleanups.pop(key, None)
                raise
            with _runtime_lock:
                _runtimes[key] = runtime
        else:
            runtime, profile_cleanup = _acquire_profile_runtime(key)
            acquired: List[Callable[[], None]] = [profile_cleanup]
            try:
                preflight_cleanup = install_exact_preflight(get_active_runtime)
                if preflight_cleanup is not None:
                    acquired.append(preflight_cleanup)
                auxiliary_cleanup = install_auxiliary_budget(get_active_runtime)
                if auxiliary_cleanup is not None:
                    acquired.append(auxiliary_cleanup)
                ctx.register_middleware("llm_request", apply_incontext)
                if callable(on_unload):
                    for cleanup in acquired[1:]:
                        on_unload(cleanup)
                    on_unload(profile_cleanup)
                else:
                    _legacy_cleanups[key] = tuple(acquired)
            except Exception:
                for cleanup in reversed(acquired):
                    cleanup()
                raise
    LOGGER.info(
        "incontext registered context=%d compression_window=%d",
        runtime.settings.context_length,
        runtime.settings.compression_window,
    )


def _reset_runtime_for_tests() -> None:
    _legacy_cleanups.clear()
    with _runtime_lock:
        _runtimes.clear()
        _active_profiles.clear()
