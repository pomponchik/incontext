"""Hermes integration and process-wide runtime lifecycle."""

from __future__ import annotations

import logging
import threading
from typing import Any

from .backend import backends
from .budget import DynamicOutputBudget
from .preflight import install as install_exact_preflight
from .settings import Environment, load_settings

LOGGER = logging.getLogger(__name__)
_runtime: DynamicOutputBudget | None = None
_runtime_lock = threading.Lock()


def build_runtime() -> DynamicOutputBudget:
    """Construct a fully validated runtime."""

    environment = Environment()
    settings = load_settings(environment=environment)
    backend = backends[environment.backend].one()
    return DynamicOutputBudget(settings, backend)


def get_runtime() -> DynamicOutputBudget:
    """Return the process-wide runtime, constructing it exactly once."""

    global _runtime  # noqa: PLW0603
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = build_runtime()
    return _runtime


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
    install_exact_preflight(runtime)
    ctx.register_middleware("llm_request", apply_incontext)
    LOGGER.info(
        "incontext registered context=%d compression_window=%d",
        runtime.settings.context_length,
        runtime.settings.compression_window,
    )


def _reset_runtime_for_tests() -> None:
    global _runtime  # noqa: PLW0603
    with _runtime_lock:
        _runtime = None
