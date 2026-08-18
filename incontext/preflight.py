"""Exact-token preflight bridge for Hermes' compression decision."""

from __future__ import annotations

import logging
import threading
from importlib import import_module
from typing import Any, Callable, Union, cast

from .budget import DynamicOutputBudget

LOGGER = logging.getLogger(__name__)
RoughEstimator = Callable[..., int]
RuntimeSource = Union[DynamicOutputBudget, Callable[[], DynamicOutputBudget]]
Cleanup = Callable[[], None]


class _ExactPreflight:
    """Callable wrapper that retains the unmodified Hermes estimator."""

    def __init__(
        self,
        runtime: RuntimeSource,
        original: RoughEstimator,
    ) -> None:
        self.runtime_source = runtime
        self.original = original

    def __call__(
        self,
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        runtime = self._runtime()
        if not isinstance(messages, list):
            return int(
                self.original(
                    messages,
                    system_prompt=system_prompt,
                    tools=tools,
                ),
            )
        provider_messages = list(messages)
        if system_prompt:
            provider_messages.insert(
                0,
                {"role": "system", "content": system_prompt},
            )
        request: dict[str, Any] = {
            "model": runtime.settings.model_name,
            "messages": provider_messages,
        }
        if isinstance(tools, list) and tools:
            request["tools"] = tools
        try:
            return runtime.backend.count(
                request,
                context_length=runtime.settings.context_length,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "incontext preflight backend_failed type=%s; "
                "using Hermes rough estimate",
                type(exc).__name__,
            )
            rough_tokens = max(
                1,
                int(
                    self.original(
                        messages,
                        system_prompt=system_prompt,
                        tools=tools,
                    ),
                ),
            )
            return rough_tokens + runtime.settings.fallback_margin_tokens

    def _runtime(self) -> DynamicOutputBudget:
        if isinstance(self.runtime_source, DynamicOutputBudget):
            return self.runtime_source
        return self.runtime_source()


_install_lock = threading.Lock()
_installed_wrappers: tuple[_ExactPreflight, ...] = ()
_installed_modules: tuple[Any, ...] = ()
_install_count = 0


def _owns_bindings(modules: tuple[Any, ...]) -> bool:
    return (
        _installed_modules == modules
        and bool(_installed_wrappers)
        and all(
            module.__dict__.get("estimate_request_tokens_rough") is wrapper
            for module, wrapper in zip(_installed_modules, _installed_wrappers)
        )
    )


def _replace_bindings(
    modules: tuple[Any, ...],
    runtime: RuntimeSource,
) -> tuple[_ExactPreflight, ...]:
    created: list[_ExactPreflight] = []
    for module in modules:
        current = module.__dict__["estimate_request_tokens_rough"]
        original = (
            current.original
            if isinstance(current, _ExactPreflight)
            else cast(RoughEstimator, current)
        )
        wrapper = _ExactPreflight(runtime, original)
        module.__dict__["estimate_request_tokens_rough"] = wrapper
        created.append(wrapper)
    return tuple(created)


def install(runtime: RuntimeSource) -> Cleanup | None:
    """Make Hermes compress from the selected backend's exact token count.

    Hermes normally decides whether to compress just before it creates the
    provider kwargs.  ``llm_request`` middleware runs afterwards, which is too
    late to prevent an already-full request.  Replacing only this estimator
    keeps Hermes' compressor and all of its recovery behaviour intact while
    making its threshold decision use the same counter as the budgeter.

    The original estimator remains the deliberate fail-open fallback: a
    temporary tokenizer outage must not prevent Hermes from making requests.
    """

    try:
        turn_context = import_module("agent.turn_context")
        conversation_loop = import_module("agent.conversation_loop")
    except ImportError:
        LOGGER.warning("incontext exact preflight unavailable: Hermes is not installed")
        return None

    modules = (turn_context, conversation_loop)
    global _install_count, _installed_modules, _installed_wrappers  # noqa: PLW0603
    with _install_lock:
        if _owns_bindings(modules):
            wrappers = _installed_wrappers
            for wrapper in wrappers:
                wrapper.runtime_source = runtime
            _install_count += 1
        else:
            wrappers = _replace_bindings(modules, runtime)
            _installed_modules = modules
            _installed_wrappers = wrappers
            _install_count = 1

    closed = False

    def cleanup() -> None:
        """Release one owner and restore every unchanged Hermes binding."""

        nonlocal closed
        global _install_count, _installed_modules, _installed_wrappers  # noqa: PLW0603
        with _install_lock:
            if closed:
                return
            closed = True
            if wrappers != _installed_wrappers:
                return
            _install_count -= 1
            if _install_count == 0:
                for module, wrapper in zip(modules, wrappers):
                    if module.__dict__.get("estimate_request_tokens_rough") is wrapper:
                        module.__dict__["estimate_request_tokens_rough"] = (
                            wrapper.original
                        )
                _installed_modules = ()
                _installed_wrappers = ()

    return cleanup
