"""Exact-token preflight bridge for Hermes' compression decision."""

from __future__ import annotations

import logging
import threading
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from .budget import DynamicOutputBudget

LOGGER = logging.getLogger(__name__)
RoughEstimator = Callable[..., int]
RuntimeSource = Union[
    DynamicOutputBudget,
    Callable[[], Optional[DynamicOutputBudget]],
]
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
        self._owners: List[Tuple[object, RuntimeSource]] = []

    def acquire(self, owner: object, runtime: RuntimeSource) -> None:
        """Attach one installation owner and make its runtime current."""

        self._owners.append((owner, runtime))

    def release(self, owner: object) -> None:
        """Release one installation owner without disturbing the others."""

        self._owners = [entry for entry in self._owners if entry[0] is not owner]

    @property
    def owned(self) -> bool:
        """Return whether this wrapper belongs to an active installation."""

        return bool(self._owners)

    def __call__(
        self,
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
        runtime = self._runtime()
        if runtime is None or not isinstance(messages, list):
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
        request: Dict[str, Any] = {
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

    def _runtime(self) -> Optional[DynamicOutputBudget]:
        owners = self._owners
        runtime_source = owners[-1][1] if owners else self.runtime_source
        if isinstance(runtime_source, DynamicOutputBudget):
            return runtime_source
        return runtime_source()


_install_lock = threading.Lock()


def install(runtime: RuntimeSource) -> Optional[Cleanup]:
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

    owner = object()
    wrappers: List[Tuple[Any, _ExactPreflight]] = []
    with _install_lock:
        modules = (turn_context, conversation_loop)
        bindings = [
            module.__dict__.get("estimate_request_tokens_rough") for module in modules
        ]
        if not all(callable(binding) for binding in bindings):
            LOGGER.warning(
                "incontext exact preflight unavailable: Hermes estimator API changed"
            )
            return None
        for module, current in zip(modules, bindings):
            if isinstance(current, _ExactPreflight) and current.owned:
                wrapper = current
            else:
                wrapper = _ExactPreflight(runtime, cast(RoughEstimator, current))
                module.__dict__["estimate_request_tokens_rough"] = wrapper
            wrapper.acquire(owner, runtime)
            wrappers.append((module, wrapper))

    closed = False

    def cleanup() -> None:
        """Release one owner and restore every unchanged Hermes binding."""

        nonlocal closed
        with _install_lock:
            if closed:
                return
            closed = True
            for module, wrapper in wrappers:
                wrapper.release(owner)
                if (
                    not wrapper.owned
                    and module.__dict__.get("estimate_request_tokens_rough") is wrapper
                ):
                    module.__dict__["estimate_request_tokens_rough"] = wrapper.original

    return cleanup
