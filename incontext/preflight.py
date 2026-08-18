"""Exact-token preflight bridge for Hermes' compression decision."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Callable, cast

from .budget import DynamicOutputBudget

LOGGER = logging.getLogger(__name__)
RoughEstimator = Callable[..., int]


class _ExactPreflight:
    """Callable wrapper that retains the unmodified Hermes estimator."""

    def __init__(
        self,
        runtime: DynamicOutputBudget,
        original: RoughEstimator,
    ) -> None:
        self.runtime = runtime
        self.original = original

    def __call__(
        self,
        messages: Any,
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> int:
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
            "model": self.runtime.settings.model_name,
            "messages": provider_messages,
        }
        if isinstance(tools, list) and tools:
            request["tools"] = tools
        try:
            return self.runtime.backend.count(
                request,
                context_length=self.runtime.settings.context_length,
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
            return rough_tokens + self.runtime.settings.fallback_margin_tokens


def install(runtime: DynamicOutputBudget) -> RoughEstimator | None:
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

    original: RoughEstimator | None = None
    for module in (turn_context, conversation_loop):
        current = module.__dict__["estimate_request_tokens_rough"]
        module_original = (
            current.original
            if isinstance(current, _ExactPreflight)
            else cast(RoughEstimator, current)
        )
        if original is None:
            original = module_original
        module.__dict__["estimate_request_tokens_rough"] = _ExactPreflight(
            runtime,
            module_original,
        )
    return original
