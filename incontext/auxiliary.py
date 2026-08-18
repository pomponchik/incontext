"""Dynamic output budgeting for Hermes auxiliary LLM requests."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Callable, Dict, cast

from .budget import DynamicOutputBudget

LOGGER = logging.getLogger(__name__)
AuxiliaryBuilder = Callable[..., Dict[str, Any]]


class _AuxiliaryBudget:
    """Wrap Hermes' request builder without replacing its provider logic."""

    def __init__(
        self,
        runtime: DynamicOutputBudget,
        original: AuxiliaryBuilder,
    ) -> None:
        self.runtime = runtime
        self.original = original

    def __call__(  # noqa: PLR0913
        self,
        provider: str,
        model: str,
        messages: list[Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[Any] | None = None,
        timeout: float = 30.0,
        extra_body: dict[str, Any] | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        original_request = self.original(
            provider,
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            timeout=timeout,
            extra_body=extra_body,
            base_url=base_url,
        )
        request = original_request
        if max_tokens is not None:
            # Hermes deliberately omits this field for most auxiliary custom
            # providers. Restore the caller's bound before incontext chooses
            # the smaller of it and the exact free compression-window space.
            request = {**request, "max_tokens": max_tokens}
        result = self.runtime(request=request)
        return original_request if result is None else result["request"]


def install(runtime: DynamicOutputBudget) -> AuxiliaryBuilder | None:
    """Apply incontext to Hermes requests that bypass ``llm_request`` middleware."""

    try:
        auxiliary_client = import_module("agent.auxiliary_client")
    except ImportError:
        LOGGER.warning(
            "incontext auxiliary budgeting unavailable: Hermes is not installed"
        )
        return None

    current = auxiliary_client.__dict__["_build_call_kwargs"]
    original = (
        current.original
        if isinstance(current, _AuxiliaryBudget)
        else cast(AuxiliaryBuilder, current)
    )
    auxiliary_client.__dict__["_build_call_kwargs"] = _AuxiliaryBudget(
        runtime,
        original,
    )
    return original
