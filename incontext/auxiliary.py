"""Dynamic output budgeting for Hermes auxiliary LLM requests."""

from __future__ import annotations

import logging
from importlib import import_module
from inspect import Parameter, Signature, signature
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
        self.signature: Signature = signature(original)

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        original_request = self.original(*args, **kwargs)
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        model = self._argument(bound.arguments, "model")
        if model != self.runtime.settings.model_name:
            return original_request
        max_tokens = self._argument(bound.arguments, "max_tokens")
        request = original_request
        has_output_cap = any(
            field in original_request
            for field in ("max_tokens", "max_completion_tokens", "max_output_tokens")
        )
        if (
            not has_output_cap
            and isinstance(max_tokens, int)
            and not isinstance(max_tokens, bool)
            and max_tokens > 0
        ):
            # Hermes deliberately omits this field for most auxiliary custom
            # providers. Restore the caller's bound before incontext chooses
            # the smaller of it and the exact free compression-window space.
            request = {**request, "max_tokens": max_tokens}
        result = self.runtime(request=request)
        return original_request if result is None else result["request"]

    def _argument(self, arguments: dict[str, Any], name: str) -> Any:
        if name in arguments:
            return arguments[name]
        for parameter in self.signature.parameters.values():
            if parameter.kind is Parameter.VAR_KEYWORD:
                extra = arguments.get(parameter.name)
                if isinstance(extra, dict) and name in extra:
                    return extra[name]
        return None


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
