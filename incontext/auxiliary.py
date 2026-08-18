"""Dynamic output budgeting for Hermes auxiliary LLM requests."""

from __future__ import annotations

import logging
import threading
from importlib import import_module
from inspect import Parameter, Signature, signature
from typing import Any, Callable, Dict, Optional, Union, cast

from .budget import DynamicOutputBudget

LOGGER = logging.getLogger(__name__)
AuxiliaryBuilder = Callable[..., Dict[str, Any]]
RuntimeSource = Union[
    DynamicOutputBudget,
    Callable[[], Optional[DynamicOutputBudget]],
]
Cleanup = Callable[[], None]


class _AuxiliaryBudget:
    """Wrap Hermes' request builder without replacing its provider logic."""

    def __init__(
        self,
        runtime: RuntimeSource,
        original: AuxiliaryBuilder,
    ) -> None:
        self.runtime_source = runtime
        self.original = original
        self.signature: Signature = signature(original)

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        original_request = self.original(*args, **kwargs)
        runtime = self._runtime()
        if runtime is None:
            return original_request
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        model = self._argument(bound.arguments, "model")
        if model != runtime.settings.model_name:
            return original_request
        provider = self._argument(bound.arguments, "provider")
        if runtime.settings.provider and provider != runtime.settings.provider:
            return original_request
        base_url = self._argument(bound.arguments, "base_url")
        if runtime.settings.base_url and (
            not isinstance(base_url, str)
            or base_url.strip().rstrip("/") != runtime.settings.base_url
        ):
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
        result = runtime(request=request)
        return original_request if result is None else result["request"]

    def _runtime(self) -> Optional[DynamicOutputBudget]:  # noqa: UP045
        if isinstance(self.runtime_source, DynamicOutputBudget):
            return self.runtime_source
        return self.runtime_source()

    def _argument(self, arguments: dict[str, Any], name: str) -> Any:
        if name in arguments:
            return arguments[name]
        for parameter in self.signature.parameters.values():
            if parameter.kind is Parameter.VAR_KEYWORD:
                extra = arguments.get(parameter.name)
                if isinstance(extra, dict) and name in extra:
                    return extra[name]
        return None


_install_lock = threading.Lock()
_installed_wrapper: _AuxiliaryBudget | None = None
_install_count = 0


def install(runtime: RuntimeSource) -> Cleanup | None:
    """Apply incontext to Hermes requests that bypass ``llm_request`` middleware."""

    try:
        auxiliary_client = import_module("agent.auxiliary_client")
    except ImportError:
        LOGGER.warning(
            "incontext auxiliary budgeting unavailable: Hermes is not installed"
        )
        return None

    global _install_count, _installed_wrapper  # noqa: PLW0603
    with _install_lock:
        current = auxiliary_client.__dict__["_build_call_kwargs"]
        if current is _installed_wrapper:
            wrapper = _installed_wrapper
            assert wrapper is not None
            wrapper.runtime_source = runtime
            _install_count += 1
        else:
            original = (
                current.original
                if isinstance(current, _AuxiliaryBudget)
                else cast(AuxiliaryBuilder, current)
            )
            wrapper = _AuxiliaryBudget(runtime, original)
            auxiliary_client.__dict__["_build_call_kwargs"] = wrapper
            _installed_wrapper = wrapper
            _install_count = 1

    closed = False

    def cleanup() -> None:
        """Release one owner and restore Hermes after the final unload."""

        nonlocal closed
        global _install_count, _installed_wrapper  # noqa: PLW0603
        with _install_lock:
            if closed:
                return
            closed = True
            if _installed_wrapper is not wrapper:
                return
            _install_count -= 1
            if _install_count == 0:
                if auxiliary_client.__dict__.get("_build_call_kwargs") is wrapper:
                    auxiliary_client.__dict__["_build_call_kwargs"] = wrapper.original
                _installed_wrapper = None

    return cleanup
