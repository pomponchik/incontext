"""Dynamic output budgeting for Hermes auxiliary LLM requests."""

from __future__ import annotations

import logging
import threading
from importlib import import_module
from inspect import Parameter, Signature, signature
from typing import Any, Callable, Dict, Optional, Union, cast

from .budget import OUTPUT_BUDGET_FIELDS, DynamicOutputBudget
from .settings import normalize_base_url

LOGGER = logging.getLogger(__name__)
AuxiliaryBuilder = Callable[..., Dict[str, Any]]
OutputCapSelector = Callable[..., Dict[str, Any]]
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
        output_cap_selector: Optional[OutputCapSelector] = None,
    ) -> None:
        self.runtime_source = runtime
        self.original = original
        self.output_cap_selector = output_cap_selector
        self.signature: Signature = signature(original)

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
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
        if runtime.settings.provider and (
            not isinstance(provider, str)
            or provider.strip().lower()
            not in {
                runtime.settings.provider,
                f"main-agent({runtime.settings.provider})",
            }
        ):
            return original_request
        base_url = self._argument(bound.arguments, "base_url")
        if runtime.settings.base_url and (
            not isinstance(base_url, str)
            or normalize_base_url(base_url) != runtime.settings.base_url
        ):
            return original_request
        max_tokens = self._argument(bound.arguments, "max_tokens")
        request = original_request
        has_output_cap = any(
            field in original_request
            for field in ("max_tokens", "max_completion_tokens", "max_output_tokens")
        )
        if not has_output_cap:
            # Ask Hermes which wire field this model accepts.  The compression
            # window is only a non-constraining seed when the caller supplied
            # no cap; incontext replaces it with the exact free remainder.
            seed = (
                max_tokens
                if isinstance(max_tokens, int)
                and not isinstance(max_tokens, bool)
                and max_tokens > 0
                else runtime.settings.compression_window
            )
            request = {**request, **self._output_cap(seed, model)}
        result = runtime(request=request)
        return original_request if result is None else result["request"]

    def _output_cap(self, value: int, model: Any) -> Dict[str, int]:
        """Select Hermes' provider-specific output-cap alias safely."""

        if self.output_cap_selector is not None:
            try:
                selected = self.output_cap_selector(value, model=model)
            except Exception:  # noqa: BLE001
                selected = None
            if isinstance(selected, dict):
                valid = {
                    field: cap
                    for field in OUTPUT_BUDGET_FIELDS
                    if isinstance((cap := selected.get(field)), int)
                    and not isinstance(cap, bool)
                    and cap > 0
                }
                if valid:
                    return valid
        return {"max_tokens": value}

    def _runtime(self) -> Optional[DynamicOutputBudget]:
        if isinstance(self.runtime_source, DynamicOutputBudget):
            return self.runtime_source
        return self.runtime_source()

    def _argument(self, arguments: Dict[str, Any], name: str) -> Any:
        if name in arguments:
            return arguments[name]
        for parameter in self.signature.parameters.values():
            if parameter.kind is Parameter.VAR_KEYWORD:
                extra = arguments.get(parameter.name)
                if isinstance(extra, dict) and name in extra:
                    return extra[name]
        return None


_install_lock = threading.Lock()
_installed_wrapper: Optional[_AuxiliaryBudget] = None
_install_count = 0


def install(runtime: RuntimeSource) -> Optional[Cleanup]:
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
        current = auxiliary_client.__dict__.get("_build_call_kwargs")
        if not callable(current):
            LOGGER.warning(
                "incontext auxiliary budgeting unavailable: "
                "Hermes auxiliary builder API changed"
            )
            return None
        output_cap_selector = auxiliary_client.__dict__.get(
            "auxiliary_max_tokens_param"
        )
        if not callable(output_cap_selector):
            output_cap_selector = None
        if current is _installed_wrapper:
            wrapper = _installed_wrapper
            assert wrapper is not None
            wrapper.runtime_source = runtime
            wrapper.output_cap_selector = output_cap_selector
            _install_count += 1
        else:
            original = (
                current.original
                if isinstance(current, _AuxiliaryBudget)
                else cast(AuxiliaryBuilder, current)
            )
            wrapper = _AuxiliaryBudget(runtime, original, output_cap_selector)
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
