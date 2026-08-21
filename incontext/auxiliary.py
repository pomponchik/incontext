"""Dynamic output budgeting for Hermes auxiliary LLM requests."""

from __future__ import annotations

import logging
import threading
from importlib import import_module
from inspect import Parameter, Signature, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast

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
        self._owners: Tuple[
            Tuple[object, RuntimeSource, Optional[OutputCapSelector]], ...
        ] = ()

    def acquire(
        self,
        owner: object,
        runtime: RuntimeSource,
        output_cap_selector: Optional[OutputCapSelector],
    ) -> None:
        """Add one installation without mutating an earlier owner's state."""
        self._owners = (*self._owners, (owner, runtime, output_cap_selector))

    def release(self, owner: object) -> None:
        """Remove exactly one installation while retaining all other owners."""
        self._owners = tuple(entry for entry in self._owners if entry[0] is not owner)

    @property
    def owned(self) -> bool:
        """Return whether at least one live installation owns this wrapper."""
        return bool(self._owners)

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        original_request = self.original(*args, **kwargs)
        runtime_source, output_cap_selector = self._configuration()
        runtime = self._runtime(runtime_source)
        if runtime is None:
            return original_request
        try:
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError:
            return original_request
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
            request = {
                **request,
                **self._output_cap(seed, model, output_cap_selector),
            }
        result = runtime(request=request)
        return original_request if result is None else result["request"]

    def _output_cap(
        self,
        value: int,
        model: Any,
        output_cap_selector: Optional[OutputCapSelector],
    ) -> Dict[str, int]:
        """Select Hermes' provider-specific output-cap alias safely."""
        if output_cap_selector is not None:
            try:
                selected = output_cap_selector(value, model=model)
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

    def _configuration(
        self,
    ) -> Tuple[RuntimeSource, Optional[OutputCapSelector]]:
        """Read one coherent owner snapshot for the complete request."""
        owners = self._owners
        if owners:
            _, runtime, output_cap_selector = owners[-1]
            return runtime, output_cap_selector
        return self.runtime_source, self.output_cap_selector

    @staticmethod
    def _runtime(runtime_source: RuntimeSource) -> Optional[DynamicOutputBudget]:
        if isinstance(runtime_source, DynamicOutputBudget):
            return runtime_source
        return runtime_source()

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


def _new_wrapper(
    runtime: RuntimeSource,
    current: Any,
    output_cap_selector: Optional[OutputCapSelector],
) -> Optional[_AuxiliaryBudget]:
    """Construct an auxiliary wrapper when its callable can be inspected."""
    original = (
        current.original
        if isinstance(current, _AuxiliaryBudget)
        else cast(AuxiliaryBuilder, current)
    )
    try:
        return _AuxiliaryBudget(runtime, original, output_cap_selector)
    except (TypeError, ValueError):
        LOGGER.warning(
            "incontext auxiliary budgeting unavailable: "
            "Hermes auxiliary builder signature is unavailable"
        )
        return None


def _load_auxiliary_builder() -> Optional[Tuple[Any, AuxiliaryBuilder]]:
    """Import and validate Hermes' optional private auxiliary binding."""
    try:
        auxiliary_client = import_module("agent.auxiliary_client")
    except ImportError:
        LOGGER.warning(
            "incontext auxiliary budgeting unavailable: Hermes is not installed"
        )
        return None
    current = auxiliary_client.__dict__.get("_build_call_kwargs")
    if not callable(current):
        LOGGER.warning(
            "incontext auxiliary budgeting unavailable: "
            "Hermes auxiliary builder API changed"
        )
        return None
    return auxiliary_client, cast(AuxiliaryBuilder, current)


def install(runtime: RuntimeSource) -> Optional[Cleanup]:
    """Apply incontext to Hermes requests that bypass ``llm_request`` middleware."""
    loaded = _load_auxiliary_builder()
    if loaded is None:
        return None
    auxiliary_client, current = loaded

    owner = object()
    global _installed_wrapper  # noqa: PLW0603
    with _install_lock:
        output_cap_selector = auxiliary_client.__dict__.get(
            "auxiliary_max_tokens_param"
        )
        if not callable(output_cap_selector):
            output_cap_selector = None
        if current is _installed_wrapper:
            wrapper = _installed_wrapper
            assert wrapper is not None
        else:
            created = _new_wrapper(runtime, current, output_cap_selector)
            if created is None:
                return None
            wrapper = created
            auxiliary_client.__dict__["_build_call_kwargs"] = wrapper
            _installed_wrapper = wrapper
        wrapper.acquire(owner, runtime, output_cap_selector)

    closed = False

    def cleanup() -> None:
        """Release one owner and restore Hermes after the final unload."""
        nonlocal closed
        global _installed_wrapper  # noqa: PLW0603
        with _install_lock:
            if closed:
                return
            closed = True
            wrapper.release(owner)
            if not wrapper.owned:
                if auxiliary_client.__dict__.get("_build_call_kwargs") is wrapper:
                    auxiliary_client.__dict__["_build_call_kwargs"] = wrapper.original
                if _installed_wrapper is wrapper:
                    _installed_wrapper = None

    return cleanup
