"""Exact-token preflight bridge for Hermes' compression decision."""

from __future__ import annotations

import logging
import threading
from collections.abc import Collection, Mapping
from importlib import import_module
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

from .budget import DynamicOutputBudget, compression_pressure_tokens
from .settings import normalize_base_url

LOGGER = logging.getLogger(__name__)
RoughEstimator = Callable[..., int]
RuntimeSource = Union[
    DynamicOutputBudget,
    Callable[[], Optional[DynamicOutputBudget]],
]
Cleanup = Callable[[], None]


class _ExactPressure(int):
    """Mark a provider-tokenized pressure value as authoritative."""


def _live_main_route() -> Optional[Tuple[str, str, str]]:
    """Read Hermes' turn-local primary route across supported releases."""

    try:
        auxiliary = import_module("agent.auxiliary_client")
    except ImportError:
        return None
    getter = auxiliary.__dict__.get("_runtime_main_value")
    try:
        if callable(getter):
            values = (
                str(getter("provider") or ""),
                str(getter("model") or ""),
                str(getter("base_url") or ""),
            )
        else:
            values = (
                str(auxiliary.__dict__.get("_RUNTIME_MAIN_PROVIDER") or ""),
                str(auxiliary.__dict__.get("_RUNTIME_MAIN_MODEL") or ""),
                str(auxiliary.__dict__.get("_RUNTIME_MAIN_BASE_URL") or ""),
            )
    except Exception:  # noqa: BLE001
        return ("", "\0route-unavailable", "")
    return values if any(values) else None


def _matches_live_route(runtime: DynamicOutputBudget) -> bool:
    """Return whether the exact backend still owns Hermes' active route."""

    route = _live_main_route()
    if route is None:
        return True
    provider, model, base_url = route
    settings = runtime.settings
    if model != settings.model_name:
        return False
    if settings.provider and provider.strip().lower() != settings.provider:
        return False
    return not (settings.base_url and normalize_base_url(base_url) != settings.base_url)


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
        **context: Any,
    ) -> int:
        if context:
            return int(
                self.original(
                    messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    **context,
                ),
            )
        runtime = self._runtime()
        if (
            runtime is None
            or not _matches_live_route(runtime)
            or not isinstance(messages, Collection)
            or isinstance(messages, (str, bytes, Mapping))
        ):
            return int(
                self.original(
                    messages,
                    system_prompt=system_prompt,
                    tools=tools,
                ),
            )
        provider_messages = []
        for stored_message in messages:
            message = stored_message
            if isinstance(stored_message, Mapping):
                message = dict(stored_message)
                api_content = message.pop("api_content", None)
                if (
                    isinstance(api_content, str)
                    and api_content
                    and message.get("role") in {"user", "assistant"}
                ):
                    message["content"] = api_content
            provider_messages.append(message)
        if system_prompt:
            provider_messages.insert(
                0,
                {"role": "system", "content": system_prompt},
            )
        request: Dict[str, Any] = {
            "model": runtime.settings.model_name,
            "messages": provider_messages,
        }
        if (
            isinstance(tools, Collection)
            and not isinstance(
                tools,
                (str, bytes, Mapping),
            )
            and tools
        ):
            request["tools"] = list(tools)
        try:
            prompt_tokens = runtime.backend.count(
                request,
                context_length=runtime.settings.context_length,
            )
            return _ExactPressure(
                compression_pressure_tokens(
                    prompt_tokens,
                    runtime.settings.min_output_tokens,
                ),
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
            return compression_pressure_tokens(
                rough_tokens + runtime.settings.fallback_margin_tokens,
                runtime.settings.min_output_tokens,
            )

    def _runtime(self) -> Optional[DynamicOutputBudget]:
        owners = self._owners
        runtime_source = owners[-1][1] if owners else self.runtime_source
        if isinstance(runtime_source, DynamicOutputBudget):
            return runtime_source
        return runtime_source()


class _ExactPreflightGate:
    """Force the exact estimate before Hermes' message-only cheap gate."""

    def __init__(self, runtime: RuntimeSource, original: Callable[..., bool]) -> None:
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

    def __call__(self, *args: Any, **kwargs: Any) -> bool:
        runtime = self._runtime()
        if runtime is not None and _matches_live_route(runtime):
            return True
        return bool(self.original(*args, **kwargs))

    def _runtime(self) -> Optional[DynamicOutputBudget]:
        owners = self._owners
        runtime_source = owners[-1][1] if owners else self.runtime_source
        if isinstance(runtime_source, DynamicOutputBudget):
            return runtime_source
        return runtime_source()


class _ExactPreflightDefer:
    """Preserve Hermes' rough-count deferral while trusting exact pressure."""

    def __init__(self, original: Callable[..., bool]) -> None:
        self.original = original
        self._owners: List[object] = []

    def acquire(self, owner: object) -> None:
        """Attach one installation owner."""

        self._owners.append(owner)

    def release(self, owner: object) -> None:
        """Release one installation owner without disturbing the others."""

        self._owners = [current for current in self._owners if current is not owner]

    @property
    def owned(self) -> bool:
        """Return whether this wrapper belongs to an active installation."""

        return bool(self._owners)

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        """Bind this callable like the Hermes instance method it replaces."""

        del owner
        return self if instance is None else MethodType(self, instance)

    def __call__(
        self,
        compressor: Any,
        pressure: int,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        if isinstance(pressure, _ExactPressure):
            return False
        return bool(self.original(compressor, pressure, *args, **kwargs))


_install_lock = threading.Lock()
PreflightWrapper = Union[_ExactPreflight, _ExactPreflightGate, _ExactPreflightDefer]
InstalledBinding = Tuple[Any, str, PreflightWrapper]


def _original_binding(binding: Any, wrapper_type: Type[Any]) -> Any:
    """Unwrap a released incontext binding before reinstalling it."""

    return binding.original if isinstance(binding, wrapper_type) else binding


def _estimator_modules(turn_context: Any) -> Tuple[Any, ...]:
    """Return every Hermes module that owns a proactive estimator binding."""

    try:
        conversation_loop = import_module("agent.conversation_loop")
    except ImportError:
        return (turn_context,)
    if not callable(
        conversation_loop.__dict__.get("estimate_request_tokens_rough"),
    ):
        return (turn_context,)
    return turn_context, conversation_loop


def _install_estimators(
    runtime: RuntimeSource,
    owner: object,
    modules: Tuple[Any, ...],
    bindings: List[Any],
) -> List[InstalledBinding]:
    """Install or share exact wrappers for independent imported bindings."""

    installed: List[InstalledBinding] = []
    for module, current in zip(modules, bindings):
        if isinstance(current, _ExactPreflight) and current.owned:
            wrapper = current
        else:
            original = _original_binding(current, _ExactPreflight)
            wrapper = _ExactPreflight(runtime, cast(RoughEstimator, original))
            module.__dict__["estimate_request_tokens_rough"] = wrapper
        wrapper.acquire(owner, runtime)
        installed.append((module, "estimate_request_tokens_rough", wrapper))
    return installed


def _install_exact_deferral(owner: object) -> Optional[InstalledBinding]:
    """Make Hermes' rough-only defer heuristic recognize exact pressure."""

    try:
        context_compressor = import_module("agent.context_compressor")
    except ImportError:
        return None
    compressor_type = context_compressor.__dict__.get("ContextCompressor")
    defer_binding = (
        compressor_type.__dict__.get("should_defer_preflight_to_real_usage")
        if isinstance(compressor_type, type)
        else None
    )
    if not callable(defer_binding):
        return None
    if isinstance(defer_binding, _ExactPreflightDefer) and defer_binding.owned:
        wrapper = defer_binding
    else:
        original = _original_binding(defer_binding, _ExactPreflightDefer)
        wrapper = _ExactPreflightDefer(cast(Callable[..., bool], original))
        type.__setattr__(
            compressor_type,
            "should_defer_preflight_to_real_usage",
            wrapper,
        )
    wrapper.acquire(owner)
    return compressor_type, "should_defer_preflight_to_real_usage", wrapper


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
    except ImportError:
        LOGGER.warning("incontext exact preflight unavailable: Hermes is not installed")
        return None

    owner = object()
    wrappers: List[InstalledBinding] = []
    with _install_lock:
        modules = _estimator_modules(turn_context)
        bindings = [
            module.__dict__.get("estimate_request_tokens_rough") for module in modules
        ]
        gate_binding = turn_context.__dict__.get("_should_run_preflight_estimate")
        if not all(callable(binding) for binding in (*bindings, gate_binding)):
            LOGGER.warning(
                "incontext exact preflight unavailable: Hermes estimator API changed"
            )
            return None
        wrappers.extend(_install_estimators(runtime, owner, modules, bindings))
        if isinstance(gate_binding, _ExactPreflightGate) and gate_binding.owned:
            gate_wrapper = gate_binding
        else:
            gate_original = _original_binding(gate_binding, _ExactPreflightGate)
            gate_wrapper = _ExactPreflightGate(
                runtime,
                cast(Callable[..., bool], gate_original),
            )
            turn_context.__dict__["_should_run_preflight_estimate"] = gate_wrapper
        gate_wrapper.acquire(owner, runtime)
        wrappers.append(
            (turn_context, "_should_run_preflight_estimate", gate_wrapper),
        )
        defer_wrapper = _install_exact_deferral(owner)
        if defer_wrapper is not None:
            wrappers.append(defer_wrapper)

    closed = False

    def cleanup() -> None:
        """Release one owner and restore every unchanged Hermes binding."""

        nonlocal closed
        with _install_lock:
            if closed:
                return
            closed = True
            for target, binding_name, wrapper in wrappers:
                wrapper.release(owner)
                if not wrapper.owned and target.__dict__.get(binding_name) is wrapper:
                    setattr(target, binding_name, wrapper.original)

    return cleanup
