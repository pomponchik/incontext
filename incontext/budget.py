"""Dynamic output-budget middleware implementation."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import Any, Callable, Dict, Optional, Tuple

from .backend import Backend
from .settings import Settings, normalize_base_url

LOGGER = logging.getLogger(__name__)
OUTPUT_BUDGET_FIELDS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def _backend_source(backend: Backend) -> str:
    """Read optional diagnostics without letting them break middleware."""
    try:
        return backend.source
    except Exception as backend_error:  # noqa: BLE001
        LOGGER.warning(
            "incontext backend_contract_failed type=%s; source unavailable",
            type(backend_error).__name__,
        )
        return "unknown-backend"


def _materialize_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Copy reusable OpenAI collections into their provider-visible shapes."""
    messages = request.get("messages")
    if not isinstance(messages, Collection) or isinstance(
        messages,
        (str, bytes, Mapping),
    ):
        return None
    prepared = dict(request)
    prepared["messages"] = list(messages)
    extra_body = request.get("extra_body")
    if isinstance(extra_body, Mapping):
        prepared_extra = dict(extra_body)
        nested_messages = prepared_extra.get("messages")
        if isinstance(nested_messages, Collection) and not isinstance(
            nested_messages,
            (str, bytes, Mapping),
        ):
            prepared_extra["messages"] = list(nested_messages)
        prepared["extra_body"] = prepared_extra
    return prepared


def compute_max_tokens(
    compression_window: int,
    prompt_tokens: int,
    *,
    safety_margin: int = 0,
    minimum_output_tokens: int = 1,
) -> Optional[int]:
    """Return viable output space, or ``None`` when compression is required.

    A technically positive completion can still be unusable for an agent.  The
    minimum keeps tiny length-truncated replies and tool calls out of the wire
    request while retaining ``1`` as the default for callers of this helper
    that only need its original positive-space contract.
    """
    if compression_window <= 0:
        raise ValueError("compression_window must be positive")
    if prompt_tokens < 0:
        raise ValueError("prompt_tokens must not be negative")
    if safety_margin < 0:
        raise ValueError("safety_margin must not be negative")
    if minimum_output_tokens <= 0:
        raise ValueError("minimum_output_tokens must be positive")
    remaining = compression_window - prompt_tokens - safety_margin
    return remaining if remaining >= minimum_output_tokens else None


def compression_pressure_tokens(
    prompt_tokens: int,
    minimum_output_tokens: int,
) -> int:
    """Include the viability reserve in Hermes' preflight pressure count."""
    if prompt_tokens < 0:
        raise ValueError("prompt_tokens must not be negative")
    if minimum_output_tokens <= 0:
        raise ValueError("minimum_output_tokens must be positive")
    # Hermes compresses at ``pressure >= compression_window``.  Subtracting
    # one preserves the useful equality case where exactly the minimum output
    # budget remains: P + (R - 1) < W iff W - P >= R.
    return prompt_tokens + minimum_output_tokens - 1


def _requested_output_cap(
    request: Dict[str, Any],
    coerce: Callable[[Any], Optional[int]],
) -> Optional[Tuple[str, int]]:
    """Return the provider field and smallest valid caller cap."""
    extra_body = request.get("extra_body")
    top_level_caps = [
        (field, value)
        for field in OUTPUT_BUDGET_FIELDS
        if (value := coerce(request.get(field))) is not None
    ]
    nested_caps = (
        [
            (field, value)
            for field in OUTPUT_BUDGET_FIELDS
            if (value := coerce(extra_body.get(field))) is not None
        ]
        if isinstance(extra_body, Mapping)
        else []
    )
    caps = [*top_level_caps, *nested_caps]
    if not caps:
        return None
    preferred = min(top_level_caps or nested_caps, key=lambda item: item[1])[0]
    return preferred, min(value for _, value in caps)


def estimate_request_tokens_rough(request: Dict[str, Any]) -> int:
    """Use Hermes' own conservative request estimator as a fallback."""
    # Hermes is intentionally an optional runtime dependency of the PyPI package.
    from agent.model_metadata import (  # type: ignore[import-not-found]  # noqa: PLC0415
        estimate_request_tokens_rough as estimator,
    )

    extra_body = request.get("extra_body")
    messages = (
        extra_body.get("messages", request.get("messages"))
        if isinstance(extra_body, Mapping)
        else request.get("messages")
    )
    tools = (
        extra_body.get("tools", request.get("tools"))
        if isinstance(extra_body, Mapping)
        else request.get("tools")
    )
    estimate = estimator(messages or [], tools=tools or None)
    if isinstance(estimate, bool):
        return 1
    try:
        return max(1, int(estimate))
    except (TypeError, ValueError):
        return 1


class DynamicOutputBudget:
    """Hermes ``llm_request`` middleware with exact-first token counting."""

    def __init__(
        self,
        settings: Settings,
        backend: Backend,
        *,
        rough_estimator: Optional[Callable[[Dict[str, Any]], int]] = None,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.rough_estimator = (
            estimate_request_tokens_rough
            if rough_estimator is None
            else rough_estimator
        )

    def __call__(
        self,
        *,
        request: Dict[str, Any],
        **context: Any,
    ) -> Optional[Dict[str, Any]]:
        """Rewrite output-cap aliases into one exact dynamic ``max_tokens``."""
        if not isinstance(request, dict):
            return None
        prepared_request = _materialize_request(request)
        if prepared_request is None:
            return None
        request = prepared_request
        if not self._matches_route(request, context):
            return None

        source = _backend_source(self.backend)
        safety_margin = 0
        try:
            prompt_tokens = self.backend.count(
                request,
                context_length=self.settings.context_length,
            )
        # Middleware must fail open for every provider/transport failure so a
        # backend outage cannot make Hermes unable to call the model.
        except Exception as exact_error:  # noqa: BLE001
            source = "rough-fallback"
            safety_margin = self.settings.fallback_margin_tokens
            LOGGER.warning(
                "incontext backend_failed type=%s; using Hermes rough estimate",
                type(exact_error).__name__,
            )
            try:
                prompt_tokens = max(1, int(self.rough_estimator(request)))
            except Exception as fallback_error:  # noqa: BLE001
                # Do not log tracebacks here: provider exceptions may contain
                # request bodies or credentials.
                LOGGER.error(  # noqa: TRY400
                    "incontext estimators_failed exact=%s fallback=%s; "
                    "request unchanged",
                    type(exact_error).__name__,
                    type(fallback_error).__name__,
                )
                return None

        resolved_budget = self._resolve_output_budget(
            request,
            self.settings.compression_window,
            prompt_tokens,
            safety_margin,
            source,
        )
        if resolved_budget is None:
            return None
        output_field, dynamic_max_tokens = resolved_budget
        rewritten = dict(request)
        for key in OUTPUT_BUDGET_FIELDS:
            rewritten.pop(key, None)
        extra_body = rewritten.get("extra_body")
        if isinstance(extra_body, Mapping):
            cleaned_extra_body = dict(extra_body)
            for key in OUTPUT_BUDGET_FIELDS:
                cleaned_extra_body.pop(key, None)
            rewritten["extra_body"] = cleaned_extra_body
        rewritten[output_field] = dynamic_max_tokens

        LOGGER.info(
            "incontext model=%s source=%s prompt_tokens=%d "
            "compression_window=%d max_tokens=%d",
            request.get("model") or "unknown",
            source,
            prompt_tokens,
            self.settings.compression_window,
            dynamic_max_tokens,
        )
        return {
            "request": rewritten,
            "source": "incontext",
            "reason": (
                f"{source}: compression_window={self.settings.compression_window}, "
                f"prompt_tokens={prompt_tokens}, max_tokens={dynamic_max_tokens}"
            ),
        }

    def _resolve_output_budget(
        self,
        request: Dict[str, Any],
        compression_window: int,
        prompt_tokens: int,
        safety_margin: int,
        source: str,
    ) -> Optional[Tuple[str, int]]:
        """Combine the window, caller cap, and backend wire constraint."""
        try:
            requested_output_cap = _requested_output_cap(
                request,
                self.backend.coerce_output_budget,
            )
            minimum_output_tokens = self.settings.min_output_tokens
            if requested_output_cap is not None:
                minimum_output_tokens = min(
                    minimum_output_tokens,
                    requested_output_cap[1],
                )
            dynamic_max_tokens = compute_max_tokens(
                compression_window,
                prompt_tokens,
                safety_margin=safety_margin,
                minimum_output_tokens=minimum_output_tokens,
            )
            output_field = "max_tokens"
            if dynamic_max_tokens is not None:
                if requested_output_cap is not None:
                    output_field, requested_cap = requested_output_cap
                    dynamic_max_tokens = min(dynamic_max_tokens, requested_cap)
                provider_limit = self.backend.output_budget_limit(
                    request,
                    context_length=self.settings.context_length,
                )
                if provider_limit is not None:
                    dynamic_max_tokens = min(dynamic_max_tokens, provider_limit)
                output_field = self.backend.output_budget_field(output_field)
        except Exception as backend_error:  # noqa: BLE001
            LOGGER.warning(
                "incontext backend_contract_failed type=%s; request unchanged",
                type(backend_error).__name__,
            )
            return None
        if dynamic_max_tokens is None or dynamic_max_tokens < minimum_output_tokens:
            # The exact preflight installed during plugin registration sees
            # this condition before Hermes builds the provider request and
            # starts compression.  Keep this middleware fail-open as a second
            # line of defence for requests from call sites that bypass that
            # preflight: an invalid sentinel cap would otherwise cause Hermes
            # to retry the same context error instead of compacting history.
            LOGGER.warning(
                "incontext model=%s source=%s prompt_tokens=%d "
                "compression_window=%d action=requires_compression",
                request.get("model") or "unknown",
                source,
                prompt_tokens,
                self.settings.compression_window,
            )
            return None
        return output_field, dynamic_max_tokens

    def _matches_route(
        self,
        request: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        extra_body = request.get("extra_body")
        model = (
            extra_body.get("model", request.get("model"))
            if isinstance(extra_body, Mapping)
            else request.get("model")
        )
        if model != self.settings.model_name:
            return False
        provider = context.get("provider")
        if (
            self.settings.provider
            and isinstance(provider, str)
            and provider.strip().lower() != self.settings.provider
        ):
            return False
        base_url = context.get("base_url")
        return not (
            self.settings.base_url
            and isinstance(base_url, str)
            and normalize_base_url(base_url) != self.settings.base_url
        )
