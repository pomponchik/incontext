"""Dynamic output-budget middleware implementation."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from .backend import Backend
from .settings import Settings

LOGGER = logging.getLogger(__name__)
OUTPUT_BUDGET_FIELDS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def compute_max_tokens(
    compression_window: int,
    prompt_tokens: int,
    *,
    safety_margin: int = 0,
) -> Optional[int]:
    """Return free output space, or ``None`` when compression is required.

    OpenAI-compatible APIs do not accept a zero-token completion.  Returning
    ``1`` for an already-full compression window is therefore unsafe: it turns
    a preflight condition into a request that the provider must reject.
    """

    if compression_window <= 0:
        raise ValueError("compression_window must be positive")
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    if safety_margin < 0:
        raise ValueError("safety_margin must not be negative")
    remaining = compression_window - prompt_tokens - safety_margin
    return remaining if remaining > 0 else None


def _requested_output_cap(
    request: Dict[str, Any],
) -> Optional[Tuple[str, int]]:
    """Return the field and smallest valid cap requested by the caller."""

    extra_body = request.get("extra_body")
    sources = [request]
    if isinstance(extra_body, dict):
        sources.append(extra_body)
    caps = [
        (field, value)
        for source in sources
        for field in OUTPUT_BUDGET_FIELDS
        if not isinstance((value := source.get(field)), bool)
        and isinstance(value, int)
        and value > 0
    ]
    return min(caps, key=lambda item: item[1]) if caps else None


def estimate_request_tokens_rough(request: Dict[str, Any]) -> int:
    """Use Hermes' own conservative request estimator as a fallback."""

    # Hermes is intentionally an optional runtime dependency of the PyPI package.
    from agent.model_metadata import (  # type: ignore[import-not-found]  # noqa: PLC0415
        estimate_request_tokens_rough as estimator,
    )

    estimate = estimator(
        request.get("messages") or [],
        tools=request.get("tools") or None,
    )
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

        if not isinstance(request, dict) or not isinstance(
            request.get("messages"),
            list,
        ):
            return None
        if not self._matches_route(request, context):
            return None

        source = self.backend.source
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

        dynamic_max_tokens = compute_max_tokens(
            self.settings.compression_window,
            prompt_tokens,
            safety_margin=safety_margin,
        )
        if dynamic_max_tokens is None:
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
        requested_output_cap = _requested_output_cap(request)
        output_field = "max_tokens"
        if requested_output_cap is not None:
            output_field, requested_cap = requested_output_cap
            dynamic_max_tokens = min(dynamic_max_tokens, requested_cap)
        rewritten = dict(request)
        for key in OUTPUT_BUDGET_FIELDS:
            rewritten.pop(key, None)
        extra_body = rewritten.get("extra_body")
        if isinstance(extra_body, dict):
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

    def _matches_route(
        self,
        request: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        extra_body = request.get("extra_body")
        model = (
            extra_body.get("model", request.get("model"))
            if isinstance(extra_body, dict)
            else request.get("model")
        )
        if model != self.settings.model_name:
            return False
        provider = context.get("provider")
        if (
            self.settings.provider
            and isinstance(provider, str)
            and provider.strip() != self.settings.provider
        ):
            return False
        base_url = context.get("base_url")
        return not (
            self.settings.base_url
            and isinstance(base_url, str)
            and base_url.strip().rstrip("/") != self.settings.base_url
        )
