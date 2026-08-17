"""Dynamic output-budget middleware implementation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from .settings import Settings
from .tokenizer import VllmTokenizer

LOGGER = logging.getLogger(__name__)
OUTPUT_BUDGET_FIELDS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


class TokenCounter(Protocol):
    """Structural contract used by the middleware's exact counter."""

    def count(self, request: dict[str, Any]) -> int:
        """Return the provider-visible prompt size."""


def compute_max_tokens(
    compression_window: int,
    prompt_tokens: int,
    *,
    safety_margin: int = 0,
) -> int:
    """Return the positive free remainder of the compression window."""

    if compression_window <= 0:
        raise ValueError("compression_window must be positive")
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    if safety_margin < 0:
        raise ValueError("safety_margin must not be negative")
    return max(1, compression_window - prompt_tokens - safety_margin)


def estimate_request_tokens_rough(request: dict[str, Any]) -> int:
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
        *,
        tokenizer: TokenCounter | None = None,
        rough_estimator: Callable[[dict[str, Any]], int] | None = None,
    ) -> None:
        self.settings = settings
        self.tokenizer = VllmTokenizer(settings) if tokenizer is None else tokenizer
        self.rough_estimator = (
            estimate_request_tokens_rough
            if rough_estimator is None
            else rough_estimator
        )

    def __call__(
        self,
        *,
        request: dict[str, Any],
        **_: Any,
    ) -> dict[str, Any] | None:
        """Rewrite output-cap aliases into one exact dynamic ``max_tokens``."""

        if not isinstance(request, dict) or not isinstance(
            request.get("messages"),
            list,
        ):
            return None

        source = "vllm-tokenize"
        safety_margin = 0
        try:
            prompt_tokens = self.tokenizer.count(request)
        # Middleware must fail open for every provider/transport failure so a
        # tokenizer outage cannot make Hermes unable to call the model.
        except Exception as exact_error:  # noqa: BLE001
            source = "rough-fallback"
            safety_margin = self.settings.fallback_margin_tokens
            LOGGER.warning(
                "incontext tokenizer_failed type=%s; using Hermes rough estimate",
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
        rewritten = dict(request)
        for key in OUTPUT_BUDGET_FIELDS:
            rewritten.pop(key, None)
        rewritten["max_tokens"] = dynamic_max_tokens

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
