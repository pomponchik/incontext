"""Configuration resolution for the incontext Hermes plugin."""

from __future__ import annotations

import inspect
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

TOKENIZER_URL_ENV = "INCONTEXT_TOKENIZER_URL"
TOKENIZER_USER_AGENT_ENV = "INCONTEXT_TOKENIZER_USER_AGENT"
TOKENIZER_TIMEOUT_ENV = "INCONTEXT_TOKENIZER_TIMEOUT_SECONDS"
FALLBACK_MARGIN_ENV = "INCONTEXT_FALLBACK_MARGIN_TOKENS"
COMPRESSION_WINDOW_ENV = "INCONTEXT_COMPRESSION_WINDOW_TOKENS"

LEGACY_TOKENIZER_URL_ENV = "HERMES_VLLM_TOKENIZER_URL"
LEGACY_TOKENIZER_USER_AGENT_ENV = "HERMES_VLLM_TOKENIZER_USER_AGENT"
LEGACY_TOKENIZER_TIMEOUT_ENV = "HERMES_VLLM_TOKENIZER_TIMEOUT_SECONDS"
LEGACY_FALLBACK_MARGIN_ENV = "HERMES_DYNAMIC_BUDGET_FALLBACK_MARGIN_TOKENS"

DEFAULT_USER_AGENT = "incontext/0.1"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_FALLBACK_MARGIN_TOKENS = 1024


class SettingsError(RuntimeError):
    """Raised when incontext cannot derive a safe runtime configuration."""


@dataclass(frozen=True)
class Settings:
    """Validated immutable runtime settings."""

    __slots__ = (
        "compression_window",
        "context_length",
        "fallback_margin_tokens",
        "tokenizer_timeout_seconds",
        "tokenizer_url",
        "tokenizer_user_agent",
    )

    context_length: int
    compression_window: int
    tokenizer_url: str
    tokenizer_user_agent: str
    tokenizer_timeout_seconds: float
    fallback_margin_tokens: int


def _strict_int(
    value: Any,
    name: str,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise SettingsError(f"{name} must be an integer")
    try:
        parsed = int(value.strip() if isinstance(value, str) else value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer") from exc
    if parsed < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise SettingsError(f"{name} must be {qualifier}")
    return parsed


def _strict_float(
    value: Any,
    name: str,
    *,
    minimum_exclusive: float,
    maximum_inclusive: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise SettingsError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= minimum_exclusive:
        raise SettingsError(f"{name} must be greater than {minimum_exclusive}")
    if maximum_inclusive is not None and parsed > maximum_inclusive:
        raise SettingsError(f"{name} must not exceed {maximum_inclusive}")
    return parsed


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SettingsError(f"Hermes {name} configuration must be a mapping")
    return value


def _environment_value(
    environ: Mapping[str, str],
    primary: str,
    legacy: str | None = None,
    *,
    default: str | None = None,
) -> str | None:
    for key in (primary, legacy):
        if key is None:
            continue
        value = environ.get(key)
        if value is not None and value.strip():
            return value.strip()
    return default


def _validated_http_url(value: str | None, name: str) -> str:
    if value is None:
        raise SettingsError(f"{name} must contain the vLLM /tokenize URL")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SettingsError(f"{name} must contain an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise SettingsError(f"{name} must not embed credentials")
    if parsed.fragment:
        raise SettingsError(f"{name} must not contain a URL fragment")
    return value


def _normalized_model_thresholds(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, float] = {}
    for key, threshold in value.items():
        if (
            isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and math.isfinite(float(threshold))
        ):
            normalized[str(key)] = float(threshold)
    return normalized


def _construct_compressor(
    compressor_class: type[Any],
    candidates: dict[str, Any],
) -> Any:
    parameters = inspect.signature(compressor_class).parameters
    accepts_arbitrary_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs = (
        candidates
        if accepts_arbitrary_keywords
        else {key: value for key, value in candidates.items() if key in parameters}
    )
    try:
        return compressor_class(**kwargs)
    except Exception as exc:
        raise SettingsError("Hermes ContextCompressor initialization failed") from exc


def _load_hermes_components() -> tuple[Callable[[], Any], type[Any]]:
    try:
        # Hermes is intentionally an optional runtime dependency of the package.
        from agent.context_compressor import (  # type: ignore[import-not-found]  # noqa: PLC0415
            ContextCompressor,
        )
        from hermes_cli.config import (  # type: ignore[import-not-found]  # noqa: PLC0415
            load_config,
        )
    except ImportError as exc:
        raise SettingsError(
            "incontext must run inside the Hermes Agent Python environment",
        ) from exc
    return load_config, ContextCompressor


def load_settings(
    *,
    environ: Mapping[str, str] | None = None,
    config_loader: Callable[[], Any] | None = None,
    compressor_class: type[Any] | None = None,
) -> Settings:
    """Load and validate Hermes plus environment configuration.

    The compression window is obtained from Hermes' real ``ContextCompressor``
    instead of duplicating its version-sensitive threshold arithmetic.
    """

    active_environ = os.environ if environ is None else environ
    if config_loader is None or compressor_class is None:
        default_loader, default_compressor = _load_hermes_components()
        config_loader = config_loader or default_loader
        compressor_class = compressor_class or default_compressor

    raw_config = config_loader()
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, Mapping):
        raise SettingsError("Hermes configuration must be a mapping")

    model = _section(raw_config, "model")
    compression = _section(raw_config, "compression")
    model_name = model.get("default")
    if not isinstance(model_name, str) or not model_name.strip():
        raise SettingsError("Hermes model.default must be a non-empty string")
    context_length = _strict_int(
        model.get("context_length"),
        "Hermes model.context_length",
        minimum=1,
    )
    threshold = _strict_float(
        compression.get("threshold", 0.50),
        "Hermes compression.threshold",
        minimum_exclusive=0.0,
        maximum_inclusive=1.0,
    )

    window_override = _environment_value(
        active_environ,
        COMPRESSION_WINDOW_ENV,
    )
    if window_override is None:
        compressor = _construct_compressor(
            compressor_class,
            {
                "model": model_name.strip(),
                "threshold_percent": threshold,
                "quiet_mode": True,
                "base_url": str(model.get("base_url") or ""),
                "config_context_length": context_length,
                "provider": str(model.get("provider") or ""),
                "api_mode": str(model.get("api_mode") or ""),
                "max_tokens": None,
                "model_thresholds": _normalized_model_thresholds(
                    compression.get("model_thresholds"),
                ),
                "threshold_tokens_cap": compression.get("threshold_tokens"),
            },
        )
        resolved_context = _strict_int(
            getattr(compressor, "context_length", None),
            "Hermes ContextCompressor.context_length",
            minimum=1,
        )
        if resolved_context != context_length:
            raise SettingsError(
                "Hermes ContextCompressor context length does not match "
                f"model.context_length: {resolved_context} != {context_length}",
            )
        compression_window = _strict_int(
            getattr(compressor, "threshold_tokens", None),
            "Hermes ContextCompressor.threshold_tokens",
            minimum=1,
        )
    else:
        compression_window = _strict_int(
            window_override,
            COMPRESSION_WINDOW_ENV,
            minimum=1,
        )

    if compression_window > context_length:
        raise SettingsError(
            "The compression window must not exceed model.context_length",
        )

    tokenizer_url = _validated_http_url(
        _environment_value(
            active_environ,
            TOKENIZER_URL_ENV,
            LEGACY_TOKENIZER_URL_ENV,
        ),
        TOKENIZER_URL_ENV,
    )
    tokenizer_user_agent = _environment_value(
        active_environ,
        TOKENIZER_USER_AGENT_ENV,
        LEGACY_TOKENIZER_USER_AGENT_ENV,
        default=DEFAULT_USER_AGENT,
    )
    assert tokenizer_user_agent is not None
    tokenizer_timeout = _strict_float(
        _environment_value(
            active_environ,
            TOKENIZER_TIMEOUT_ENV,
            LEGACY_TOKENIZER_TIMEOUT_ENV,
            default=str(DEFAULT_TIMEOUT_SECONDS),
        ),
        TOKENIZER_TIMEOUT_ENV,
        minimum_exclusive=0.0,
    )
    fallback_margin = _strict_int(
        _environment_value(
            active_environ,
            FALLBACK_MARGIN_ENV,
            LEGACY_FALLBACK_MARGIN_ENV,
            default=str(DEFAULT_FALLBACK_MARGIN_TOKENS),
        ),
        FALLBACK_MARGIN_ENV,
        minimum=0,
    )
    if fallback_margin >= compression_window:
        raise SettingsError(
            f"{FALLBACK_MARGIN_ENV} must be below the compression window",
        )

    return Settings(
        context_length=context_length,
        compression_window=compression_window,
        tokenizer_url=tokenizer_url,
        tokenizer_user_agent=tokenizer_user_agent,
        tokenizer_timeout_seconds=tokenizer_timeout,
        fallback_margin_tokens=fallback_margin,
    )
