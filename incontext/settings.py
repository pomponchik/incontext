"""Configuration resolution for the incontext Hermes plugin."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple, Type, cast
from urllib.parse import urlsplit, urlunsplit

from skelet import EnvSource, Field, Storage


class SettingsError(RuntimeError):
    """Raised when incontext cannot derive a safe runtime configuration."""


def _optional_positive_integer_text(value: str) -> bool:
    """Validate an optional integer environment value after whitespace removal."""

    if not value:
        return True
    try:
        return int(value) > 0
    except ValueError:
        return False


class Environment(
    Storage,
    sources=cast(
        Any,
        [
            *EnvSource.for_library("incontext"),
            *EnvSource.for_library("hermes_dynamic_budget"),
        ],
    ),
):
    """Typed environment configuration resolved entirely by skelet."""

    backend: str = Field(
        "vllm",
        conversion=lambda value: value.strip(),
        validation={
            "backend must not be blank": lambda value: bool(value),
        },
        read_only=True,
    )
    fallback_margin_tokens: int = Field(
        1024,
        validation={
            "fallback_margin_tokens must be at least 0": lambda value: value >= 0,
        },
        read_only=True,
    )
    compression_window_tokens: int = Field(
        0,
        validation={
            "compression_window_tokens must be positive": lambda value: value > 0,
        },
        validate_default=False,
        read_only=True,
    )


class HermesEnvironment(
    Storage,
    sources=cast(Any, [*EnvSource.for_library("hermes")]),
):
    """Hermes-owned environment overrides that affect compression policy."""

    max_tokens: str = Field(
        "",
        conversion=lambda value: value.strip(),
        validation={
            "max_tokens must be a positive integer or blank": (
                _optional_positive_integer_text
            ),
        },
        read_only=True,
    )


@dataclass(frozen=True)
class Settings:
    """Validated immutable runtime settings."""

    __slots__ = (
        "base_url",
        "compression_window",
        "context_length",
        "fallback_margin_tokens",
        "model_name",
        "provider",
    )

    model_name: str
    context_length: int
    compression_window: int
    fallback_margin_tokens: int
    provider: str
    base_url: str


def normalize_base_url(value: Any) -> str:
    """Return a stable route identity for equivalent HTTP endpoint spellings."""

    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError:
        return text
    if not parsed.scheme or parsed.hostname is None:
        return text
    if parsed.username is not None or parsed.password is not None:
        return text
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = {"http": 80, "https": 443}.get(scheme)
    if port is not None and port != default_port:
        host = f"{host}:{port}"
    return urlunsplit(
        (scheme, host, parsed.path.rstrip("/"), parsed.query, parsed.fragment),
    )


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
    maximum_inclusive: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise SettingsError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as exc:
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


def _normalized_model_thresholds(value: Any) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    normalized: Dict[str, float] = {}
    for key, threshold in value.items():
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            continue
        try:
            converted = float(threshold)
        except OverflowError:
            continue
        if math.isfinite(converted):
            normalized[str(key)] = converted
    return normalized


def _construct_compressor(
    compressor_class: Type[Any],
    candidates: Dict[str, Any],
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


def _load_hermes_components() -> Tuple[Callable[[], Any], Type[Any]]:
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


def _effective_compression_threshold(
    configured: float,
    *,
    model: str,
    provider: str,
    compression: Mapping[str, Any],
) -> float:
    """Reuse Hermes' installed model-specific threshold policy when available."""

    try:
        from agent.auxiliary_client import (  # type: ignore[import-not-found]  # noqa: PLC0415
            _compression_threshold_for_model,
        )
    except (ImportError, AttributeError):
        return configured

    try:
        allow_codex_autoraise = str(
            compression.get("codex_gpt55_autoraise", True),
        ).lower() in {"true", "1", "yes"}
        model_threshold = _compression_threshold_for_model(
            model,
            provider,
            allow_codex_gpt55_autoraise=allow_codex_autoraise,
        )
    except Exception:  # noqa: BLE001
        return configured

    try:
        from agent.agent_init import (  # type: ignore[import-not-found]  # noqa: PLC0415
            _resolve_compression_threshold,
        )
        from agent.auxiliary_client import (  # noqa: PLC0415
            _is_codex_gpt54_or_gpt55,
            _is_codex_spark,
        )
    except (ImportError, AttributeError):
        # Hermes 2026.7.1 applied the model override directly, before the
        # shared resolver and expanded Codex classifiers were introduced.
        effective = configured if model_threshold is None else model_threshold
    else:
        try:
            effective, _ = _resolve_compression_threshold(
                configured,
                model_threshold,
                model=model,
                is_codex_autoraise=(
                    _is_codex_gpt54_or_gpt55(model, provider)
                    or _is_codex_spark(model, provider)
                ),
            )
        except Exception:  # noqa: BLE001
            # Hermes itself treats model-policy lookup as best effort and keeps the
            # configured global threshold if those private helpers fail.
            return configured
    return _strict_float(
        effective,
        "Hermes effective compression threshold",
        minimum_exclusive=0.0,
        maximum_inclusive=1.0,
    )


def _normalized_provider_selector(value: Any) -> str:
    """Normalize the menu spelling Hermes uses for named providers."""

    return "-".join(str(value or "").strip().lower().replace("_", "-").split())


def _custom_provider_aliases(display_name: Any, provider_key: Any) -> Set[str]:
    """Return normalized durable identities accepted by Hermes custom routes."""

    aliases: Set[str] = set()
    for value in (display_name, provider_key):
        normalized = _normalized_provider_selector(value)
        if not normalized:
            continue
        aliases.add(normalized)
        aliases.add(
            normalized if normalized.startswith("custom:") else f"custom:{normalized}"
        )
        if normalized.startswith("custom:"):
            suffix = normalized.split(":", 1)[1]
            if suffix:
                aliases.update({suffix, f"custom:{normalized}"})
    return aliases


def _provider_enabled(configured: Mapping[str, Any]) -> bool:
    """Interpret Hermes' enabled flag for a modern provider entry."""

    flag = configured.get("enabled", True)
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.strip().lower() not in {"false", "0", "no", "off"}
    return bool(flag)


def _named_provider_config(
    providers: Mapping[str, Any],
    selector: str,
) -> Optional[Mapping[str, Any]]:
    """Find a providers entry by mapping key or normalized display name."""

    target = _normalized_provider_selector(selector)
    for key, configured in providers.items():
        display_name = (
            configured.get("name")
            if isinstance(configured, Mapping) and configured.get("name")
            else key
        )
        if target not in _custom_provider_aliases(display_name, key):
            continue
        if not isinstance(configured, Mapping):
            raise SettingsError("Hermes providers entry must be a mapping")
        if not _provider_enabled(configured):
            continue
        return configured
    return None


def _legacy_provider_config(
    custom_providers: Any,
    selector: str,
) -> Optional[Mapping[str, Any]]:
    """Find a saved list-style custom provider still supported by Hermes."""

    if not isinstance(custom_providers, list):
        return None
    target = _normalized_provider_selector(selector)
    for configured in custom_providers:
        if not isinstance(configured, Mapping):
            continue
        if target in _custom_provider_aliases(
            configured.get("name"),
            configured.get("provider_key"),
        ):
            return configured
    return None


def _configured_provider(
    providers: Mapping[str, Any],
    custom_providers: Any,
    selector: str,
) -> Optional[Mapping[str, Any]]:
    """Resolve the new mapping before Hermes' legacy provider list."""

    configured = _named_provider_config(providers, selector)
    return (
        configured
        if configured is not None
        else _legacy_provider_config(custom_providers, selector)
    )


def _resolved_builtin_provider(provider: str) -> Optional[str]:
    """Return Hermes' canonical built-in identity when its registry accepts it."""

    try:
        from hermes_cli.auth import (  # type: ignore[import-not-found]  # noqa: PLC0415
            resolve_provider,
        )
    except (ImportError, AttributeError):
        return None
    try:
        resolved = _normalized_provider_selector(resolve_provider(provider))
    except Exception:  # noqa: BLE001
        return None
    return resolved or None


def _effective_model_name(model: str, provider: str) -> str:
    """Mirror Hermes' provider-aware model normalization when available."""

    try:
        from hermes_cli.model_normalize import (  # type: ignore[import-not-found]  # noqa: PLC0415
            _AGGREGATOR_PROVIDERS,
            normalize_model_for_provider,
        )
    except (ImportError, AttributeError):
        return model
    if provider in _AGGREGATOR_PROVIDERS:
        return model
    try:
        normalized = normalize_model_for_provider(model, provider)
    except Exception:  # noqa: BLE001
        return model
    return normalized if isinstance(normalized, str) and normalized.strip() else model


def _effective_bare_provider(
    provider: str,
    providers: Mapping[str, Any],
    custom_providers: Any,
) -> Tuple[str, Mapping[str, Any], bool]:
    """Resolve a non-empty, non-custom selector using Hermes' precedence."""

    canonical = _resolved_builtin_provider(provider)
    if canonical == provider:
        return canonical, {}, False
    configured = _configured_provider(providers, custom_providers, provider)
    if configured is not None:
        return "custom", configured, True
    if canonical is not None:
        return canonical, {}, False
    if provider in {"vllm", "ollama", "llamacpp"}:
        return "custom", {}, False
    return provider, {}, False


def _effective_provider_route(
    model: Mapping[str, Any],
    providers: Mapping[str, Any],
    custom_providers: Any = None,
) -> Tuple[str, str, Mapping[str, Any]]:
    """Resolve Hermes' selector into its live provider and endpoint identity."""

    provider_selector = _normalized_provider_selector(model.get("provider"))
    provider = provider_selector
    provider_config: Mapping[str, Any] = {}
    named = False
    if ":" in provider_selector:
        provider_prefix, provider_name = provider_selector.split(":", 1)
        configured_provider = _configured_provider(
            providers,
            custom_providers,
            provider_name,
        )
        if not provider_prefix or not provider_name or configured_provider is None:
            raise SettingsError(
                "Hermes named model.provider must reference providers.<name>",
            )
        provider = "custom"
        provider_config = configured_provider
        named = True
    elif provider_selector not in {"", "custom"}:
        provider, provider_config, named = _effective_bare_provider(
            provider_selector,
            providers,
            custom_providers,
        )
    else:
        configured_provider = providers.get(provider)
        if configured_provider is None and provider == "custom":
            configured_provider = _legacy_provider_config(
                custom_providers,
                provider,
            )
            named = configured_provider is not None
        if configured_provider is not None:
            if not isinstance(configured_provider, Mapping):
                raise SettingsError("Hermes providers entry must be a mapping")
            provider_config = configured_provider
    provider_endpoint = (
        provider_config.get("api")
        or provider_config.get("url")
        or provider_config.get("base_url")
    )
    base_url = normalize_base_url(
        provider_endpoint if named else model.get("base_url") or provider_endpoint,
    )
    return provider, base_url, provider_config


def _effective_max_tokens(
    model: Mapping[str, Any],
    provider_config: Mapping[str, Any],
    environment: HermesEnvironment,
) -> Optional[int]:
    """Resolve Hermes' output allowance in the same precedence order."""

    if environment.max_tokens:
        return int(environment.max_tokens)
    configured = model.get("max_tokens")
    if configured is not None:
        return _strict_int(configured, "Hermes model.max_tokens", minimum=1)
    provider_configured = provider_config.get("max_output_tokens")
    if provider_configured is None:
        provider_configured = provider_config.get("max_tokens")
    if provider_configured is not None:
        return _strict_int(
            provider_configured,
            "Hermes provider max_output_tokens",
            minimum=1,
        )
    return None


def _validate_context_engine(
    context: Mapping[str, Any],
    window_override: int,
) -> None:
    """Require a known boundary for non-default Hermes context engines."""

    context_engine = str(context.get("engine") or "compressor").strip().lower()
    if window_override == 0 and context_engine != "compressor":
        raise SettingsError(
            "Hermes context.engine must be compressor unless an explicit "
            "compression_window_tokens override is configured",
        )


def load_settings(
    *,
    environment: Optional[Environment] = None,
    config_loader: Optional[Callable[[], Any]] = None,
    compressor_class: Optional[Type[Any]] = None,
) -> Settings:
    """Load and validate Hermes plus environment configuration.

    The compression window is obtained from Hermes' real ``ContextCompressor``
    instead of duplicating its version-sensitive threshold arithmetic.
    """

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
    context = _section(raw_config, "context")
    providers = _section(raw_config, "providers")
    configured_model_name = model.get("default")
    if not isinstance(configured_model_name, str) or not configured_model_name.strip():
        raise SettingsError("Hermes model.default must be a non-empty string")
    context_length = _strict_int(
        model.get("context_length"),
        "Hermes model.context_length",
        minimum=1,
    )
    provider, base_url, provider_config = _effective_provider_route(
        model,
        providers,
        raw_config.get("custom_providers"),
    )
    model_name = _effective_model_name(configured_model_name.strip(), provider)
    threshold = _strict_float(
        compression.get("threshold", 0.50),
        "Hermes compression.threshold",
        minimum_exclusive=0.0,
        maximum_inclusive=1.0,
    )
    threshold = _effective_compression_threshold(
        threshold,
        model=model_name,
        provider=provider,
        compression=compression,
    )

    try:
        environment = Environment() if environment is None else environment
        hermes_environment = HermesEnvironment()
    except (TypeError, ValueError) as exc:
        raise SettingsError(str(exc)) from exc

    max_tokens = _effective_max_tokens(model, provider_config, hermes_environment)

    window_override = environment.compression_window_tokens
    _validate_context_engine(context, window_override)
    if window_override == 0:
        compressor = _construct_compressor(
            compressor_class,
            {
                "model": model_name,
                "threshold_percent": threshold,
                "quiet_mode": True,
                "base_url": base_url,
                "config_context_length": context_length,
                "provider": provider,
                "api_mode": str(model.get("api_mode") or ""),
                "max_tokens": max_tokens,
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
        compression_window = window_override

    if compression_window > context_length:
        raise SettingsError(
            "The compression window must not exceed model.context_length",
        )

    fallback_margin = environment.fallback_margin_tokens
    if fallback_margin >= compression_window:
        raise SettingsError(
            "fallback_margin_tokens must be below the compression window",
        )

    return Settings(
        model_name=model_name,
        context_length=context_length,
        compression_window=compression_window,
        fallback_margin_tokens=fallback_margin,
        provider=provider,
        base_url=base_url,
    )
