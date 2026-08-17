from __future__ import annotations

import builtins
import types
from collections.abc import Mapping
from typing import Any, ClassVar

import pytest

from incontext import settings

BASE_ENV = {settings.TOKENIZER_URL_ENV: "https://inference.example/tokenize"}
BASE_CONFIG = {
    "model": {
        "default": "qwen-test",
        "context_length": 65_536,
        "base_url": "https://inference.example/v1",
        "provider": "custom",
        "api_mode": "chat_completions",
    },
    "compression": {"threshold": 0.5},
}


class ModernCompressor:
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(
        self,
        model: str,
        threshold_percent: float,
        quiet_mode: bool,
        base_url: str,
        config_context_length: int,
        provider: str,
        api_mode: str,
        max_tokens: int | None,
        model_thresholds: dict[str, float],
        threshold_tokens_cap: Any,
    ) -> None:
        self.calls.append(dict(locals()))
        self.context_length = config_context_length
        self.threshold_tokens = int(config_context_length * 0.85)


def load(
    *,
    environment: Mapping[str, str] | None = None,
    config: Any = BASE_CONFIG,
    compressor: type[Any] = ModernCompressor,
) -> settings.Settings:
    return settings.load_settings(
        environ=BASE_ENV if environment is None else environment,
        config_loader=lambda: config,
        compressor_class=compressor,
    )


def test_settings_remains_slotted_on_python_38() -> None:
    assert not hasattr(load(), "__dict__")


@pytest.mark.parametrize(("value", "expected"), [(1, 1), (" 42 ", 42), (0, 0)])
def test_strict_int_accepts_exact_integers(value: Any, expected: int) -> None:
    assert settings._strict_int(value, "value", minimum=0) == expected


@pytest.mark.parametrize("value", [True, 1.5, None, object()])
def test_strict_int_rejects_non_integers(value: Any) -> None:
    with pytest.raises(settings.SettingsError, match="must be an integer"):
        settings._strict_int(value, "value", minimum=0)


def test_strict_int_rejects_malformed_string() -> None:
    with pytest.raises(settings.SettingsError, match="must be an integer"):
        settings._strict_int("1.5", "value", minimum=0)


@pytest.mark.parametrize(
    ("minimum", "message"),
    [(1, "must be positive"), (3, "must be at least 3")],
)
def test_strict_int_enforces_minimum(minimum: int, message: str) -> None:
    with pytest.raises(settings.SettingsError, match=message):
        settings._strict_int(0, "value", minimum=minimum)


@pytest.mark.parametrize("value", [1, 0.5, "0.25"])
def test_strict_float_accepts_finite_numbers(value: Any) -> None:
    assert settings._strict_float(
        value,
        "value",
        minimum_exclusive=0,
        maximum_inclusive=1,
    ) == float(value)


@pytest.mark.parametrize("value", [True, None, object()])
def test_strict_float_rejects_non_numeric_types(value: Any) -> None:
    with pytest.raises(settings.SettingsError, match="must be numeric"):
        settings._strict_float(value, "value", minimum_exclusive=0)


def test_strict_float_rejects_malformed_string() -> None:
    with pytest.raises(settings.SettingsError, match="must be numeric"):
        settings._strict_float("invalid", "value", minimum_exclusive=0)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_strict_float_rejects_non_positive_or_non_finite(value: float) -> None:
    with pytest.raises(settings.SettingsError, match="must be greater"):
        settings._strict_float(value, "value", minimum_exclusive=0)


def test_strict_float_enforces_maximum() -> None:
    with pytest.raises(settings.SettingsError, match="must not exceed"):
        settings._strict_float(
            1.01,
            "value",
            minimum_exclusive=0,
            maximum_inclusive=1,
        )


def test_section_accepts_missing_and_none() -> None:
    assert settings._section({}, "model") == {}
    assert settings._section({"model": None}, "model") == {}


def test_section_rejects_non_mapping() -> None:
    with pytest.raises(settings.SettingsError, match="must be a mapping"):
        settings._section({"model": []}, "model")


def test_environment_value_prefers_primary_and_skips_blanks() -> None:
    environment = {"NEW": " value ", "OLD": "legacy"}
    assert settings._environment_value(environment, "NEW", "OLD") == "value"
    environment["NEW"] = "  "
    assert settings._environment_value(environment, "NEW", "OLD") == "legacy"
    assert settings._environment_value({}, "NEW", default="fallback") == "fallback"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "must contain the vLLM"),
        ("ftp://example.test/tokenize", "HTTP"),
        ("https:///tokenize", "HTTP"),
        ("https://user:pass@example.test/tokenize", "credentials"),
        ("https://example.test/tokenize#fragment", "fragment"),
    ],
)
def test_validated_http_url_rejects_unsafe_values(
    value: str | None,
    message: str,
) -> None:
    with pytest.raises(settings.SettingsError, match=message):
        settings._validated_http_url(value, "URL")


@pytest.mark.parametrize(
    "value",
    ["http://127.0.0.1:8000/tokenize", "https://example.test/tokenize?mode=1"],
)
def test_validated_http_url_accepts_http_urls(value: str) -> None:
    assert settings._validated_http_url(value, "URL") == value


def test_normalized_model_thresholds_keeps_only_finite_numbers() -> None:
    assert settings._normalized_model_thresholds(
        {
            "qwen": 0.75,
            123: 1,
            "bool": True,
            "string": "0.5",
            "nan": float("nan"),
        },
    ) == {"qwen": 0.75, "123": 1.0}
    assert settings._normalized_model_thresholds([]) == {}


def test_construct_compressor_filters_unknown_keywords() -> None:
    class Legacy:
        def __init__(self, model: str) -> None:
            self.model = model

    instance = settings._construct_compressor(
        Legacy,
        {"model": "qwen", "new_option": True},
    )
    assert instance.model == "qwen"


def test_construct_compressor_passes_all_keywords_to_flexible_class() -> None:
    class Flexible:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    instance = settings._construct_compressor(Flexible, {"model": "qwen"})
    assert instance.kwargs == {"model": "qwen"}


def test_construct_compressor_wraps_initialization_failure() -> None:
    class Broken:
        def __init__(self, **kwargs: Any) -> None:
            raise ValueError("boom")

    with pytest.raises(settings.SettingsError, match="initialization failed"):
        settings._construct_compressor(Broken, {})


def test_load_settings_uses_real_compressor_threshold() -> None:
    ModernCompressor.calls.clear()
    result = load()
    assert result == settings.Settings(
        context_length=65_536,
        compression_window=55_705,
        tokenizer_url="https://inference.example/tokenize",
        tokenizer_user_agent=settings.DEFAULT_USER_AGENT,
        tokenizer_timeout_seconds=30.0,
        fallback_margin_tokens=1024,
    )
    call = ModernCompressor.calls[-1]
    assert call["model"] == "qwen-test"
    assert call["threshold_percent"] == 0.5
    assert call["quiet_mode"] is True
    assert call["max_tokens"] is None


def test_load_settings_passes_modern_compression_options() -> None:
    ModernCompressor.calls.clear()
    config = {
        **BASE_CONFIG,
        "compression": {
            "threshold": 0.6,
            "threshold_tokens": 50_000,
            "model_thresholds": {"qwen": 0.7, "ignored": "0.9"},
        },
    }
    load(config=config)
    call = ModernCompressor.calls[-1]
    assert call["model_thresholds"] == {"qwen": 0.7}
    assert call["threshold_tokens_cap"] == 50_000


def test_load_settings_explicit_window_avoids_compressor_construction() -> None:
    class MustNotRun:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("compressor should not run")

    result = load(
        environment={**BASE_ENV, settings.COMPRESSION_WINDOW_ENV: "50000"},
        compressor=MustNotRun,
    )
    assert result.compression_window == 50_000


def test_load_settings_prefers_new_environment_names() -> None:
    environment = {
        settings.TOKENIZER_URL_ENV: "https://new.example/tokenize",
        settings.LEGACY_TOKENIZER_URL_ENV: "https://old.example/tokenize",
        settings.TOKENIZER_USER_AGENT_ENV: "new-agent",
        settings.LEGACY_TOKENIZER_USER_AGENT_ENV: "old-agent",
        settings.TOKENIZER_TIMEOUT_ENV: "12.5",
        settings.LEGACY_TOKENIZER_TIMEOUT_ENV: "99",
        settings.FALLBACK_MARGIN_ENV: "0",
        settings.LEGACY_FALLBACK_MARGIN_ENV: "999",
    }
    result = load(environment=environment)
    assert result.tokenizer_url == "https://new.example/tokenize"
    assert result.tokenizer_user_agent == "new-agent"
    assert result.tokenizer_timeout_seconds == 12.5
    assert result.fallback_margin_tokens == 0


def test_load_settings_supports_legacy_environment_names() -> None:
    environment = {
        settings.LEGACY_TOKENIZER_URL_ENV: "https://old.example/tokenize",
        settings.LEGACY_TOKENIZER_USER_AGENT_ENV: "old-agent",
        settings.LEGACY_TOKENIZER_TIMEOUT_ENV: "5",
        settings.LEGACY_FALLBACK_MARGIN_ENV: "100",
    }
    result = load(environment=environment)
    assert result.tokenizer_url == "https://old.example/tokenize"
    assert result.tokenizer_user_agent == "old-agent"
    assert result.tokenizer_timeout_seconds == 5
    assert result.fallback_margin_tokens == 100


def test_load_settings_uses_default_components_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "_load_hermes_components",
        lambda: (lambda: BASE_CONFIG, ModernCompressor),
    )
    assert settings.load_settings(environ=BASE_ENV).compression_window == 55_705


def test_load_settings_fills_only_missing_default_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "_load_hermes_components",
        lambda: (lambda: BASE_CONFIG, ModernCompressor),
    )
    assert (
        settings.load_settings(
            environ=BASE_ENV,
            config_loader=lambda: BASE_CONFIG,
        ).compression_window
        == 55_705
    )
    assert (
        settings.load_settings(
            environ=BASE_ENV,
            compressor_class=ModernCompressor,
        ).compression_window
        == 55_705
    )


def test_load_hermes_components_imports_expected_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_package = types.ModuleType("agent")
    agent_package.__path__ = []  # type: ignore[attr-defined]
    compressor_module = types.ModuleType("agent.context_compressor")
    compressor_module.ContextCompressor = ModernCompressor  # type: ignore[attr-defined]
    cli_package = types.ModuleType("hermes_cli")
    cli_package.__path__ = []  # type: ignore[attr-defined]
    config_module = types.ModuleType("hermes_cli.config")

    def loader() -> dict[str, Any]:
        return BASE_CONFIG

    config_module.load_config = loader  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "agent", agent_package)
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent.context_compressor",
        compressor_module,
    )
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli", cli_package)
    monkeypatch.setitem(
        __import__("sys").modules,
        "hermes_cli.config",
        config_module,
    )
    assert settings._load_hermes_components() == (loader, ModernCompressor)


def test_load_hermes_components_reports_missing_hermes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def rejecting_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "agent.context_compressor":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)
    with pytest.raises(settings.SettingsError, match="must run inside the Hermes"):
        settings._load_hermes_components()


@pytest.mark.parametrize("config", [None, {}])
def test_load_settings_handles_empty_config(config: Any) -> None:
    with pytest.raises(settings.SettingsError, match=r"model\.default"):
        load(config=config)


def test_load_settings_rejects_non_mapping_config() -> None:
    with pytest.raises(settings.SettingsError, match="configuration must be a mapping"):
        load(config=[])


@pytest.mark.parametrize("section", ["model", "compression"])
def test_load_settings_rejects_non_mapping_sections(section: str) -> None:
    config = dict(BASE_CONFIG)
    config[section] = []
    with pytest.raises(settings.SettingsError, match=section):
        load(config=config)


@pytest.mark.parametrize("model_name", [None, "", "   ", 123])
def test_load_settings_requires_model_name(model_name: Any) -> None:
    config = {**BASE_CONFIG, "model": {**BASE_CONFIG["model"], "default": model_name}}
    with pytest.raises(settings.SettingsError, match=r"model\.default"):
        load(config=config)


def test_load_settings_rejects_context_mismatch() -> None:
    class Mismatch(ModernCompressor):
        def __init__(self, **kwargs: Any) -> None:
            self.context_length = 32_000
            self.threshold_tokens = 20_000

    with pytest.raises(settings.SettingsError, match="does not match"):
        load(compressor=Mismatch)


def test_load_settings_rejects_window_above_context() -> None:
    environment = {**BASE_ENV, settings.COMPRESSION_WINDOW_ENV: "65537"}
    with pytest.raises(settings.SettingsError, match="must not exceed"):
        load(environment=environment)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({}, settings.TOKENIZER_URL_ENV),
        ({**BASE_ENV, settings.TOKENIZER_TIMEOUT_ENV: "0"}, "greater than"),
        ({**BASE_ENV, settings.FALLBACK_MARGIN_ENV: "55705"}, "below"),
    ],
)
def test_load_settings_rejects_unsafe_runtime_environment(
    environment: Mapping[str, str],
    message: str,
) -> None:
    with pytest.raises(settings.SettingsError, match=message):
        load(environment=environment)
