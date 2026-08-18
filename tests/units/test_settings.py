from __future__ import annotations

import builtins
import sys
import types
from collections.abc import Mapping
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from incontext import settings

base_environment: dict[str, str] = {}
base_config = {
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
    config: Any = base_config,
    compressor: type[Any] = ModernCompressor,
) -> settings.Settings:
    active_environment = base_environment if environment is None else environment
    with patch.dict("os.environ", active_environment, clear=True):
        return settings.load_settings(
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


def test_strict_float_wraps_unrepresentable_integer() -> None:
    """Convert a YAML-sized integer overflow into the settings error contract.

    Python and YAML accept integers much larger than a platform float.  Such a
    configured threshold must fail as ``SettingsError`` like every other
    malformed numeric value instead of leaking ``OverflowError`` from startup.
    """

    with pytest.raises(settings.SettingsError, match="must be numeric"):
        settings._strict_float(10**10_000, "value", minimum_exclusive=0)


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


def test_skelet_environment_prefers_primary_and_converts_fields() -> None:
    with patch.dict(
        "os.environ",
        {
            "INCONTEXT_BACKEND": " custom_backend ",
            "INCONTEXT_FALLBACK_MARGIN_TOKENS": "256",
            "HERMES_DYNAMIC_BUDGET_FALLBACK_MARGIN_TOKENS": "999",
        },
        clear=True,
    ):
        environment = settings.Environment()

    assert environment.backend == "custom_backend"
    assert environment.fallback_margin_tokens == 256
    assert environment.compression_window_tokens == 0


def test_skelet_environment_reads_process_environment() -> None:
    with patch.dict(
        "os.environ",
        {
            "INCONTEXT_FALLBACK_MARGIN_TOKENS": "256",
            "INCONTEXT_COMPRESSION_WINDOW_TOKENS": "50000",
        },
        clear=True,
    ):
        environment = settings.Environment()

    assert environment.backend == "vllm"
    assert environment.fallback_margin_tokens == 256
    assert environment.compression_window_tokens == 50_000


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


def test_normalized_model_thresholds_skips_unrepresentable_integer() -> None:
    """Ignore per-model thresholds that cannot be represented as finite floats.

    A single arbitrary-precision YAML integer must not abort loading otherwise
    valid model overrides; it belongs to the same rejected category as NaN and
    infinity and is therefore omitted from the normalized mapping.
    """

    assert settings._normalized_model_thresholds({"huge": 10**10_000}) == {}


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
        model_name="qwen-test",
        context_length=65_536,
        compression_window=55_705,
        fallback_margin_tokens=1024,
        provider="custom",
        base_url="https://inference.example/v1",
    )
    call = ModernCompressor.calls[-1]
    assert call["model"] == "qwen-test"
    assert call["threshold_percent"] == 0.5
    assert call["quiet_mode"] is True
    assert call["max_tokens"] is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (" relative/path/ ", "relative/path"),
        ("https://EXAMPLE.test:443/v1/", "https://example.test/v1"),
        ("http://EXAMPLE.test:8080/v1/", "http://example.test:8080/v1"),
        ("http://[::1]:80/v1/", "http://[::1]/v1"),
        ("https://user:pass@example.test/v1", "https://user:pass@example.test/v1"),
        ("https://example.test:notaport/v1", "https://example.test:notaport/v1"),
    ],
)
def test_base_url_normalization_preserves_route_identity(
    value: Any,
    expected: str,
) -> None:
    """Canonicalize only URL spellings that preserve endpoint identity.

    Hermes and its HTTP clients may lowercase hosts, remove default ports, or
    append a trailing slash.  Those representations must compare equal, while
    relative, credential-bearing, and malformed authorities stay untouched so
    normalization never invents a different route.
    """

    assert settings.normalize_base_url(value) == expected


def test_load_settings_normalizes_provider_like_hermes() -> None:
    """Store the canonical provider and endpoint exposed by live Hermes.

    Hermes strips and lowercases provider IDs before middleware dispatch, and
    its HTTP client canonicalizes base URLs.  Applying the same normalization
    while constructing Settings prevents mixed-case configuration from making
    the primary route look like an unrelated fallback.
    """

    config = {
        **base_config,
        "model": {
            **base_config["model"],
            "provider": " Custom ",
            "base_url": "https://INFERENCE.EXAMPLE:443/v1/",
        },
    }

    result = load(config=config)

    assert result.provider == "custom"
    assert result.base_url == "https://inference.example/v1"


def test_load_settings_passes_modern_compression_options() -> None:
    ModernCompressor.calls.clear()
    config = {
        **base_config,
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


def test_load_settings_uses_hermes_model_specific_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derive the window from the same effective threshold as Hermes itself.

    Hermes applies installed model policies before constructing its live
    compressor (for example, Trinity and Codex autoraises).  Passing only the
    raw global config would make incontext use a different compression boundary
    even though both instantiate the same ``ContextCompressor`` class.
    """

    class ThresholdAwareCompressor:
        def __init__(
            self,
            *,
            threshold_percent: float,
            config_context_length: int,
            **options: Any,
        ) -> None:
            del options
            self.context_length = config_context_length
            self.threshold_tokens = int(
                config_context_length * threshold_percent,
            )

    calls: list[tuple[float, str, str, Mapping[str, Any]]] = []

    def effective(
        configured: float,
        *,
        model: str,
        provider: str,
        compression: Mapping[str, Any],
    ) -> float:
        calls.append((configured, model, provider, compression))
        return 0.75

    monkeypatch.setattr(settings, "_effective_compression_threshold", effective)

    result = load(compressor=ThresholdAwareCompressor)

    assert result.compression_window == 49_152
    assert calls == [
        (0.5, "qwen-test", "custom", base_config["compression"]),
    ]


def test_effective_threshold_delegates_to_installed_hermes_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse Hermes helpers including opt-out and route classification inputs.

    The policy is version-sensitive and contains distinct Codex/Spark branches.
    This test supplies the installed-module contract and verifies incontext
    forwards model, provider, opt-out state, and the final autoraise verdict
    rather than reimplementing those rules.
    """

    agent = types.ModuleType("agent")
    agent.__path__ = []  # type: ignore[attr-defined]
    agent_init = types.ModuleType("agent.agent_init")
    auxiliary = types.ModuleType("agent.auxiliary_client")
    calls: list[tuple[Any, ...]] = []

    def model_threshold(
        model: str,
        provider: str,
        *,
        allow_codex_gpt55_autoraise: bool,
    ) -> float:
        calls.append(("threshold", model, provider, allow_codex_gpt55_autoraise))
        return 0.7

    def resolve(
        configured: float,
        override: float,
        *,
        model: str,
        is_codex_autoraise: bool,
    ) -> tuple[float, None]:
        calls.append(("resolve", configured, override, model, is_codex_autoraise))
        return override, None

    agent_init._resolve_compression_threshold = resolve  # type: ignore[attr-defined]
    auxiliary._compression_threshold_for_model = model_threshold  # type: ignore[attr-defined]
    auxiliary._is_codex_gpt54_or_gpt55 = (  # type: ignore[attr-defined]
        lambda model, provider: False
    )
    auxiliary._is_codex_spark = lambda model, provider: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.agent_init", agent_init)
    monkeypatch.setitem(
        sys.modules,
        "agent.auxiliary_client",
        auxiliary,
    )

    result = settings._effective_compression_threshold(
        0.5,
        model="spark",
        provider="openai-codex",
        compression={"codex_gpt55_autoraise": False},
    )

    assert result == 0.7
    assert calls == [
        ("threshold", "spark", "openai-codex", False),
        ("resolve", 0.5, 0.7, "spark", True),
    ]


@pytest.mark.parametrize(
    ("model_override", "expected"),
    [(0.85, 0.85), (None, 0.5)],
)
def test_effective_threshold_uses_legacy_hermes_policy(
    monkeypatch: pytest.MonkeyPatch,
    model_override: float | None,
    expected: float,
) -> None:
    """Mirror the threshold contract exposed by Hermes 2026.7.1.

    That release already supplied its model-specific threshold function but
    predates the shared resolver and expanded Codex/Spark classifiers.  The
    returned override must therefore replace the global value directly, while
    a ``None`` result retains the configured threshold, so incontext and the
    installed compressor always derive the same boundary.
    """

    agent = types.ModuleType("agent")
    agent.__path__ = []  # type: ignore[attr-defined]
    auxiliary = types.ModuleType("agent.auxiliary_client")
    calls: list[tuple[Any, ...]] = []

    def model_threshold(
        model: str,
        provider: str,
        *,
        allow_codex_gpt55_autoraise: bool,
    ) -> float | None:
        calls.append((model, provider, allow_codex_gpt55_autoraise))
        return model_override

    auxiliary._compression_threshold_for_model = model_threshold  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.delitem(sys.modules, "agent.agent_init", raising=False)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary)

    result = settings._effective_compression_threshold(
        0.5,
        model="gpt-5.5",
        provider="openai-codex",
        compression={"codex_gpt55_autoraise": True},
    )

    assert result == expected
    assert calls == [("gpt-5.5", "openai-codex", True)]


@pytest.mark.parametrize("failure_stage", ["threshold", "resolver"])
def test_effective_threshold_falls_back_when_hermes_policy_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """Match Hermes' best-effort behavior for private policy failures.

    These helpers are intentionally reused from the installed Hermes version.
    If a future compatible release raises while resolving optional model policy,
    incontext must retain the validated global threshold just as Hermes does.
    """

    agent = types.ModuleType("agent")
    agent.__path__ = []  # type: ignore[attr-defined]
    agent_init = types.ModuleType("agent.agent_init")
    auxiliary = types.ModuleType("agent.auxiliary_client")

    def model_policy(*args: Any, **kwargs: Any) -> float:
        if failure_stage == "threshold":
            raise RuntimeError("policy unavailable")
        return 0.9

    def resolver(*args: Any, **kwargs: Any) -> tuple[float, None]:
        if failure_stage == "resolver":
            raise RuntimeError("resolver unavailable")
        return 0.9, None

    agent_init._resolve_compression_threshold = resolver  # type: ignore[attr-defined]
    auxiliary._compression_threshold_for_model = model_policy  # type: ignore[attr-defined]
    auxiliary._is_codex_gpt54_or_gpt55 = (  # type: ignore[attr-defined]
        lambda model, provider: False
    )
    auxiliary._is_codex_spark = lambda model, provider: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.agent_init", agent_init)
    monkeypatch.setitem(
        sys.modules,
        "agent.auxiliary_client",
        auxiliary,
    )

    assert (
        settings._effective_compression_threshold(
            0.6,
            model="qwen",
            provider="custom",
            compression={},
        )
        == 0.6
    )


def test_load_settings_reserves_hermes_configured_output_budget() -> None:
    """Reconstruct the same compression boundary as the live Hermes agent.

    Hermes passes ``model.max_tokens`` to ``ContextCompressor`` because the
    configured completion allowance reduces the safe pre-compression input
    threshold.  Dropping that value here produces a larger, fictitious window
    and lets incontext budget requests beyond Hermes' actual boundary.
    """

    ModernCompressor.calls.clear()
    config = {
        **base_config,
        "model": {**base_config["model"], "max_tokens": "8192"},
    }

    load(config=config)

    assert ModernCompressor.calls[-1]["max_tokens"] == 8192


@pytest.mark.parametrize("value", [True, 0, -1, "invalid"])
def test_load_settings_rejects_invalid_hermes_output_budget(value: Any) -> None:
    """Reject malformed output reserves before deriving a false window.

    Values that Hermes cannot interpret as a positive token allowance must not
    be silently converted into a different compression threshold by incontext.
    The validation mirrors the strict integer rules used for context length.
    """

    config = {
        **base_config,
        "model": {**base_config["model"], "max_tokens": value},
    }

    with pytest.raises(settings.SettingsError, match=r"model\.max_tokens"):
        load(config=config)


def test_load_settings_explicit_window_avoids_compressor_construction() -> None:
    class MustNotRun:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("compressor should not run")

    result = load(
        environment={
            **base_environment,
            "INCONTEXT_COMPRESSION_WINDOW_TOKENS": "50000",
        },
        compressor=MustNotRun,
    )
    assert result.compression_window == 50_000


def test_load_settings_prefers_new_environment_names() -> None:
    environment = {
        "INCONTEXT_BACKEND": "other_backend",
        "INCONTEXT_FALLBACK_MARGIN_TOKENS": "0",
        "HERMES_DYNAMIC_BUDGET_FALLBACK_MARGIN_TOKENS": "999",
    }
    with patch.dict("os.environ", environment, clear=True):
        resolved_environment = settings.Environment()
        result = settings.load_settings(
            environment=resolved_environment,
            config_loader=lambda: base_config,
            compressor_class=ModernCompressor,
        )
    assert resolved_environment.backend == "other_backend"
    assert result.fallback_margin_tokens == 0


def test_load_settings_supports_legacy_environment_names() -> None:
    environment = {
        "HERMES_DYNAMIC_BUDGET_FALLBACK_MARGIN_TOKENS": "100",
    }
    result = load(environment=environment)
    assert result.fallback_margin_tokens == 100


def test_load_settings_uses_default_components_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "_load_hermes_components",
        lambda: (lambda: base_config, ModernCompressor),
    )
    with patch.dict("os.environ", base_environment, clear=True):
        assert settings.load_settings().compression_window == 55_705


def test_load_settings_fills_only_missing_default_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "_load_hermes_components",
        lambda: (lambda: base_config, ModernCompressor),
    )
    with patch.dict("os.environ", base_environment, clear=True):
        assert (
            settings.load_settings(
                config_loader=lambda: base_config,
            ).compression_window
            == 55_705
        )
        assert (
            settings.load_settings(
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
        return base_config

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
    config = dict(base_config)
    config[section] = []
    with pytest.raises(settings.SettingsError, match=section):
        load(config=config)


@pytest.mark.parametrize("model_name", [None, "", "   ", 123])
def test_load_settings_requires_model_name(model_name: Any) -> None:
    config = {
        **base_config,
        "model": {**base_config["model"], "default": model_name},
    }
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
    environment = {
        **base_environment,
        "INCONTEXT_COMPRESSION_WINDOW_TOKENS": "65537",
    }
    with pytest.raises(settings.SettingsError, match="must not exceed"):
        load(environment=environment)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"INCONTEXT_BACKEND": "  "}, "must not be blank"),
        (
            {**base_environment, "INCONTEXT_FALLBACK_MARGIN_TOKENS": "-1"},
            "at least 0",
        ),
        (
            {**base_environment, "INCONTEXT_COMPRESSION_WINDOW_TOKENS": "0"},
            "positive",
        ),
        (
            {**base_environment, "INCONTEXT_FALLBACK_MARGIN_TOKENS": "55705"},
            "below",
        ),
    ],
)
def test_load_settings_rejects_unsafe_runtime_environment(
    environment: Mapping[str, str],
    message: str,
) -> None:
    with pytest.raises(settings.SettingsError, match=message):
        load(environment=environment)
