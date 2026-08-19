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
        "providers": {"custom": {"api": "https://fallback.invalid/v1"}},
    }

    result = load(config=config)

    assert result.provider == "custom"
    assert result.base_url == "https://inference.example/v1"


def test_load_settings_uses_effective_named_custom_route() -> None:
    """Scope budgeting to the route Hermes actually sends over the wire.

    Hermes resolves ``model.provider: custom:local`` through ``providers.local``
    before invoking request middleware: the live context contains provider
    ``custom`` and the provider entry's API URL.  Keeping the selector literal
    and an empty raw ``model.base_url`` makes both public and auxiliary
    budgeting reject the configured primary route entirely.
    """

    config = {
        **base_config,
        "model": {
            "default": "qwen-test",
            "provider": "custom:local",
            "context_length": 65_536,
        },
        "providers": {
            "local": {
                "api": "https://INFERENCE.EXAMPLE:443/v1/",
                "default_model": "qwen-test",
            },
        },
    }

    result = load(config=config)

    assert result.provider == "custom"
    assert result.base_url == "https://inference.example/v1"


@pytest.mark.parametrize(
    ("selector", "providers"),
    [
        (
            "local",
            {
                "local": {
                    "api": "https://inference.example/v1",
                    "default_model": "qwen-test",
                },
            },
        ),
        (
            "custom:local-display",
            {
                "unrelated": {"api": "https://unrelated.invalid/v1"},
                "local-key": {
                    "name": "Local Display",
                    "api": "https://inference.example/v1",
                    "default_model": "qwen-test",
                },
            },
        ),
    ],
)
def test_load_settings_matches_all_hermes_named_provider_selectors(
    selector: str,
    providers: Mapping[str, Any],
) -> None:
    """Scope budgeting to the canonical runtime identity of a named provider.

    Hermes accepts a providers mapping key, its normalized display name, and
    the ``custom:<name>`` menu spelling.  All resolve to live provider
    ``custom`` before middleware and auxiliary builders run.  Retaining a
    selector literal makes both route guards reject the actual primary route.
    """

    result = load(
        config={
            **base_config,
            "model": {
                "default": "qwen-test",
                "provider": selector,
                "context_length": 65_536,
            },
            "providers": providers,
        },
    )

    assert result.provider == "custom"
    assert result.base_url == "https://inference.example/v1"


@pytest.mark.parametrize("selector", ["custom:local-vllm", "custom:edge_key"])
def test_load_settings_resolves_legacy_custom_provider_route(selector: str) -> None:
    """Honor the list-style custom-provider route still resolved by Hermes.

    Hermes accepts legacy ``custom_providers`` entries by normalized display
    name or ``provider_key`` and exposes either as provider ``custom`` plus the
    entry endpoint.  Ignoring that persisted schema rejects a valid profile or
    makes route guards miss every request; its output allowance must also reach
    the reconstructed ``ContextCompressor`` boundary.
    """

    ModernCompressor.calls.clear()
    result = load(
        config={
            "model": {
                "default": "qwen-test",
                "provider": selector,
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            "custom_providers": [
                None,
                {"name": "Unrelated", "base_url": "https://unused.invalid/v1"},
                {
                    "name": "Local vLLM",
                    "provider_key": "edge_key",
                    "base_url": "https://INFERENCE.EXAMPLE:443/v1/",
                    "model": "qwen-test",
                    "max_output_tokens": 2048,
                },
            ],
        },
    )

    assert result.provider == "custom"
    assert result.base_url == "https://inference.example/v1"
    assert ModernCompressor.calls[-1]["max_tokens"] == 2048


@pytest.mark.parametrize(
    "provider_sections",
    [
        {
            "providers": {
                "custom:edge": {"api": "https://inference.example/v1"},
            },
        },
        {
            "custom_providers": [
                {
                    "name": "Edge Display",
                    "provider_key": "custom:edge",
                    "base_url": "https://inference.example/v1",
                },
            ],
        },
    ],
)
def test_prefixed_custom_identity_resolves_in_both_provider_schemas(
    provider_sections: Mapping[str, Any],
) -> None:
    """Accept the durable prefixed custom identity understood by Hermes.

    Hermes includes ``custom:edge`` and its suffix in the alias set for both
    modern mapping keys and legacy ``provider_key`` values.  Stripping the
    request prefix without canonicalizing the stored identity makes a valid
    live route fail plugin registration.
    """

    result = load(
        config={
            "model": {
                "default": "qwen-test",
                "provider": "custom:edge",
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            **provider_sections,
        },
    )

    assert result.provider == "custom"
    assert result.base_url == "https://inference.example/v1"
    assert "" not in settings._custom_provider_aliases("custom:", "")


@pytest.mark.parametrize("disabled", [False, "off", 0])
def test_disabled_modern_provider_falls_through_to_legacy_entry(
    monkeypatch: pytest.MonkeyPatch,
    disabled: Any,
) -> None:
    """Follow Hermes when a modern provider entry is explicitly disabled.

    Hermes excludes disabled ``providers`` records from route resolution and
    can still resolve a same-name legacy entry.  Selecting the disabled record
    first gives incontext a different endpoint from the running agent and
    silently disables exact budgeting on the valid legacy route.  Boolean,
    string, and truth-value forms must follow Hermes' compatibility parser.
    """

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    config_module = types.ModuleType("hermes_cli.config")

    def is_provider_enabled(configured: dict[str, Any]) -> bool:
        flag = configured.get("enabled", True)
        if isinstance(flag, str):
            return flag.strip().lower() not in {"false", "0", "no", "off"}
        return bool(flag)

    config_module.is_provider_enabled = is_provider_enabled  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config_module)

    result = load(
        config={
            "model": {
                "default": "qwen-test",
                "provider": "custom:edge",
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            "providers": {
                "edge": {
                    "enabled": disabled,
                    "api": "https://disabled.example/v1",
                },
            },
            "custom_providers": [
                {
                    "name": "Edge",
                    "base_url": "https://live.example/v1",
                },
            ],
        },
    )

    assert result.base_url == "https://live.example/v1"


def test_older_hermes_does_not_apply_a_future_provider_enabled_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follow the installed Hermes version instead of cloning newer policy.

    Releases predating ``hermes_cli.config.is_provider_enabled`` treat an
    ``enabled`` key as inert provider metadata and still route through that
    entry.  Locally reimplementing a newer truthiness rule would make
    incontext scope itself to a legacy fallback while the installed agent uses
    the modern endpoint, silently disabling budgeting on the live request.
    """

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.delitem(sys.modules, "hermes_cli.config", raising=False)

    result = load(
        config={
            "model": {
                "default": "qwen-test",
                "provider": "custom:edge",
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            "providers": {
                "edge": {
                    "enabled": False,
                    "api": "https://version-active.example/v1",
                },
            },
            "custom_providers": [
                {"name": "Edge", "base_url": "https://legacy.example/v1"},
            ],
        },
    )

    assert result.base_url == "https://version-active.example/v1"


def test_provider_enabled_fails_open_when_installed_helper_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep route discovery usable across a broken private Hermes helper.

    The version-owned enabled parser is optional integration code.  If an
    additive Hermes change makes it reject the copied mapping, incontext must
    retain the provider as older releases did rather than silently switching
    to another endpoint before the agent itself resolves the route.
    """

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    config_module = types.ModuleType("hermes_cli.config")

    def fail(configured: dict[str, Any]) -> bool:
        del configured
        raise RuntimeError("private helper changed")

    config_module.is_provider_enabled = fail  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config_module)

    assert settings._provider_enabled({"enabled": False}) is True


def test_legacy_custom_provider_lookup_returns_none_without_a_match() -> None:
    """Do not attach an unrelated persisted endpoint to the primary route.

    The legacy list can contain several valid providers.  Exhausting it
    without a matching normalized name must leave resolution to Hermes' normal
    built-in path; selecting the last unrelated entry would scope the vLLM
    tokenizer and context window to the wrong external destination.
    """

    assert (
        settings._legacy_provider_config(
            [{"name": "Unrelated", "base_url": "https://unused.invalid/v1"}],
            "missing",
        )
        is None
    )


@pytest.mark.parametrize(
    ("provider_entry", "model_base_url"),
    [
        ({"api": "https://inference.example/v1"}, "https://stale.invalid/v1"),
        ({"url": "https://inference.example/v1"}, None),
        ({"base_url": "https://inference.example/v1"}, None),
    ],
)
def test_named_provider_uses_hermes_effective_endpoint(
    provider_entry: Mapping[str, Any],
    model_base_url: str | None,
) -> None:
    """Use the endpoint selected by Hermes' named-provider resolver.

    Once a named entry is selected, Hermes accepts its ``api``, ``url``, and
    ``base_url`` aliases and does not revive a stale model-level URL.  Diverging
    either disables budgeting on the primary route or removes the URL guard and
    permits the primary tokenizer on an unrelated custom fallback.
    """

    model: dict[str, Any] = {
        "default": "qwen-test",
        "provider": "custom:local",
        "context_length": 65_536,
    }
    if model_base_url is not None:
        model["base_url"] = model_base_url

    result = load(
        config={
            **base_config,
            "model": model,
            "providers": {
                "local": {
                    **provider_entry,
                    "default_model": "qwen-test",
                },
            },
        },
    )

    assert result.base_url == "https://inference.example/v1"


def test_named_provider_uses_hermes_camelcase_base_url_alias() -> None:
    """Retain endpoint scoping for Hermes-normalized provider fields.

    Hermes accepts ``baseUrl`` in hand-written modern entries and maps it to
    ``base_url`` before runtime resolution.  Dropping that endpoint leaves only
    the shared provider label ``custom``, allowing a same-model fallback to be
    budgeted with the primary tokenizer and context window.
    """

    result = load(
        config={
            "model": {
                "default": "qwen-test",
                "provider": "custom:edge",
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            "providers": {
                "edge": {
                    "baseUrl": "https://INFERENCE.EXAMPLE:443/v1/",
                    "defaultModel": "qwen-test",
                },
            },
        },
    )

    assert result.base_url == "https://inference.example/v1"


def test_legacy_entry_precedes_compatibility_normalized_modern_base_url() -> None:
    """Mirror Hermes' two-stage lookup for camelCase-only modern entries.

    The direct modern fast path does not read ``baseUrl``.  Hermes next builds
    its compatibility list with persisted legacy entries first, so a matching
    legacy route wins before the modern record is camelCase-normalized.  Taking
    modern ``baseUrl`` immediately gives incontext a different endpoint guard
    from the agent that will send the request.
    """

    result = load(
        config={
            "model": {
                "default": "qwen-test",
                "provider": "custom:edge",
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            "providers": {
                "edge": {"baseUrl": "https://modern.example/v1"},
            },
            "custom_providers": [
                {"name": "Edge", "base_url": "https://legacy.example/v1"},
            ],
        },
    )

    assert result.base_url == "https://legacy.example/v1"


@pytest.mark.parametrize(
    "providers",
    [
        {"edge": "not-a-mapping"},
        {"unrelated": {"baseUrl": "https://unrelated.example/v1"}},
        {"edge": {"baseUrl": "not-a-url"}},
    ],
)
def test_compatibility_provider_lookup_ignores_unusable_entries(
    providers: Mapping[str, Any],
) -> None:
    """Return no route for entries Hermes' compatibility view cannot expose.

    CamelCase normalization is a fallback after direct and legacy lookup, not
    permission to accept malformed mappings, unrelated identities, or invalid
    URLs.  Treating any of those as selected would remove or corrupt endpoint
    isolation for the tokenizer-backed budget.
    """

    assert settings._compatible_modern_provider_config(providers, "edge") is None


def test_compatibility_provider_lookup_skips_version_disabled_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply installed Hermes enabled policy in the compatibility fallback.

    A camelCase-only endpoint bypasses the direct provider fast path and is
    normalized later.  When the installed release supports disabling entries,
    that later path must skip it too or incontext revives a route Hermes hides.
    """

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    config_module = types.ModuleType("hermes_cli.config")
    config_module.is_provider_enabled = (  # type: ignore[attr-defined]
        lambda configured: bool(configured.get("enabled", True))
    )
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config_module)

    assert (
        settings._compatible_modern_provider_config(
            {
                "edge": {
                    "enabled": False,
                    "baseUrl": "https://disabled.example/v1",
                },
            },
            "edge",
        )
        is None
    )


def test_legacy_provider_accepts_runtime_url_placeholder() -> None:
    """Preserve URL templates that Hermes expands after configuration load.

    Hermes deliberately accepts both environment references and bare-brace URL
    templates before runtime substitution.  Rejecting them locally would hide
    a valid provider from incontext and make plugin startup disagree with the
    agent's later resolved route.
    """

    assert settings._valid_provider_endpoint("https://${REGION}.example/v1") == (
        "https://${REGION}.example/v1"
    )


@pytest.mark.parametrize(
    ("provider_entry", "expected"),
    [
        (
            {
                "name": "Edge",
                "base_url": "https://LIVE.EXAMPLE:443/v1/",
                "url": "https://older.invalid/v1",
                "api": "https://stale.invalid/v1",
            },
            "https://live.example/v1",
        ),
        (
            {"name": "Edge", "baseUrl": "https://CAMEL.EXAMPLE:443/v1/"},
            "https://camel.example/v1",
        ),
        (
            {"name": "Edge", "url": "https://URL.EXAMPLE:443/v1/"},
            "https://url.example/v1",
        ),
        (
            {"name": "Edge", "api": "https://API.EXAMPLE:443/v1/"},
            "https://api.example/v1",
        ),
    ],
)
def test_legacy_provider_uses_hermes_normalized_endpoint(
    provider_entry: Mapping[str, Any],
    expected: str,
) -> None:
    """Resolve raw legacy URL aliases through Hermes' canonical precedence.

    Legacy entries can retain multiple fields after migrations.  Hermes makes
    ``base_url`` authoritative, supports ``baseUrl``, then falls back to
    ``url`` and ``api``.  Using modern precedence stores a different route and
    makes middleware reject the endpoint the running agent actually calls.
    """

    result = load(
        config={
            "model": {
                "default": "qwen-test",
                "provider": "custom:edge",
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            "custom_providers": [provider_entry],
        },
    )

    assert result.base_url == expected


def test_incomplete_provider_entries_fall_through_to_usable_legacy_entry() -> None:
    """Ignore named entries that cannot form a runtime endpoint.

    Hermes continues through both schemas when matching records have no usable
    URL.  Treating either incomplete record as selected removes endpoint
    isolation and hides a valid legacy route with the same durable identity.
    """

    result = load(
        config={
            "model": {
                "default": "qwen-test",
                "provider": "custom:edge",
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            "providers": {"edge": {"name": "Edge"}},
            "custom_providers": [
                {"name": "Edge"},
                {"name": "Edge", "base_url": "https://live.example/v1"},
            ],
        },
    )

    assert result.base_url == "https://live.example/v1"


def test_bare_custom_incomplete_modern_entry_falls_through_to_legacy() -> None:
    """Resolve a usable legacy route for Hermes' literal custom identity.

    A bare ``provider: custom`` can name an entry literally called ``custom``.
    Hermes skips a same-key modern record without an endpoint and continues to
    the compatibility list.  Stopping at the incomplete mapping removes the
    URL scope and can apply the primary tokenizer to another custom route.
    """

    result = load(
        config={
            "model": {
                "default": "qwen-test",
                "provider": "custom",
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            "providers": {"custom": {"name": "custom"}},
            "custom_providers": [
                {"name": "custom", "base_url": "https://legacy.example/v1"},
            ],
        },
    )

    assert result.base_url == "https://legacy.example/v1"


def test_provider_selector_preserves_underscores_as_identity() -> None:
    """Keep distinct Hermes provider keys distinct during route resolution.

    Hermes lowercases names and replaces spaces with hyphens, but it never
    rewrites underscores.  Collapsing ``edge_key`` into ``edge-key`` can select
    the first colliding provider and store an endpoint that the live agent does
    not use, causing every exact-budget route guard to miss.
    """

    result = load(
        config={
            "model": {
                "default": "qwen-test",
                "provider": "custom:edge_key",
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            "providers": {
                "edge-key": {"api": "https://hyphen.example/v1"},
                "edge_key": {"api": "https://underscore.example/v1"},
            },
        },
    )

    assert result.base_url == "https://underscore.example/v1"


def test_colon_bearing_provider_selector_matches_its_literal_key() -> None:
    """Do not treat every colon as the reserved ``custom:`` menu prefix.

    Hermes permits a modern mapping key such as ``tenant:edge`` and compares
    that complete identity against provider aliases.  Blindly discarding the
    prefix resolves the unrelated ``edge`` entry instead, attaching the wrong
    endpoint and context policy to otherwise valid requests.
    """

    result = load(
        config={
            "model": {
                "default": "qwen-test",
                "provider": "tenant:edge",
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            "providers": {
                "edge": {"api": "https://suffix.example/v1"},
                "tenant:edge": {"api": "https://literal.example/v1"},
            },
        },
    )

    assert result.base_url == "https://literal.example/v1"


def test_legacy_provider_skips_malformed_higher_precedence_url() -> None:
    """Select the first valid legacy URL rather than the first truthy value.

    Hermes validates each alias in precedence order and continues after a
    malformed ``base_url`` to a usable ``url``.  Accepting the malformed value
    as route identity makes incontext reject the valid endpoint actually used
    by generation and silently disables exact budgeting.
    """

    result = load(
        config={
            "model": {
                "default": "qwen-test",
                "provider": "custom:edge",
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            "custom_providers": [
                {
                    "name": "Edge",
                    "base_url": "not-a-url",
                    "url": "https://live.example/v1",
                },
            ],
        },
    )

    assert result.base_url == "https://live.example/v1"


@pytest.mark.parametrize("alias", ["vllm", "ollama", "llamacpp"])
def test_load_settings_uses_live_identity_for_local_provider_alias(alias: str) -> None:
    """Canonicalize Hermes local-provider aliases before route scoping.

    Hermes maps its local runtime aliases to the live provider label ``custom``
    before invoking request middleware.  Keeping the configured alias would
    silently disable exact budgeting even though the endpoint and model match.
    """

    result = load(
        config={
            **base_config,
            "model": {**base_config["model"], "provider": alias},
        },
    )

    assert result.provider == "custom"
    assert result.base_url == "https://inference.example/v1"


@pytest.mark.parametrize(
    "config",
    [
        {
            **base_config,
            "model": {**base_config["model"], "provider": "custom:missing"},
        },
        {
            **base_config,
            "model": {**base_config["model"], "provider": "custom:"},
            "providers": {"custom": {}},
        },
        {
            **base_config,
            "providers": {"custom": "invalid"},
        },
        {
            **base_config,
            "model": {**base_config["model"], "provider": "custom:local"},
            "providers": {"local": "invalid"},
        },
        {
            **base_config,
            "model": {**base_config["model"], "provider": "tenant:missing"},
        },
    ],
)
def test_load_settings_rejects_invalid_provider_route_configuration(
    config: Mapping[str, Any],
) -> None:
    """Fail startup when provider scoping cannot match Hermes' live route.

    Missing named entries, empty selectors, and non-mapping provider records
    cannot yield a reliable provider/base-URL identity.  Silently accepting
    them would either disable dynamic budgeting or apply the primary tokenizer
    and context window to a different fallback endpoint.
    """

    with pytest.raises(settings.SettingsError, match="provider"):
        load(config=config)


def test_load_settings_preserves_nonlocal_builtin_provider() -> None:
    """Leave ordinary Hermes provider identities unchanged during scoping.

    Only named custom profiles and documented local-runtime aliases canonicalize
    to ``custom``.  A built-in provider without a matching profiles entry must
    retain its live middleware label while still using the configured model URL.
    """

    result = load(
        config={
            **base_config,
            "model": {**base_config["model"], "provider": "openai"},
        },
    )

    assert result.provider == "openai"
    assert result.base_url == "https://inference.example/v1"


def test_load_settings_uses_hermes_live_identity_for_builtin_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror the provider identity exposed by Hermes request dispatch.

    Hermes resolves supported aliases such as ``z-ai`` to ``zai`` before
    constructing ``AIAgent`` and passes that canonical ID to request
    middleware.  Retaining the configuration spelling makes both primary and
    auxiliary route guards reject the intended request before exact counting.
    """

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    auth = types.ModuleType("hermes_cli.auth")
    auth.resolve_provider = (  # type: ignore[attr-defined]
        lambda provider: "zai" if provider == "z-ai" else provider
    )
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", auth)

    result = load(
        config={
            **base_config,
            "model": {**base_config["model"], "provider": "z-ai"},
        },
    )

    assert result.provider == "zai"


@pytest.mark.parametrize(
    "provider_sections",
    [
        {"providers": {"anthropic": {"api": "https://shadow.invalid/v1"}}},
        {
            "custom_providers": [
                {"name": "anthropic", "base_url": "https://shadow.invalid/v1"}
            ]
        },
    ],
)
def test_canonical_builtin_provider_is_not_shadowed_by_custom_entry(
    monkeypatch: pytest.MonkeyPatch,
    provider_sections: Mapping[str, Any],
) -> None:
    """Preserve Hermes' canonical built-in route over a colliding custom name.

    Hermes asks its auth registry whether a bare selector is already canonical
    before scanning either custom-provider schema.  Treating an ``anthropic``
    entry as authoritative changes the live provider to ``custom`` and makes
    both route guards reject every real primary request, silently disabling
    exact budgeting.
    """

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    auth = types.ModuleType("hermes_cli.auth")
    auth.resolve_provider = lambda provider: provider  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", auth)

    result = load(
        config={
            "model": {
                "default": "claude-sonnet-4.6",
                "provider": "anthropic",
                "base_url": "https://api.anthropic.example/v1",
                "context_length": 65_536,
            },
            "compression": {"threshold": 0.5},
            **provider_sections,
        },
    )

    assert result.provider == "anthropic"
    assert result.base_url == "https://api.anthropic.example/v1"


def test_load_settings_keeps_provider_when_hermes_alias_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep startup fail-open if Hermes cannot resolve an ordinary selector.

    Provider discovery may involve an optional runtime registry whose lookup
    can fail while the explicitly configured endpoint remains usable.  Hermes
    itself treats later provider setup as authoritative, so incontext must
    preserve the normalized selector rather than disable plugin registration.
    """

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    auth = types.ModuleType("hermes_cli.auth")

    def fail(provider: str) -> str:
        del provider
        raise RuntimeError("registry unavailable")

    auth.resolve_provider = fail  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", auth)

    result = load(
        config={
            **base_config,
            "model": {**base_config["model"], "provider": "openai"},
        },
    )

    assert result.provider == "openai"


def test_load_settings_uses_hermes_normalized_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the same model identifier as ``AIAgent`` and its provider request.

    Hermes normalizes ``zai/glm-5.1`` to ``glm-5.1`` before constructing its
    compressor and request.  Keeping the raw configuration value makes route
    scoping skip exact budgeting and can evaluate model-specific compression
    policy under a different identity.
    """

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    model_normalize = types.ModuleType("hermes_cli.model_normalize")
    calls: list[tuple[str, str]] = []

    def normalize(model: str, provider: str) -> str:
        calls.append((model, provider))
        return model.split("/", 1)[-1]

    model_normalize._AGGREGATOR_PROVIDERS = frozenset(  # type: ignore[attr-defined]
        {"openrouter"}
    )
    model_normalize.normalize_model_for_provider = normalize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.model_normalize",
        model_normalize,
    )
    ModernCompressor.calls.clear()

    result = load(
        config={
            **base_config,
            "model": {
                **base_config["model"],
                "default": "zai/glm-5.1",
                "provider": "zai",
            },
        },
    )

    assert result.model_name == "glm-5.1"
    assert ModernCompressor.calls[-1]["model"] == "glm-5.1"
    assert calls == [("zai/glm-5.1", "zai")]


@pytest.mark.parametrize("provider", ["openrouter", "zai"])
def test_model_normalization_remains_best_effort_like_hermes(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    """Preserve Hermes' skip and failure behavior around model normalization.

    Aggregators intentionally keep vendor-qualified model IDs and are never
    passed through the direct-provider normalizer.  For other providers Hermes
    catches normalization failures during agent initialization; incontext must
    retain the configured model under the same conditions instead of making an
    optional compatibility helper a startup dependency.
    """

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    model_normalize = types.ModuleType("hermes_cli.model_normalize")

    def fail(model: str, selected_provider: str) -> str:
        del model, selected_provider
        raise RuntimeError("normalizer unavailable")

    model_normalize._AGGREGATOR_PROVIDERS = frozenset(  # type: ignore[attr-defined]
        {"openrouter"}
    )
    model_normalize.normalize_model_for_provider = fail  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.model_normalize",
        model_normalize,
    )

    assert settings._effective_model_name("vendor/model", provider) == "vendor/model"


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


def test_load_settings_reserves_hermes_environment_output_budget() -> None:
    """Rebuild the same boundary when Hermes' environment override is active.

    Hermes gives ``HERMES_MAX_TOKENS`` precedence over model and provider
    completion allowances before constructing its live compressor.  The
    override is not folded into ``load_config()``, so it must be read through a
    typed skelet storage or incontext would derive a larger unsafe window.
    """

    ModernCompressor.calls.clear()
    config = {
        **base_config,
        "model": {**base_config["model"], "max_tokens": 4096},
        "providers": {"custom": {"max_output_tokens": 2048}},
    }

    load(environment={"HERMES_MAX_TOKENS": "8192"}, config=config)

    assert ModernCompressor.calls[-1]["max_tokens"] == 8192


def test_load_settings_reserves_provider_output_budget() -> None:
    """Use the selected provider allowance when no higher-priority cap exists.

    Hermes promotes ``providers.<route>.max_output_tokens`` into the agent's
    effective completion allowance.  Passing that same value to the rebuilt
    compressor keeps its threshold identical for both named and direct custom
    routes instead of budgeting beyond the live compression boundary.
    """

    ModernCompressor.calls.clear()
    config = {
        **base_config,
        "providers": {"custom": {"max_output_tokens": "2048"}},
    }

    load(config=config)

    assert ModernCompressor.calls[-1]["max_tokens"] == 2048


def test_load_settings_reserves_provider_max_tokens_alias() -> None:
    """Honor both output-cap names accepted by Hermes provider resolution.

    Hermes lifts ``providers.<name>.max_tokens`` into the runtime completion
    allowance used by gateway-created agents.  Ignoring that supported alias
    gives incontext a larger fictitious compression window than the active
    compressor for the same named provider.
    """

    ModernCompressor.calls.clear()
    config = {
        **base_config,
        "model": {
            "default": "qwen-test",
            "provider": "custom:local",
            "context_length": 65_536,
        },
        "providers": {
            "local": {
                "api": "https://inference.example/v1",
                "default_model": "qwen-test",
                "max_tokens": 2048,
            },
        },
    }

    load(config=config)

    assert ModernCompressor.calls[-1]["max_tokens"] == 2048


def test_blank_hermes_max_tokens_falls_back_to_model_configuration() -> None:
    """Treat an empty Hermes environment override as absent.

    Hermes checks ``HERMES_MAX_TOKENS`` for a non-empty value before parsing;
    a conventional blank dotenv assignment therefore falls through to
    ``model.max_tokens``.  Native skelet conversion and validation must mirror
    that behavior instead of failing plugin registration.
    """

    ModernCompressor.calls.clear()
    config = {
        **base_config,
        "model": {**base_config["model"], "max_tokens": 4096},
    }

    load(environment={"HERMES_MAX_TOKENS": ""}, config=config)

    assert ModernCompressor.calls[-1]["max_tokens"] == 4096


def test_invalid_hermes_max_tokens_fails_native_environment_validation() -> None:
    """Reject a non-integer Hermes override through skelet's field contract.

    The optional environment value is parsed by the same declarative skelet
    field that trims blank assignments.  A non-empty malformed value must not
    leak a raw conversion exception or silently fall through to YAML because
    Hermes itself would be unable to construct a matching token allowance.
    """

    with pytest.raises(
        settings.SettingsError,
        match="max_tokens must be a positive integer or blank",
    ):
        load(environment={"HERMES_MAX_TOKENS": "invalid"})


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


def test_load_settings_rejects_invalid_provider_output_budget() -> None:
    """Reject a malformed effective provider reserve before window derivation.

    An invalid selected provider allowance cannot be silently omitted because
    Hermes may reject it or derive a different threshold; startup failure is
    safer than claiming exact budgeting against a fictitious compressor.
    """

    config = {
        **base_config,
        "providers": {"custom": {"max_output_tokens": "invalid"}},
    }

    with pytest.raises(settings.SettingsError, match="provider max_output_tokens"):
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


@pytest.mark.parametrize("disabled", [False, "false", "0", "no"])
def test_disabled_hermes_compression_uses_the_complete_context_window(
    disabled: Any,
) -> None:
    """Do not enforce a threshold whose automatic compaction is disabled.

    Hermes still constructs a ContextCompressor for metadata when
    ``compression.enabled`` is false, but it never runs automatic preflight or
    reactive compaction at that object's threshold.  Reusing the inactive
    threshold would make incontext prematurely refuse valid requests even
    though Hermes intentionally allows them up to the model context limit.
    """

    class MustNotRun:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("inactive compressor threshold must not be used")

    result = load(
        config={
            **base_config,
            "compression": {"enabled": disabled, "threshold": 0.5},
        },
        compressor=MustNotRun,
    )

    assert result.compression_window == base_config["model"]["context_length"]


def test_load_settings_rejects_non_builtin_context_engine() -> None:
    """Never derive a budget from an inactive built-in compressor.

    Hermes selects context management through ``context.engine``.  External
    engines own their compaction threshold and the ``compression`` block is
    specific to the built-in ``ContextCompressor``.  Constructing that inactive
    class would silently give request middleware an unrelated window, which can
    truncate valid output or cross the active engine's real boundary.
    """

    config = {**base_config, "context": {"engine": "lcm"}}

    with pytest.raises(
        settings.SettingsError,
        match=r"context\.engine.*compressor",
    ):
        load(config=config)


def test_explicit_window_supports_external_context_engine_safely() -> None:
    """Allow an external engine only when its real boundary is supplied.

    A positive incontext override is an explicit operator assertion of the
    selected engine's active threshold.  In that case no inactive built-in
    compressor is constructed, so the plugin can budget against the stated
    external boundary without inventing engine-specific policy.
    """

    result = load(
        environment={"INCONTEXT_COMPRESSION_WINDOW_TOKENS": "50000"},
        config={**base_config, "context": {"engine": "lcm"}},
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


@pytest.mark.parametrize("section", ["model", "compression", "context", "providers"])
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
