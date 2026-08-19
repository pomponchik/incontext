from __future__ import annotations

import pytest

from incontext.settings import Environment, HermesEnvironment, Settings
from incontext.vllm import VllmEnvironment


@pytest.fixture(autouse=True)
def reset_skelet_environment_caches() -> None:
    """Keep skelet's process-environment cache isolated between unit tests."""

    for storage in (Environment, HermesEnvironment, VllmEnvironment):
        for source in storage.__sources__.sources:
            source.__dict__.pop("data", None)


@pytest.fixture
def runtime_settings() -> Settings:
    return Settings(
        model_name="qwen",
        context_length=65_536,
        compression_window=55_705,
        fallback_margin_tokens=1024,
        provider="",
        base_url="",
    )
