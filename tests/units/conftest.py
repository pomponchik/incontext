from __future__ import annotations

import pytest

from incontext.settings import Settings


@pytest.fixture
def runtime_settings() -> Settings:
    return Settings(
        context_length=65_536,
        compression_window=55_705,
        tokenizer_url="https://inference.example/tokenize",
        tokenizer_user_agent="incontext-tests",
        tokenizer_timeout_seconds=3.5,
        fallback_margin_tokens=1024,
    )
