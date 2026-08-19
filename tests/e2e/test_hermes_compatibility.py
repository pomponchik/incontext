"""Black-box compatibility checks for Hermes' public plugin contract."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("INCONTEXT_E2E") != "1",
        reason="set INCONTEXT_E2E=1 inside a Hermes Agent environment",
    ),
]


class TokenizerHandler(BaseHTTPRequestHandler):
    """Deterministic in-process stand-in for vLLM's /tokenize endpoint."""

    prompt_tokens: ClassVar[int] = 12_345
    context_length: ClassVar[int] = 65_536
    requests: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        self.requests.append(json.loads(body))
        response = json.dumps(
            {
                "count": self.prompt_tokens,
                "max_model_len": self.context_length,
            },
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, message_format: str, *args: Any) -> None:
        """Keep successful CI logs focused on the compatibility assertions."""


@pytest.fixture
def tokenizer_server() -> Iterator[str]:
    TokenizerHandler.requests.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), TokenizerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/tokenize"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def write_hermes_config(home: Path) -> None:
    """Write the smallest real config that resolves all plugin invariants."""

    home.mkdir(parents=True)
    (home / "empty-bundled-plugins").mkdir()
    (home / "config.yaml").write_text(
        """\
model:
  default: qwen-e2e
  provider: custom
  base_url: http://inference.invalid/v1
  api_mode: chat_completions
  context_length: 65536
compression:
  enabled: true
  threshold: 0.5
plugins:
  enabled:
    - incontext
  disabled: []
""",
        encoding="utf-8",
    )


def test_pypi_entrypoint_rewrites_a_real_hermes_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_server: str,
) -> None:
    """Load through metadata, register, and execute through Hermes itself."""

    home = tmp_path / "hermes"
    write_hermes_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(
        "HERMES_BUNDLED_PLUGINS",
        str(home / "empty-bundled-plugins"),
    )
    monkeypatch.setenv("INCONTEXT_TOKENIZER_URL", tokenizer_server)

    # Import after setting HERMES_HOME: Hermes resolves some paths at import time.
    from hermes_cli.middleware import (  # type: ignore[import-not-found]  # noqa: PLC0415
        apply_llm_request_middleware,
    )
    from hermes_cli.plugins import (  # type: ignore[import-not-found]  # noqa: PLC0415
        get_plugin_manager,
    )

    from incontext.hermes import _reset_runtime_for_tests, get_runtime  # noqa: PLC0415
    from incontext.vllm import VllmBackend  # noqa: PLC0415

    # Container images may preload plugins from sitecustomize or pytest entry
    # points before this test installs its isolated Hermes home. Reset only the
    # package cache so discovery validates the fixture configuration itself.
    _reset_runtime_for_tests()
    manager = get_plugin_manager()
    manager.discover_and_load(force=True)
    loaded = manager._plugins["incontext"]
    assert loaded.enabled is True
    assert loaded.error is None
    assert loaded.manifest.source == "entrypoint"
    assert loaded.middleware_registered == ["llm_request"]

    original = {
        "model": "qwen-e2e",
        "messages": [{"role": "user", "content": "Use the real middleware"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "long_tool", "parameters": {}},
            },
        ],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    result = apply_llm_request_middleware(original, session_id="incontext-e2e")
    runtime = get_runtime()

    assert isinstance(runtime.backend, VllmBackend)
    assert result.changed is True
    assert result.original_payload == original
    assert result.payload is not original
    assert result.payload["max_tokens"] == (
        runtime.settings.compression_window - TokenizerHandler.prompt_tokens
    )
    assert "max_completion_tokens" not in result.payload
    assert "max_output_tokens" not in result.payload
    assert result.trace == [
        {
            "source": "incontext",
            "reason": (
                "vllm-tokenize: compression_window="
                f"{runtime.settings.compression_window}, "
                f"prompt_tokens={TokenizerHandler.prompt_tokens}, "
                f"max_tokens={result.payload['max_tokens']}"
            ),
        },
    ]
    assert "max_tokens" not in original

    assert TokenizerHandler.requests == [
        {
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": True},
            "messages": original["messages"],
            "model": "qwen-e2e",
            "tools": original["tools"],
        },
    ]

    bounded = {
        **original,
        "max_tokens": 8192,
        "max_completion_tokens": 4096,
        "max_output_tokens": 2048,
    }
    bounded_result = apply_llm_request_middleware(
        bounded,
        session_id="incontext-bounded-e2e",
    )
    assert bounded_result.payload["max_tokens"] == 2048
    assert "max_completion_tokens" not in bounded_result.payload
    assert "max_output_tokens" not in bounded_result.payload
    assert bounded["max_tokens"] == 8192

    # Hermes' auxiliary builder intentionally drops max_tokens for custom
    # providers. The plugin must cover this path as well as public middleware,
    # otherwise compression summaries silently regain the provider's full
    # context remainder.
    from agent.auxiliary_client import (  # type: ignore[import-not-found]  # noqa: PLC0415
        _build_call_kwargs,
    )

    auxiliary_bounded = _build_call_kwargs(
        "custom",
        "qwen-e2e",
        [{"role": "user", "content": "Bound this compression summary"}],
        max_tokens=2048,
        base_url="http://inference.invalid/v1",
    )
    assert auxiliary_bounded["max_tokens"] == 2048

    auxiliary_dynamic = _build_call_kwargs(
        "custom",
        "qwen-e2e",
        [{"role": "user", "content": "Budget this title dynamically"}],
        base_url="http://inference.invalid/v1",
    )
    assert auxiliary_dynamic["max_tokens"] == (
        runtime.settings.compression_window - TokenizerHandler.prompt_tokens
    )

    # The bounded public request above has the same provider-visible prompt as
    # the first request, so VllmBackend correctly serves it from cache. The two
    # distinct auxiliary prompts each require one additional tokenizer call.
    assert len(TokenizerHandler.requests) == 3
