"""Black-box compatibility and auto-compression checks against real Hermes."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("INCONTEXT_E2E") != "1",
        reason="set INCONTEXT_E2E=1 inside a Hermes Agent environment",
    ),
]

E2E_SUMMARY_MARKER = "E2E compression checkpoint created"
E2E_SUMMARY_RESPONSE = f"""\
## Active Task
Prove the complete incontext auto-compression path.

## Goal
{E2E_SUMMARY_MARKER} by the real Hermes context compressor.

## Completed Actions
Earlier synthetic turns were compacted.

## Remaining Work
Answer the latest user message.
"""
E2E_FINAL_RESPONSE = "Completed after automatic context compression"
COMPRESSED_PROMPT_TOKENS = 10_000


class TokenizerHandler(BaseHTTPRequestHandler):
    """Deterministic OpenAI/vLLM server for the complete Hermes request path."""

    prompt_tokens: ClassVar[int] = 12_345
    context_length: ClassVar[int] = 65_536
    requests: ClassVar[list[dict[str, Any]]] = []
    chat_requests: ClassVar[list[dict[str, Any]]] = []
    count_resolver: ClassVar[Any] = None

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        payload = json.loads(body)
        path = self.path.split("?", 1)[0]
        if path.endswith("/tokenize"):
            self.requests.append(payload)
            prompt_tokens = (
                type(self).count_resolver(payload)
                if type(self).count_resolver is not None
                else self.prompt_tokens
            )
            self._send_json(
                {
                    "count": prompt_tokens,
                    "max_model_len": self.context_length,
                },
            )
            return
        if path.endswith("/chat/completions"):
            self.chat_requests.append(payload)
            serialized = json.dumps(payload.get("messages", [])).lower()
            is_summary = "you are a summarization agent" in serialized
            content = E2E_SUMMARY_RESPONSE if is_summary else E2E_FINAL_RESPONSE
            if payload.get("stream") is True:
                self._send_chat_stream(content)
                return
            self._send_json(
                {
                    "id": "chatcmpl-incontext-e2e",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "qwen-e2e",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content,
                            },
                            "finish_reason": "stop",
                        },
                    ],
                    "usage": {
                        "prompt_tokens": COMPRESSED_PROMPT_TOKENS,
                        "completion_tokens": 100,
                        "total_tokens": COMPRESSED_PROMPT_TOKENS + 100,
                    },
                },
            )
            return
        self.send_error(404)

    def _send_chat_stream(self, content: str) -> None:
        chunks = [
            {
                "id": "chatcmpl-incontext-e2e",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "qwen-e2e",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": None,
                    },
                ],
            },
            {
                "id": "chatcmpl-incontext-e2e",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "qwen-e2e",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    },
                ],
                "usage": {
                    "prompt_tokens": COMPRESSED_PROMPT_TOKENS,
                    "completion_tokens": 100,
                    "total_tokens": COMPRESSED_PROMPT_TOKENS + 100,
                },
            },
        ]
        response = (
            b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks)
            + b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _send_json(self, payload: dict[str, Any]) -> None:
        response = json.dumps(payload).encode()
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
    TokenizerHandler.chat_requests.clear()
    TokenizerHandler.count_resolver = None
    TokenizerHandler.prompt_tokens = 12_345
    server = ThreadingHTTPServer(("127.0.0.1", 0), TokenizerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/tokenize"
    finally:
        TokenizerHandler.count_resolver = None
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def write_hermes_config(home: Path, inference_base_url: str) -> None:
    """Write the smallest real config that resolves all plugin invariants."""

    home.mkdir(parents=True)
    (home / "empty-bundled-plugins").mkdir()
    (home / "config.yaml").write_text(
        f"""\
model:
  default: qwen-e2e
  provider: custom
  base_url: {inference_base_url}
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


def assert_viable_output_boundary(
    runtime: Any,
    apply_middleware: Any,
) -> None:
    """Exercise the exact reserve boundary through real Hermes bindings."""

    from agent import turn_context  # type: ignore[import-not-found]  # noqa: PLC0415

    previous_prompt_tokens = TokenizerHandler.prompt_tokens
    previous_request_count = len(TokenizerHandler.requests)
    boundary_request = {
        "model": "qwen-e2e",
        "messages": [{"role": "user", "content": "Cross the viable boundary"}],
    }
    try:
        TokenizerHandler.prompt_tokens = (
            runtime.settings.compression_window - runtime.settings.min_output_tokens + 1
        )
        runtime.backend.clear_cache()
        pressure = turn_context.estimate_request_tokens_rough(  # type: ignore[attr-defined]
            boundary_request["messages"],
        )
        assert pressure == runtime.settings.compression_window

        boundary_result = apply_middleware(
            boundary_request,
            session_id="incontext-boundary-e2e",
        )
        assert boundary_result.changed is False
        assert "max_tokens" not in boundary_result.payload
        assert len(TokenizerHandler.requests) == previous_request_count + 1
    finally:
        TokenizerHandler.prompt_tokens = previous_prompt_tokens


def assert_complete_auto_compression(
    runtime: Any,
    inference_base_url: str,
) -> None:
    """Run an oversized turn through real Hermes compression and inference."""

    token_request_start = len(TokenizerHandler.requests)
    chat_request_start = len(TokenizerHandler.chat_requests)
    oversized_prompt_tokens = (
        runtime.settings.compression_window - runtime.settings.min_output_tokens + 1
    )

    def resolve_count(payload: dict[str, Any]) -> int:
        serialized = json.dumps(payload.get("messages", [])).lower()
        if (
            E2E_SUMMARY_MARKER.lower() in serialized
            or "you are a summarization agent" in serialized
        ):
            return COMPRESSED_PROMPT_TOKENS
        return oversized_prompt_tokens

    history: list[dict[str, Any]] = []
    for index in range(30):
        history.extend(
            [
                {
                    "role": "user",
                    "content": f"Archived question {index}: " + "u" * 1200,
                },
                {
                    "role": "assistant",
                    "content": f"Archived answer {index}: " + "a" * 1200,
                },
            ],
        )
    original_turn_message_count = len(history) + 1
    status_messages: list[tuple[str, str]] = []

    TokenizerHandler.count_resolver = resolve_count
    runtime.backend.clear_cache()
    try:
        with patch(
            "run_agent.get_tool_definitions",
            return_value=[],
        ), patch(
            "run_agent.check_toolset_requirements",
            return_value={},
        ):
            from run_agent import (  # type: ignore[import-not-found]  # noqa: PLC0415
                AIAgent,
            )

            agent = AIAgent(
                api_key="incontext-e2e-key",
                base_url=inference_base_url,
                provider="custom",
                api_mode="chat_completions",
                model="qwen-e2e",
                max_iterations=2,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                save_trajectories=False,
            )

        agent._cached_system_prompt = "You are the incontext e2e agent."
        agent._use_prompt_caching = False
        agent._disable_streaming = True
        agent._compression_feasibility_checked = True
        agent.tool_delay = 0
        agent.status_callback = lambda event, message: status_messages.append(
            (event, message),
        )
        with patch.object(
            agent,
            "_persist_session",
        ), patch.object(
            agent,
            "_save_trajectory",
        ), patch.object(
            agent,
            "_cleanup_task_resources",
        ), patch.object(
            agent,
            "_build_system_prompt",
            return_value="You are the incontext e2e agent.",
        ):
            result = agent.run_conversation(
                "Answer only after compacting the prior history.",
                conversation_history=history,
            )
    finally:
        TokenizerHandler.count_resolver = None
        runtime.backend.clear_cache()

    assert result["completed"] is True
    assert result["final_response"] == E2E_FINAL_RESPONSE
    assert agent.context_compressor.compression_count >= 1
    assert len(result["messages"]) < original_turn_message_count
    assert E2E_SUMMARY_MARKER in json.dumps(result["messages"])
    assert any(
        event == "lifecycle" and "Preflight compression" in message
        for event, message in status_messages
    )

    token_requests = TokenizerHandler.requests[token_request_start:]
    assert E2E_SUMMARY_MARKER not in json.dumps(token_requests[0])
    assert any(E2E_SUMMARY_MARKER in json.dumps(item) for item in token_requests[1:])

    chat_requests = TokenizerHandler.chat_requests[chat_request_start:]
    summary_requests = [
        item
        for item in chat_requests
        if "you are a summarization agent" in json.dumps(item).lower()
    ]
    main_requests = [item for item in chat_requests if item not in summary_requests]
    assert len(summary_requests) == 1
    assert len(main_requests) == 1
    assert E2E_SUMMARY_MARKER in json.dumps(main_requests[0]["messages"])
    assert main_requests[0]["max_tokens"] == (
        runtime.settings.compression_window - COMPRESSED_PROMPT_TOKENS
    )
    assert main_requests[0]["max_tokens"] >= runtime.settings.min_output_tokens


def test_pypi_entrypoint_runs_the_complete_hermes_compression_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_server: str,
) -> None:
    """Load through metadata, register, and execute through Hermes itself."""

    home = tmp_path / "hermes"
    inference_base_url = tokenizer_server.rsplit("/", 1)[0] + "/v1"
    write_hermes_config(home, inference_base_url)
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
        base_url=inference_base_url,
    )
    assert auxiliary_bounded["max_tokens"] == 2048

    auxiliary_dynamic = _build_call_kwargs(
        "custom",
        "qwen-e2e",
        [{"role": "user", "content": "Budget this title dynamically"}],
        base_url=inference_base_url,
    )
    assert auxiliary_dynamic["max_tokens"] == (
        runtime.settings.compression_window - TokenizerHandler.prompt_tokens
    )

    # The bounded public request above has the same provider-visible prompt as
    # the first request, so VllmBackend correctly serves it from cache. The two
    # distinct auxiliary prompts each require one additional tokenizer call.
    assert len(TokenizerHandler.requests) == 3

    # One token below the viable reserve is a compression condition, not a
    # request for a tiny length-truncated completion. The real Hermes preflight
    # sees pressure exactly at its threshold, while middleware fails open if it
    # is invoked directly before that compression has happened.
    assert_viable_output_boundary(runtime, apply_llm_request_middleware)

    # Finally drive a complete real Hermes turn: exact preflight, summary API,
    # ContextCompressor assembly, post-compression budgeting, and main API.
    assert_complete_auto_compression(runtime, inference_base_url)
