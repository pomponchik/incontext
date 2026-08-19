from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from importlib.metadata import version
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import pytest

from incontext.vllm import VllmBackend, VllmEnvironment


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class RecordingOpener:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[urllib.request.Request, float]] = []

    def __call__(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> FakeResponse:
        self.calls.append((request, timeout))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return FakeResponse(json.dumps(response).encode())


def response(count: Any = 123, context: Any = 65_536) -> dict[str, Any]:
    return {"count": count, "max_model_len": context}


def make_backend(
    responses: list[Any] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    cache_entries: Any = 64,
) -> tuple[VllmBackend, RecordingOpener]:
    opener = RecordingOpener([] if responses is None else responses)
    values = {
        "INCONTEXT_TOKENIZER_URL": "https://inference.example/tokenize",
        "INCONTEXT_TOKENIZER_USER_AGENT": "incontext-tests",
        "INCONTEXT_TOKENIZER_TIMEOUT_SECONDS": "3.5",
    }
    if environment is not None:
        values = dict(environment)
    with patch.dict("os.environ", values, clear=True):
        backend = VllmBackend(cache_entries=cache_entries, opener=opener)
    return backend, opener


def test_source_is_stable_and_non_sensitive() -> None:
    backend, _ = make_backend()
    assert backend.source == "vllm-tokenize"


def test_vllm_maps_responses_output_cap_to_chat_completions() -> None:
    """Avoid a silently ignored output budget on vLLM's chat endpoint.

    ``max_output_tokens`` belongs to the Responses API and the deployed vLLM
    Chat Completions request model ignores it.  The bundled backend therefore
    translates only that incompatible alias and preserves aliases that the
    endpoint already understands.
    """

    backend, _ = make_backend()

    assert backend.output_budget_field("max_output_tokens") == "max_tokens"
    assert backend.output_budget_field("max_completion_tokens") == (
        "max_completion_tokens"
    )


@pytest.mark.parametrize(
    ("wire_value", "expected"),
    [
        ("5", 5),
        ("5.0", 5),
        ("1_0", 10),
        ("\u0661", None),
        (5.0, 5),
        (True, 1),
        (0, None),
        (1.5, None),
    ],
)
def test_vllm_coerces_output_caps_like_chat_request_validation(
    wire_value: Any,
    expected: int | None,
) -> None:
    """Expose only positive caps that vLLM's request model will enforce.

    Its Pydantic schema accepts zero-fraction decimal strings, ASCII underscore
    separators, integral floats, and booleans, while rejecting Unicode digit
    spellings.  Dynamic budgeting must preserve exactly those caller bounds;
    accepting or dropping a different spelling changes the provider request.
    """

    backend, _ = make_backend()

    assert backend.coerce_output_budget(wire_value) == expected


def test_default_user_agent_tracks_distribution_version() -> None:
    """Keep the tokenizer transport identity aligned with package metadata.

    Tokenizer-server logs, operational metrics, and proxy policies consume the
    default User-Agent.  Comparing it with installed distribution metadata
    prevents a copied version literal from silently identifying a newer client
    as an older release after future version bumps.
    """

    with patch.dict(
        "os.environ",
        {"INCONTEXT_TOKENIZER_URL": "https://inference.test/tokenize"},
        clear=True,
    ):
        environment = VllmEnvironment()

    assert environment.tokenizer_user_agent == f"incontext/{version('incontext')}"


def test_build_payload_minimal_shape() -> None:
    request = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    assert VllmBackend._build_payload(request) == {
        "model": "qwen",
        "messages": request["messages"],
        "add_generation_prompt": True,
    }


def test_build_payload_includes_tools_and_template_kwargs() -> None:
    request = {
        "model": "qwen",
        "messages": [],
        "tools": [{"type": "function"}],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    payload = VllmBackend._build_payload(request)
    assert payload["tools"] == request["tools"]
    assert payload["tools"] is not request["tools"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_build_payload_honors_read_only_extra_body_overrides() -> None:
    """Count Mapping overrides with the same precedence as OpenAI's SDK.

    ``extra_body`` accepts any reusable Mapping and is shallow-merged after
    ordinary request parameters.  Restricting recognition to mutable dicts
    tokenizes the superseded model, messages, and tools, so exact budgeting can
    use an unrelated chat template and undercount the actual prompt.
    """

    wire_messages = ({"role": "user", "content": "wire"},)
    wire_tools = ({"type": "function", "function": {"name": "wire"}},)

    payload = VllmBackend._build_payload(
        {
            "model": "top-model",
            "messages": [],
            "tools": [],
            "extra_body": MappingProxyType(
                {
                    "model": "wire-model",
                    "messages": wire_messages,
                    "tools": wire_tools,
                },
            ),
        },
    )

    assert payload["model"] == "wire-model"
    assert payload["messages"] == list(wire_messages)
    assert payload["tools"] == list(wire_tools)


def test_build_payload_materializes_tuple_tools_like_openai_sdk() -> None:
    """Tokenize tool schemas accepted through OpenAI's Iterable API surface.

    The official client accepts ``tools`` as an iterable and converts a tuple
    to a JSON array before sending generation.  Dropping that same reusable
    sequence from ``/tokenize`` removes model-visible definitions from the
    rendered prompt and undercounts it.  Materialization must not mutate the
    caller-owned tuple that Hermes can reuse for retries.
    """

    tools = (
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object"},
            },
        },
    )

    payload = VllmBackend._build_payload(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "use the tool"}],
            "tools": tools,
        },
    )

    assert payload["tools"] == list(tools)
    assert isinstance(payload["tools"], list)


def test_build_payload_recursively_materializes_openai_wire_mappings() -> None:
    """Mirror OpenAI's recursive conversion of reusable typed mappings.

    The SDK accepts read-only Mapping instances at message, content-part,
    tool, function, and outer-parameter positions and serializes them as JSON
    objects.  Leaving any nested object unchanged makes ``json.dumps`` reject
    the tokenizer payload even though generation reaches vLLM normally,
    disabling exact counting.  Copies must also preserve caller ownership.
    """

    content_part = MappingProxyType({"type": "text", "text": "hello"})
    content_parts = {"only": content_part}
    message = MappingProxyType(
        {"role": "user", "content": content_parts.values()},
    )
    function = MappingProxyType(
        {"name": "lookup", "parameters": MappingProxyType({"type": "object"})},
    )
    tool = MappingProxyType({"type": "function", "function": function})

    payload = VllmBackend._build_payload(
        {
            "model": "qwen",
            "messages": (message,),
            "tools": (tool,),
        },
    )

    assert json.loads(json.dumps(payload)) == {
        "model": "qwen",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            },
        ],
        "add_generation_prompt": True,
    }
    assert next(iter(message["content"])) is content_part
    assert tool["function"] is function


def test_build_payload_mirrors_every_prompt_affecting_vllm_option() -> None:
    """Tokenize exactly the chat prompt that vLLM will use for generation.

    Continuations, custom templates, special-token policy, multimodal processor
    options, and template kwargs can all alter the rendered prompt.  OpenAI's
    client merges ``extra_body`` over standard request fields, so the tokenizer
    payload must forward the same options with the same precedence instead of
    silently restoring ``add_generation_prompt=True``.
    """

    request = {
        "model": "qwen",
        "messages": [{"role": "assistant", "content": "prefix"}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"source": "top-level"},
        "extra_body": {
            "add_generation_prompt": False,
            "continue_final_message": True,
            "add_special_tokens": False,
            "chat_template": "{{ messages }}",
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {"num_crops": 4},
        },
    }

    assert VllmBackend._build_payload(request) == {
        "model": "qwen",
        "messages": request["messages"],
        "add_generation_prompt": False,
        "continue_final_message": True,
        "add_special_tokens": False,
        "chat_template": "{{ messages }}",
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"num_crops": 4},
    }


def test_build_payload_applies_extra_body_to_core_chat_fields() -> None:
    """Mirror OpenAI's final shallow merge for model, messages, and tools.

    ``extra_body`` wins over generated request parameters on the HTTP wire.
    Tokenizing the pre-merge values can count a different model, transcript, or
    tool schema; an explicit empty tool list must also remain distinguishable
    from an omitted field for vLLM's chat renderer.
    """

    override_messages = [{"role": "user", "content": "override"}]
    request = {
        "model": "original",
        "messages": [{"role": "user", "content": "original"}],
        "tools": [{"type": "function", "function": {"name": "original"}}],
        "extra_body": {
            "model": "override",
            "messages": override_messages,
            "tools": [],
        },
    }

    assert VllmBackend._build_payload(request) == {
        "model": "override",
        "messages": override_messages,
        "tools": [],
        "add_generation_prompt": True,
    }


def test_build_payload_reproduces_vllm_reasoning_and_rag_rendering() -> None:
    """Translate chat-only fields into the equivalent tokenize parameters.

    vLLM injects ``documents`` and ``reasoning_effort`` into Jinja template
    kwargs and derives ``enable_thinking`` when the caller did not set it.
    Media IO options are forwarded directly.  Missing this transformation can
    make Qwen's generation prompt tens of tokens different before any content
    or tool-schema growth is considered.
    """

    request = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "answer from context"}],
        "reasoning_effort": "high",
        "documents": [{"title": "top", "text": "ignored by extra_body"}],
        "chat_template_kwargs": {"custom": "value"},
        "media_io_kwargs": {"image": {"num_frames": 1}},
        "extra_body": {
            "reasoning_effort": "none",
            "documents": [{"title": "final", "text": "document"}],
            "media_io_kwargs": {"video": {"num_frames": 4}},
        },
    }

    payload = VllmBackend._build_payload(request)

    assert payload["chat_template_kwargs"] == {
        "custom": "value",
        "documents": [{"title": "final", "text": "document"}],
        "reasoning_effort": "none",
        "enable_thinking": False,
    }
    assert payload["media_io_kwargs"] == {"video": {"num_frames": 4}}


def test_build_payload_normalizes_deprecated_reasoning_content() -> None:
    """Tokenize the same assistant reasoning field that generation renders.

    vLLM's chat-completion validator migrates legacy ``reasoning_content`` to
    ``reasoning`` before rendering, while its tokenize request accepts only the
    new field.  Mirroring the migration avoids an exact-count drift and must
    not mutate the request that Hermes may reuse for retries.  An explicit
    modern null is considered unset, while a null legacy value is simply
    removed, matching the provider's null-aware validator exactly.
    """

    legacy = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "private trace",
    }
    modern = {
        "role": "assistant",
        "content": "answer",
        "reasoning": "preferred trace",
        "reasoning_content": "obsolete trace",
    }
    explicit_null = {
        "role": "assistant",
        "content": "answer",
        "reasoning": None,
        "reasoning_content": "fallback trace",
    }
    empty_legacy = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": None,
    }
    messages: list[Any] = [
        "invalid-provider-value",
        legacy,
        modern,
        explicit_null,
        empty_legacy,
    ]

    payload = VllmBackend._build_payload({"model": "qwen", "messages": messages})

    assert payload["messages"] == [
        "invalid-provider-value",
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": "private trace",
        },
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": "preferred trace",
        },
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": "fallback trace",
        },
        {
            "role": "assistant",
            "content": "answer",
        },
    ]
    assert legacy["reasoning_content"] == "private trace"
    assert modern["reasoning_content"] == "obsolete trace"
    assert explicit_null["reasoning"] is None
    assert empty_legacy["reasoning_content"] is None


def test_reasoning_normalization_preserves_unvalidated_message_shapes() -> None:
    """Leave non-list message input for vLLM to validate on both endpoints.

    Incontext owns prompt equivalence, not provider schema repair.  Returning a
    malformed scalar unchanged ensures ``/tokenize`` rejects the same value as
    chat generation instead of silently manufacturing a different prompt.
    """

    assert VllmBackend._normalize_messages("invalid") == "invalid"


def test_build_payload_preserves_explicit_thinking_override() -> None:
    """Let caller template kwargs override vLLM's reasoning-derived default.

    The chat-completions renderer derives ``enable_thinking`` only when that
    key is absent.  An explicit value must survive even when reasoning effort
    would otherwise imply the opposite setting.
    """

    payload = VllmBackend._build_payload(
        {
            "model": "qwen",
            "messages": [],
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )

    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "none",
    }


def test_build_payload_forwards_invalid_template_kwargs_for_vllm_validation() -> None:
    """Do not silently replace a malformed provider-visible template value.

    OpenAI's shallow merge sends the value to vLLM, whose Pydantic contract owns
    validation.  Keeping the invalid shape in ``/tokenize`` makes exact counting
    fail open consistently instead of tokenizing defaults for an inference
    request that will later be rejected or interpreted differently.
    """

    payload = VllmBackend._build_payload(
        {
            "model": "qwen",
            "messages": [],
            "extra_body": {"chat_template_kwargs": ["invalid"]},
        },
    )

    assert payload["chat_template_kwargs"] == ["invalid"]


@pytest.mark.parametrize(
    ("tools", "extra_body"),
    [("invalid", []), (None, None)],
)
def test_build_payload_ignores_non_effective_optional_fields(
    tools: Any,
    extra_body: Any,
) -> None:
    payload = VllmBackend._build_payload(
        {"model": "qwen", "messages": [], "tools": tools, "extra_body": extra_body},
    )
    assert "tools" not in payload
    assert "chat_template_kwargs" not in payload


def test_build_payload_preserves_explicit_empty_tools() -> None:
    """Keep ``tools=[]`` because it survives into vLLM's wire request.

    Treating an explicit empty list as field absence changes the Pydantic input
    from a list to ``None`` and can select different server-side rendering
    defaults, so exact tokenization must retain the caller's shape.
    """

    payload = VllmBackend._build_payload(
        {"model": "qwen", "messages": [], "tools": []},
    )

    assert payload["tools"] == []


@pytest.mark.parametrize("value", [None, True, False, 0, -1, "1", 1.5])
def test_positive_response_integer_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(VllmBackend.VllmBackendError, match="invalid count"):
        VllmBackend._positive_response_integer({"count": value}, "count")


def test_positive_response_integer_accepts_positive_integer() -> None:
    assert VllmBackend._positive_response_integer({"count": 1}, "count") == 1


def test_count_accepts_and_caches_an_empty_rendered_prompt() -> None:
    """Accept a legitimately empty token sequence reported by vLLM.

    ``/tokenize`` defines count as ``len(input_ids)``, which can be zero when a
    valid template emits no text and generation/special-token insertion are
    disabled.  Rejecting zero discards an exact full-window budget and falls
    back to a positive rough estimate plus margin; model context length remains
    independently required to be positive.
    """

    backend, opener = make_backend([response(0)])
    request = {
        "model": "qwen",
        "messages": [],
        "add_generation_prompt": False,
        "add_special_tokens": False,
    }

    assert backend.count(request, context_length=65_536) == 0
    assert backend.count(request, context_length=65_536) == 0
    assert len(opener.calls) == 1


@pytest.mark.parametrize("value", [True, -1, 1.5, "0"])
def test_nonnegative_response_integer_rejects_non_counts(value: Any) -> None:
    """Keep malformed tokenizer counts outside the exact-budget contract."""

    with pytest.raises(VllmBackend.VllmBackendError, match="invalid count"):
        VllmBackend._nonnegative_response_integer({"count": value}, "count")


@pytest.mark.parametrize("cache_entries", [True, "1"])
def test_backend_rejects_invalid_cache_capacity_type(cache_entries: Any) -> None:
    with pytest.raises(TypeError, match="positive integer"):
        make_backend(cache_entries=cache_entries)


@pytest.mark.parametrize("cache_entries", [0, -1])
def test_backend_rejects_non_positive_cache_capacity(cache_entries: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        make_backend(cache_entries=cache_entries)


def test_environment_prefers_primary_names_and_converts_values() -> None:
    backend, _ = make_backend(
        environment={
            "INCONTEXT_TOKENIZER_URL": " https://primary.test/tokenize ",
            "HERMES_VLLM_TOKENIZER_URL": "https://legacy.test/tokenize",
            "INCONTEXT_TOKENIZER_USER_AGENT": " primary-agent ",
            "HERMES_VLLM_TOKENIZER_USER_AGENT": "legacy-agent",
            "INCONTEXT_TOKENIZER_TIMEOUT_SECONDS": " 12.5 ",
            "HERMES_VLLM_TOKENIZER_TIMEOUT_SECONDS": "99",
        },
    )
    assert backend._environment.tokenizer_url == "https://primary.test/tokenize"
    assert backend._environment.tokenizer_user_agent == "primary-agent"
    assert backend._environment.tokenizer_timeout_seconds == 12.5


def test_environment_supports_legacy_names() -> None:
    backend, _ = make_backend(
        environment={
            "HERMES_VLLM_TOKENIZER_URL": "https://legacy.test/tokenize",
            "HERMES_VLLM_TOKENIZER_USER_AGENT": "legacy-agent",
            "HERMES_VLLM_TOKENIZER_TIMEOUT_SECONDS": "8",
        },
    )
    assert backend._environment.tokenizer_url == "https://legacy.test/tokenize"
    assert backend._environment.tokenizer_user_agent == "legacy-agent"
    assert backend._environment.tokenizer_timeout_seconds == 8


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({}, "HTTP"),
        ({"INCONTEXT_TOKENIZER_URL": "  "}, "must not be blank"),
        ({"INCONTEXT_TOKENIZER_URL": "ftp://example.test/tokenize"}, "HTTP"),
        ({"INCONTEXT_TOKENIZER_URL": "https:///tokenize"}, "HTTP"),
        ({"INCONTEXT_TOKENIZER_URL": "https://example.test:not-a-port"}, "valid"),
        ({"INCONTEXT_TOKENIZER_URL": "https://example.test:65536"}, "valid"),
        ({"INCONTEXT_TOKENIZER_URL": "https://example.test:0"}, "valid TCP"),
        ({"INCONTEXT_TOKENIZER_URL": "https://[broken/tokenize"}, "valid"),
        (
            {"INCONTEXT_TOKENIZER_URL": "https://user:pass@example.test/tokenize"},
            "credentials",
        ),
        (
            {"INCONTEXT_TOKENIZER_URL": "https://example.test/tokenize#fragment"},
            "fragment",
        ),
        (
            {
                "INCONTEXT_TOKENIZER_URL": "https://example.test/tokenize",
                "INCONTEXT_TOKENIZER_USER_AGENT": "  ",
            },
            "must not be blank",
        ),
        (
            {
                "INCONTEXT_TOKENIZER_URL": "https://example.test/tokenize",
                "INCONTEXT_TOKENIZER_TIMEOUT_SECONDS": "0",
            },
            "greater than",
        ),
        (
            {
                "INCONTEXT_TOKENIZER_URL": "https://example.test/tokenize",
                "INCONTEXT_TOKENIZER_TIMEOUT_SECONDS": "not-a-number",
            },
            "float",
        ),
    ],
)
def test_backend_rejects_unsafe_environment(
    environment: Mapping[str, str],
    message: str,
) -> None:
    with pytest.raises(VllmBackend.VllmBackendError, match=message):
        make_backend(environment=environment)


def test_backend_accepts_http_url_with_query() -> None:
    """Accept ordinary HTTP transport components after strict validation."""

    backend, _ = make_backend(
        environment={
            "INCONTEXT_TOKENIZER_URL": "http://127.0.0.1:8080/tokenize?mode=1",
        },
    )
    assert backend._environment.tokenizer_url.endswith("?mode=1")


def test_backend_accepts_an_injected_environment() -> None:
    with patch.dict(
        "os.environ",
        {"INCONTEXT_TOKENIZER_URL": "https://injected.test/tokenize"},
        clear=True,
    ):
        environment = VllmEnvironment()
    backend = VllmBackend(environment=environment, opener=RecordingOpener([]))
    assert backend._environment is environment


def test_backend_sends_exact_request_and_caches_result() -> None:
    backend, opener = make_backend([response(321)])
    request = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "Привет"}],
        "tools": [{"type": "function", "function": {"name": "test"}}],
    }
    assert backend.count(request, context_length=65_536) == 321
    assert backend.count(request, context_length=65_536) == 321
    assert len(opener.calls) == 1
    http_request, timeout = opener.calls[0]
    assert http_request.full_url == "https://inference.example/tokenize"
    assert http_request.method == "POST"
    assert http_request.get_header("Content-type") == "application/json"
    assert http_request.get_header("User-agent") == "incontext-tests"
    assert timeout == 3.5
    assert json.loads(http_request.data or b"") == VllmBackend._build_payload(request)


def test_backend_preserves_prompt_observable_json_key_order() -> None:
    """Keep schema insertion order identical to the inference request.

    Chat templates can iterate JSON-schema mappings in insertion order, so
    recursively sorting keys before ``/tokenize`` may render a different prompt
    from the one vLLM receives for generation.  Requests with different schema
    order must also occupy different cache entries instead of sharing a count.
    """

    backend, opener = make_backend([response(10), response(11)])
    first_properties = {
        "z_first": {"type": "string"},
        "a_second": {"type": "string"},
    }
    second_properties = {
        "a_second": {"type": "string"},
        "z_first": {"type": "string"},
    }

    def request(properties: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": "qwen",
            "messages": [{"role": "user", "content": "use the tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ordered",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                        },
                    },
                },
            ],
        }

    first_request = request(first_properties)
    second_request = request(second_properties)
    assert backend.count(first_request, context_length=65_536) == 10
    assert backend.count(second_request, context_length=65_536) == 11

    assert len(opener.calls) == 2
    first_wire = opener.calls[0][0].data
    assert (
        first_wire
        == json.dumps(
            VllmBackend._build_payload(first_request),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    )
    assert first_wire.index(b'"z_first"') < first_wire.index(b'"a_second"')


def test_cache_does_not_bypass_context_length_validation() -> None:
    backend, opener = make_backend([response(1), response(1)])
    request = {"model": "qwen", "messages": []}
    assert backend.count(request, context_length=65_536) == 1
    with pytest.raises(VllmBackend.VllmBackendError, match="does not match"):
        backend.count(request, context_length=32_000)
    assert len(opener.calls) == 2


def test_count_honors_positive_prompt_truncation() -> None:
    """Return the count after vLLM truncates a long rendered prompt.

    The tokenize endpoint reports the complete rendered prompt, whereas chat
    generation applies ``truncate_prompt_tokens`` before inference.  Capping
    the raw exact count reproduces the provider-visible input size and avoids
    unnecessary compression caused by budgeting from tokens vLLM discards.
    """

    backend, _ = make_backend([response(120)])
    request = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "long prompt"}],
        "truncate_prompt_tokens": 50,
    }

    assert backend.count(request, context_length=65_536) == 50


def test_count_honors_extra_body_prompt_truncation_override() -> None:
    """Apply the final OpenAI wire value when extra_body overrides truncation.

    OpenAI clients shallow-merge ``extra_body`` after ordinary parameters.
    Mirroring that precedence keeps counting aligned when a caller replaces a
    top-level truncation limit without mutating the request passed to Hermes.
    """

    backend, _ = make_backend([response(120)])
    request = {
        "model": "qwen",
        "messages": [],
        "truncate_prompt_tokens": 80,
        "extra_body": {"truncate_prompt_tokens": 30},
    }

    assert backend.count(request, context_length=65_536) == 30


@pytest.mark.parametrize(
    ("wire_value", "expected"),
    [(True, 1), ("2", 2), ("50.0", 50), ("1_0", 10), (3.0, 3)],
)
def test_count_honors_prompt_truncation_values_coerced_by_vllm(
    wire_value: Any,
    expected: int,
) -> None:
    """Match vLLM ChatCompletionRequest's non-strict integer validation.

    vLLM coerces JSON booleans, integer and zero-fraction decimal strings, and
    integral floats before applying prompt truncation.  Because ``/tokenize``
    returns the untruncated rendering, ignoring an accepted wire value counts
    tokens generation drops and can trigger premature compression.
    """

    backend, _ = make_backend([response(120)])

    assert (
        backend.count(
            {
                "model": "qwen",
                "messages": [],
                "truncate_prompt_tokens": wire_value,
            },
            context_length=65_536,
        )
        == expected
    )


@pytest.mark.parametrize(("wire_value", "expected"), [(0, 0), ("0", 0), (False, 0)])
def test_count_honors_zero_prompt_truncation(
    wire_value: Any,
    expected: int,
) -> None:
    """Mirror vLLM's valid empty prompt after non-strict coercion.

    ChatCompletionRequest accepts integer zero, its string form, and boolean
    false as a zero-token truncation bound.  Returning the raw tokenizer count
    for those values invents prompt tokens generation discards and can trigger
    unnecessary compression instead of exposing the full output window.
    """

    backend, _ = make_backend([response(120)])

    assert (
        backend.count(
            {
                "model": "qwen",
                "messages": [],
                "truncate_prompt_tokens": wire_value,
            },
            context_length=65_536,
        )
        == expected
    )


@pytest.mark.parametrize(
    "wire_value",
    [1.5, float("nan"), "invalid", "\u0661", object()],
)
def test_count_does_not_invent_invalid_prompt_truncation_coercions(
    wire_value: Any,
) -> None:
    """Keep the raw safe count for values vLLM cannot coerce to an integer.

    Treating a rejected or fractional value as a truncation limit could
    undercount a prompt if the provider ignores or rejects that field.  Only
    coercions known to match the provider contract may reduce the tokenizer's
    complete rendered count.
    """

    backend, _ = make_backend([response(120)])

    assert (
        backend.count(
            {
                "model": "qwen",
                "messages": [],
                "truncate_prompt_tokens": wire_value,
            },
            context_length=65_536,
        )
        == 120
    )


def test_count_does_not_cap_multimodal_prompt_after_media_expansion() -> None:
    """Keep vLLM's final expanded count for truncated multimodal input.

    vLLM applies ``truncate_prompt_tokens`` to rendered text token IDs before
    image tokens replace their placeholders.  Therefore a 50-token text limit
    can still produce a 120-token model prompt; applying ``min(count, 50)``
    would over-allocate output and violate the compression window.  The
    provider-visible ``extra_body.messages`` override is authoritative here.
    """

    backend, _ = make_backend([response(120)])
    request = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "superseded text"}],
        "truncate_prompt_tokens": 50,
        "extra_body": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:"}},
                    ],
                },
            ],
        },
    }

    assert backend.count(request, context_length=65_536) == 120


def test_count_detects_reusable_multimodal_content_materialized_by_openai() -> None:
    """Keep the expanded count for reusable non-list content collections.

    The OpenAI Python client accepts an iterable of content parts and converts
    ``dict_values`` to a JSON array before vLLM renders it.  Looking only for
    sequences misclassifies the image as text-only and clamps the expanded
    prompt, which over-allocates completion tokens.
    """

    backend, _ = make_backend([response(120)])
    request = {
        "model": "qwen",
        "messages": [
            {
                "role": "user",
                "content": {
                    "text": {"type": "text", "text": "describe"},
                    "image": {
                        "type": "image_url",
                        "image_url": {"url": "data:"},
                    },
                }.values(),
            },
        ],
        "truncate_prompt_tokens": 50,
    }

    assert backend.count(request, context_length=65_536) == 120


def test_payload_materializes_reusable_tool_collections() -> None:
    """Tokenize every tool that the OpenAI client serializes on the wire.

    OpenAI accepts reusable iterables and materializes ``dict_values`` into a
    JSON array.  Dropping that collection from ``/tokenize`` undercounts tool
    schemas even though generation receives them, allowing an unsafe output
    budget whenever those schemas cross the compression boundary.
    """

    tools_by_name = {
        "first": {"type": "function", "function": {"name": "first"}},
        "second": {"type": "function", "function": {"name": "second"}},
    }

    payload = VllmBackend._build_payload(
        {
            "model": "qwen",
            "messages": [],
            "tools": tools_by_name.values(),
        },
    )

    assert payload["tools"] == list(tools_by_name.values())


@pytest.mark.parametrize(
    ("wire_value", "expected"),
    [(90, 10), ("90", 10), (0, 100), (-1, None), ("invalid", None)],
)
def test_output_budget_limit_matches_vllm_truncation_validation(
    wire_value: Any,
    expected: int | None,
) -> None:
    """Couple completion length to vLLM's explicit input truncation cap.

    vLLM rejects a chat request unless ``truncate_prompt_tokens`` is at most
    ``context_length - max_tokens``.  A short actual prompt does not relax that
    schema invariant, so the backend must expose ``C - T`` as an independent
    ceiling while leaving the dynamic ``-1`` sentinel and invalid values to
    provider validation.
    """

    backend, _ = make_backend()

    assert (
        backend.output_budget_limit(
            {
                "model": "qwen",
                "messages": [],
                "truncate_prompt_tokens": wire_value,
            },
            context_length=100,
        )
        == expected
    )


def test_multimodal_truncation_still_limits_vllm_output_budget() -> None:
    """Separate expanded prompt counting from vLLM's request validation.

    Media expansion prevents the textual truncation value from capping the
    final prompt count, but vLLM still validates that same wire value against
    ``context_length - max_tokens`` before rendering media.  Exact counting
    must therefore keep the expanded tokenizer result while the independent
    output ceiling remains active.
    """

    backend, _ = make_backend()
    request = {
        "model": "qwen",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:"}}],
            },
        ],
        "truncate_prompt_tokens": 90,
    }

    assert VllmBackend._prompt_truncation_limit(request) is None
    assert backend.output_budget_limit(request, context_length=100) == 10


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ("invalid", False),
        (["invalid", {"content": "plain"}], False),
        ([{"content": {"image_url": "data:"}}], True),
        ([{"content": [123]}], True),
        ([{"content": ["plain string"]}], False),
        ([{"content": [{"type": "text", "text": "plain"}]}], False),
        ([{"content": [{"type": "input_text", "text": "plain"}]}], False),
    ],
)
def test_multimodal_detection_is_conservative_for_wire_message_shapes(
    messages: Any,
    expected: bool,
) -> None:
    """Distinguish known text-only forms from media or ambiguous content.

    A false negative can undercount expanded media and exceed the compression
    boundary, while a false positive merely forgoes an optimization and keeps
    the raw tokenizer count.  Non-message values remain the provider's
    validation concern and do not themselves imply media expansion.
    """

    assert VllmBackend._has_multimodal_content({"messages": messages}) is expected


@pytest.mark.parametrize("part_type", ["output_text", "refusal", "thinking"])
def test_count_truncates_vllm_structured_text_content_parts(
    part_type: str,
) -> None:
    """Do not mistake vLLM-supported structured text for multimodal input.

    vLLM converts refusal, thinking, and output-text content parts to ordinary
    strings before tokenization.  No media placeholders are expanded, so a
    positive truncation value bounds the final provider-visible prompt exactly.
    Classifying these parts as media would under-allocate completion space and
    can trigger premature compression.
    """

    backend, _ = make_backend([response(120)])
    request = {
        "model": "qwen",
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": part_type, part_type: "plain text"}],
            },
        ],
        "truncate_prompt_tokens": 50,
    }

    assert backend.count(request, context_length=65_536) == 50


def test_count_truncates_vllm_tool_reference_content_parts() -> None:
    """Apply text-only prompt truncation to deferred tool references.

    vLLM explicitly passes ``tool_reference`` parts through chat-template
    expansion without creating media placeholders.  Generation therefore
    truncates the final token sequence exactly like other structured text.
    Treating the reference as media keeps the raw tokenizer count, causing
    premature compression and a needlessly smaller output allowance.
    """

    backend, _ = make_backend([response(120)])
    request = {
        "model": "qwen",
        "messages": [
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": [{"type": "tool_reference", "name": "lookup"}],
            },
        ],
        "truncate_prompt_tokens": 50,
    }

    assert backend.count(request, context_length=65_536) == 50


def test_cached_raw_count_supports_distinct_truncation_limits() -> None:
    """Reuse one rendered count without conflating provider-visible limits.

    Truncation does not alter chat-template rendering, so otherwise identical
    requests should share the expensive tokenizer response.  The cache must
    retain the raw count and apply each request's limit afterwards; caching an
    already-truncated value would let the first caller poison later budgets.
    """

    backend, opener = make_backend([response(120)])
    base = {"model": "qwen", "messages": []}

    assert (
        backend.count(
            {**base, "truncate_prompt_tokens": 50},
            context_length=65_536,
        )
        == 50
    )
    assert (
        backend.count(
            {**base, "truncate_prompt_tokens": 20},
            context_length=65_536,
        )
        == 20
    )
    assert len(opener.calls) == 1


def test_count_uses_disaggregated_decode_prompt_token_ids() -> None:
    """Count the token IDs vLLM uses instead of rendering stale messages.

    On a disaggregated decode node, ``kv_transfer_params.prompt_token_ids``
    bypass chat-template rendering because the prefill node has already
    produced the prompt.  Incontext must use that list length, while still
    calling ``/tokenize`` once to verify the server's advertised context
    length.  Token-ID changes can then reuse that validation cache safely.
    """

    backend, opener = make_backend([response(999)])
    request = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "not the decode prompt"}],
        "truncate_prompt_tokens": -1,
        "kv_transfer_params": {"prompt_token_ids": [1, 2]},
        "extra_body": {
            "kv_transfer_params": {
                "prompt_token_ids": (0, 4, 8, 15, 16, 23, 42),
            },
        },
    }

    assert backend.count(request, context_length=65_536) == 7
    request["extra_body"]["kv_transfer_params"]["prompt_token_ids"] = (3, 5, 8)
    assert backend.count(request, context_length=65_536) == 3
    assert len(opener.calls) == 1


@pytest.mark.parametrize(
    "token_ids",
    [[True], [-1], ["1"]],
)
def test_count_rejects_invalid_disaggregated_prompt_token_ids(
    token_ids: Any,
) -> None:
    """Fail open rather than claim exactness for malformed reused prompts.

    vLLM requires a non-empty integer token sequence.  Falling back to message
    rendering for an invalid but provider-visible sequence could allocate an
    unrelated output budget, so the backend must surface a contract error and
    let the middleware use its conservative rough estimator.
    """

    backend, opener = make_backend()

    with pytest.raises(
        VllmBackend.VllmBackendError,
        match="prompt_token_ids",
    ):
        backend.count(
            {
                "model": "qwen",
                "messages": [],
                "kv_transfer_params": {"prompt_token_ids": token_ids},
            },
            context_length=65_536,
        )
    assert opener.calls == []


@pytest.mark.parametrize("token_ids", [None, False, 0, "", {}, [], ()])
def test_falsy_disaggregated_prompt_ids_render_messages_normally(
    token_ids: Any,
) -> None:
    """Treat falsy decode-side IDs as absent exactly as vLLM does.

    vLLM consumes ``prompt_token_ids`` with an ``or None`` fallback and follows
    normal message rendering for any falsy JSON value.  Rejecting those valid
    sentinels disables exact counting and applies the rough fallback margin to
    a request generation can serve normally.
    """

    backend, opener = make_backend([response(120)])

    assert (
        backend.count(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "render me"}],
                "kv_transfer_params": {"prompt_token_ids": token_ids},
            },
            context_length=65_536,
        )
        == 120
    )
    assert len(opener.calls) == 1


def test_count_ignores_kv_transfer_metadata_without_reused_prompt_ids() -> None:
    """Render messages normally when transfer metadata has no prompt IDs.

    ``kv_transfer_params`` carries fields for several transfer phases.  Only a
    concrete ``prompt_token_ids`` key replaces the chat prompt; unrelated
    metadata must not disable the ordinary exact tokenizer path.
    """

    backend, opener = make_backend([response(120)])

    assert (
        backend.count(
            {
                "model": "qwen",
                "messages": [],
                "kv_transfer_params": {"do_remote_decode": False},
            },
            context_length=65_536,
        )
        == 120
    )
    assert len(opener.calls) == 1


@pytest.mark.parametrize("wire_value", [-1, "-1"])
def test_count_uses_raw_count_for_dynamic_minus_one_prompt_truncation(
    wire_value: Any,
) -> None:
    """Keep exact counting for vLLM's dynamic ``-1`` truncation sentinel.

    Incontext emits output ``O <= W - P`` and validates compression window
    ``W <= C``.  vLLM therefore resolves ``-1`` to an input limit
    ``C - O >= C - W + P >= P``: a request below the boundary cannot truncate
    its raw P-token prompt.  Skipping ``/tokenize`` would unnecessarily replace
    this exact count with a margin-adjusted rough estimate.
    """

    backend, opener = make_backend([response(60, 100)])

    assert (
        backend.count(
            {
                "model": "qwen",
                "messages": [],
                "truncate_prompt_tokens": wire_value,
            },
            context_length=100,
        )
        == 60
    )
    assert len(opener.calls) == 1


def test_default_opener_is_resolved_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = RecordingOpener([response()])
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    with patch.dict(
        "os.environ",
        {"INCONTEXT_TOKENIZER_URL": "https://inference.test/tokenize"},
        clear=True,
    ):
        backend = VllmBackend()
    assert (
        backend.count({"model": "qwen", "messages": []}, context_length=65_536) == 123
    )


def test_lru_evicts_oldest_entry() -> None:
    backend, opener = make_backend(
        [response(1), response(2), response(3), response(4)],
        cache_entries=2,
    )
    first = {"model": "qwen", "messages": [{"content": "first"}]}
    second = {"model": "qwen", "messages": [{"content": "second"}]}
    third = {"model": "qwen", "messages": [{"content": "third"}]}
    assert backend.count(first, context_length=65_536) == 1
    assert backend.count(second, context_length=65_536) == 2
    assert backend.count(first, context_length=65_536) == 1
    assert backend.count(third, context_length=65_536) == 3
    assert backend.count(second, context_length=65_536) == 4
    assert len(opener.calls) == 4


def test_clear_cache_forces_new_request() -> None:
    backend, _ = make_backend([response(1), response(2)])
    request = {"model": "qwen", "messages": []}
    assert backend.count(request, context_length=65_536) == 1
    backend.clear_cache()
    assert backend.count(request, context_length=65_536) == 2


def test_clear_cache_invalidates_an_inflight_tokenizer_response() -> None:
    """Prevent an old HTTP result from repopulating a freshly cleared cache.

    Cache invalidation can race a slow ``/tokenize`` call during a model or
    configuration transition.  The in-flight caller may finish with its valid
    old response, but that response must not become a cache hit after
    ``clear_cache``; the next caller has to observe the new tokenizer result.
    """

    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def opener(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> FakeResponse:
        del request, timeout
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            started.set()
            assert release.wait(timeout=2)
        return FakeResponse(json.dumps(response(calls[-1])).encode())

    with patch.dict(
        "os.environ",
        {"INCONTEXT_TOKENIZER_URL": "https://inference.test/tokenize"},
        clear=True,
    ):
        backend = VllmBackend(opener=opener)
    request = {"model": "qwen", "messages": []}
    first_result: list[int] = []
    worker = threading.Thread(
        target=lambda: first_result.append(
            backend.count(request, context_length=65_536),
        ),
    )

    worker.start()
    assert started.wait(timeout=2)
    backend.clear_cache()
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert first_result == [1]
    assert backend.count(request, context_length=65_536) == 2
    assert calls == [1, 2]


def test_transport_failure_is_not_cached() -> None:
    backend, _ = make_backend([urllib.error.URLError("offline"), response(5)])
    request = {"model": "qwen", "messages": []}
    with pytest.raises(urllib.error.URLError):
        backend.count(request, context_length=65_536)
    assert backend.count(request, context_length=65_536) == 5


def test_non_serializable_payload_is_rejected() -> None:
    backend, _ = make_backend()
    with pytest.raises(TypeError):
        backend.count(
            {"model": "qwen", "messages": [object()]},
            context_length=65_536,
        )


def test_non_object_response_is_rejected() -> None:
    backend, _ = make_backend([[1, 2, 3]])
    with pytest.raises(VllmBackend.VllmBackendError, match="non-object"):
        backend.count({"model": "qwen", "messages": []}, context_length=65_536)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"max_model_len": 65_536}, "invalid count"),
        ({"count": 1}, "invalid max_model_len"),
        (response(1, 32_000), "does not match"),
    ],
)
def test_response_contract_is_validated(
    payload: dict[str, Any],
    message: str,
) -> None:
    backend, _ = make_backend([payload])
    with pytest.raises(VllmBackend.VllmBackendError, match=message):
        backend.count({"model": "qwen", "messages": []}, context_length=65_536)
