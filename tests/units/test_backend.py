from __future__ import annotations

from typing import Any, cast, get_type_hints
from unittest.mock import patch

import pytest
from pristan.errors import OneResolutionError, PrimadonnaPluginError

import incontext
from incontext.backend import Backend, backends
from incontext.budget import DynamicOutputBudget
from incontext.hermes import apply_incontext, register
from incontext.settings import load_settings
from incontext.vllm import VllmBackend


class ReplacementBackend(Backend):
    @property
    def source(self) -> str:
        return "replacement"

    def count(
        self,
        request: dict[str, Any],
        *,
        context_length: int,
    ) -> int:
        del request
        return context_length

    def clear_cache(self) -> None:
        return None


def test_backend_contract_is_public_and_abstract() -> None:
    """Expose the extension contract at the documented package boundary."""
    assert incontext.Backend is Backend
    assert incontext.backends is backends
    with pytest.raises(TypeError):
        Backend()  # type: ignore[abstract]


def test_backend_preserves_output_budget_alias_by_default() -> None:
    """Keep third-party backend behavior unchanged after extending the API.

    Existing plugins inherit the neutral implementation, so provider-selected
    fields remain untouched unless a backend explicitly documents a wire-level
    incompatibility such as vLLM Chat Completions' ignored Responses alias.
    """
    assert ReplacementBackend().output_budget_field("max_output_tokens") == (
        "max_output_tokens"
    )


def test_backend_adds_no_provider_output_limit_by_default() -> None:
    """Keep third-party backends neutral when no wire constraint is declared.

    Provider-specific schemas may couple prompt truncation to output length,
    but existing backend plugins only promise exact token counting.  The
    default extension must therefore leave their dynamic budget untouched
    until a backend explicitly reports an additional limit.
    """
    assert (
        ReplacementBackend().output_budget_limit(
            {"model": "replacement", "messages": []},
            context_length=1024,
        )
        is None
    )


def test_public_type_hints_resolve_on_every_supported_python() -> None:
    """Keep public annotations introspectable down to the Python 3.8 floor.

    Postponed annotations avoid import-time evaluation but do not backport
    PEP 585 built-in generics or PEP 604 unions.  Plugin frameworks commonly
    call ``typing.get_type_hints`` on contracts and middleware, so every public
    callable must resolve at runtime on each interpreter declared in package
    metadata, not merely parse successfully there.
    """
    targets = (
        cast(property, Backend.__dict__["source"]).fget,
        Backend.count,
        Backend.clear_cache,
        Backend.output_budget_field,
        Backend.output_budget_limit,
        Backend.coerce_output_budget,
        backends,
        DynamicOutputBudget.__init__,
        DynamicOutputBudget.__call__,
        apply_incontext,
        register,
        VllmBackend.__init__,
        cast(property, VllmBackend.__dict__["source"]).fget,
        VllmBackend.count,
        VllmBackend.clear_cache,
        VllmBackend.output_budget_field,
        VllmBackend.output_budget_limit,
        VllmBackend.coerce_output_budget,
        load_settings,
    )

    for target in targets:
        assert target is not None
        assert get_type_hints(target)


def test_bundled_vllm_provider_is_selected_by_name() -> None:
    # Importing the entry point module is what pristan package discovery does.
    import incontext.vllm_provider  # noqa: F401, PLC0415

    with patch.dict(
        "os.environ",
        {"INCONTEXT_TOKENIZER_URL": "https://inference.test/tokenize"},
        clear=True,
    ):
        backend = backends["vllm"].one()

    assert isinstance(backend, VllmBackend)


def test_named_backend_can_replace_the_bundled_backend() -> None:
    """Resolve one named plugin and reject an ambiguous duplicate provider.

    Third-party packages register through the slot without repeating its
    uniqueness policy.  A second distribution using the same selected name
    must fail during discovery instead of making startup order-dependent.
    """
    expected = ReplacementBackend()

    @backends.plugin("unit_replacement")
    def provide_replacement() -> Backend:
        return expected

    assert backends["unit_replacement"].one() is expected

    with pytest.raises(PrimadonnaPluginError):

        @backends.plugin("unit_replacement")
        def provide_duplicate() -> Backend:
            return ReplacementBackend()


def test_unknown_backend_name_fails_single_plugin_resolution() -> None:
    with pytest.raises(OneResolutionError, match="cannot choose one"):
        backends["missing_backend"].one()
