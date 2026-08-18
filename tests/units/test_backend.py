from __future__ import annotations

from typing import Any, get_type_hints
from unittest.mock import patch

import pytest
from pristan.errors import OneResolutionError

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
        return context_length

    def clear_cache(self) -> None:
        return None


def test_backend_contract_is_abstract() -> None:
    with pytest.raises(TypeError):
        Backend()  # type: ignore[abstract]


def test_public_type_hints_resolve_on_every_supported_python() -> None:
    """Keep public annotations introspectable down to the Python 3.8 floor.

    Postponed annotations avoid import-time evaluation but do not backport
    PEP 585 built-in generics or PEP 604 unions.  Plugin frameworks commonly
    call ``typing.get_type_hints`` on contracts and middleware, so every public
    callable must resolve at runtime on each interpreter declared in package
    metadata, not merely parse successfully there.
    """

    targets = (
        Backend.count,
        DynamicOutputBudget.__init__,
        DynamicOutputBudget.__call__,
        apply_incontext,
        register,
        VllmBackend.__init__,
        VllmBackend.count,
        load_settings,
    )

    for target in targets:
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
    expected = ReplacementBackend()

    @backends.plugin("unit_replacement", unique=True)
    def provide_replacement() -> Backend:
        return expected

    assert backends["unit_replacement"].one() is expected


def test_unknown_backend_name_fails_single_plugin_resolution() -> None:
    with pytest.raises(OneResolutionError, match="cannot choose one"):
        backends["missing_backend"].one()
