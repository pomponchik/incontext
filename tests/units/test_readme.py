from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

README = Path(__file__).parents[2] / "README.md"


def test_readme_user_agent_matches_distribution_version() -> None:
    """Keep copied deployment configuration aligned with package metadata.

    Operators commonly copy the explicit environment example rather than rely
    on the default.  A stale literal makes server logs and proxy policy report
    the wrong client release even when the implementation itself was updated,
    so every documented value must track the installed distribution version.
    """

    contents = README.read_text(encoding="utf-8")
    expected = f"incontext/{version('incontext')}"

    assert contents.count(expected) == 2
    assert "incontext/0.0.1" not in contents


def test_readme_backend_example_runs_on_python_38() -> None:
    """Use annotation syntax accepted by the declared oldest interpreter.

    The package supports Python 3.8, where built-in generic aliases such as
    ``dict[str, Any]`` cannot be resolved by ``typing.get_type_hints`` even
    under postponed annotations.  The public plugin example must therefore use
    ``typing.Dict`` so a copied backend remains importable and introspectable
    across the complete supported matrix.
    """

    contents = README.read_text(encoding="utf-8")

    assert "from typing import Any, Dict" in contents
    assert "request: Dict[str, Any]" in contents
