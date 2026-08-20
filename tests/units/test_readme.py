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


def test_readme_pypi_install_cannot_select_an_older_release() -> None:
    """Tie the documented PyPI path to the behavior described by this release.

    Development documentation can advance before its distribution reaches
    PyPI.  An unbounded install then succeeds with an older wheel lacking the
    documented preflight and auxiliary safety mechanisms, so operators must be
    given a version floor that fails visibly until the matching release exists.
    """

    contents = README.read_text(encoding="utf-8")
    current = version("incontext")

    assert f"python -m pip install 'incontext>={current}'" in contents
    assert "Until the first PyPI release" not in contents


def test_readme_does_not_describe_a_mutable_branch_as_reviewed_revision() -> None:
    """Avoid an immutable-sounding assurance for a moving Git branch.

    The development installation intentionally tracks ``develop`` so a commit
    hash cannot remain current in this repository's own README.  Calling that
    mutable target a reviewed revision would overstate what the command pins;
    the text must identify it honestly as the current branch instead.
    """

    contents = README.read_text(encoding="utf-8")

    assert "install the current `develop`\nbranch" in contents
    assert "reviewed `develop`" not in contents


def test_readme_documents_the_viable_output_algorithm() -> None:
    contents = README.read_text(encoding="utf-8")

    assert "## Algorithm" in contents
    assert "INCONTEXT_MIN_OUTPUT_TOKENS" in contents
    assert "preflight_pressure = P + R - 1" in contents
    assert "W - P < R" in contents
    assert "exactly `R` tokens of output space remains valid" in contents
    assert "does not turn an\nexhausted window into `max_tokens=1`" in contents
