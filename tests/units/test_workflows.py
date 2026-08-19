from __future__ import annotations

from pathlib import Path

WORKFLOWS = Path(__file__).parents[2] / ".github" / "workflows"


def test_release_waits_for_every_behavioral_quality_workflow() -> None:
    """Prevent a main-branch push from publishing before its checks finish.

    GitHub starts independent push workflows concurrently, so merely running
    lint, unit, distribution, and real-Hermes checks elsewhere cannot protect
    PyPI.  The release workflow must invoke each reusable quality workflow and
    make its trusted-publishing job depend on all three successful results.
    """

    release = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")

    assert "uses: ./.github/workflows/lint.yml" in release
    assert "uses: ./.github/workflows/tests_and_coverage.yml" in release
    assert "uses: ./.github/workflows/hermes_e2e.yml" in release
    assert (
        "needs:\n      - lint\n      - tests-and-coverage\n      - hermes-e2e"
        in release
    )


def test_release_quality_workflows_expose_reusable_entry_points() -> None:
    """Keep every release gate callable and aligned with Python support.

    A local workflow reference is accepted only when its target declares
    ``workflow_call``.  Checking all three files guards against a seemingly
    harmless trigger cleanup silently breaking the dependency chain that
    protects the package index.  Lint and unit jobs must also exercise every
    interpreter promised by package metadata, including free-threaded Python.
    """

    for name in ("lint.yml", "tests_and_coverage.yml", "hermes_e2e.yml"):
        contents = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "  workflow_call:\n" in contents

    python_matrix = (
        'python-version: ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", '
        '"3.14", "3.14t", "3.15.0-beta.1"]'
    )
    for name in ("lint.yml", "tests_and_coverage.yml"):
        contents = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert python_matrix in contents


def test_distribution_workflow_imports_wheel_code_in_isolation() -> None:
    """Reject metadata-only distribution checks that accept an empty wheel.

    Installing a wheel creates importlib metadata even when it contains no
    importable ``incontext`` package.  Running from the checkout can also hide
    that defect by importing source files.  The release gate must use isolated
    Python, prove the module came from site-packages, and load both published
    plugin entry points from the installed wheel.
    """

    workflow = (WORKFLOWS / "tests_and_coverage.yml").read_text(encoding="utf-8")

    assert "wheel-check/bin/python -I" in workflow
    assert "import incontext" in workflow
    assert '"site-packages" in Path(incontext.__file__).parts' in workflow
    assert '("hermes_agent.plugins", "incontext")' in workflow
    assert '("incontext.backends", "vllm")' in workflow
    assert "selected[0].load()" in workflow
