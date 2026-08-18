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
    """Keep every release gate callable from the publishing workflow.

    A local workflow reference is accepted only when its target declares
    ``workflow_call``.  Checking all three files guards against a seemingly
    harmless trigger cleanup silently breaking the dependency chain that
    protects the package index.
    """

    for name in ("lint.yml", "tests_and_coverage.yml", "hermes_e2e.yml"):
        contents = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "  workflow_call:\n" in contents
