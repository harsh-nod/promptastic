"""Tests for optimize.verification module."""

from promptastic.optimize._types import DiagnosticIssue, DiagnosticReport, MetricTarget
from promptastic.optimize.structure import parse_prompt
from promptastic.optimize.verification import VerificationResult, verify_candidate


def _report_for(section: str) -> DiagnosticReport:
    metric = f"retention_ratio_{section}"
    issue = DiagnosticIssue(
        metric_name=metric,
        value=0.2,
        satisfaction=0.2,
        target=MetricTarget(name=metric, direction="above", ideal=0.8),
        suggested_mutation="meta_rewrite",
        reason="low score",
    )
    return DiagnosticReport(issues=[issue], overall_score=0.5, num_failing=1, num_total=3)


def test_accepts_when_targeted_section_changes():
    current = parse_prompt(
        "## Rules\nOld rules.\n\n## Examples\nExample.\n",
        {
            "system_prompt": {
                "regions": [
                    {"name": "rules", "start_marker": "## Rules"},
                    {"name": "examples", "start_marker": "## Examples"},
                ],
            }
        },
    )
    candidate = parse_prompt(
        "## Rules\nImproved rules.\n\n## Examples\nExample.\n",
        {
            "system_prompt": {
                "regions": [
                    {"name": "rules", "start_marker": "## Rules"},
                    {"name": "examples", "start_marker": "## Examples"},
                ],
            }
        },
    )

    result = verify_candidate(current, candidate, _report_for("rules"))
    assert isinstance(result, VerificationResult)
    assert result.accepted
    assert result.changed_sections == ["rules"]


def test_rejects_when_targeted_section_unchanged():
    current = parse_prompt(
        "## Rules\nOld rules.\n\n## Examples\nExample.\n",
        {
            "system_prompt": {
                "regions": [
                    {"name": "rules", "start_marker": "## Rules"},
                    {"name": "examples", "start_marker": "## Examples"},
                ],
            }
        },
    )
    candidate = parse_prompt(
        "## Rules\nOld rules.\n\n## Examples\nExample.\n",
        {
            "system_prompt": {
                "regions": [
                    {"name": "rules", "start_marker": "## Rules"},
                    {"name": "examples", "start_marker": "## Examples"},
                ],
            }
        },
    )

    result = verify_candidate(current, candidate, _report_for("rules"))
    assert not result.accepted
    assert "no_sections_changed" in result.reasons
    assert any(reason.startswith("target_section_unchanged") for reason in result.reasons)


def test_rejects_when_length_delta_too_large():
    current = parse_prompt(
        "## Rules\nOld rules.\n",
        {"system_prompt": {"regions": [{"name": "rules", "start_marker": "## Rules"}]}},
    )
    candidate = parse_prompt(
        "## Rules\n" + ("Very long text.\n" * 50),
        {"system_prompt": {"regions": [{"name": "rules", "start_marker": "## Rules"}]}},
    )

    result = verify_candidate(current, candidate, _report_for("rules"), length_tolerance=0.2)
    assert not result.accepted
    assert any(reason.startswith("length_delta:") for reason in result.reasons)
