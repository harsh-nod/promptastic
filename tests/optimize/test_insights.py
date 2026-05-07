"""Tests for developer-facing insights helpers."""

from promptastic.optimize._types import DiagnosticIssue, DiagnosticReport, MetricTarget
from promptastic.optimize.insights import (
    build_fix_plan,
    changed_sections,
    summarize_prompt_diff,
)
from promptastic.optimize.structure import parse_prompt


def _config():
    return {
        "system_prompt": {
            "regions": [
                {"name": "rules", "start_marker": "## Rules"},
                {"name": "examples", "start_marker": "## Examples"},
            ],
        }
    }


def _report():
    issue = DiagnosticIssue(
        metric_name="retention_ratio_rules",
        value=0.2,
        satisfaction=0.3,
        target=MetricTarget(name="retention_ratio_rules", direction="above", ideal=0.8),
        suggested_mutation="duplicate_summary",
        reason="Low retention on rules section",
    )
    return DiagnosticReport(issues=[issue], overall_score=0.5, num_failing=1, num_total=5)


def test_build_fix_plan_generates_actions():
    plan = build_fix_plan(_report())
    assert len(plan) == 1
    assert "retention_ratio_rules" in plan[0]


def test_changed_sections_detects_body_change():
    before = parse_prompt(
        "## Rules\nOld rules.\n\n## Examples\nExample.\n",
        _config(),
    )
    after = parse_prompt(
        "## Rules\nImproved rules.\n\n## Examples\nExample.\n",
        _config(),
    )
    sections = changed_sections(before, after)
    assert sections == ["rules"]


def test_summarize_prompt_diff_returns_snippets():
    before = parse_prompt(
        "## Rules\nOld rules.\n\n## Examples\nExample.\n",
        _config(),
    )
    after = parse_prompt(
        "## Rules\nImproved rules.\n\n## Examples\nExample.\n",
        _config(),
    )
    diff = summarize_prompt_diff(before, after, ["rules"])
    assert "Improved" in diff[0]
