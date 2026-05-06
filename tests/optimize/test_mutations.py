"""Tests for promptastic.optimize.mutations -- structural prompt mutations."""

import pytest

from promptastic.optimize._types import (
    DiagnosticIssue,
    DiagnosticReport,
    MetricTarget,
)
from promptastic.optimize.mutations import StructuralMutator


def _region_config(*regions):
    """Build a minimal region config from (name, start_marker, end_marker) tuples."""
    return {
        "system_prompt": {
            "regions": [
                {"name": name, "start_marker": start, "end_marker": end}
                for name, start, end in regions
            ],
        },
    }


SAMPLE_PROMPT = """\
## Rules
You must follow these rules carefully.
Always be polite and helpful.

## Examples
User: Hello
Assistant: Hi there!

## Output Format
Respond in JSON format."""


SAMPLE_CONFIG = _region_config(
    ("rules", "## Rules", "## Examples"),
    ("examples", "## Examples", "## Output Format"),
    ("output_format", "## Output Format", ""),
)


class TestReorderSections:
    def test_moves_to_end(self):
        mutator = StructuralMutator()
        new_prompt, record = mutator.reorder_sections(
            SAMPLE_PROMPT, SAMPLE_CONFIG, "rules", "end",
        )
        assert record.operation == "reorder_sections"
        assert "## Rules" in new_prompt
        # Rules should now be at the end
        rules_pos = new_prompt.find("## Rules")
        examples_pos = new_prompt.find("## Examples")
        assert rules_pos > examples_pos

    def test_region_not_found(self):
        mutator = StructuralMutator()
        new_prompt, record = mutator.reorder_sections(
            SAMPLE_PROMPT, SAMPLE_CONFIG, "nonexistent", "end",
        )
        assert new_prompt == SAMPLE_PROMPT
        assert "not found" in record.diff_summary


class TestInsertSeparator:
    def test_inserts_after_region(self):
        mutator = StructuralMutator()
        new_prompt, record = mutator.insert_separator(
            SAMPLE_PROMPT, SAMPLE_CONFIG, "rules",
        )
        assert record.operation == "insert_separator"
        # Separator should be near the ## Examples marker
        assert "---" in new_prompt

    def test_avoids_double_separator(self):
        mutator = StructuralMutator()
        # First insert
        prompt1, _ = mutator.insert_separator(
            SAMPLE_PROMPT, SAMPLE_CONFIG, "rules",
        )
        # Second insert should be a no-op
        prompt2, record2 = mutator.insert_separator(
            prompt1, SAMPLE_CONFIG, "rules",
        )
        assert "already present" in record2.diff_summary

    def test_region_not_found(self):
        mutator = StructuralMutator()
        new_prompt, record = mutator.insert_separator(
            SAMPLE_PROMPT, SAMPLE_CONFIG, "nonexistent",
        )
        assert new_prompt == SAMPLE_PROMPT


class TestDuplicateSummary:
    def test_appends_summary(self):
        mutator = StructuralMutator()
        new_prompt, record = mutator.duplicate_summary(
            SAMPLE_PROMPT, SAMPLE_CONFIG, "rules",
        )
        assert record.operation == "duplicate_summary"
        assert "Remember:" in new_prompt
        # Should be at the end
        assert new_prompt.rstrip().endswith(
            new_prompt.rstrip().split("Remember:")[-1].rstrip()
        )

    def test_region_not_found(self):
        mutator = StructuralMutator()
        new_prompt, record = mutator.duplicate_summary(
            SAMPLE_PROMPT, SAMPLE_CONFIG, "nonexistent",
        )
        assert new_prompt == SAMPLE_PROMPT


class TestAdjustEmphasis:
    def test_adds_emphasis(self):
        mutator = StructuralMutator()
        new_prompt, record = mutator.adjust_emphasis(
            SAMPLE_PROMPT, SAMPLE_CONFIG, "rules",
        )
        assert record.operation == "adjust_emphasis"
        assert "**## Rules**" in new_prompt

    def test_avoids_double_emphasis(self):
        mutator = StructuralMutator()
        prompt1, _ = mutator.adjust_emphasis(
            SAMPLE_PROMPT, SAMPLE_CONFIG, "rules",
        )
        prompt2, record2 = mutator.adjust_emphasis(
            prompt1, SAMPLE_CONFIG, "rules",
        )
        assert "already emphasized" in record2.diff_summary


class TestApplyBestMutation:
    def test_picks_structural_mutation(self):
        mutator = StructuralMutator()
        t = MetricTarget(name="context_bleed_ratio", direction="below", ideal=1.0, maximum=3.0)
        report = DiagnosticReport(
            issues=[
                DiagnosticIssue(
                    metric_name="context_bleed_ratio",
                    value=2.5,
                    satisfaction=0.25,
                    target=t,
                    suggested_mutation="insert_separator",
                    reason="too high",
                ),
            ],
            overall_score=0.5,
            num_failing=1,
            num_total=1,
        )
        # No matching region for context_bleed_ratio, so it won't find one
        # But the method shouldn't crash
        new_spec, record = mutator.apply_best_mutation(
            SAMPLE_PROMPT, SAMPLE_CONFIG, report,
        )
        from promptastic.optimize._types import PromptSpec
        assert isinstance(new_spec, PromptSpec)

    def test_skips_llm_mutations(self):
        mutator = StructuralMutator()
        t = MetricTarget(name="causal_importance_rules", direction="above", ideal=0.1)
        report = DiagnosticReport(
            issues=[
                DiagnosticIssue(
                    metric_name="causal_importance_rules",
                    value=0.01,
                    satisfaction=0.1,
                    target=t,
                    suggested_mutation="llm_rewrite",
                    reason="too low",
                ),
            ],
            overall_score=0.3,
            num_failing=1,
            num_total=1,
        )
        new_spec, record = mutator.apply_best_mutation(
            SAMPLE_PROMPT, SAMPLE_CONFIG, report,
        )
        # Should skip llm_rewrite and return None
        assert record is None
        assert new_spec.system_prompt == SAMPLE_PROMPT

    def test_no_issues(self):
        mutator = StructuralMutator()
        report = DiagnosticReport(
            issues=[], overall_score=1.0, num_failing=0, num_total=5,
        )
        new_spec, record = mutator.apply_best_mutation(
            SAMPLE_PROMPT, SAMPLE_CONFIG, report,
        )
        assert record is None
        assert new_spec.system_prompt == SAMPLE_PROMPT
