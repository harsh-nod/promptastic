"""Tests for the trajectory-aware meta-rewriter (N+1 prompt generation)."""

from unittest.mock import MagicMock

import pytest

from promptastic.optimize._types import (
    DiagnosticIssue,
    DiagnosticReport,
    IterationRecord,
    MetricScore,
    MetricTarget,
    MutationRecord,
    OptimizationConfig,
    OptimizationScore,
    PromptSpec,
    TrajectoryEntry,
)
from promptastic.optimize.meta_rewriter import MetaRewriter
from promptastic.optimize.policy import CandidateDecision, CandidateEvaluator
from promptastic.optimize.structure import parse_prompt


# ======================================================================
# Helpers
# ======================================================================


def _make_score(total: float, n_sat: int = 3, n_total: int = 5) -> OptimizationScore:
    return OptimizationScore(
        total=total,
        per_metric={},
        num_satisfied=n_sat,
        num_total=n_total,
    )


def _make_record(
    iteration: int,
    score: float = 0.5,
    prompt: str = "test prompt",
    mutation: MutationRecord | None = None,
    response_samples: list[str] | None = None,
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        prompt_text=prompt,
        regions={},
        metrics={"retention_ratio_rules": 0.6, "context_bleed_ratio": 0.1},
        score=_make_score(score),
        mutation_applied=mutation,
        forward_passes=iteration + 1,
        wall_time_seconds=1.0,
        response_samples=response_samples or [],
    )


def _make_mutation(operation: str = "reorder_sections") -> MutationRecord:
    return MutationRecord(
        mutation_type="structural",
        operation=operation,
        target_region="rules",
        reason="test",
        diff_summary="test diff",
    )


def _make_targets() -> dict[str, MetricTarget]:
    return {
        "retention_ratio_rules": MetricTarget(
            name="retention_ratio_rules",
            direction="above",
            weight=1.0,
            minimum=0.3,
            ideal=0.8,
        ),
        "context_bleed_ratio": MetricTarget(
            name="context_bleed_ratio",
            direction="below",
            weight=1.0,
            ideal=0.05,
            maximum=0.3,
        ),
    }


def _make_report(n_issues: int = 2) -> DiagnosticReport:
    issues = [
        DiagnosticIssue(
            metric_name="retention_ratio_rules",
            value=0.4,
            satisfaction=0.3,
            target=MetricTarget(
                name="retention_ratio_rules", direction="above",
                minimum=0.3, ideal=0.8,
            ),
            suggested_mutation="llm_rewrite",
            reason="Low retention",
        ),
    ][:n_issues]
    return DiagnosticReport(
        issues=issues,
        overall_score=0.5,
        num_failing=n_issues,
        num_total=5,
    )


def _region_config():
    return {
        "system_prompt": {
            "regions": [
                {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
                {"name": "examples", "start_marker": "## Examples", "end_marker": ""},
            ],
        },
    }


# ======================================================================
# TrajectoryEntry type tests
# ======================================================================


class TestTrajectoryEntry:

    def test_defaults(self):
        entry = TrajectoryEntry(
            iteration=0,
            prompt_text="test",
            metrics={},
            score_total=0.5,
            score_delta=0.0,
            num_satisfied=3,
            num_total=5,
            mutation_applied=None,
        )
        assert entry.response_samples == []

    def test_with_responses(self):
        entry = TrajectoryEntry(
            iteration=0,
            prompt_text="test",
            metrics={"m1": 0.5},
            score_total=0.5,
            score_delta=0.0,
            num_satisfied=3,
            num_total=5,
            mutation_applied=None,
            response_samples=["Hello!", "Hi there."],
        )
        assert len(entry.response_samples) == 2
        assert entry.response_samples[0] == "Hello!"


class TestIterationRecordResponseSamples:

    def test_backward_compat(self):
        rec = _make_record(0)
        assert rec.response_samples == []

    def test_with_responses(self):
        rec = _make_record(0, response_samples=["foo", "bar"])
        assert rec.response_samples == ["foo", "bar"]


class TestOptimizationConfigMeta:

    def test_defaults(self):
        cfg = OptimizationConfig()
        assert cfg.meta_window_size == 5
        assert cfg.meta_include_responses is False
        assert cfg.meta_response_max_tokens == 64

    def test_custom(self):
        cfg = OptimizationConfig(
            meta_window_size=3,
            meta_include_responses=True,
            meta_response_max_tokens=128,
        )
        assert cfg.meta_window_size == 3
        assert cfg.meta_include_responses is True


# ======================================================================
# build_trajectory
# ======================================================================


class TestBuildTrajectory:

    def test_empty_history(self):
        result = MetaRewriter.build_trajectory([], window_size=5)
        assert result == []

    def test_single_entry(self):
        history = [_make_record(0, score=0.5)]
        result = MetaRewriter.build_trajectory(history, window_size=5)
        assert len(result) == 1
        assert result[0].iteration == 0
        assert result[0].score_delta == 0.0

    def test_two_entries(self):
        history = [
            _make_record(0, score=0.4),
            _make_record(1, score=0.6, mutation=_make_mutation()),
        ]
        result = MetaRewriter.build_trajectory(history, window_size=5)
        assert len(result) == 2
        assert result[0].score_delta == 0.0
        assert result[1].score_delta == pytest.approx(0.2)

    def test_windowing(self):
        history = [_make_record(i, score=0.1 * i) for i in range(10)]
        result = MetaRewriter.build_trajectory(history, window_size=5)
        assert len(result) == 5
        # Always includes iteration 0
        iters = [e.iteration for e in result]
        assert 0 in iters
        # Always includes best (iteration 9)
        assert 9 in iters

    def test_best_always_included(self):
        # Best at iteration 2, most recent at 7
        history = [_make_record(i, score=0.1 * i) for i in range(8)]
        history[2] = _make_record(2, score=0.99)
        result = MetaRewriter.build_trajectory(history, window_size=3)
        iters = [e.iteration for e in result]
        assert 0 in iters
        assert 2 in iters  # best

    def test_score_deltas(self):
        history = [
            _make_record(0, score=0.3),
            _make_record(1, score=0.5),
            _make_record(2, score=0.4),
        ]
        result = MetaRewriter.build_trajectory(history, window_size=5)
        deltas = [e.score_delta for e in result]
        assert deltas[0] == 0.0
        assert deltas[1] == pytest.approx(0.2)
        assert deltas[2] == pytest.approx(-0.1)


# ======================================================================
# Context formatting
# ======================================================================


class TestFormatTrajectory:

    def test_structure(self):
        history = [
            _make_record(0, score=0.4),
            _make_record(1, score=0.6, mutation=_make_mutation()),
        ]
        trajectory = MetaRewriter.build_trajectory(history)
        text = MetaRewriter._format_trajectory(trajectory)
        assert "### Iteration 0" in text
        assert "### Iteration 1" in text
        assert "Score:" in text
        assert "delta:" in text

    def test_long_prompt_truncation(self):
        long_prompt = "A" * 5000
        history = [_make_record(0, score=0.5, prompt=long_prompt)]
        trajectory = MetaRewriter.build_trajectory(history)
        text = MetaRewriter._format_trajectory(trajectory)
        assert "chars omitted" in text
        assert len(text) < len(long_prompt)

    def test_with_responses(self):
        history = [
            _make_record(0, score=0.5, response_samples=["Hello there!"]),
        ]
        trajectory = MetaRewriter.build_trajectory(history)
        text = MetaRewriter._format_responses(trajectory)
        assert "Iteration 0 responses" in text
        assert "Hello there!" in text

    def test_without_responses(self):
        history = [_make_record(0, score=0.5)]
        trajectory = MetaRewriter.build_trajectory(history)
        text = MetaRewriter._format_responses(trajectory)
        assert text == ""


# ======================================================================
# Metric trends
# ======================================================================


class TestFormatTrends:

    def test_improving(self):
        targets = _make_targets()
        entries = [
            TrajectoryEntry(i, "p", {"retention_ratio_rules": 0.3 + 0.1 * i},
                            0.5, 0.0, 3, 5, None)
            for i in range(4)
        ]
        text = MetaRewriter._format_trends(entries, targets)
        assert "improving" in text

    def test_regressing(self):
        targets = _make_targets()
        entries = [
            TrajectoryEntry(i, "p", {"retention_ratio_rules": 0.8 - 0.1 * i},
                            0.5, 0.0, 3, 5, None)
            for i in range(4)
        ]
        text = MetaRewriter._format_trends(entries, targets)
        assert "regressing" in text

    def test_stuck(self):
        targets = _make_targets()
        entries = [
            TrajectoryEntry(i, "p", {"retention_ratio_rules": 0.5},
                            0.5, 0.0, 3, 5, None)
            for i in range(4)
        ]
        text = MetaRewriter._format_trends(entries, targets)
        assert "stuck" in text

    def test_insufficient_history(self):
        targets = _make_targets()
        entries = [TrajectoryEntry(0, "p", {}, 0.5, 0.0, 3, 5, None)]
        text = MetaRewriter._format_trends(entries, targets)
        assert "insufficient" in text


# ======================================================================
# Response parsing
# ======================================================================


class TestParseResponse:

    def test_well_formed(self):
        raw = (
            "<reasoning>The rules section needs emphasis.</reasoning>\n"
            "<changes>\n- Added bold markers\n- Reordered sections\n</changes>\n"
            "<confidence>0.8</confidence>\n"
            "<prompt_structure>"
            '{"leading_text": "", "sections": ['
            '{"name": "rules", "body": "\\n**Follow these rules.**\\n"},'
            '{"name": "examples", "body": "\\nExample."}'
            ']}'
            "</prompt_structure>"
        )
        result = MetaRewriter._parse_response(raw)
        assert result["structure"]["sections"][0]["name"] == "rules"
        assert "**Follow these rules.**" in result["structure"]["sections"][0]["body"]
        assert "emphasis" in result["reasoning"]
        assert len(result["changes"]) == 2
        assert result["confidence"] == pytest.approx(0.8)

    def test_missing_tags(self):
        raw = "Just a plain prompt without any tags."
        result = MetaRewriter._parse_response(raw)
        assert result["prompt"] == raw
        assert result["reasoning"] == ""
        assert result["changes"] == []
        assert result["confidence"] == 0.5

    def test_partial_tags(self):
        raw = "<prompt>New prompt here</prompt>"
        result = MetaRewriter._parse_response(raw)
        assert result["prompt"] == "New prompt here"
        assert result["confidence"] == 0.5
        assert result["reasoning"] == ""

    def test_confidence_clamping(self):
        raw = "<confidence>1.5</confidence>\n<prompt>test</prompt>"
        result = MetaRewriter._parse_response(raw)
        assert result["confidence"] == 1.0

        raw2 = "<confidence>-0.3</confidence>\n<prompt>test</prompt>"
        result2 = MetaRewriter._parse_response(raw2)
        assert result2["confidence"] == 0.0

    def test_confidence_non_numeric(self):
        raw = "<confidence>high</confidence>\n<prompt>test</prompt>"
        result = MetaRewriter._parse_response(raw)
        assert result["confidence"] == 0.5


class TestApplyStructureUpdate:

    def test_reorders_and_updates_sections(self):
        prompt = (
            "## Intro\nIntro text.\n\n"
            "## Rules\nOld rules.\n\n"
            "## Examples\nOld example.\n"
        )
        region_config = {
            "system_prompt": {
                "regions": [
                    {"name": "intro", "start_marker": "## Intro", "end_marker": "## Rules"},
                    {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
                    {"name": "examples", "start_marker": "## Examples", "end_marker": ""},
                ],
            }
        }
        structured = parse_prompt(prompt, region_config)
        payload = {
            "sections": [
                {"name": "examples", "body": "\nNew example.\n", "order": 0},
                {"name": "rules", "body": "\nImproved rules.\n", "order": 1},
                {"name": "intro", "body": "\nIntro text.\n", "order": 2},
            ]
        }
        updated = MetaRewriter._apply_structure_update(structured, payload)
        assert updated is not None
        order = [section.name for section in updated.sections]
        assert order == ["examples", "rules", "intro"]
        assert "Improved rules." in updated.render()

    def test_missing_or_invalid_payload_returns_none(self):
        prompt = "## Intro\nIntro text.\n"
        structured = parse_prompt(prompt, _region_config())
        assert MetaRewriter._apply_structure_update(structured, None) is None
        assert MetaRewriter._apply_structure_update(structured, {"sections": []}) is None
# ======================================================================
# Marker validation
# ======================================================================


class TestValidateMarkers:

    def test_all_present(self):
        prompt = "## Rules\nBe nice.\n\n## Examples\nExample here."
        missing = MetaRewriter._validate_markers(prompt, _region_config())
        assert missing == []

    def test_missing_marker(self):
        prompt = "## Rules\nBe nice.\n\nSome other section."
        missing = MetaRewriter._validate_markers(prompt, _region_config())
        assert "## Examples" in missing

    def test_empty_config(self):
        missing = MetaRewriter._validate_markers("anything", {"system_prompt": {"regions": []}})
        assert missing == []


# ======================================================================
# Integration (mocked Anthropic client)
# ======================================================================


class TestProposeRewrite:

    def _mock_rewriter(self, candidate_evaluator=None):
        if candidate_evaluator is None:
            candidate_evaluator = CandidateEvaluator()
        rewriter = MetaRewriter(
            model="test-model",
            api_key="test-key",
            candidate_evaluator=candidate_evaluator,
        )
        mock_client = MagicMock()
        rewriter._client = mock_client
        return rewriter, mock_client

    def test_calls_api(self):
        rewriter, mock_client = self._mock_rewriter()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=(
            "<reasoning>Improved clarity</reasoning>\n"
            "<changes>\n- Reworded rules\n</changes>\n"
            "<confidence>0.7</confidence>\n"
            "<prompt_structure>"
            '{"sections": ['
            '{"name": "rules", "body": "\\nBetter rules.\\n"},'
            '{"name": "examples", "body": "\\nExample.\\n"}'
            ']}'
            "</prompt_structure>"
        ))]
        mock_client.messages.create.return_value = mock_response

        history = [
            _make_record(0, score=0.4, prompt="## Rules\nOld.\n\n## Examples\nEx."),
            _make_record(1, score=0.5, prompt="## Rules\nOld v2.\n\n## Examples\nEx."),
        ]

        new_prompt, record = rewriter.propose_rewrite(
            "## Rules\nOld v2.\n\n## Examples\nEx.",
            _region_config(),
            history,
            _make_targets(),
            _make_report(),
        )

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["temperature"] == 0

        assert "Better rules." in new_prompt
        assert record.mutation_type == "meta_rewrite"
        assert record.operation == "trajectory_rewrite"
        assert "policy=" in record.diff_summary

    def test_rejects_missing_markers(self):
        rewriter, mock_client = self._mock_rewriter()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=(
            "<prompt>No markers here at all.</prompt>"
        ))]
        mock_client.messages.create.return_value = mock_response

        history = [_make_record(0), _make_record(1)]

        new_prompt, record = rewriter.propose_rewrite(
            "## Rules\nOld.\n\n## Examples\nEx.",
            _region_config(),
            history,
            _make_targets(),
            _make_report(),
        )

        # Should return original prompt unchanged
        assert new_prompt == "## Rules\nOld.\n\n## Examples\nEx."
        assert "marker validation failed" in record.diff_summary

    def test_rejects_via_verification(self):
        rewriter, mock_client = self._mock_rewriter()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=(
            "<reasoning>No change needed</reasoning>\n"
            "<changes>\n- kept original\n</changes>\n"
            "<confidence>0.5</confidence>\n"
            "<prompt_structure>"
            '{"sections": ['
            '{"name": "rules", "body": "\\nOld.\\n\\n"},'
            '{"name": "examples", "body": "\\nEx."}'
            ']}'
            "</prompt_structure>"
        ))]
        mock_client.messages.create.return_value = mock_response

        history = [
            _make_record(0, score=0.3, prompt="## Rules\nOld.\n\n## Examples\nEx."),
            _make_record(1, score=0.4, prompt="## Rules\nOld.\n\n## Examples\nEx."),
        ]

        new_prompt, record = rewriter.propose_rewrite(
            "## Rules\nOld.\n\n## Examples\nEx.",
            _region_config(),
            history,
            _make_targets(),
            _make_report(),
        )

        assert new_prompt == "## Rules\nOld.\n\n## Examples\nEx."
        assert record is not None
        assert "verification" in record.reason.lower()

    def test_rejects_via_policy(self):
        class RejectingEvaluator:
            def evaluate(self, context):
                return CandidateDecision(False, "stub rejection", 0.0)

        rewriter, mock_client = self._mock_rewriter(candidate_evaluator=RejectingEvaluator())
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=(
            "<reasoning>Change everything</reasoning>\n"
            "<changes>\n- rewrite rules\n</changes>\n"
            "<confidence>0.8</confidence>\n"
            "<prompt_structure>"
            '{"sections": ['
            '{"name": "rules", "body": "\\nImproved.\\n"},'
            '{"name": "examples", "body": "\\nExample.\\n"}'
            ']}'
            "</prompt_structure>"
        ))]
        mock_client.messages.create.return_value = mock_response

        history = [
            _make_record(0, score=0.3, prompt="## Rules\nOld.\n\n## Examples\nEx."),
            _make_record(1, score=0.4, prompt="## Rules\nOld.\n\n## Examples\nEx."),
        ]

        new_prompt, record = rewriter.propose_rewrite(
            "## Rules\nOld.\n\n## Examples\nEx.",
            _region_config(),
            history,
            _make_targets(),
            _make_report(),
        )

        assert new_prompt == "## Rules\nOld.\n\n## Examples\nEx."
        assert "policy" in record.reason.lower()


class TestApplyMetaRewrite:

    def test_insufficient_history(self):
        rewriter = MetaRewriter(model="test-model")
        # Only 1 iteration — not enough trajectory
        history = [_make_record(0)]
        prompt, record = rewriter.apply_meta_rewrite(
            "test prompt", _region_config(), history,
            _make_targets(), _make_report(),
        )
        assert prompt == "test prompt"
        assert record is None

    def test_empty_history(self):
        rewriter = MetaRewriter(model="test-model")
        prompt, record = rewriter.apply_meta_rewrite(
            "test prompt", _region_config(), [],
            _make_targets(), _make_report(),
        )
        assert prompt == "test prompt"
        assert record is None

    def test_sufficient_history_calls_propose(self):
        rewriter = MetaRewriter(model="test-model")
        mock_client = MagicMock()
        rewriter._client = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=(
            "<prompt>## Rules\nNew.\n\n## Examples\nNew ex.</prompt>"
        ))]
        mock_client.messages.create.return_value = mock_response

        history = [_make_record(0, score=0.3), _make_record(1, score=0.5)]
        prompt, record = rewriter.apply_meta_rewrite(
            "## Rules\nOld.\n\n## Examples\nOld ex.",
            _region_config(),
            history,
            _make_targets(),
            _make_report(),
        )
        assert record is not None
        assert record.mutation_type == "meta_rewrite"


# ======================================================================
# Format targets
# ======================================================================


class TestFormatTargets:

    def test_basic(self):
        targets = _make_targets()
        text = MetaRewriter._format_targets(targets)
        assert "retention_ratio_rules" in text
        assert "context_bleed_ratio" in text
        assert "weight=" in text

    def test_empty(self):
        text = MetaRewriter._format_targets({})
        assert "no targets" in text


# ======================================================================
# Truncation
# ======================================================================


class TestTruncatePrompt:

    def test_short_prompt_unchanged(self):
        text = "Short prompt"
        assert MetaRewriter._truncate_prompt(text) == text

    def test_long_prompt_truncated(self):
        text = "A" * 5000
        result = MetaRewriter._truncate_prompt(text, max_chars=2000)
        assert len(result) < 5000
        assert "chars omitted" in result
        # Starts and ends with original content
        assert result.startswith("A" * 100)
        assert result.endswith("A" * 100)
