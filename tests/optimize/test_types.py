"""Tests for promptastic.optimize._types -- dataclass construction."""

from promptastic.optimize._types import (
    DiagnosticIssue,
    DiagnosticReport,
    IterationRecord,
    MetricScore,
    MetricTarget,
    MutationRecord,
    OptimizationConfig,
    OptimizationResult,
    OptimizationScore,
)


def test_metric_target_defaults():
    t = MetricTarget(name="foo", direction="above")
    assert t.weight == 1.0
    assert t.minimum == 0.0
    assert t.ideal == 0.0
    assert t.category == ""


def test_metric_target_above():
    t = MetricTarget(name="x", direction="above", minimum=0.0, ideal=0.5, weight=1.5)
    assert t.direction == "above"
    assert t.ideal == 0.5
    assert t.weight == 1.5


def test_metric_target_below():
    t = MetricTarget(name="x", direction="below", ideal=1.0, maximum=3.0)
    assert t.maximum == 3.0


def test_optimization_config_defaults():
    c = OptimizationConfig()
    assert c.max_iterations == 10
    assert c.target_score == 0.85
    assert c.patience == 3
    assert c.mutation_strategy == "hybrid"
    assert c.profile == "general"
    assert c.structural_iterations == 3
    assert c.verification_enabled is True
    assert c.verification_length_tolerance == 0.4
    assert c.use_candidate_policy is True


def test_optimization_score_construction():
    s = OptimizationScore(total=0.8, per_metric={}, num_satisfied=3, num_total=5)
    assert s.total == 0.8
    assert s.num_satisfied == 3


def test_mutation_record():
    m = MutationRecord(
        mutation_type="structural",
        operation="reorder_sections",
        target_region="rules",
        reason="low attention",
        diff_summary="Moved 200 chars to end",
    )
    assert m.mutation_type == "structural"
    assert m.target_region == "rules"


def test_diagnostic_issue():
    t = MetricTarget(name="x", direction="above", ideal=0.5)
    issue = DiagnosticIssue(
        metric_name="x",
        value=0.1,
        satisfaction=0.2,
        target=t,
        suggested_mutation="reorder_sections",
        reason="too low",
    )
    assert issue.satisfaction == 0.2


def test_optimization_result():
    score = OptimizationScore(total=0.9, per_metric={}, num_satisfied=5, num_total=5)
    result = OptimizationResult(
        best_prompt="test",
        best_regions={},
        best_score=score,
        best_iteration=3,
        history=[],
        total_forward_passes=10,
        total_wall_time_seconds=45.0,
        converged=True,
        convergence_reason="target_reached",
    )
    assert result.converged
    assert result.convergence_reason == "target_reached"
