"""Tests for promptastic.optimize.diagnostics -- diagnostic analysis."""

from promptastic.optimize._types import MetricScore, MetricTarget, OptimizationScore
from promptastic.optimize.diagnostics import diagnose, format_diagnostic_report


def _make_score(metrics: dict[str, tuple[float, float, MetricTarget]]) -> OptimizationScore:
    """Build an OptimizationScore from {name: (value, satisfaction, target)}."""
    per_metric = {}
    for name, (value, satisfaction, target) in metrics.items():
        per_metric[name] = MetricScore(
            value=value, satisfaction=satisfaction, weight=target.weight, target=target,
        )
    total = (
        sum(m.satisfaction * m.weight for m in per_metric.values())
        / sum(m.weight for m in per_metric.values())
        if per_metric
        else 0.0
    )
    num_satisfied = sum(1 for m in per_metric.values() if m.satisfaction >= 0.9)
    return OptimizationScore(
        total=total, per_metric=per_metric,
        num_satisfied=num_satisfied, num_total=len(per_metric),
    )


def test_diagnose_all_satisfied():
    t = MetricTarget(name="x", direction="above", ideal=0.5, weight=1.0)
    score = _make_score({"x": (0.8, 0.95, t)})
    report = diagnose(score)
    assert report.num_failing == 0
    assert report.issues == []


def test_diagnose_one_failing():
    t = MetricTarget(name="context_bleed_ratio", direction="below", ideal=1.0, maximum=3.0, weight=1.0)
    score = _make_score({"context_bleed_ratio": (2.5, 0.25, t)})
    report = diagnose(score)
    assert report.num_failing == 1
    assert report.issues[0].metric_name == "context_bleed_ratio"
    assert report.issues[0].suggested_mutation == "insert_separator"


def test_diagnose_sorted_by_satisfaction():
    t1 = MetricTarget(name="terminal_attention_rules", direction="above", ideal=0.05, weight=1.0)
    t2 = MetricTarget(name="context_bleed_ratio", direction="below", ideal=1.0, maximum=3.0, weight=1.0)
    score = _make_score({
        "terminal_attention_rules": (0.01, 0.2, t1),
        "context_bleed_ratio": (2.0, 0.5, t2),
    })
    report = diagnose(score)
    assert len(report.issues) == 2
    # Worst first
    assert report.issues[0].satisfaction <= report.issues[1].satisfaction


def test_diagnose_mutation_suggestions():
    targets_and_expected = [
        ("context_bleed_ratio", "insert_separator"),
        ("terminal_attention_rules", "reorder_sections"),
        ("retention_ratio_rules", "duplicate_summary"),
        ("peak_layer_frac_rules", "adjust_emphasis"),
        ("density_cv", "adjust_section_length"),
        ("causal_importance_rules", "llm_rewrite"),
    ]
    for metric_name, expected_mutation in targets_and_expected:
        t = MetricTarget(name=metric_name, direction="above", ideal=1.0, weight=1.0)
        score = _make_score({metric_name: (0.0, 0.0, t)})
        report = diagnose(score)
        assert report.issues[0].suggested_mutation == expected_mutation, (
            f"Expected {expected_mutation} for {metric_name}, "
            f"got {report.issues[0].suggested_mutation}"
        )


def test_format_diagnostic_report_no_issues():
    t = MetricTarget(name="x", direction="above", ideal=0.5, weight=1.0)
    score = _make_score({"x": (0.8, 0.95, t)})
    report = diagnose(score)
    text = format_diagnostic_report(report)
    assert "All metrics satisfied" in text


def test_format_diagnostic_report_with_issues():
    t = MetricTarget(name="context_bleed_ratio", direction="below", ideal=1.0, maximum=3.0, weight=1.0)
    score = _make_score({"context_bleed_ratio": (2.5, 0.25, t)})
    report = diagnose(score)
    text = format_diagnostic_report(report)
    assert "context_bleed_ratio" in text
    assert "insert_separator" in text
