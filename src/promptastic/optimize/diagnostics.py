"""Diagnostic analysis of optimization metrics.

Identifies failing metrics and suggests appropriate mutation strategies
to address each issue.
"""

from __future__ import annotations

from ._types import DiagnosticIssue, DiagnosticReport, MetricTarget, OptimizationScore


# Maps metric name patterns to suggested structural mutations
_MUTATION_RULES: list[tuple[str, str, str]] = [
    # (metric pattern, suggested mutation, reason template)
    (
        "context_bleed_ratio",
        "insert_separator",
        "Context bleed ratio {value:.2f} exceeds target; separators may reduce history influence",
    ),
    (
        "terminal_attention_",
        "reorder_sections",
        "Terminal attention to {region} is {value:.4f}; moving section closer to end may increase salience",
    ),
    (
        "retention_ratio_",
        "duplicate_summary",
        "Retention ratio for {region} is {value:.2f}; adding a summary echo near prompt end may help",
    ),
    (
        "peak_layer_frac_",
        "adjust_emphasis",
        "Peak layer fraction for {region} is {value:.2f}; emphasis markers may force earlier absorption",
    ),
    (
        "density_cv",
        "adjust_section_length",
        "Density CV is {value:.2f}; balancing section lengths may improve attention distribution",
    ),
    (
        "causal_importance_",
        "llm_rewrite",
        "Causal importance of {region} is {value:.4f}; may need semantic restructuring",
    ),
    (
        "head_variance_",
        "llm_rewrite",
        "Head variance for {region} is {value:.6f}; content revision may help specialization",
    ),
]


def _extract_region(metric_name: str) -> str:
    """Extract the region name from a region-specific metric name."""
    for prefix in (
        "terminal_attention_",
        "retention_ratio_",
        "peak_layer_frac_",
        "peak_value_",
        "causal_importance_",
        "critical_layer_spread_",
        "attn_causal_corr_",
        "head_variance_",
        "specialist_head_count_",
    ):
        if metric_name.startswith(prefix):
            return metric_name[len(prefix):]
    return ""


def _suggest_mutation(metric_name: str, value: float) -> tuple[str, str]:
    """Find the best mutation suggestion for a failing metric."""
    region = _extract_region(metric_name)

    for pattern, mutation, reason_template in _MUTATION_RULES:
        if metric_name.startswith(pattern) or metric_name == pattern:
            reason = reason_template.format(value=value, region=region)
            return mutation, reason

    return "llm_rewrite", f"Metric {metric_name} = {value:.4f} is below target"


def diagnose(
    score: OptimizationScore,
    satisfaction_threshold: float = 0.9,
) -> DiagnosticReport:
    """Analyze optimization score and identify failing metrics.

    Returns a diagnostic report with issues sorted by impact (lowest
    satisfaction first) and suggested mutations for each.
    """
    issues: list[DiagnosticIssue] = []

    for name, metric_score in score.per_metric.items():
        if metric_score.satisfaction >= satisfaction_threshold:
            continue

        mutation, reason = _suggest_mutation(name, metric_score.value)
        issues.append(DiagnosticIssue(
            metric_name=name,
            value=metric_score.value,
            satisfaction=metric_score.satisfaction,
            target=metric_score.target,
            suggested_mutation=mutation,
            reason=reason,
        ))

    # Sort by satisfaction ascending (worst first)
    issues.sort(key=lambda i: i.satisfaction)

    num_failing = len(issues)
    num_total = score.num_total

    return DiagnosticReport(
        issues=issues,
        overall_score=score.total,
        num_failing=num_failing,
        num_total=num_total,
    )


def format_diagnostic_report(report: DiagnosticReport) -> str:
    """Format a diagnostic report as human-readable text."""
    lines: list[str] = []
    lines.append(
        f"Score: {report.overall_score:.3f} "
        f"({report.num_total - report.num_failing}/{report.num_total} metrics satisfied)"
    )

    if not report.issues:
        lines.append("All metrics satisfied.")
        return "\n".join(lines)

    lines.append(f"\nFailing metrics ({report.num_failing}):")
    for issue in report.issues:
        lines.append(
            f"  {issue.metric_name}: {issue.value:.4f} "
            f"(satisfaction={issue.satisfaction:.2f}, "
            f"target={issue.target.direction} {issue.target.ideal})"
        )
        lines.append(f"    -> {issue.suggested_mutation}: {issue.reason}")

    return "\n".join(lines)
