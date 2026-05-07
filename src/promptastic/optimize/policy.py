"""Stubbed candidate evaluation policy for prompt optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ._types import DiagnosticReport
from .verification import VerificationResult


@dataclass
class CandidateContext:
    """Context passed to the candidate evaluator."""

    iteration: int
    mutation_type: str
    report: DiagnosticReport
    verification: VerificationResult | None
    changed_sections: Sequence[str]
    reasoning: str


@dataclass
class CandidateDecision:
    """Decision produced by the candidate evaluator."""

    accept: bool
    rationale: str
    confidence: float


class CandidateEvaluator:
    """Stubbed learned-scoring hook.

    Replaces historical heuristics with a pluggable interface that can be
    swapped for a learned reward model.  The current implementation uses
    simple targeting heuristics and reports a confidence score in [0, 1].
    """

    def evaluate(self, context: CandidateContext) -> CandidateDecision:
        targeted = _targeted_sections(context.report)
        changed = set(context.changed_sections)
        hits = len(changed & targeted)
        total_targets = len(targeted) if targeted else 1

        # Confidence is higher when we touch the sections diagnostics flagged.
        confidence = min(1.0, 0.2 + 0.6 * (hits / total_targets))
        rationale = "stub: pass-through"

        if targeted:
            if hits == 0:
                rationale = "stub: changed sections miss diagnostic targets"
                confidence = 0.1
            else:
                rationale = f"stub: touched {hits}/{len(targeted)} targeted sections"

        return CandidateDecision(
            accept=True,
            rationale=rationale,
            confidence=round(confidence, 2),
        )


def _targeted_sections(report: DiagnosticReport) -> set[str]:
    sections: set[str] = set()
    for issue in getattr(report, "issues", []):
        region = _extract_region(issue.metric_name)
        if region:
            sections.add(region)
    return sections


def _extract_region(metric_name: str) -> str:
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
