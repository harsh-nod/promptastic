"""Verification gate for candidate prompt rewrites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ._types import DiagnosticReport
from .structure import StructuredPrompt


@dataclass
class VerificationResult:
    """Outcome of candidate prompt verification."""

    accepted: bool
    reasons: list[str]
    changed_sections: list[str]
    length_ratio: float


def verify_candidate(
    current: StructuredPrompt,
    candidate: StructuredPrompt,
    report: DiagnosticReport,
    *,
    length_tolerance: float = 0.4,
) -> VerificationResult:
    """Decide whether to accept ``candidate`` as the next prompt version."""
    reasons: list[str] = []
    changed_sections = _compute_changed_sections(current, candidate)

    if not changed_sections:
        reasons.append("no_sections_changed")

    targeted_sections = _targeted_sections(report)
    untouched_targets = sorted(
        sect for sect in targeted_sections if sect not in changed_sections
    )
    for section in untouched_targets:
        reasons.append(f"target_section_unchanged:{section}")

    length_ratio = _length_ratio(current.render(), candidate.render())
    if abs(length_ratio - 1.0) > length_tolerance:
        reasons.append(f"length_delta:{length_ratio:.2f}")

    current_names = {section.name for section in current.sections}
    candidate_names = {section.name for section in candidate.sections}
    dropped_sections = sorted(current_names - candidate_names)
    if dropped_sections:
        reasons.append(f"sections_missing:{','.join(dropped_sections)}")

    return VerificationResult(
        accepted=not reasons,
        reasons=reasons,
        changed_sections=changed_sections,
        length_ratio=length_ratio,
    )


def _compute_changed_sections(
    current: StructuredPrompt,
    candidate: StructuredPrompt,
) -> list[str]:
    changed: list[str] = []
    current_lookup = {section.name: section for section in current.sections}
    for section in candidate.sections:
        original = current_lookup.get(section.name)
        if original is None:
            continue
        if original.body != section.body or original.prefix != section.prefix:
            changed.append(section.name)
    return changed


def _targeted_sections(report: DiagnosticReport) -> set[str]:
    targets: set[str] = set()
    for issue in getattr(report, "issues", []):
        region = _region_from_metric(issue.metric_name)
        if region:
            targets.add(region)
    return targets


def _region_from_metric(metric_name: str) -> str:
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


def _length_ratio(current_text: str, candidate_text: str) -> float:
    current_len = max(1, len(current_text))
    candidate_len = len(candidate_text)
    return candidate_len / current_len
