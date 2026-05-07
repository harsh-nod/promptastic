"""Developer-facing summaries for prompt optimization trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ._types import DiagnosticReport
from .structure import StructuredPrompt


def build_fix_plan(report: DiagnosticReport, limit: int = 4) -> list[str]:
    """Generate a short list of actionable fixes from diagnostics."""
    plan: list[str] = []
    for issue in getattr(report, "issues", [])[:limit]:
        target = issue.metric_name
        suggestion = issue.suggested_mutation or "review"
        summary = issue.reason or ""
        plan.append(f"{target}: {summary} -> {suggestion}")
    if not plan:
        plan.append("All metrics satisfied.")
    return plan


def changed_sections(
    before: StructuredPrompt,
    after: StructuredPrompt,
) -> list[str]:
    """Return section names whose bodies or prefixes changed."""
    before_lookup = {section.name: section for section in before.sections}
    changed: list[str] = []
    for section in after.sections:
        previous = before_lookup.get(section.name)
        if previous is None:
            continue
        if previous.body != section.body or previous.prefix != section.prefix:
            changed.append(section.name)
    if before.trailing_text != after.trailing_text:
        changed.append("(trailing)")
    return changed


def summarize_prompt_diff(
    before: StructuredPrompt,
    after: StructuredPrompt,
    sections: Sequence[str],
    *,
    limit: int = 4,
    snippet_chars: int = 80,
) -> list[str]:
    summaries: list[str] = []
    after_lookup = {section.name: section for section in after.sections}
    before_lookup = {section.name: section for section in before.sections}

    for name in sections:
        if name == "(trailing)":
            snippet = after.trailing_text.strip()[:snippet_chars]
            summaries.append(f"{name}: {snippet or '(empty)'}")
            continue
        current = after_lookup.get(name)
        previous = before_lookup.get(name)
        if current is None or previous is None:
            continue
        snippet = current.body.strip().replace("\n", " ")[:snippet_chars]
        summaries.append(f"{name}: {snippet or '(empty)'}")
        if len(summaries) >= limit:
            break
    return summaries
