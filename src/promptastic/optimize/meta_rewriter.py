"""Trajectory-aware meta-rewriter for N+1 prompt generation.

Uses the history of previous prompt iterations — their metrics, scores,
mutations, and optionally actual model responses — to generate the next
optimal prompt via an external LLM.  Unlike the region-scoped
:class:`LLMRewriter`, this module returns a *full* rewritten prompt
because it makes holistic decisions based on the optimization trajectory.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ._types import (
    DiagnosticReport,
    IterationRecord,
    MetricTarget,
    MutationRecord,
    TrajectoryEntry,
)
from .diagnostics import format_diagnostic_report
from .policy import CandidateContext, CandidateDecision
from .structure import PromptSection, StructuredPrompt, parse_prompt
from .verification import VerificationResult, verify_candidate


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_META_TEMPLATE = """\
You are an expert prompt engineer optimizing a system prompt for a
transformer language model.  You have the full history of optimization
attempts and their measured outcomes.  Your job is to produce the next
version of the prompt that will score highest on the target metrics.

## Optimization Targets

{targets_section}

## Optimization Trajectory

{trajectory_section}

## Metric Trends

{trends_section}

## Current Diagnostic Issues

{diagnostic_section}

## Current Prompt (version {current_iteration})

{current_prompt}
## Current Prompt Structure (JSON)

{structure_section}
{response_section}
## Instructions

Analyse the trajectory above.  Consider:
- Which mutations improved scores and which didn't
- Metrics that are stuck or regressing despite multiple attempts
- The overall coherence and clarity of the prompt
- Response quality patterns (if response samples are provided)

Then produce the optimal next version of the prompt.

Constraints:
- Preserve ALL section headers and markers exactly as they appear
  (they are used for automated region detection)
- You may restructure, reword, reorder, add, or remove content
  *within* or *between* sections
- Keep approximately the same overall length (+/- 30%)

Respond using these XML tags:

<reasoning>
Why you are making these specific changes, referencing the trajectory.
</reasoning>

<changes>
- Bullet list of specific changes made
</changes>

<confidence>
A number between 0.0 and 1.0 representing your confidence this will improve the score.
</confidence>

<prompt_structure>
JSON matching this schema:
{{
  "leading_text": string (optional; defaults to current leading text),
  "sections": [
    {{
      "name": string (must match an existing section name),
      "body": string with updated content for that section,
      "prefix": string (optional; omit to keep existing prefix),
      "order": integer (optional; lower numbers appear earlier)
    }},
    ...
  ],
  "trailing_text": string (optional; defaults to current trailing text)
}}
</prompt_structure>
"""


# ---------------------------------------------------------------------------
# MetaRewriter
# ---------------------------------------------------------------------------


class MetaRewriter:
    """Generate the optimal N+1 prompt from optimization history."""

    def __init__(
        self,
        model: str = "",
        api_key: str = "",
        window_size: int = 5,
        *,
        verification_enabled: bool = True,
        verification_length_tolerance: float = 0.4,
        candidate_evaluator: Any | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.window_size = window_size
        self.verification_enabled = verification_enabled
        self.verification_length_tolerance = verification_length_tolerance
        self._candidate_evaluator = candidate_evaluator
        self._client: Any = None

    # -- Anthropic client (lazy) -------------------------------------------

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = anthropic.Anthropic(**kwargs)
        except ImportError as exc:
            raise ImportError(
                "Meta-rewriting requires the 'anthropic' package. "
                "Install with: pip install anthropic"
            ) from exc

    # -- Trajectory building -----------------------------------------------

    @staticmethod
    def build_trajectory(
        history: list[IterationRecord],
        window_size: int = 5,
    ) -> list[TrajectoryEntry]:
        """Convert iteration history into a windowed trajectory.

        Always includes iteration 0 (baseline) and the best-scoring
        iteration.  Remaining slots go to the most recent iterations.
        """
        if not history:
            return []

        # Find the best-scoring iteration
        best_idx = max(range(len(history)), key=lambda i: history[i].score.total)

        # Determine which indices to include
        must_include = {0, best_idx}
        remaining_slots = max(0, window_size - len(must_include))
        # Fill with most recent iterations (excluding already-included)
        recent = [
            i for i in range(len(history) - 1, -1, -1)
            if i not in must_include
        ][:remaining_slots]
        selected = sorted(must_include | set(recent))

        entries: list[TrajectoryEntry] = []
        for idx in selected:
            rec = history[idx]
            prev_score = history[idx - 1].score.total if idx > 0 else 0.0
            delta = rec.score.total - prev_score if idx > 0 else 0.0
            entries.append(TrajectoryEntry(
                iteration=rec.iteration,
                prompt_text=rec.prompt_text,
                metrics=dict(rec.metrics),
                score_total=rec.score.total,
                score_delta=delta,
                num_satisfied=rec.score.num_satisfied,
                num_total=rec.score.num_total,
                mutation_applied=rec.mutation_applied,
                response_samples=list(rec.response_samples),
            ))
        return entries

    # -- Formatting helpers ------------------------------------------------

    @staticmethod
    def _format_targets(targets: dict[str, MetricTarget]) -> str:
        lines: list[str] = []
        for name, t in sorted(targets.items()):
            direction_desc = {
                "above": f"above {t.ideal}",
                "below": f"below {t.ideal}",
                "range": f"between {t.minimum} and {t.maximum}",
            }.get(t.direction, t.direction)
            lines.append(
                f"- **{name}** (weight={t.weight:.1f}): target {direction_desc}"
            )
        return "\n".join(lines) if lines else "(no targets defined)"

    @staticmethod
    def _truncate_prompt(text: str, max_chars: int = 2000) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        omitted = len(text) - max_chars
        return f"{text[:half]}\n[...{omitted} chars omitted...]\n{text[-half:]}"

    @staticmethod
    def _format_structure(structured: StructuredPrompt) -> str:
        payload = {
            "leading_text": structured.leading_text,
            "sections": [
                {
                    "name": section.name,
                    "start_marker": section.start_marker,
                    "header": section.header,
                    "prefix": section.prefix,
                    "body": section.body,
                }
                for section in structured.sections
            ],
            "trailing_text": structured.trailing_text,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _format_trajectory(trajectory: list[TrajectoryEntry]) -> str:
        blocks: list[str] = []
        for entry in trajectory:
            lines = [
                f"### Iteration {entry.iteration}",
                f"Score: {entry.score_total:.3f} (delta: {entry.score_delta:+.3f})",
                f"Satisfied: {entry.num_satisfied}/{entry.num_total}",
            ]
            if entry.mutation_applied:
                m = entry.mutation_applied
                lines.append(
                    f"Mutation: {m.operation} on {m.target_region} — {m.diff_summary}"
                )
            else:
                lines.append("Mutation: (baseline, no mutation)")

            # Top failing metrics (satisfaction < 0.9, show up to 5)
            failing = sorted(
                ((k, v) for k, v in entry.metrics.items()),
                key=lambda kv: kv[1],
            )[:5]
            if failing:
                lines.append("Key metrics:")
                for k, v in failing:
                    lines.append(f"  {k} = {v:.4f}")

            lines.append(
                f"Prompt:\n```\n{MetaRewriter._truncate_prompt(entry.prompt_text)}\n```"
            )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @staticmethod
    def _format_trends(
        trajectory: list[TrajectoryEntry],
        targets: dict[str, MetricTarget],
    ) -> str:
        if len(trajectory) < 2:
            return "(insufficient history for trend analysis)"

        # Gather metric series across trajectory entries
        metric_names = set()
        for entry in trajectory:
            metric_names.update(entry.metrics.keys())

        lines: list[str] = []
        for name in sorted(metric_names):
            if name not in targets:
                continue
            values = [
                entry.metrics[name]
                for entry in trajectory
                if name in entry.metrics
            ]
            if len(values) < 2:
                continue

            # Determine trend from last 3 values
            recent = values[-3:]
            t = targets[name]
            if t.direction == "above":
                improving = all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1))
                regressing = all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1))
            elif t.direction == "below":
                improving = all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1))
                regressing = all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1))
            else:
                improving = False
                regressing = False

            total_change = abs(values[-1] - values[0])
            if total_change < 0.01:
                trend = "stuck"
            elif improving:
                trend = "improving"
            elif regressing:
                trend = "regressing"
            else:
                trend = "mixed"

            lines.append(
                f"- **{name}**: {trend} "
                f"(current={values[-1]:.4f}, best={max(values) if t.direction == 'above' else min(values):.4f})"
            )

        return "\n".join(lines) if lines else "(no targetable metrics tracked)"

    @staticmethod
    def _format_responses(trajectory: list[TrajectoryEntry]) -> str:
        blocks: list[str] = []
        for entry in trajectory:
            if not entry.response_samples:
                continue
            lines = [f"### Iteration {entry.iteration} responses"]
            for i, sample in enumerate(entry.response_samples):
                lines.append(f"Response {i + 1}: {sample[:500]}")
            blocks.append("\n".join(lines))
        if not blocks:
            return ""
        return (
            "\n## Response Samples\n\n"
            "These are actual outputs the model generated with each prompt version.\n\n"
            + "\n\n".join(blocks)
            + "\n"
        )

    # -- Response parsing --------------------------------------------------

    @staticmethod
    def _parse_response(raw_text: str) -> dict[str, Any]:
        """Parse the structured LLM response.

        Returns dict with keys: prompt, reasoning, changes, confidence.
        Falls back gracefully if tags are missing.
        """
        def _extract_tag(text: str, tag: str) -> str:
            pattern = rf"<{tag}>(.*?)</{tag}>"
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else ""

        structure_raw = _extract_tag(raw_text, "prompt_structure")
        prompt = _extract_tag(raw_text, "prompt")
        reasoning = _extract_tag(raw_text, "reasoning")
        changes_raw = _extract_tag(raw_text, "changes")
        confidence_raw = _extract_tag(raw_text, "confidence")

        structure: dict[str, Any] | None = None
        if structure_raw:
            try:
                structure = json.loads(structure_raw)
            except json.JSONDecodeError:
                structure = None

        # Parse changes into list
        changes: list[str] = []
        if changes_raw:
            changes = [
                line.lstrip("- ").strip()
                for line in changes_raw.splitlines()
                if line.strip()
            ]

        # Parse confidence
        try:
            confidence = float(confidence_raw)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        # Fallback: if no <prompt> tag, treat entire response as prompt
        if not prompt:
            prompt = raw_text.strip()

        return {
            "prompt": prompt,
            "reasoning": reasoning,
            "changes": changes,
            "confidence": confidence,
            "structure": structure,
        }

    @staticmethod
    def _apply_structure_update(
        structured: StructuredPrompt,
        update_payload: dict[str, Any] | None,
    ) -> StructuredPrompt | None:
        if not update_payload:
            return None

        sections_payload = update_payload.get("sections")
        if not isinstance(sections_payload, list) or not sections_payload:
            return None

        section_lookup = {section.name: section for section in structured.sections}
        used_names: set[str] = set()
        ordered_sections: list[PromptSection] = []

        sortable_entries: list[tuple[float, dict[str, Any]]] = []
        for index, entry in enumerate(sections_payload):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name or name not in section_lookup:
                continue
            order_value = entry.get("order")
            if isinstance(order_value, (int, float)):
                sort_key = float(order_value)
            else:
                sort_key = float(index)
            sortable_entries.append((sort_key, entry))

        for _, entry in sorted(sortable_entries, key=lambda item: item[0]):
            name = entry.get("name")
            if not name or name not in section_lookup:
                continue
            base_section = section_lookup[name]
            used_names.add(name)

            body = entry.get("body")
            if body is None:
                body = base_section.body

            prefix = entry.get("prefix")
            if prefix is None:
                prefix = base_section.prefix

            ordered_sections.append(
                PromptSection(
                    name=name,
                    start_marker=base_section.start_marker,
                    prefix=prefix,
                    header=base_section.header,
                    body=body,
                    metadata=dict(getattr(base_section, "metadata", {})),
                )
            )

        for section in structured.sections:
            if section.name in used_names:
                continue
            ordered_sections.append(section)

        leading_text = update_payload.get("leading_text")
        if leading_text is None:
            leading_text = structured.leading_text

        trailing_text = update_payload.get("trailing_text")
        if trailing_text is None:
            trailing_text = structured.trailing_text

        return StructuredPrompt(
            original_text=structured.original_text,
            leading_text=leading_text,
            sections=ordered_sections,
            trailing_text=trailing_text,
            missing_regions=list(structured.missing_regions),
        )

    @staticmethod
    def _changed_sections(
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

    # -- Marker validation -------------------------------------------------

    @staticmethod
    def _validate_markers(
        new_prompt: str,
        region_config: dict[str, Any],
    ) -> list[str]:
        """Check that all region markers are present in the new prompt.

        Returns list of missing markers (empty = valid).
        """
        missing: list[str] = []
        sys_regions = region_config.get("system_prompt", {}).get("regions", [])
        for rdef in sys_regions:
            marker = rdef.get("start_marker", "")
            if marker and marker not in new_prompt:
                missing.append(marker)
        return missing

    # -- Main entry points -------------------------------------------------

    def propose_rewrite(
        self,
        current_prompt: str,
        region_config: dict[str, Any],
        history: list[IterationRecord],
        targets: dict[str, MetricTarget],
        report: DiagnosticReport,
    ) -> tuple[str, MutationRecord]:
        """Generate the N+1 prompt from optimization trajectory.

        Returns (new_prompt, mutation_record).
        """
        self._ensure_client()

        structured_prompt = parse_prompt(current_prompt, region_config)

        trajectory = self.build_trajectory(history, self.window_size)

        # Build the meta-rewrite prompt
        has_responses = any(e.response_samples for e in trajectory)
        response_section = self._format_responses(trajectory) if has_responses else ""

        current_iteration = history[-1].iteration if history else 0

        prompt_text = _META_TEMPLATE.format(
            targets_section=self._format_targets(targets),
            trajectory_section=self._format_trajectory(trajectory),
            trends_section=self._format_trends(trajectory, targets),
            diagnostic_section=format_diagnostic_report(report),
            current_prompt=current_prompt,
            structure_section=self._format_structure(structured_prompt),
            response_section=response_section,
            current_iteration=current_iteration,
        )

        message = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt_text}],
        )

        raw_response = message.content[0].text.strip()
        parsed = self._parse_response(raw_response)

        new_structured = self._apply_structure_update(
            structured_prompt,
            parsed.get("structure"),
        )

        candidate_structured = new_structured
        if candidate_structured is None:
            new_prompt = parsed["prompt"]
            if not new_prompt:
                new_prompt = current_prompt
            candidate_structured = parse_prompt(new_prompt, region_config)
        else:
            new_prompt = candidate_structured.render()

        # Validate region markers before verification logic.
        missing = self._validate_markers(new_prompt, region_config)
        if missing:
            return current_prompt, MutationRecord(
                mutation_type="meta_rewrite",
                operation="trajectory_rewrite",
                target_region="(full prompt)",
                reason=f"Rejected: missing markers {missing}",
                diff_summary="No change — marker validation failed",
            )

        verification: VerificationResult | None = None
        if self.verification_enabled:
            verification = verify_candidate(
                structured_prompt,
                candidate_structured,
                report,
                length_tolerance=self.verification_length_tolerance,
            )
            if not verification.accepted:
                reason = "; ".join(verification.reasons) or "verification_failed"
                return current_prompt, MutationRecord(
                    mutation_type="meta_rewrite",
                    operation="trajectory_rewrite",
                    target_region="(full prompt)",
                    reason=f"Rejected by verification: {reason}",
                    diff_summary=f"Verification rejected candidate ({reason})",
                )

        policy_decision: CandidateDecision | None = None
        changed_sections = self._changed_sections(structured_prompt, candidate_structured)
        if self._candidate_evaluator is not None:
            policy_decision = self._candidate_evaluator.evaluate(
                CandidateContext(
                    iteration=current_iteration,
                    mutation_type="meta_rewrite",
                    report=report,
                    verification=verification,
                    changed_sections=changed_sections,
                    reasoning=parsed.get("reasoning", ""),
                )
            )
            if not policy_decision.accept:
                rationale = policy_decision.rationale or "policy_rejected"
                return current_prompt, MutationRecord(
                    mutation_type="meta_rewrite",
                    operation="trajectory_rewrite",
                    target_region="(full prompt)",
                    reason=f"Rejected by policy: {rationale}",
                    diff_summary="No change — policy rejected candidate",
                )

        reasoning = parsed["reasoning"][:500] if parsed["reasoning"] else "N+1 trajectory-based rewrite"
        changes = parsed["changes"]
        changes_str = "; ".join(changes[:5]) if changes else "full prompt rewrite"
        if verification and verification.changed_sections:
            changes_str += f" | sections:{','.join(verification.changed_sections)}"
        elif policy_decision and changed_sections:
            changes_str += f" | sections:{','.join(changed_sections)}"
        if policy_decision is not None:
            changes_str += f" | policy={policy_decision.confidence:.2f}"

        return new_prompt, MutationRecord(
            mutation_type="meta_rewrite",
            operation="trajectory_rewrite",
            target_region="(full prompt)",
            reason=reasoning,
            diff_summary=(
                f"Rewrote full prompt ({len(current_prompt)} -> {len(new_prompt)} chars, "
                f"confidence={parsed['confidence']:.1f}): {changes_str}"
            ),
        )

    def apply_meta_rewrite(
        self,
        current_prompt: str,
        region_config: dict[str, Any],
        history: list[IterationRecord],
        targets: dict[str, MetricTarget],
        report: DiagnosticReport,
    ) -> tuple[str, MutationRecord | None]:
        """Top-level dispatch for the meta-rewriter.

        Returns ``(current_prompt, None)`` when there is insufficient
        history (fewer than 2 iterations) to learn from.
        """
        if len(history) < 2:
            return current_prompt, None

        return self.propose_rewrite(
            current_prompt, region_config, history, targets, report,
        )
