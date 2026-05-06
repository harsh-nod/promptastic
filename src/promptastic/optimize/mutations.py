"""Structural prompt mutations for optimization.

Programmatic operations that modify prompt structure without requiring
an external LLM.  Each mutation preserves region marker integrity so
that re-annotation continues to work after transformation.
"""

from __future__ import annotations

import re
from typing import Any

from ._types import DiagnosticReport, MutationRecord, PromptSpec


class StructuralMutator:
    """Apply deterministic structural transformations to prompts."""

    def reorder_sections(
        self,
        prompt: str,
        region_config: dict[str, Any],
        target_region: str,
        position: str = "end",
    ) -> tuple[str, MutationRecord]:
        """Move a section closer to the end (or start) of the prompt.

        Identifies the section boundaries from region config markers and
        relocates the text block.
        """
        sys_regions = region_config.get("system_prompt", {}).get("regions", [])
        region_def = None
        for rdef in sys_regions:
            if rdef.get("name") == target_region:
                region_def = rdef
                break

        if region_def is None:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="reorder_sections",
                target_region=target_region,
                reason=f"Region {target_region} not found in config",
                diff_summary="No change (region not found)",
            )

        # Extract section text using markers
        start_marker = region_def.get("start_marker", "")
        end_marker = region_def.get("end_marker", "")

        if not start_marker:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="reorder_sections",
                target_region=target_region,
                reason="No start_marker defined",
                diff_summary="No change (no markers)",
            )

        start_idx = prompt.find(start_marker)
        if start_idx == -1:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="reorder_sections",
                target_region=target_region,
                reason="Start marker not found in prompt",
                diff_summary="No change (marker not found)",
            )

        if end_marker:
            end_idx = prompt.find(end_marker, start_idx + len(start_marker))
            if end_idx == -1:
                end_idx = len(prompt)
        else:
            end_idx = len(prompt)

        section_text = prompt[start_idx:end_idx]
        remaining = prompt[:start_idx] + prompt[end_idx:]

        # Clean up double newlines from removal
        remaining = re.sub(r"\n{3,}", "\n\n", remaining)

        if position == "end":
            new_prompt = remaining.rstrip() + "\n\n" + section_text.strip() + "\n"
        else:
            new_prompt = section_text.strip() + "\n\n" + remaining.lstrip()

        return new_prompt, MutationRecord(
            mutation_type="structural",
            operation="reorder_sections",
            target_region=target_region,
            reason=f"Moved {target_region} to {position} of prompt",
            diff_summary=f"Relocated {len(section_text)} chars to {position}",
        )

    def insert_separator(
        self,
        prompt: str,
        region_config: dict[str, Any],
        after_region: str,
        separator: str = "\n---\n",
    ) -> tuple[str, MutationRecord]:
        """Insert a separator after a region to reduce context bleed."""
        sys_regions = region_config.get("system_prompt", {}).get("regions", [])
        region_def = None
        for rdef in sys_regions:
            if rdef.get("name") == after_region:
                region_def = rdef
                break

        if region_def is None:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="insert_separator",
                target_region=after_region,
                reason=f"Region {after_region} not found",
                diff_summary="No change",
            )

        end_marker = region_def.get("end_marker", "")
        if end_marker:
            insert_pos = prompt.find(end_marker)
            if insert_pos != -1:
                insert_pos += len(end_marker)
            else:
                insert_pos = -1
        else:
            # Use start_marker + region text to estimate end
            start_marker = region_def.get("start_marker", "")
            if start_marker:
                start_idx = prompt.find(start_marker)
                if start_idx != -1:
                    # Find end of the paragraph
                    next_double_newline = prompt.find("\n\n", start_idx + len(start_marker))
                    insert_pos = next_double_newline if next_double_newline != -1 else len(prompt)
                else:
                    insert_pos = -1
            else:
                insert_pos = -1

        if insert_pos == -1:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="insert_separator",
                target_region=after_region,
                reason="Could not locate region boundary",
                diff_summary="No change",
            )

        # Avoid double separators
        if separator.strip() in prompt[max(0, insert_pos - 20):insert_pos + 20]:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="insert_separator",
                target_region=after_region,
                reason="Separator already present",
                diff_summary="No change (already present)",
            )

        new_prompt = prompt[:insert_pos] + separator + prompt[insert_pos:]

        return new_prompt, MutationRecord(
            mutation_type="structural",
            operation="insert_separator",
            target_region=after_region,
            reason=f"Inserted separator after {after_region}",
            diff_summary=f"Added '{separator.strip()}' after {after_region}",
        )

    def duplicate_summary(
        self,
        prompt: str,
        region_config: dict[str, Any],
        target_region: str,
        summary_prefix: str = "Remember: ",
    ) -> tuple[str, MutationRecord]:
        """Add a brief echo of a region near the end of the prompt.

        Extracts the first sentence of the region and appends it as a
        reminder at the end.
        """
        sys_regions = region_config.get("system_prompt", {}).get("regions", [])
        region_def = None
        for rdef in sys_regions:
            if rdef.get("name") == target_region:
                region_def = rdef
                break

        if region_def is None:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="duplicate_summary",
                target_region=target_region,
                reason=f"Region {target_region} not found",
                diff_summary="No change",
            )

        start_marker = region_def.get("start_marker", "")
        if not start_marker:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="duplicate_summary",
                target_region=target_region,
                reason="No start_marker",
                diff_summary="No change",
            )

        start_idx = prompt.find(start_marker)
        if start_idx == -1:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="duplicate_summary",
                target_region=target_region,
                reason="Start marker not found",
                diff_summary="No change",
            )

        # Extract first sentence after the marker
        content_start = start_idx + len(start_marker)
        rest = prompt[content_start:content_start + 500].strip()
        # Find first sentence boundary
        for delim in (". ", ".\n", "!\n", "?\n"):
            pos = rest.find(delim)
            if pos != -1:
                first_sentence = rest[: pos + 1]
                break
        else:
            first_sentence = rest[:200].strip()

        if not first_sentence:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="duplicate_summary",
                target_region=target_region,
                reason="Could not extract summary sentence",
                diff_summary="No change",
            )

        summary = f"\n\n{summary_prefix}{first_sentence}\n"
        new_prompt = prompt.rstrip() + summary

        return new_prompt, MutationRecord(
            mutation_type="structural",
            operation="duplicate_summary",
            target_region=target_region,
            reason=f"Added summary echo of {target_region} at end",
            diff_summary=f"Appended {len(summary)} char summary",
        )

    def adjust_emphasis(
        self,
        prompt: str,
        region_config: dict[str, Any],
        target_region: str,
        emphasis_markers: tuple[str, str] = ("**", "**"),
    ) -> tuple[str, MutationRecord]:
        """Add emphasis markers around a region's key content."""
        sys_regions = region_config.get("system_prompt", {}).get("regions", [])
        region_def = None
        for rdef in sys_regions:
            if rdef.get("name") == target_region:
                region_def = rdef
                break

        if region_def is None:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="adjust_emphasis",
                target_region=target_region,
                reason=f"Region {target_region} not found",
                diff_summary="No change",
            )

        start_marker = region_def.get("start_marker", "")
        if not start_marker:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="adjust_emphasis",
                target_region=target_region,
                reason="No start_marker",
                diff_summary="No change",
            )

        start_idx = prompt.find(start_marker)
        if start_idx == -1:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="adjust_emphasis",
                target_region=target_region,
                reason="Start marker not found",
                diff_summary="No change",
            )

        # Add emphasis around the start marker line
        open_m, close_m = emphasis_markers
        if open_m in prompt[start_idx:start_idx + len(start_marker) + 10]:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="adjust_emphasis",
                target_region=target_region,
                reason="Emphasis already present",
                diff_summary="No change (already emphasized)",
            )

        new_marker = f"{open_m}{start_marker}{close_m}"
        new_prompt = prompt[:start_idx] + new_marker + prompt[start_idx + len(start_marker):]

        return new_prompt, MutationRecord(
            mutation_type="structural",
            operation="adjust_emphasis",
            target_region=target_region,
            reason=f"Added emphasis to {target_region} header",
            diff_summary=f"Wrapped header in {open_m}...{close_m}",
        )

    def apply_best_mutation(
        self,
        spec: str | PromptSpec,
        region_config: dict[str, Any],
        report: DiagnosticReport,
    ) -> tuple[PromptSpec, MutationRecord | None]:
        """Apply the most impactful structural mutation based on diagnostics.

        Picks the first failing issue that has a structural mutation
        suggestion and applies it.  Accepts either a plain prompt string
        or a ``PromptSpec`` and always returns a ``PromptSpec``.
        """
        if isinstance(spec, str):
            spec = PromptSpec.from_string(spec)

        structural_ops = {
            "insert_separator",
            "reorder_sections",
            "duplicate_summary",
            "adjust_emphasis",
            "adjust_section_length",
            "split_to_turns",
        }

        for issue in report.issues:
            if issue.suggested_mutation not in structural_ops:
                continue

            # Handle multi-turn split separately.
            if issue.suggested_mutation == "split_to_turns":
                if spec.has_been_split:
                    continue  # already split, skip
                from .split import split_to_turns

                return split_to_turns(
                    spec.system_prompt, region_config, report,
                )

            region = _extract_region_from_metric(issue.metric_name)

            if issue.suggested_mutation == "insert_separator":
                new_text, record = self.insert_separator(
                    spec.system_prompt, region_config, region,
                )
            elif issue.suggested_mutation == "reorder_sections":
                new_text, record = self.reorder_sections(
                    spec.system_prompt, region_config, region,
                )
            elif issue.suggested_mutation == "duplicate_summary":
                new_text, record = self.duplicate_summary(
                    spec.system_prompt, region_config, region,
                )
            elif issue.suggested_mutation == "adjust_emphasis":
                new_text, record = self.adjust_emphasis(
                    spec.system_prompt, region_config, region,
                )
            else:
                continue

            return PromptSpec(
                system_prompt=new_text,
                prefix_turns=list(spec.prefix_turns),
                has_been_split=spec.has_been_split,
            ), record

        return spec, None


def _extract_region_from_metric(metric_name: str) -> str:
    """Extract region name from a metric name like 'terminal_attention_rules'."""
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
