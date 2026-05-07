"""Structural prompt mutations for optimization.

Programmatic operations that modify prompt structure without requiring
an external LLM.  Each mutation preserves region marker integrity so
that re-annotation continues to work after transformation.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Callable

from ._types import DiagnosticReport, MutationRecord, PromptSpec
from .structure import PromptSection, StructuredPrompt, parse_prompt


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
        structured = parse_prompt(prompt, region_config)
        names = [section.name for section in structured.sections]
        if target_region not in names:
            return prompt, _no_region_record("reorder_sections", target_region)

        sections = list(structured.sections)
        index = names.index(target_region)
        section = sections.pop(index)

        if position == "start":
            section = replace(section, prefix="")
            sections.insert(0, section)
        else:
            # Default to moving towards the end.
            section = replace(section, prefix="\n\n" if sections else "")
            sections.append(section)

        # Ensure the first section has no prefix and subsequent sections have spacing.
        normalized_sections: list[PromptSection] = []
        for idx, sec in enumerate(sections):
            if idx == 0:
                normalized_sections.append(replace(sec, prefix=""))
            else:
                prefix = sec.prefix if sec.prefix.strip() else "\n\n"
                normalized_sections.append(replace(sec, prefix=prefix))

        new_structured = StructuredPrompt(
            original_text=structured.original_text,
            leading_text=structured.leading_text,
            sections=normalized_sections,
            trailing_text=structured.trailing_text,
            missing_regions=list(structured.missing_regions),
        )

        new_prompt = new_structured.render()
        return new_prompt, MutationRecord(
            mutation_type="structural",
            operation="reorder_sections",
            target_region=target_region,
            reason=f"Moved {target_region} to {position} of prompt",
            diff_summary=f"Reordered sections ({target_region} -> {position})",
        )

    def insert_separator(
        self,
        prompt: str,
        region_config: dict[str, Any],
        after_region: str,
        separator: str = "\n---\n",
    ) -> tuple[str, MutationRecord]:
        """Insert a separator after a region to reduce context bleed."""
        structured = parse_prompt(prompt, region_config)
        target_section = structured.get_section(after_region)
        if target_section is None:
            return prompt, _no_region_record("insert_separator", after_region)

        body = target_section.body
        if separator.strip() in body[-80:]:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="insert_separator",
                target_region=after_region,
                reason="Separator already present",
                diff_summary="No change (already present)",
            )

        new_body = body.rstrip() + separator + "\n"
        updated_sections: list[PromptSection] = []
        for sec in structured.sections:
            if sec.name == after_region:
                updated_sections.append(replace(sec, body=new_body))
            else:
                updated_sections.append(sec)

        new_structured = StructuredPrompt(
            original_text=structured.original_text,
            leading_text=structured.leading_text,
            sections=updated_sections,
            trailing_text=structured.trailing_text,
            missing_regions=list(structured.missing_regions),
        )

        return new_structured.render(), MutationRecord(
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
        structured = parse_prompt(prompt, region_config)
        target_section = structured.get_section(target_region)
        if target_section is None:
            return prompt, _no_region_record("duplicate_summary", target_region)

        snippet = target_section.body.strip()
        first_sentence = _extract_first_sentence(snippet)
        if not first_sentence:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="duplicate_summary",
                target_region=target_region,
                reason="Could not extract summary sentence",
                diff_summary="No change",
            )

        summary = f"\n\n{summary_prefix}{first_sentence}\n"
        new_trailing = structured.trailing_text.rstrip() + summary

        new_structured = StructuredPrompt(
            original_text=structured.original_text,
            leading_text=structured.leading_text,
            sections=list(structured.sections),
            trailing_text=new_trailing,
            missing_regions=list(structured.missing_regions),
        )

        return new_structured.render(), MutationRecord(
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
        structured = parse_prompt(prompt, region_config)
        target_section = structured.get_section(target_region)
        if target_section is None:
            return prompt, _no_region_record("adjust_emphasis", target_region)

        open_m, close_m = emphasis_markers
        emphasised = f"{open_m}{target_section.start_marker}{close_m}"
        if emphasised in prompt:
            return prompt, MutationRecord(
                mutation_type="structural",
                operation="adjust_emphasis",
                target_region=target_region,
                reason="Emphasis already present",
                diff_summary="No change (already emphasized)",
            )

        new_header = emphasised
        updated_sections: list[PromptSection] = []
        for sec in structured.sections:
            if sec.name == target_region:
                updated_sections.append(replace(sec, header=new_header))
            else:
                updated_sections.append(sec)

        new_structured = StructuredPrompt(
            original_text=structured.original_text,
            leading_text=structured.leading_text,
            sections=updated_sections,
            trailing_text=structured.trailing_text,
            missing_regions=list(structured.missing_regions),
        )

        return new_structured.render(), MutationRecord(
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

        if spec.structured_prompt is None:
            structured = parse_prompt(spec.system_prompt, region_config)
            spec = spec.with_structured(structured)
        else:
            structured = spec.structured_prompt

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

            new_structured = parse_prompt(new_text, region_config)
            new_spec = spec.with_structured(new_structured)
            return new_spec, record

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


def _no_region_record(operation: str, target_region: str) -> MutationRecord:
    return MutationRecord(
        mutation_type="structural",
        operation=operation,
        target_region=target_region,
        reason=f"Region {target_region} not found",
        diff_summary="No change (region not found)",
    )


def _extract_first_sentence(text: str) -> str:
    snippet = text.strip()
    if not snippet:
        return ""
    for delim in (". ", ".\n", "!\n", "?\n", "! ", "? "):
        idx = snippet.find(delim)
        if idx != -1:
            return snippet[: idx + 1].strip()
    return snippet.splitlines()[0].strip()
