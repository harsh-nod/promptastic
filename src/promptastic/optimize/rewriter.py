"""LLM-guided prompt rewriting for optimization.

Uses an external LLM (separate from the model being analysed) to
propose semantic rewrites of prompt regions based on MI diagnostics.
"""

from __future__ import annotations

from typing import Any

from ._types import DiagnosticReport, MutationRecord
from .diagnostics import format_diagnostic_report


_REWRITE_TEMPLATE = """\
You are optimizing a system prompt for a transformer model. Below are
mechanistic interpretability diagnostics showing how the model processes
the prompt internally.

## Diagnostic Report
{diagnostic_report}

## Current System Prompt (with region annotations)
{annotated_prompt}

## Target Region to Modify
{target_region}

## Current Text of Target Region
{region_text}

## Task
Rewrite the "{target_region}" region to address the diagnostic issues.

Constraints:
- Preserve the semantic meaning and instructions
- Keep approximately the same length (+/- 20%)
- Maintain any section headers or markers so region detection still works
- Make the content clearer and more structurally distinct

Return ONLY the rewritten region text, with no explanation or preamble.
"""


class LLMRewriter:
    """Propose prompt rewrites using an external LLM."""

    def __init__(
        self,
        model: str = "",
        api_key: str = "",
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._client: Any = None

    def _ensure_client(self) -> None:
        """Lazy-init the Anthropic client."""
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
                "LLM rewriting requires the 'anthropic' package. "
                "Install with: pip install anthropic"
            ) from exc

    def _extract_region_text(
        self,
        prompt: str,
        region_config: dict[str, Any],
        target_region: str,
    ) -> str:
        """Extract the text of a region from the prompt."""
        sys_regions = region_config.get("system_prompt", {}).get("regions", [])
        for rdef in sys_regions:
            if rdef.get("name") != target_region:
                continue
            start_marker = rdef.get("start_marker", "")
            end_marker = rdef.get("end_marker", "")
            if not start_marker:
                return ""
            start_idx = prompt.find(start_marker)
            if start_idx == -1:
                return ""
            if end_marker:
                end_idx = prompt.find(end_marker, start_idx)
                if end_idx == -1:
                    end_idx = len(prompt)
            else:
                end_idx = len(prompt)
            return prompt[start_idx:end_idx]
        return ""

    def propose_rewrite(
        self,
        prompt: str,
        region_config: dict[str, Any],
        report: DiagnosticReport,
        target_region: str,
    ) -> tuple[str, MutationRecord]:
        """Ask the LLM to rewrite a specific region.

        Returns (new_prompt, mutation_record).
        """
        self._ensure_client()

        region_text = self._extract_region_text(prompt, region_config, target_region)
        if not region_text:
            return prompt, MutationRecord(
                mutation_type="llm_rewrite",
                operation="propose_rewrite",
                target_region=target_region,
                reason="Could not extract region text",
                diff_summary="No change",
            )

        rewrite_prompt = _REWRITE_TEMPLATE.format(
            diagnostic_report=format_diagnostic_report(report),
            annotated_prompt=prompt,
            target_region=target_region,
            region_text=region_text,
        )

        message = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": rewrite_prompt}],
        )

        new_text = message.content[0].text.strip()

        # Replace the region text in the prompt
        new_prompt = prompt.replace(region_text, new_text, 1)

        return new_prompt, MutationRecord(
            mutation_type="llm_rewrite",
            operation="propose_rewrite",
            target_region=target_region,
            reason=f"LLM rewrote {target_region} based on {report.num_failing} failing metrics",
            diff_summary=(
                f"Replaced {len(region_text)} chars with {len(new_text)} chars "
                f"via {self.model}"
            ),
        )

    def apply_best_rewrite(
        self,
        prompt: str,
        region_config: dict[str, Any],
        report: DiagnosticReport,
    ) -> tuple[str, MutationRecord | None]:
        """Rewrite the region associated with the worst-scoring metric."""
        if not report.issues:
            return prompt, None

        # Find the failing issue with a region that we can rewrite
        for issue in report.issues:
            region = _extract_region_from_metric(issue.metric_name)
            if region:
                return self.propose_rewrite(
                    prompt, region_config, report, region,
                )

        return prompt, None


def _extract_region_from_metric(metric_name: str) -> str:
    """Extract region name from a metric name."""
    for prefix in (
        "terminal_attention_",
        "retention_ratio_",
        "peak_layer_frac_",
        "causal_importance_",
        "head_variance_",
    ):
        if metric_name.startswith(prefix):
            return metric_name[len(prefix):]
    return ""
