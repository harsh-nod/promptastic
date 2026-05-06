"""Multi-turn split mutation.

Detects when a monolithic system prompt should be decomposed into
multiple conversation turns and determines what those turns should be.
For example, few-shot examples become explicit user/assistant pairs
while rules/policy stays in the system prompt.
"""

from __future__ import annotations

import re
from typing import Any

from ._types import DiagnosticReport, MutationRecord, PromptSpec

# ======================================================================
# Region classification
# ======================================================================

# Regions that should always stay in the system prompt.
_KEEP_IN_SYSTEM = frozenset({
    "rules", "directive", "entity_rules", "passage_rules",
    "policy", "output_format", "instructions", "constraints",
})

# Regions that are candidates for extraction as user/assistant turn pairs.
_EXAMPLE_PATTERNS = frozenset({
    "examples", "few_shot", "demonstrations", "approved_responses",
    "sample_responses", "templates",
})

# Regions that are candidates for extraction as context-setting turns.
_CONTEXT_PATTERNS = frozenset({
    "context", "background", "history", "reference",
    "knowledge_base", "documents",
})

# Turn label patterns recognised when parsing example content.
_TURN_PATTERNS = re.compile(
    r"^(User|Customer|Human|Input)\s*:\s*",
    re.MULTILINE | re.IGNORECASE,
)
_RESPONSE_PATTERNS = re.compile(
    r"^(Assistant|Agent|AI|Output|Bot)\s*:\s*",
    re.MULTILINE | re.IGNORECASE,
)


def classify_region(name: str, content: str = "") -> str:
    """Classify a region as ``keep``, ``example``, or ``context``.

    Uses the region name first, then falls back to content analysis.
    """
    lower = name.lower()
    if lower in _KEEP_IN_SYSTEM:
        return "keep"
    if lower in _EXAMPLE_PATTERNS:
        return "example"
    if lower in _CONTEXT_PATTERNS:
        return "context"

    # Content-based fallback: if the text contains User:/Agent: pairs
    # it's an example region.
    if content and _TURN_PATTERNS.search(content) and _RESPONSE_PATTERNS.search(content):
        return "example"

    # Default: leave in system prompt.
    return "keep"


# ======================================================================
# Example turn parser
# ======================================================================

def _parse_example_turns(text: str) -> list[dict[str, str]]:
    """Parse ``User:/Assistant:``-style pairs into message dicts.

    Returns a list of ``{"role": "user"|"assistant", "content": ...}``
    dicts.  If no turn markers are found, wraps the entire text as a
    single user turn with an assistant acknowledgment.
    """
    # Build a combined regex that captures both sides.
    combined = re.compile(
        r"^(?:User|Customer|Human|Input)\s*:\s*(.*?)(?=^(?:Assistant|Agent|AI|Output|Bot)\s*:|$)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    response_split = re.compile(
        r"^(?:Assistant|Agent|AI|Output|Bot)\s*:\s*(.*?)(?=^(?:User|Customer|Human|Input)\s*:|$)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )

    user_matches = list(combined.finditer(text))
    assistant_matches = list(response_split.finditer(text))

    if not user_matches or not assistant_matches:
        # No recognisable turn structure — wrap as a single exchange.
        return [
            {"role": "user", "content": f"Here are reference examples:\n{text.strip()}"},
            {"role": "assistant", "content": "Understood, I'll follow these examples."},
        ]

    turns: list[dict[str, str]] = []
    # Pair them up in document order.
    pairs = min(len(user_matches), len(assistant_matches))
    for i in range(pairs):
        user_text = user_matches[i].group(1).strip()
        asst_text = assistant_matches[i].group(1).strip()
        if user_text:
            turns.append({"role": "user", "content": user_text})
        if asst_text:
            turns.append({"role": "assistant", "content": asst_text})

    # If there are leftover user turns without a response, add them
    # with a generic acknowledgment.
    for i in range(pairs, len(user_matches)):
        user_text = user_matches[i].group(1).strip()
        if user_text:
            turns.append({"role": "user", "content": user_text})
            turns.append({"role": "assistant", "content": "Understood."})

    return turns or [
        {"role": "user", "content": f"Here are reference examples:\n{text.strip()}"},
        {"role": "assistant", "content": "Understood, I'll follow these examples."},
    ]


# ======================================================================
# Region extraction helpers
# ======================================================================

def _extract_region_text(
    prompt: str,
    region_def: dict[str, Any],
) -> tuple[str, int, int] | None:
    """Locate a region in the prompt using its markers.

    Returns ``(text, start_idx, end_idx)`` or ``None``.
    """
    start_marker = region_def.get("start_marker", "")
    if not start_marker:
        return None

    start_idx = prompt.find(start_marker)
    if start_idx == -1:
        return None

    end_marker = region_def.get("end_marker")
    if end_marker:
        end_idx = prompt.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            end_idx = len(prompt)
    else:
        end_idx = len(prompt)

    return prompt[start_idx:end_idx], start_idx, end_idx


# ======================================================================
# Main split function
# ======================================================================

def split_to_turns(
    prompt: str,
    region_config: dict[str, Any],
    report: DiagnosticReport | None = None,
    metrics: dict[str, float] | None = None,
) -> tuple[PromptSpec, MutationRecord]:
    """Split a monolithic system prompt into multi-turn format.

    Examines each region, classifies it, and extracts ``example`` and
    ``context`` regions into prefix conversation turns.  ``keep`` regions
    remain in the system prompt.

    Returns a ``PromptSpec`` with the reduced system prompt and the
    extracted prefix turns, plus a ``MutationRecord`` describing what
    was done.
    """
    sys_regions = region_config.get("system_prompt", {}).get("regions", [])

    if not sys_regions:
        return PromptSpec.from_string(prompt), MutationRecord(
            mutation_type="structural",
            operation="split_to_turns",
            target_region="",
            reason="No regions defined in config",
            diff_summary="No change (no regions)",
        )

    prefix_turns: list[dict[str, str]] = []
    extracted_names: list[str] = []
    # Track regions to remove — collect spans, remove in reverse order
    # to preserve earlier indices.
    removal_spans: list[tuple[int, int]] = []

    for rdef in sys_regions:
        name = rdef.get("name", "")
        extraction = _extract_region_text(prompt, rdef)
        if extraction is None:
            continue

        text, start_idx, end_idx = extraction
        classification = classify_region(name, text)

        if classification == "example":
            turns = _parse_example_turns(text)
            prefix_turns.extend(turns)
            removal_spans.append((start_idx, end_idx))
            extracted_names.append(name)

        elif classification == "context":
            content = text.strip()
            # Remove the header marker from the content for cleaner turns.
            start_marker = rdef.get("start_marker", "")
            if start_marker and content.startswith(start_marker):
                content = content[len(start_marker):].strip()
            prefix_turns.append({"role": "user", "content": content})
            prefix_turns.append(
                {"role": "assistant", "content": "Understood, I'll keep this in mind."},
            )
            removal_spans.append((start_idx, end_idx))
            extracted_names.append(name)

    if not extracted_names:
        return PromptSpec.from_string(prompt), MutationRecord(
            mutation_type="structural",
            operation="split_to_turns",
            target_region="",
            reason="No regions suitable for extraction",
            diff_summary="No change (nothing to split)",
        )

    # Remove extracted regions from the prompt (reverse order to
    # preserve indices).
    reduced = prompt
    for start, end in sorted(removal_spans, reverse=True):
        reduced = reduced[:start] + reduced[end:]

    # Clean up excess whitespace.
    reduced = re.sub(r"\n{3,}", "\n\n", reduced).strip() + "\n"

    spec = PromptSpec(
        system_prompt=reduced,
        prefix_turns=prefix_turns,
        has_been_split=True,
    )

    num_turns = len(prefix_turns)
    regions_str = ", ".join(extracted_names)

    return spec, MutationRecord(
        mutation_type="structural",
        operation="split_to_turns",
        target_region=regions_str,
        reason=f"Extracted {len(extracted_names)} region(s) into {num_turns} prefix turns",
        diff_summary=(
            f"Split: {regions_str} -> {num_turns} turns; "
            f"system prompt reduced by {len(prompt) - len(reduced)} chars"
        ),
    )
