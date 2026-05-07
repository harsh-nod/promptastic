"""Structured prompt representation and helpers.

This module treats the system prompt as an ordered collection of named
sections, preserving the text that appears between sections so we can
round-trip back to the original string without losing formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class PromptSection:
    """One logical section inside the system prompt.

    Attributes
    ----------
    name:
        Region name from the region configuration.
    start_marker:
        Marker string that announces the section (typically a header).
    prefix:
        Exact text that appeared immediately before the start marker.
        For the first section this is always an empty string; the leading
        text lives on the parent ``StructuredPrompt``.
    header:
        The literal text slice that matched ``start_marker`` inside the
        original prompt.  We store it separately so callers can replace
        the body without having to recreate the marker.
    body:
        The text between the header and the next section (or end of
        prompt).  Includes trailing newlines exactly as they appeared.
    metadata:
        Optional bag for downstream algorithms.
    """

    name: str
    start_marker: str
    prefix: str
    header: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Render the section back to plain text."""
        return f"{self.prefix}{self.header}{self.body}"


@dataclass
class StructuredPrompt:
    """Structured view over a system prompt."""

    original_text: str
    leading_text: str
    sections: list[PromptSection]
    trailing_text: str
    missing_regions: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render the structured prompt back to plain text."""
        parts = [self.leading_text]
        parts.extend(section.render() for section in self.sections)
        parts.append(self.trailing_text)
        return "".join(parts)

    def get_section(self, name: str) -> PromptSection | None:
        """Return the section with the given name if it exists."""
        for section in self.sections:
            if section.name == name:
                return section
        return None


def parse_prompt(
    prompt_text: str,
    region_config: dict[str, Any],
    *,
    region_names: Iterable[str] | None = None,
) -> StructuredPrompt:
    """Parse ``prompt_text`` into a ``StructuredPrompt``.

    Parameters
    ----------
    prompt_text:
        Full system prompt to parse.
    region_config:
        Region configuration dictionary used throughout the optimizer.
    region_names:
        Optional explicit ordering of region names.  When omitted we use
        the order of regions as they appear in the configuration.
    """
    sys_regions = list(region_config.get("system_prompt", {}).get("regions", []))

    if region_names is not None:
        # Filter the configuration down to the requested names, preserving
        # the supplied order.
        name_set = set(region_names)
        sys_regions = [r for r in sys_regions if r.get("name") in name_set]
        lookup = {r.get("name"): r for r in sys_regions}
        sys_regions = [lookup[name] for name in region_names if name in lookup]

    discovered: list[tuple[int, dict[str, Any]]] = []
    missing: list[str] = []

    search_cursor = 0

    for rdef in sys_regions:
        marker = rdef.get("start_marker", "")
        name = rdef.get("name", "")
        if not marker or not name:
            continue
        idx = prompt_text.find(marker, search_cursor)
        if idx == -1:
            missing.append(name)
            continue
        discovered.append((idx, rdef))
        search_cursor = idx + len(marker)

    discovered.sort(key=lambda item: item[0])

    sections: list[PromptSection] = []
    cursor = 0
    leading_text = ""

    for index, (position, rdef) in enumerate(discovered):
        start_marker = rdef["start_marker"]
        name = rdef["name"]

        if not sections:
            leading_text = prompt_text[:position]
            prefix = ""
        else:
            prefix = prompt_text[cursor:position]

        header = prompt_text[position:position + len(start_marker)]
        body_start = position + len(start_marker)

        next_position = (
            discovered[index + 1][0] if index + 1 < len(discovered) else len(prompt_text)
        )
        body = prompt_text[body_start:next_position]

        sections.append(
            PromptSection(
                name=name,
                start_marker=start_marker,
                prefix=prefix,
                header=header,
                body=body,
            )
        )
        cursor = next_position

    trailing_text = prompt_text[cursor:]

    return StructuredPrompt(
        original_text=prompt_text,
        leading_text=leading_text,
        sections=sections,
        trailing_text=trailing_text,
        missing_regions=missing,
    )
