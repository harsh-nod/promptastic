"""Region annotation engine -- resolve named text spans to character offsets.

Given a JSON configuration describing named regions of a prompt, this module
locates each region's character boundaries within the source text.  Downstream
pipeline stages (the analysis engine) convert these character-level spans into
token-level spans via cumulative-decode mapping.

Three boundary strategies are supported:

    Marker-based   -- literal ``start_marker`` / ``end_marker`` strings
    Regex-based    -- ``start_pattern`` / ``end_pattern`` regular expressions
    Character-range -- explicit ``start_char`` / ``end_char`` integers

Region config layout (regions.json)::

    {
      "system_prompt": {
        "regions": [
          {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
          {"name": "examples", "start_marker": "## Examples", "end_marker": null}
        ]
      },
      "user_message": {
        "regions": [
          {"name": "context", "start_pattern": "(?i)previous:", "end_pattern": "(?i)current:"},
          {"name": "request", "start_char": 0, "end_char": 120}
        ]
      },
      "query_positions": {"terminal": "last_token"},
      "tracked_tokens": ["<", "folder_a"]
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .._types import CharRegionInfo

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_region_config(path: str) -> dict[str, Any]:
    """Read and validate a region configuration JSON file.

    Raises ``FileNotFoundError`` when *path* does not exist and
    ``json.JSONDecodeError`` on malformed JSON.  Basic structural checks
    ensure the top-level value is a mapping.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Region config not found: {path}")

    raw = config_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError(
            f"Region config must be a JSON object, got {type(data).__name__}"
        )

    return data


# ---------------------------------------------------------------------------
# Boundary resolution helpers
# ---------------------------------------------------------------------------

_MARKER_KEYS = ("start_marker", "end_marker")
_PATTERN_KEYS = ("start_pattern", "end_pattern")
_CHAR_KEYS = ("start_char", "end_char")


def _resolve_char_range(
    defn: dict[str, Any],
) -> tuple[int | None, int | None]:
    """Resolve an explicit character-range definition."""

    start: int | None = defn.get("start_char")
    end: int | None = defn.get("end_char")
    return start, end


def _resolve_regex(
    text: str, defn: dict[str, Any]
) -> tuple[int | None, int | None]:
    """Resolve a regex-based definition against *text*."""

    start_pat = defn.get("start_pattern")
    if start_pat is None:
        return None, None

    match_start = re.search(start_pat, text)
    if match_start is None:
        return None, None

    begin = match_start.start()

    end: int | None = None
    end_pat = defn.get("end_pattern")
    if end_pat:
        # Search only the portion of text *after* the start match so that the
        # end pattern cannot overlap with the start match itself.
        match_end = re.search(end_pat, text[match_start.end() :])
        if match_end is not None:
            end = match_start.end() + match_end.start()

    return begin, end


def _resolve_markers(
    text: str, defn: dict[str, Any]
) -> tuple[int | None, int | None]:
    """Resolve a literal-marker definition against *text*."""

    start_marker: str | None = defn.get("start_marker")
    if start_marker is None:
        return None, None

    pos = text.find(start_marker)
    if pos == -1:
        return None, None

    end: int | None = None
    end_marker: str | None = defn.get("end_marker")
    if end_marker:
        search_from = pos + len(start_marker)
        end_pos = text.find(end_marker, search_from)
        if end_pos != -1:
            end = end_pos

    return pos, end


def _locate_span(
    text: str, defn: dict[str, Any]
) -> tuple[int | None, int | None]:
    """Pick the right strategy and return ``(start, end)`` for a region.

    Strategy priority:
        1. Character range  (cheapest, no search needed)
        2. Regex pattern    (most flexible)
        3. Literal markers  (simplest / most common)
    """

    if "start_char" in defn:
        return _resolve_char_range(defn)

    if "start_pattern" in defn:
        return _resolve_regex(text, defn)

    if "start_marker" in defn:
        return _resolve_markers(text, defn)

    # No recognised boundary keys at all.
    return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_text(
    text: str,
    region_defs: list[dict[str, Any]],
    text_offset: int = 0,
) -> dict[str, CharRegionInfo]:
    """Map every region definition to concrete character positions in *text*.

    Parameters
    ----------
    text:
        The source string to scan.
    region_defs:
        Sequence of region descriptors.  Each must contain a ``"name"`` key
        plus one set of boundary specifiers (marker, regex, or char-range).
        May optionally contain a nested ``"regions"`` list for sub-regions
        scoped within the parent span.
    text_offset:
        A constant added to every returned position.  Useful when *text*
        is a slice of a larger assembled prompt and you need positions
        relative to the full prompt.

    Returns
    -------
    dict mapping region name to ``{"char_start": int, "char_end": int}``.
    """

    found: dict[str, CharRegionInfo] = {}

    for defn in region_defs:
        name = defn["name"]

        span_start, span_end = _locate_span(text, defn)

        if span_start is None:
            print(f"  [regions] WARNING: could not locate region '{name}'")
            continue

        # An open-ended region stretches to the end of the text.
        if span_end is None:
            span_end = len(text)

        found[name] = {
            "char_start": span_start + text_offset,
            "char_end": span_end + text_offset,
        }

        # Recurse into nested sub-regions, scoping them to the parent span.
        nested_defs = defn.get("regions")
        if nested_defs:
            parent_slice = text[span_start:span_end]
            nested = annotate_text(
                parent_slice,
                nested_defs,
                text_offset=span_start + text_offset,
            )
            found.update(nested)

    return found


def parse_query_positions(config: dict[str, Any]) -> dict[str, Any]:
    """Pull query-position definitions out of a region config.

    The ``"query_positions"`` block tells the engine which token positions
    to use as the query dimension when extracting attention rows.
    """

    return config.get("query_positions", {})


def parse_tracked_tokens(config: dict[str, Any]) -> list[str]:
    """Pull the tracked-token list out of a region config.

    Tracked tokens are surface-form strings whose logit-lens rank is
    monitored across every layer of the forward pass.
    """

    return list(config.get("tracked_tokens", []))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m promptastic.prep.regions <regions.json> [text_file]")
        sys.exit(1)

    cfg = load_region_config(sys.argv[1])
    print(json.dumps(cfg, indent=2))

    if len(sys.argv) >= 3:
        sample_text = Path(sys.argv[2]).read_text(encoding="utf-8")
        sys_defs = cfg.get("system_prompt", {}).get("regions", [])
        result = annotate_text(sample_text, sys_defs)
        print("\nAnnotated regions:")
        for rname, info in result.items():
            print(f"  {rname}: chars {info['char_start']}..{info['char_end']}")
