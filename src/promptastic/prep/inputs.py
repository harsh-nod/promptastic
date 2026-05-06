"""Test-case assembly -- combine prompt, regions, and conversations into analysis input.

Reads a system prompt text file, a region configuration JSON, and a
conversations JSON array, then produces the unified ``test_cases.json``
consumed by the analysis engine.

Usage::

    python -m promptastic.prep.inputs \\
        --prompt system_prompt.txt \\
        --regions regions.json \\
        --conversations conversations.json \\
        --output test_cases.json \\
        [--captures captures.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .regions import (
    annotate_text,
    load_region_config,
    parse_query_positions,
    parse_tracked_tokens,
)


def _load_capture_config(path: str | None) -> dict[str, Any] | None:
    """Optionally load a capture-configuration JSON file.

    Returns ``None`` when *path* is ``None`` or the file does not exist,
    so callers can treat the result as "use engine defaults."
    """

    if path is None:
        return None

    p = Path(path)
    if not p.exists():
        print(f"  [inputs] WARNING: capture config not found at {path}, skipping")
        return None

    raw = p.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"Capture config must be a JSON object, got {type(data).__name__}"
        )
    return data


def _annotate_case_section(
    text: str,
    global_defs: list[dict[str, Any]],
    per_case_defs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Annotate a single text section (user message or response).

    Per-case region definitions, when present, take priority over the
    global definitions from the region config.
    """

    defs = per_case_defs if per_case_defs is not None else global_defs
    return annotate_text(text, defs)


def build_test_cases(
    system_prompt: str,
    conversations: list[dict[str, Any]],
    region_config: dict[str, Any],
    capture_config: dict[str, Any] | None = None,
    prefix_turns: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Assemble the full test-cases structure from its constituent parts.

    Parameters
    ----------
    system_prompt:
        Complete system-prompt text.
    conversations:
        List of conversation dicts.  Each must contain ``"id"`` and
        ``"user_message"``.  Optional keys: ``"response"``,
        ``"user_regions"`` (list of region defs), ``"response_regions"``.
    region_config:
        Loaded region configuration (the dict returned by
        ``load_region_config``).
    capture_config:
        Optional dict describing which engine capture modes to enable
        (attention, logit_lens, per_head, patching, etc.).
    prefix_turns:
        Optional list of user/assistant message dicts to insert between
        the system prompt and the final user message.  Used when the
        optimizer has split a monolithic prompt into multi-turn format.

    Returns
    -------
    A dict ready to be serialised as ``test_cases.json``.
    """

    # -- system-prompt regions ------------------------------------------------
    sys_defs = region_config.get("system_prompt", {}).get("regions", [])
    system_regions = annotate_text(system_prompt, sys_defs)

    # -- global per-section region defs (applied to every case) ---------------
    user_global_defs = region_config.get("user_message", {}).get("regions", [])
    resp_global_defs = region_config.get("response", {}).get("regions", [])

    # -- iterate conversations and build cases --------------------------------
    cases: list[dict[str, Any]] = []

    for conv in conversations:
        user_msg: str = conv["user_message"]
        response: str = conv.get("response", "")

        user_regions = _annotate_case_section(
            user_msg,
            user_global_defs,
            conv.get("user_regions"),
        )

        response_regions = _annotate_case_section(
            response,
            resp_global_defs,
            conv.get("response_regions"),
        )

        cases.append(
            {
                "id": conv["id"],
                "user_message": user_msg,
                "response": response,
                "user_regions": user_regions,
                "response_regions": response_regions,
            }
        )

    # -- top-level metadata ---------------------------------------------------
    query_positions = parse_query_positions(region_config)
    tracked_tokens = parse_tracked_tokens(region_config)

    output: dict[str, Any] = {
        "system_prompt": system_prompt,
        "system_regions": system_regions,
        "query_positions": query_positions,
        "tracked_tokens": tracked_tokens,
        "capture_config": capture_config if capture_config is not None else {},
        "cases": cases,
    }

    if prefix_turns:
        output["prefix_turns"] = prefix_turns

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``python -m promptastic.prep.inputs``."""

    parser = argparse.ArgumentParser(
        description="Build test_cases.json for the promptastic analysis engine.",
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Path to the system-prompt text file.",
    )
    parser.add_argument(
        "--regions",
        required=True,
        help="Path to the region-config JSON file.",
    )
    parser.add_argument(
        "--conversations",
        required=True,
        help="Path to a JSON array of conversation objects "
        "(each with id, user_message, and optionally response).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the generated test_cases.json.",
    )
    parser.add_argument(
        "--captures",
        default=None,
        help="Optional path to a capture-config JSON file.",
    )

    args = parser.parse_args()

    # -- Load inputs ----------------------------------------------------------
    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        print(f"ERROR: prompt file not found: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    system_prompt = prompt_path.read_text(encoding="utf-8")

    region_config = load_region_config(args.regions)

    conv_path = Path(args.conversations)
    if not conv_path.exists():
        print(
            f"ERROR: conversations file not found: {args.conversations}",
            file=sys.stderr,
        )
        sys.exit(1)

    conversations: list[dict[str, Any]] = json.loads(
        conv_path.read_text(encoding="utf-8")
    )

    capture_config = _load_capture_config(args.captures)

    print(f"System prompt: {len(system_prompt)} characters")
    print(f"Conversations: {len(conversations)} entries")

    # -- Build ----------------------------------------------------------------
    test_cases = build_test_cases(
        system_prompt,
        conversations,
        region_config,
        capture_config,
    )

    # -- Write ----------------------------------------------------------------
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(test_cases, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Written: {out}")
    print(f"  System regions: {list(test_cases['system_regions'].keys())}")
    print(f"  Cases: {len(test_cases['cases'])}")

    if test_cases["tracked_tokens"]:
        print(f"  Tracked tokens: {test_cases['tracked_tokens']}")
    if test_cases["capture_config"]:
        print(f"  Capture config keys: {list(test_cases['capture_config'].keys())}")


if __name__ == "__main__":
    main()
