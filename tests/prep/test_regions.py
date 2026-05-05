"""Tests for promptastic.prep.regions -- region annotation engine."""

import json
import tempfile
from pathlib import Path

import pytest

from promptastic.prep.regions import (
    annotate_text,
    load_region_config,
    parse_query_positions,
    parse_tracked_tokens,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


SAMPLE_TEXT = (
    "## Rules\n"
    "You must follow these rules carefully.\n"
    "## Examples\n"
    "Here is an example of good output.\n"
    "## Output Format\n"
    "Return JSON only."
)


# ---------------------------------------------------------------
# Marker-based detection
# ---------------------------------------------------------------


def test_marker_based_detection():
    defs = [
        {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
    ]
    result = annotate_text(SAMPLE_TEXT, defs)
    assert "rules" in result
    info = result["rules"]
    assert info["char_start"] == SAMPLE_TEXT.index("## Rules")
    assert info["char_end"] == SAMPLE_TEXT.index("## Examples")


def test_marker_based_end_marker_none():
    """When end_marker is None, region extends to end of text."""
    defs = [
        {"name": "output_format", "start_marker": "## Output Format", "end_marker": None},
    ]
    result = annotate_text(SAMPLE_TEXT, defs)
    assert "output_format" in result
    assert result["output_format"]["char_end"] == len(SAMPLE_TEXT)


def test_marker_based_missing_marker():
    """Missing start_marker should produce no entry (not crash)."""
    defs = [
        {"name": "missing", "start_marker": "## NONEXISTENT", "end_marker": None},
    ]
    result = annotate_text(SAMPLE_TEXT, defs)
    assert "missing" not in result


# ---------------------------------------------------------------
# Regex-based detection
# ---------------------------------------------------------------


def test_regex_based_detection():
    defs = [
        {
            "name": "rules_regex",
            "start_pattern": r"## Rules",
            "end_pattern": r"## Examples",
        },
    ]
    result = annotate_text(SAMPLE_TEXT, defs)
    assert "rules_regex" in result
    info = result["rules_regex"]
    assert info["char_start"] == SAMPLE_TEXT.index("## Rules")
    assert info["char_end"] == SAMPLE_TEXT.index("## Examples")


def test_regex_case_insensitive():
    text = "Previous: old data\nCurrent: new data"
    defs = [
        {
            "name": "context",
            "start_pattern": r"(?i)previous:",
            "end_pattern": r"(?i)current:",
        },
    ]
    result = annotate_text(text, defs)
    assert "context" in result
    assert result["context"]["char_start"] == 0


def test_regex_no_match():
    defs = [
        {"name": "no_match", "start_pattern": r"ZZZZZ", "end_pattern": r"YYYYY"},
    ]
    result = annotate_text(SAMPLE_TEXT, defs)
    assert "no_match" not in result


def test_regex_no_end_pattern():
    """start_pattern without end_pattern should extend to end of text."""
    defs = [
        {"name": "tail", "start_pattern": r"## Output Format"},
    ]
    result = annotate_text(SAMPLE_TEXT, defs)
    assert "tail" in result
    assert result["tail"]["char_end"] == len(SAMPLE_TEXT)


# ---------------------------------------------------------------
# Character range detection
# ---------------------------------------------------------------


def test_char_range_detection():
    defs = [
        {"name": "head", "start_char": 0, "end_char": 20},
    ]
    result = annotate_text(SAMPLE_TEXT, defs)
    assert "head" in result
    assert result["head"]["char_start"] == 0
    assert result["head"]["char_end"] == 20


def test_char_range_end_none():
    """end_char=None means extend to end of text."""
    defs = [
        {"name": "rest", "start_char": 10, "end_char": None},
    ]
    result = annotate_text(SAMPLE_TEXT, defs)
    assert "rest" in result
    assert result["rest"]["char_start"] == 10
    assert result["rest"]["char_end"] == len(SAMPLE_TEXT)


# ---------------------------------------------------------------
# Nested sub-regions
# ---------------------------------------------------------------


def test_nested_sub_regions():
    """Regions with a nested 'regions' list should produce sub-entries."""
    defs = [
        {
            "name": "system",
            "start_marker": "## Rules",
            "end_marker": None,
            "regions": [
                {
                    "name": "examples",
                    "start_marker": "## Examples",
                    "end_marker": "## Output Format",
                },
            ],
        },
    ]
    result = annotate_text(SAMPLE_TEXT, defs)
    assert "system" in result
    assert "examples" in result
    # The nested region's char_start should be within the parent span
    assert result["examples"]["char_start"] >= result["system"]["char_start"]
    assert result["examples"]["char_end"] <= result["system"]["char_end"]


# ---------------------------------------------------------------
# text_offset
# ---------------------------------------------------------------


def test_text_offset_applied():
    defs = [
        {"name": "head", "start_char": 0, "end_char": 10},
    ]
    offset = 500
    result = annotate_text(SAMPLE_TEXT, defs, text_offset=offset)
    assert result["head"]["char_start"] == 0 + offset
    assert result["head"]["char_end"] == 10 + offset


def test_text_offset_with_markers():
    defs = [
        {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
    ]
    offset = 200
    result = annotate_text(SAMPLE_TEXT, defs, text_offset=offset)
    assert result["rules"]["char_start"] == SAMPLE_TEXT.index("## Rules") + offset
    assert result["rules"]["char_end"] == SAMPLE_TEXT.index("## Examples") + offset


# ---------------------------------------------------------------
# parse_query_positions / parse_tracked_tokens
# ---------------------------------------------------------------


def test_parse_query_positions_present():
    config = {"query_positions": {"terminal": "last_token"}}
    result = parse_query_positions(config)
    assert result == {"terminal": "last_token"}


def test_parse_query_positions_absent():
    config = {}
    result = parse_query_positions(config)
    assert result == {}


def test_parse_tracked_tokens_present():
    config = {"tracked_tokens": ["<", "folder_a"]}
    result = parse_tracked_tokens(config)
    assert result == ["<", "folder_a"]


def test_parse_tracked_tokens_absent():
    config = {}
    result = parse_tracked_tokens(config)
    assert result == []


def test_parse_tracked_tokens_tuple_input():
    """Even if the config stores a tuple, the output should be a list."""
    config = {"tracked_tokens": ("<", "folder_a")}
    result = parse_tracked_tokens(config)
    assert isinstance(result, list)
    assert result == ["<", "folder_a"]


# ---------------------------------------------------------------
# load_region_config
# ---------------------------------------------------------------


def test_load_region_config_valid():
    data = {"system_prompt": {"regions": []}}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        f.flush()
        result = load_region_config(f.name)
    assert result == data


def test_load_region_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_region_config("/nonexistent/path.json")


def test_load_region_config_not_a_dict():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump([1, 2, 3], f)
        f.flush()
        with pytest.raises(ValueError, match="JSON object"):
            load_region_config(f.name)


# ---------------------------------------------------------------
# No recognized boundary keys
# ---------------------------------------------------------------


def test_no_recognized_keys():
    """A definition with no boundary keys should be skipped."""
    defs = [{"name": "orphan"}]
    result = annotate_text(SAMPLE_TEXT, defs)
    assert "orphan" not in result
