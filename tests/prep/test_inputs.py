"""Tests for promptastic.prep.inputs -- test-case assembly."""

from promptastic.prep.inputs import build_test_cases


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _simple_region_config():
    return {
        "system_prompt": {
            "regions": [
                {"name": "rules", "start_marker": "## Rules", "end_marker": "## End"},
            ],
        },
        "user_message": {
            "regions": [
                {"name": "request", "start_char": 0, "end_char": 10},
            ],
        },
        "response": {
            "regions": [],
        },
        "query_positions": {"terminal": "last_token"},
        "tracked_tokens": ["<"],
    }


def _simple_system_prompt():
    return "## Rules\nDo stuff.\n## End\nDone."


def _simple_conversations():
    return [
        {
            "id": "case_01",
            "user_message": "Hello world, how are you?",
            "response": "I am fine.",
        },
        {
            "id": "case_02",
            "user_message": "Another question here.",
        },
    ]


# ---------------------------------------------------------------
# build_test_cases
# ---------------------------------------------------------------


def test_build_test_cases_basic_structure():
    result = build_test_cases(
        _simple_system_prompt(),
        _simple_conversations(),
        _simple_region_config(),
    )
    assert "system_prompt" in result
    assert "system_regions" in result
    assert "cases" in result
    assert "query_positions" in result
    assert "tracked_tokens" in result
    assert "capture_config" in result


def test_system_regions_annotated():
    result = build_test_cases(
        _simple_system_prompt(),
        _simple_conversations(),
        _simple_region_config(),
    )
    sys_regions = result["system_regions"]
    assert "rules" in sys_regions
    info = sys_regions["rules"]
    assert "char_start" in info
    assert "char_end" in info


def test_case_count_matches_conversations():
    convs = _simple_conversations()
    result = build_test_cases(
        _simple_system_prompt(),
        convs,
        _simple_region_config(),
    )
    assert len(result["cases"]) == len(convs)


def test_case_ids_preserved():
    result = build_test_cases(
        _simple_system_prompt(),
        _simple_conversations(),
        _simple_region_config(),
    )
    ids = [c["id"] for c in result["cases"]]
    assert ids == ["case_01", "case_02"]


def test_per_case_user_regions():
    result = build_test_cases(
        _simple_system_prompt(),
        _simple_conversations(),
        _simple_region_config(),
    )
    case = result["cases"][0]
    assert "user_regions" in case
    # The global user_message regions define "request" at chars 0..10
    assert "request" in case["user_regions"]


def test_per_case_response_regions():
    result = build_test_cases(
        _simple_system_prompt(),
        _simple_conversations(),
        _simple_region_config(),
    )
    case = result["cases"][0]
    assert "response_regions" in case


def test_response_defaults_to_empty_string():
    result = build_test_cases(
        _simple_system_prompt(),
        _simple_conversations(),
        _simple_region_config(),
    )
    # case_02 has no "response" key
    case = result["cases"][1]
    assert case["response"] == ""


def test_capture_config_included_when_provided():
    capture = {"attention": True, "logit_lens": True}
    result = build_test_cases(
        _simple_system_prompt(),
        _simple_conversations(),
        _simple_region_config(),
        capture_config=capture,
    )
    assert result["capture_config"] == capture


def test_capture_config_empty_when_not_provided():
    result = build_test_cases(
        _simple_system_prompt(),
        _simple_conversations(),
        _simple_region_config(),
    )
    assert result["capture_config"] == {}


def test_query_positions_from_config():
    result = build_test_cases(
        _simple_system_prompt(),
        _simple_conversations(),
        _simple_region_config(),
    )
    assert result["query_positions"] == {"terminal": "last_token"}


def test_tracked_tokens_from_config():
    result = build_test_cases(
        _simple_system_prompt(),
        _simple_conversations(),
        _simple_region_config(),
    )
    assert result["tracked_tokens"] == ["<"]


def test_per_case_user_regions_override():
    """When a conversation provides its own user_regions, those take priority."""
    convs = [
        {
            "id": "custom",
            "user_message": "Hello world, how are you?",
            "user_regions": [
                {"name": "greeting", "start_char": 0, "end_char": 5},
            ],
        },
    ]
    result = build_test_cases(
        _simple_system_prompt(),
        convs,
        _simple_region_config(),
    )
    case = result["cases"][0]
    # Should use the per-case override, not the global definition
    assert "greeting" in case["user_regions"]
    # The global "request" should NOT be present
    assert "request" not in case["user_regions"]
