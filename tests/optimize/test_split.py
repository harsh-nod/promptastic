"""Tests for multi-turn split mutation."""

import pytest

from promptastic.optimize.split import (
    _parse_example_turns,
    classify_region,
    split_to_turns,
)
from promptastic.optimize._types import PromptSpec


# ======================================================================
# classify_region
# ======================================================================


class TestClassifyRegion:

    def test_rules_kept(self):
        assert classify_region("rules") == "keep"
        assert classify_region("policy") == "keep"
        assert classify_region("directive") == "keep"
        assert classify_region("output_format") == "keep"

    def test_examples_extracted(self):
        assert classify_region("examples") == "example"
        assert classify_region("few_shot") == "example"
        assert classify_region("approved_responses") == "example"
        assert classify_region("demonstrations") == "example"

    def test_context_extracted(self):
        assert classify_region("context") == "context"
        assert classify_region("background") == "context"
        assert classify_region("history") == "context"

    def test_unknown_defaults_to_keep(self):
        assert classify_region("custom_region") == "keep"

    def test_content_fallback_detects_turns(self):
        content = "Customer: I need help\nAgent: Sure thing"
        assert classify_region("my_region", content) == "example"

    def test_content_fallback_no_turns(self):
        content = "This is just some text without turn markers."
        assert classify_region("my_region", content) == "keep"


# ======================================================================
# _parse_example_turns
# ======================================================================


class TestParseExampleTurns:

    def test_customer_agent_pairs(self):
        text = (
            "Customer: I want a refund\n"
            "Agent: Let me check your order\n\n"
            "Customer: My product broke\n"
            "Agent: I'll help with that"
        )
        turns = _parse_example_turns(text)
        assert len(turns) == 4
        assert turns[0] == {"role": "user", "content": "I want a refund"}
        assert turns[1] == {"role": "assistant", "content": "Let me check your order"}
        assert turns[2] == {"role": "user", "content": "My product broke"}
        assert turns[3] == {"role": "assistant", "content": "I'll help with that"}

    def test_user_assistant_pairs(self):
        text = (
            "User: Hello\n"
            "Assistant: Hi there\n"
        )
        turns = _parse_example_turns(text)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"

    def test_no_markers_fallback(self):
        text = "Some example text without any turn markers."
        turns = _parse_example_turns(text)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert "reference examples" in turns[0]["content"]
        assert turns[1]["role"] == "assistant"

    def test_human_ai_pairs(self):
        text = (
            "Human: What is Python?\n"
            "AI: Python is a programming language.\n"
        )
        turns = _parse_example_turns(text)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert "Python" in turns[0]["content"]
        assert turns[1]["role"] == "assistant"


# ======================================================================
# split_to_turns
# ======================================================================

SAMPLE_PROMPT = """\
You are a customer support agent.

## Policy
- Be polite
- Offer refunds within 30 days

## Approved Responses
Customer: I want a refund
Agent: Let me check your order details.

Customer: Product broke
Agent: I'm sorry to hear that."""

SAMPLE_CONFIG = {
    "system_prompt": {
        "regions": [
            {"name": "policy", "start_marker": "## Policy", "end_marker": "## Approved Responses"},
            {"name": "approved_responses", "start_marker": "## Approved Responses", "end_marker": None},
        ]
    }
}


class TestSplitToTurns:

    def test_extracts_examples_keeps_rules(self):
        spec, record = split_to_turns(SAMPLE_PROMPT, SAMPLE_CONFIG)
        assert isinstance(spec, PromptSpec)
        assert spec.has_been_split is True
        # Policy should remain in system prompt
        assert "## Policy" in spec.system_prompt
        assert "Be polite" in spec.system_prompt
        # Approved responses should be extracted
        assert "## Approved Responses" not in spec.system_prompt
        assert len(spec.prefix_turns) > 0

    def test_prefix_turns_are_user_assistant_pairs(self):
        spec, _ = split_to_turns(SAMPLE_PROMPT, SAMPLE_CONFIG)
        roles = [t["role"] for t in spec.prefix_turns]
        # Should have user/assistant pairs
        assert "user" in roles
        assert "assistant" in roles

    def test_mutation_record(self):
        _, record = split_to_turns(SAMPLE_PROMPT, SAMPLE_CONFIG)
        assert record.operation == "split_to_turns"
        assert record.mutation_type == "structural"
        assert "approved_responses" in record.target_region

    def test_no_regions_no_change(self):
        spec, record = split_to_turns(SAMPLE_PROMPT, {"system_prompt": {"regions": []}})
        assert spec.system_prompt == SAMPLE_PROMPT
        assert spec.prefix_turns == []
        assert "No regions" in record.diff_summary or "no regions" in record.reason.lower()

    def test_all_keep_regions_no_change(self):
        config = {
            "system_prompt": {
                "regions": [
                    {"name": "policy", "start_marker": "## Policy", "end_marker": "## Approved Responses"},
                ]
            }
        }
        spec, record = split_to_turns(SAMPLE_PROMPT, config)
        # Only "policy" region, which is classified as "keep"
        assert spec.prefix_turns == []
        assert "nothing to split" in record.diff_summary.lower() or "No change" in record.diff_summary

    def test_system_prompt_shrinks(self):
        spec, _ = split_to_turns(SAMPLE_PROMPT, SAMPLE_CONFIG)
        assert len(spec.system_prompt) < len(SAMPLE_PROMPT)

    def test_idempotent_via_has_been_split(self):
        """PromptSpec.has_been_split prevents the mutation system from
        re-splitting, but split_to_turns itself should still work on
        raw text (it's the mutation dispatcher that checks the flag)."""
        spec1, _ = split_to_turns(SAMPLE_PROMPT, SAMPLE_CONFIG)
        assert spec1.has_been_split is True
        # The flag is checked by StructuralMutator, not by split_to_turns


class TestSplitWithContextRegion:

    def test_context_becomes_turn(self):
        prompt = (
            "You are a helpful assistant.\n\n"
            "## Context\n"
            "The user is working on Project Alpha.\n\n"
            "## Rules\n"
            "Be concise.\n"
        )
        config = {
            "system_prompt": {
                "regions": [
                    {"name": "context", "start_marker": "## Context", "end_marker": "## Rules"},
                    {"name": "rules", "start_marker": "## Rules", "end_marker": None},
                ]
            }
        }
        spec, record = split_to_turns(prompt, config)
        assert spec.has_been_split is True
        assert "## Context" not in spec.system_prompt
        assert "## Rules" in spec.system_prompt
        # Context should become a user/assistant pair
        assert len(spec.prefix_turns) == 2
        assert spec.prefix_turns[0]["role"] == "user"
        assert "Project Alpha" in spec.prefix_turns[0]["content"]
        assert spec.prefix_turns[1]["role"] == "assistant"
