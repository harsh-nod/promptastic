"""Tests for PromptSpec dataclass."""

from promptastic.optimize._types import PromptSpec


def test_from_string_basic():
    spec = PromptSpec.from_string("Hello world")
    assert spec.system_prompt == "Hello world"
    assert spec.prefix_turns == []
    assert spec.has_been_split is False
    assert spec.structured_prompt is None


def test_to_messages_no_prefix():
    spec = PromptSpec.from_string("You are a helpful assistant.")
    msgs = spec.to_messages("What is 2+2?")
    assert len(msgs) == 2
    assert msgs[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert msgs[1] == {"role": "user", "content": "What is 2+2?"}


def test_to_messages_with_response():
    spec = PromptSpec.from_string("System prompt")
    msgs = spec.to_messages("Hello", "Hi there!")
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2] == {"role": "assistant", "content": "Hi there!"}


def test_to_messages_with_prefix_turns():
    spec = PromptSpec(
        system_prompt="System prompt",
        prefix_turns=[
            {"role": "user", "content": "Example question"},
            {"role": "assistant", "content": "Example answer"},
        ],
    )
    msgs = spec.to_messages("Real question", "Real answer")
    assert len(msgs) == 5
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "Example question"}
    assert msgs[2] == {"role": "assistant", "content": "Example answer"}
    assert msgs[3] == {"role": "user", "content": "Real question"}
    assert msgs[4] == {"role": "assistant", "content": "Real answer"}


def test_to_messages_empty_response_excluded():
    spec = PromptSpec.from_string("System")
    msgs = spec.to_messages("Question", "")
    assert len(msgs) == 2
    assert msgs[-1]["role"] == "user"


def test_prefix_turns_default_empty():
    spec = PromptSpec(system_prompt="test")
    assert spec.prefix_turns == []
    assert spec.has_been_split is False
    assert spec.structured_prompt is None


def test_has_been_split_flag():
    spec = PromptSpec(
        system_prompt="test",
        prefix_turns=[{"role": "user", "content": "x"}],
        has_been_split=True,
    )
    assert spec.has_been_split is True


def test_with_structured_round_trip():
    from promptastic.optimize.structure import PromptSection, StructuredPrompt

    section = PromptSection(
        name="rules",
        start_marker="## Rules",
        prefix="",
        header="## Rules",
        body="\n- Follow instructions.\n",
    )
    structured = StructuredPrompt(
        original_text="## Rules\n- Follow instructions.\n",
        leading_text="",
        sections=[section],
        trailing_text="",
        missing_regions=[],
    )

    spec = PromptSpec.from_string("placeholder").with_structured(structured)
    assert spec.system_prompt == "## Rules\n- Follow instructions.\n"
    assert spec.structured_prompt is structured
