"""Tests for structured prompt parsing utilities."""

from promptastic.optimize.structure import (
    PromptSection,
    StructuredPrompt,
    parse_prompt,
)


def _region_config():
    return {
        "system_prompt": {
            "regions": [
                {"name": "intro", "start_marker": "## Intro"},
                {"name": "rules", "start_marker": "## Rules"},
                {"name": "examples", "start_marker": "## Examples"},
            ],
        }
    }


def test_parse_prompt_basic_round_trip():
    prompt = (
        "Preamble text\n\n"
        "## Intro\n"
        "You are an expert assistant.\n\n"
        "## Rules\n"
        "1. Follow instructions.\n"
        "2. Stay on topic.\n\n"
        "## Examples\n"
        "- Q: Hello\n"
        "- A: Hi there!\n"
        "\nTrailing footer"
    )

    structured = parse_prompt(prompt, _region_config())
    assert isinstance(structured, StructuredPrompt)
    assert structured.leading_text == "Preamble text\n\n"
    assert structured.trailing_text == ""
    assert structured.missing_regions == []
    assert [s.name for s in structured.sections] == ["intro", "rules", "examples"]

    intro = structured.get_section("intro")
    assert intro is not None
    assert intro.header == "## Intro"
    assert intro.body.startswith("\nYou are an expert assistant.")
    examples = structured.get_section("examples")
    assert examples is not None
    assert examples.body.endswith("\n\nTrailing footer")

    # Round-trip should produce the original prompt verbatim.
    assert structured.render() == prompt


def test_parse_prompt_missing_region():
    prompt = "## Intro\nHello"
    structured = parse_prompt(prompt, _region_config())

    assert structured.missing_regions == ["rules", "examples"]
    assert structured.sections[0].name == "intro"
    assert structured.render() == prompt


def test_prompt_section_render_includes_prefix():
    section = PromptSection(
        name="rules",
        start_marker="## Rules",
        prefix="\n\n",
        header="## Rules",
        body="\n- A\n",
    )
    assert section.render() == "\n\n## Rules\n- A\n"
