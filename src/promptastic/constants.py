"""Shared constants for analysis and rendering phases.

All phase boundaries scale dynamically to any layer count.
Never hardcode layer numbers in renderers or analysis code.
"""

from __future__ import annotations

FINAL_LAYERS = 4

# Display phases (4-phase model for visualizations)
_DISPLAY_PHASE_FRACTIONS = [
    ("Rules absorbed", 0.0, 0.125),
    ("Internal computation", 0.125, 0.5),
    ("Focus narrows", 0.5, 0.75),
    ("Output formatting", 0.75, 1.0),
]


def display_phases(num_layers: int) -> list[tuple[str, int, int]]:
    """Return display phase boundaries scaled to the given layer count."""
    return [
        (name, round(start * (num_layers - 1)), round(end * (num_layers - 1)))
        for name, start, end in _DISPLAY_PHASE_FRACTIONS
    ]


# Analysis phases (5-phase detailed model)
_ANALYSIS_PHASE_FRACTIONS = {
    "P1_broad_read": (0.0, 0.094),
    "P2_absorption": (0.109, 0.172),
    "P3_compression": (0.188, 0.484),
    "P4_reengagement": (0.5, 0.734),
    "P5_output_prep": (0.75, 1.0),
}


def analysis_phases(num_layers: int) -> dict[str, tuple[int, int]]:
    """Return analysis phase boundaries scaled to the given layer count."""
    return {
        name: (round(start * (num_layers - 1)), round(end * (num_layers - 1)))
        for name, (start, end) in _ANALYSIS_PHASE_FRACTIONS.items()
    }


# Container regions that should never be plotted as individual curves
SKIP_REGIONS = {
    "chat_template",
    "system_prompt",
    "user_message",
    "response",
    "thinking_section",
    "entities_section",
}

# Canonical ordering for cooking curve display
DEFAULT_DISPLAY_REGIONS = [
    "directive",
    "entity_rules",
    "passage_rules",
    "examples",
    "conversation_turns",
    "current_message",
    "output_format",
]

# Capture mode defaults
DEFAULT_CAPTURES = {"attention", "residual", "logit_lens"}
DEFAULT_PATCHING_METHOD = "zero"
DEFAULT_GRADIENT_METHOD = "vanilla"
DEFAULT_MAX_NEW_TOKENS = 32
MAX_FEATURES_PER_REGION = 10
PATCHING_LAYER_STRIDE = 4
