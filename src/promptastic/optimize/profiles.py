"""Pre-built optimization profiles.

Each profile defines a set of target overrides and strategy preferences
for common optimization goals.
"""

from __future__ import annotations

from ._types import MetricTarget, OptimizationConfig
from .targets import _target


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

def _general_overrides() -> list[MetricTarget]:
    """Balanced profile with moderate targets across all categories."""
    return [
        _target("context_bleed_ratio", "below", ideal=1.5, maximum=3.0, weight=1.2, category="attention"),
        _target("density_cv", "below", ideal=0.8, maximum=2.0, weight=1.0, category="attention"),
    ]


def _anti_bleed_overrides() -> list[MetricTarget]:
    """Aggressive context bleed reduction."""
    return [
        _target("context_bleed_ratio", "below", ideal=0.8, maximum=1.5, weight=2.0, category="attention"),
        _target("density_cv", "below", ideal=0.5, maximum=1.5, weight=1.0, category="attention"),
    ]


def _maximize_rules_overrides() -> list[MetricTarget]:
    """Maximize rules/directive adherence."""
    return [
        _target("context_bleed_ratio", "below", ideal=1.0, maximum=2.5, weight=1.0, category="attention"),
        _target("density_cv", "below", ideal=0.6, maximum=2.0, weight=0.8, category="attention"),
    ]


def _balanced_attention_overrides() -> list[MetricTarget]:
    """Even attention distribution across all regions."""
    return [
        _target("context_bleed_ratio", "below", ideal=1.0, maximum=2.0, weight=1.0, category="attention"),
        _target("density_cv", "below", ideal=0.3, maximum=1.0, weight=2.0, category="attention"),
    ]


PROFILES: dict[str, tuple[list[MetricTarget], dict[str, float]]] = {
    "general": (_general_overrides(), {}),
    "anti_bleed": (_anti_bleed_overrides(), {"context_bleed_ratio": 2.0}),
    "maximize_rules_adherence": (
        _maximize_rules_overrides(),
        # Boost weight for rules-related region metrics
    {},
    ),
    "balanced_attention": (_balanced_attention_overrides(), {"density_cv": 2.0}),
}


# Region-specific target overrides per profile
PROFILE_REGION_OVERRIDES: dict[str, dict[str, dict[str, float]]] = {
    "maximize_rules_adherence": {
        "rules": {"terminal_attention_min": 0.08, "retention_ratio_min": 0.4},
        "directive": {"terminal_attention_min": 0.08, "retention_ratio_min": 0.4},
    },
    "anti_bleed": {
        "conversation_turns": {"terminal_attention_max": 0.15},
    },
}


def get_profile(name: str) -> tuple[list[MetricTarget], dict[str, float]]:
    """Look up a profile by name.

    Returns (target_overrides, weight_overrides).
    Raises KeyError if the profile doesn't exist.
    """
    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise KeyError(f"Unknown profile {name!r}. Available: {available}")
    return PROFILES[name]


def list_profiles() -> list[str]:
    """Return available profile names."""
    return sorted(PROFILES)


def apply_profile(config: OptimizationConfig) -> tuple[list[MetricTarget], dict[str, float]]:
    """Resolve the profile from config and return target/weight overrides."""
    return get_profile(config.profile)
