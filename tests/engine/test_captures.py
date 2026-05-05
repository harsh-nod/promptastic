"""Tests for promptastic.engine.captures -- capture mode registry."""

import pytest

from promptastic.engine.captures import (
    CaptureMode,
    get_active_modes,
    get_mode,
    validate_compatibility,
)


# ---------------------------------------------------------------
# Registry: all 10 built-in modes are present
# ---------------------------------------------------------------

EXPECTED_MODES = [
    "attention",
    "per_head",
    "residual",
    "logit_lens",
    "mlp",
    "tuned_lens",
    "sae",
    "patching",
    "gradients",
    "generation",
]


def test_all_builtin_modes_registered():
    for name in EXPECTED_MODES:
        mode = get_mode(name)
        assert mode.name == name


def test_builtin_mode_count():
    """Exactly 10 modes should be registered."""
    # Retrieve all modes by name to verify
    modes = [get_mode(n) for n in EXPECTED_MODES]
    assert len(modes) == 10


# ---------------------------------------------------------------
# get_mode
# ---------------------------------------------------------------


def test_get_mode_returns_correct_object():
    mode = get_mode("attention")
    assert isinstance(mode, CaptureMode)
    assert mode.name == "attention"
    assert "attn" in mode.hook_targets


def test_get_mode_unknown_raises():
    with pytest.raises(KeyError, match="Unknown capture mode"):
        get_mode("nonexistent_mode")


# ---------------------------------------------------------------
# Mode properties
# ---------------------------------------------------------------


def test_attention_mode_properties():
    mode = get_mode("attention")
    assert not mode.requires_grad
    assert not mode.requires_multiple_passes
    assert mode.hook_targets == ["attn"]


def test_gradients_mode_requires_grad():
    mode = get_mode("gradients")
    assert mode.requires_grad is True


def test_patching_mode_requires_multiple_passes():
    mode = get_mode("patching")
    assert mode.requires_multiple_passes is True


def test_generation_mode_properties():
    mode = get_mode("generation")
    assert mode.requires_multiple_passes is True
    assert "attn" in mode.hook_targets
    assert "layer" in mode.hook_targets


def test_logit_lens_no_hooks():
    mode = get_mode("logit_lens")
    assert mode.hook_targets == []


# ---------------------------------------------------------------
# get_active_modes
# ---------------------------------------------------------------


def test_get_active_modes_basic():
    config = {"attention": True, "residual": True, "logit_lens": False}
    active = get_active_modes(config)
    names = {m.name for m in active}
    assert "attention" in names
    assert "residual" in names
    assert "logit_lens" not in names


def test_get_active_modes_empty_config():
    active = get_active_modes({})
    assert active == []


def test_get_active_modes_all_true():
    config = {name: True for name in EXPECTED_MODES}
    active = get_active_modes(config)
    assert len(active) == 10


def test_get_active_modes_ignores_non_boolean_keys():
    """Non-boolean config keys like 'patching_method' should not activate modes."""
    config = {"patching_method": "zero", "sae_weights_path": "/some/path"}
    active = get_active_modes(config)
    assert active == []


# ---------------------------------------------------------------
# validate_compatibility
# ---------------------------------------------------------------


def test_validate_no_warnings_for_compatible():
    """attention + residual + logit_lens should produce no warnings."""
    modes = [get_mode("attention"), get_mode("residual"), get_mode("logit_lens")]
    warnings = validate_compatibility(modes)
    assert warnings == []


def test_validate_warns_grad_plus_no_grad():
    """Mixing gradients (requires_grad) with attention (no grad) should warn."""
    modes = [get_mode("gradients"), get_mode("attention")]
    warnings = validate_compatibility(modes)
    assert len(warnings) >= 1
    assert any("gradient" in w.lower() for w in warnings)


def test_validate_warns_multiple_multi_pass():
    """patching + generation both require multiple passes."""
    modes = [get_mode("patching"), get_mode("generation")]
    warnings = validate_compatibility(modes)
    assert any("multi-pass" in w.lower() for w in warnings)


def test_validate_warns_hook_target_conflict():
    """Single-pass mode sharing hook targets with multi-pass mode should warn."""
    # residual uses "layer", patching uses "layer" + requires_multiple_passes
    modes = [get_mode("residual"), get_mode("patching")]
    warnings = validate_compatibility(modes)
    assert any("hook targets" in w.lower() for w in warnings)


def test_validate_empty_modes():
    warnings = validate_compatibility([])
    assert warnings == []
