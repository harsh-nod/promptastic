"""Tests for promptastic.render._shared -- visualization primitives."""

import numpy as np
import pytest

from promptastic.render._shared import (
    gaussian_smooth,
    get_colormap,
    is_newline_token,
    normalize_weights,
    parse_layer_spec,
    sanitize_token,
    text_color_for_bg,
)


# ---------------------------------------------------------------
# gaussian_smooth
# ---------------------------------------------------------------


def test_gaussian_smooth_preserves_sum():
    values = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    smoothed = gaussian_smooth(values, sigma=1.0)
    # Sum should be approximately preserved (convolution in "same" mode)
    assert abs(smoothed.sum() - values.sum()) < 0.01


def test_gaussian_smooth_zero_sigma():
    values = np.array([1.0, 2.0, 3.0])
    smoothed = gaussian_smooth(values, sigma=0.0)
    np.testing.assert_array_equal(smoothed, values)


def test_gaussian_smooth_negative_sigma():
    values = np.array([1.0, 2.0, 3.0])
    smoothed = gaussian_smooth(values, sigma=-1.0)
    np.testing.assert_array_equal(smoothed, values)


def test_gaussian_smooth_output_length():
    values = np.random.rand(50)
    smoothed = gaussian_smooth(values, sigma=2.0)
    assert len(smoothed) == len(values)


def test_gaussian_smooth_uniform_stays_uniform():
    """Interior values of a uniform signal stay unchanged after smoothing.
    Edge values drop due to boundary effects of convolution."""
    values = np.ones(40) * 5.0
    smoothed = gaussian_smooth(values, sigma=2.0)
    # Interior (far from edges) stays close to 5.0
    np.testing.assert_allclose(smoothed[10:30], 5.0, atol=0.1)


# ---------------------------------------------------------------
# normalize_weights
# ---------------------------------------------------------------


def test_normalize_weights_output_range():
    weights = np.array([0.01, 0.05, 0.1, 0.5, 1.0])
    normed = normalize_weights(weights)
    assert np.all(normed >= 0.0)
    assert np.all(normed <= 1.0)


def test_normalize_weights_highest_gets_one():
    weights = np.array([0.1, 0.5, 0.9, 0.2, 0.3])
    normed = normalize_weights(weights, clip_low=0.0)
    # The highest value should map to 1.0
    assert normed[2] == 1.0


def test_normalize_weights_all_zero():
    """When all weights are zero they are tied, so they all get the same rank.
    With clip_low=5 the average rank gets clipped but the output is uniform."""
    weights = np.zeros(10)
    normed = normalize_weights(weights)
    # All values are the same (tied ranks -> same normalized value)
    assert normed.min() == normed.max()


def test_normalize_weights_single_element():
    weights = np.array([0.5])
    normed = normalize_weights(weights, clip_low=0.0)
    # Single element gets rank 0 out of 0 -> normed = 0/max(1,0) = 0
    assert normed[0] == 0.0


def test_normalize_weights_with_mask():
    weights = np.array([0.1, 0.5, 0.9, 0.2, 0.3])
    mask = np.array([True, True, True, False, False])
    normed = normalize_weights(weights, clip_low=0.0, mask=mask)
    # Masked-out positions should be 0
    assert normed[3] == 0.0
    assert normed[4] == 0.0
    # Unmasked should have values
    assert normed[2] > 0.0


def test_normalize_weights_empty_mask():
    weights = np.array([0.1, 0.2, 0.3])
    mask = np.array([False, False, False])
    normed = normalize_weights(weights, mask=mask)
    np.testing.assert_array_equal(normed, np.zeros(3))


# ---------------------------------------------------------------
# parse_layer_spec
# ---------------------------------------------------------------


def test_parse_layer_spec_final():
    layers = parse_layer_spec("final", num_layers=64)
    # FINAL_LAYERS = 4 -> [60, 61, 62, 63]
    assert layers == [60, 61, 62, 63]


def test_parse_layer_spec_final_small_model():
    layers = parse_layer_spec("final", num_layers=8)
    assert layers == [4, 5, 6, 7]


def test_parse_layer_spec_all():
    layers = parse_layer_spec("all", num_layers=10)
    assert layers == list(range(10))


def test_parse_layer_spec_single():
    layers = parse_layer_spec("48", num_layers=64)
    assert layers == [48]


def test_parse_layer_spec_comma_separated():
    layers = parse_layer_spec("0,16,32", num_layers=64)
    assert layers == [0, 16, 32]


def test_parse_layer_spec_range():
    layers = parse_layer_spec("60-63", num_layers=64)
    assert layers == [60, 61, 62, 63]


def test_parse_layer_spec_mixed():
    layers = parse_layer_spec("0,16,60-63", num_layers=64)
    assert layers == [0, 16, 60, 61, 62, 63]


def test_parse_layer_spec_case_insensitive():
    layers = parse_layer_spec("FINAL", num_layers=64)
    assert layers == [60, 61, 62, 63]


def test_parse_layer_spec_whitespace():
    layers = parse_layer_spec("  all  ", num_layers=8)
    assert layers == list(range(8))


# ---------------------------------------------------------------
# text_color_for_bg
# ---------------------------------------------------------------


def test_text_color_dark_bg_returns_white():
    # Black background -> white text
    result = text_color_for_bg(0, 0, 0)
    assert result == "#ffffff"


def test_text_color_light_bg_returns_black():
    # White background -> black text
    result = text_color_for_bg(255, 255, 255)
    assert result == "#000000"


def test_text_color_medium_dark():
    # Dark blue -> white text
    result = text_color_for_bg(0, 0, 128)
    assert result == "#ffffff"


def test_text_color_medium_light():
    # Light yellow -> black text
    result = text_color_for_bg(255, 255, 200)
    assert result == "#000000"


# ---------------------------------------------------------------
# sanitize_token
# ---------------------------------------------------------------


def test_sanitize_token_normal():
    assert sanitize_token("hello") == "hello"


def test_sanitize_token_tab():
    assert sanitize_token("\t") == "  "  # tab -> two spaces


def test_sanitize_token_carriage_return():
    assert sanitize_token("\r") == ""  # stripped


def test_sanitize_token_control_char():
    result = sanitize_token("\x01")
    assert result == "\\x01"


def test_sanitize_token_newline_preserved():
    assert sanitize_token("\n") == "\n"


def test_sanitize_token_mixed():
    result = sanitize_token("a\tb\x02c")
    assert result == "a  b\\x02c"


# ---------------------------------------------------------------
# is_newline_token
# ---------------------------------------------------------------


def test_is_newline_token_true():
    assert is_newline_token("\n") is True


def test_is_newline_token_crlf():
    assert is_newline_token("\r\n") is True


def test_is_newline_token_false():
    assert is_newline_token("hello") is False


def test_is_newline_token_space():
    assert is_newline_token(" ") is False


def test_is_newline_token_empty():
    assert is_newline_token("") is False


# ---------------------------------------------------------------
# get_colormap
# ---------------------------------------------------------------


def test_get_colormap_inferno():
    lut = get_colormap("inferno")
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8


def test_get_colormap_viridis():
    lut = get_colormap("viridis")
    assert lut.shape == (256, 3)


def test_get_colormap_hot():
    lut = get_colormap("hot")
    assert lut.shape == (256, 3)


def test_get_colormap_coolwarm():
    lut = get_colormap("coolwarm")
    assert lut.shape == (256, 3)


def test_get_colormap_unknown_raises():
    with pytest.raises(ValueError, match="Unknown colormap"):
        get_colormap("nonexistent_cmap")
