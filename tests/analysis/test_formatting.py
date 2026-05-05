"""Tests for promptastic.analysis.formatting -- output helpers."""

from math import nan

from promptastic.analysis.formatting import delta_str, fmt, pct


# ---------------------------------------------------------------
# fmt
# ---------------------------------------------------------------


def test_fmt_normal_small():
    result = fmt(0.1234)
    assert "0.1234" in result


def test_fmt_normal_medium():
    result = fmt(12.345)
    assert "12.345" in result


def test_fmt_normal_large():
    result = fmt(123.456)
    assert "123.5" in result


def test_fmt_zero():
    result = fmt(0.0)
    assert "0.0000" in result


def test_fmt_nan():
    result = fmt(nan)
    assert "N/A" in result


def test_fmt_very_small():
    result = fmt(1e-6)
    assert "e" in result.lower()  # scientific notation


def test_fmt_small_but_above_threshold():
    """Values between 1e-4 and 1e-3 should get 5 decimal places."""
    result = fmt(0.00055)
    assert "0.00055" in result


def test_fmt_custom_width():
    result = fmt(1.234, width=15)
    assert len(result) == 15


def test_fmt_negative():
    result = fmt(-5.678)
    assert "-" in result


# ---------------------------------------------------------------
# pct
# ---------------------------------------------------------------


def test_pct_normal():
    result = pct(0.5)
    assert "50.000%" in result


def test_pct_small():
    result = pct(0.001)
    assert "0.100%" in result


def test_pct_nan():
    result = pct(nan)
    assert "N/A" in result


def test_pct_zero():
    result = pct(0.0)
    assert "0.000%" in result


def test_pct_custom_width():
    result = pct(0.5, width=14)
    assert len(result) == 14


# ---------------------------------------------------------------
# delta_str
# ---------------------------------------------------------------


def test_delta_str_positive():
    result = delta_str(1.5, 1.0)
    assert "+" in result
    assert "50.0%" in result


def test_delta_str_negative():
    result = delta_str(0.5, 1.0)
    assert "-50.0%" in result


def test_delta_str_zero_reference():
    result = delta_str(1.0, 0.0)
    assert "N/A" in result


def test_delta_str_nan_val():
    result = delta_str(nan, 1.0)
    assert "N/A" in result


def test_delta_str_nan_ref():
    result = delta_str(1.0, nan)
    assert "N/A" in result


def test_delta_str_no_change():
    result = delta_str(1.0, 1.0)
    assert "+0.0%" in result


def test_delta_str_negative_reference():
    """Negative ref should use abs(ref) for percentage calculation."""
    result = delta_str(-0.5, -1.0)
    assert "+" in result  # -0.5 is "more" than -1.0 (val-ref = 0.5, positive)
