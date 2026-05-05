"""Tests for promptastic.engine.hooks -- activation cache classes.

Mocks torch tensors with types.SimpleNamespace where the hook internals
are not exercised. For ResidualCache/MLPCache we construct minimal mock
tensors that satisfy .dim(), .unsqueeze(), .float(), .cpu() chains.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

# We need to mock torch before importing hooks, since hooks.py
# imports torch at module level.
_mock_torch = MagicMock()
sys.modules.setdefault("torch", _mock_torch)

from promptastic.engine.hooks import (
    AttentionCache,
    BaseCache,
    MLPCache,
    ResidualCache,
)


# ---------------------------------------------------------------
# BaseCache key generation
# ---------------------------------------------------------------


def test_base_cache_key_format():
    key = BaseCache._key("resid", 5, "terminal")
    assert key == "resid_L5_terminal"


# ---------------------------------------------------------------
# ResidualCache
# ---------------------------------------------------------------


def test_residual_cache_store_and_retrieve():
    cache = ResidualCache()
    # Manually insert data using the internal key format
    key = BaseCache._key("resid", 3, "terminal")
    sentinel = object()
    cache.data[key] = sentinel
    assert cache.get(3, "terminal") is sentinel


def test_residual_cache_missing_returns_none():
    cache = ResidualCache()
    assert cache.get(0, "terminal") is None


def test_residual_cache_clear():
    cache = ResidualCache()
    cache.data["resid_L0_terminal"] = "value"
    cache.data["resid_L1_terminal"] = "value"
    assert len(cache.data) == 2
    cache.clear()
    assert len(cache.data) == 0


# ---------------------------------------------------------------
# AttentionCache
# ---------------------------------------------------------------


def test_attention_cache_per_head_false_stores_region_weights():
    cache = AttentionCache(per_head=False)
    # Simulate what the hook would store
    key = BaseCache._key("attn", 5, "terminal")
    cache.data[key] = {"rules": 0.3, "examples": 0.2}
    result = cache.get(5, "terminal")
    assert result == {"rules": 0.3, "examples": 0.2}


def test_attention_cache_per_head_true_stores_head_data():
    cache = AttentionCache(per_head=True)
    # Simulate per-head data storage
    ph_key = BaseCache._key("attn_ph", 5, "terminal")
    cache.data[ph_key] = {"rules": [0.1, 0.2, 0.3], "examples": [0.05, 0.1, 0.15]}
    result = cache.get_per_head(5, "terminal")
    assert result is not None
    assert len(result["rules"]) == 3


def test_attention_cache_get_per_head_missing():
    cache = AttentionCache(per_head=False)
    assert cache.get_per_head(0, "terminal") is None


def test_attention_cache_per_token():
    cache = AttentionCache(capture_per_token=True)
    pt_key = BaseCache._key("attn_pt", 2, "terminal")
    cache.data[pt_key] = [0.01, 0.02, 0.03, 0.04]
    result = cache.get_per_token(2, "terminal")
    assert result == [0.01, 0.02, 0.03, 0.04]


def test_attention_cache_per_token_missing():
    cache = AttentionCache(capture_per_token=True)
    assert cache.get_per_token(99, "terminal") is None


def test_attention_cache_clear():
    cache = AttentionCache(per_head=True, capture_per_token=True)
    cache.data["attn_L0_terminal"] = {"r": 0.1}
    cache.data["attn_ph_L0_terminal"] = {"r": [0.1]}
    cache.data["attn_pt_L0_terminal"] = [0.1]
    assert len(cache.data) == 3
    cache.clear()
    assert len(cache.data) == 0


# ---------------------------------------------------------------
# MLPCache
# ---------------------------------------------------------------


def test_mlp_cache_make_hook_raises():
    """MLPCache.make_hook should raise -- use make_input_hook/make_output_hook."""
    cache = MLPCache()
    with pytest.raises(NotImplementedError):
        cache.make_hook()


def test_mlp_cache_store_input_and_output():
    cache = MLPCache()
    in_key = BaseCache._key("mlp_in", 3, "terminal")
    out_key = BaseCache._key("mlp_out", 3, "terminal")
    cache.data[in_key] = "input_tensor"
    cache.data[out_key] = "output_tensor"
    assert cache.get_input(3, "terminal") == "input_tensor"
    assert cache.get_output(3, "terminal") == "output_tensor"


def test_mlp_cache_missing_returns_none():
    cache = MLPCache()
    assert cache.get_input(0, "terminal") is None
    assert cache.get_output(0, "terminal") is None


def test_mlp_cache_clear():
    cache = MLPCache()
    cache.data["mlp_in_L0_terminal"] = "a"
    cache.data["mlp_out_L0_terminal"] = "b"
    cache.clear()
    assert len(cache.data) == 0
