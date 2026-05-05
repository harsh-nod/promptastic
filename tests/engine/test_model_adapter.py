"""Tests for promptastic.engine.model_adapter -- architecture auto-discovery.

All model objects are mocked with types.SimpleNamespace so no GPU or
real model is required.
"""

import types

import pytest

from promptastic.engine.model_adapter import ModelAdapter


# ---------------------------------------------------------------
# Helpers -- build mock models
# ---------------------------------------------------------------


def _make_attn_module():
    return types.SimpleNamespace(name="self_attn")


def _make_mlp_module():
    return types.SimpleNamespace(name="mlp")


def _make_layer(has_mlp=True):
    layer = types.SimpleNamespace(
        self_attn=_make_attn_module(),
    )
    if has_mlp:
        layer.mlp = _make_mlp_module()
    return layer


def _make_llama_model(num_layers=4):
    """Llama-style: model.model.layers[].self_attn, model.model.norm, model.lm_head."""
    layers = [_make_layer() for _ in range(num_layers)]
    inner = types.SimpleNamespace(
        layers=layers,
        norm=types.SimpleNamespace(name="final_norm"),
    )
    config = types.SimpleNamespace(
        num_hidden_layers=num_layers,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        vocab_size=32000,
        _name_or_path="meta-llama/test",
    )
    model = types.SimpleNamespace(
        model=inner,
        lm_head=types.SimpleNamespace(name="lm_head"),
        config=config,
    )
    return model


def _make_gpt2_model(num_layers=4):
    """GPT-2 style: model.transformer.h[].attn, model.transformer.ln_f, model.lm_head."""
    layers = []
    for _ in range(num_layers):
        layer = types.SimpleNamespace(
            attn=_make_attn_module(),
            mlp=_make_mlp_module(),
        )
        layers.append(layer)
    transformer = types.SimpleNamespace(
        h=layers,
        ln_f=types.SimpleNamespace(name="ln_f"),
    )
    config = types.SimpleNamespace(
        num_hidden_layers=num_layers,
        num_attention_heads=12,
        hidden_size=768,
        vocab_size=50257,
        _name_or_path="gpt2",
    )
    model = types.SimpleNamespace(
        transformer=transformer,
        lm_head=types.SimpleNamespace(name="lm_head"),
        config=config,
    )
    return model


# ---------------------------------------------------------------
# from_model factory
# ---------------------------------------------------------------


def test_from_model_llama():
    model = _make_llama_model(num_layers=4)
    adapter = ModelAdapter.from_model(model)
    assert adapter.num_layers == 4
    assert adapter.num_query_heads == 32
    assert adapter.num_kv_heads == 8
    assert adapter.hidden_size == 4096
    assert adapter.vocab_size == 32000
    assert adapter.model_name == "meta-llama/test"


def test_from_model_gpt2():
    model = _make_gpt2_model(num_layers=4)
    adapter = ModelAdapter.from_model(model)
    assert adapter.num_layers == 4
    assert adapter.num_query_heads == 12
    # GPT-2 has no num_key_value_heads, should fall back to num_attention_heads
    assert adapter.num_kv_heads == 12
    assert adapter.hidden_size == 768
    assert adapter.vocab_size == 50257


# ---------------------------------------------------------------
# get_* methods
# ---------------------------------------------------------------


def test_get_attention_modules_llama():
    model = _make_llama_model(num_layers=4)
    adapter = ModelAdapter.from_model(model)
    attn_mods = adapter.get_attention_modules()
    assert len(attn_mods) == 4
    for idx, mod in attn_mods:
        assert hasattr(mod, "name")
        assert mod.name == "self_attn"


def test_get_attention_modules_gpt2():
    model = _make_gpt2_model(num_layers=4)
    adapter = ModelAdapter.from_model(model)
    attn_mods = adapter.get_attention_modules()
    assert len(attn_mods) == 4


def test_get_layer_modules():
    model = _make_llama_model(num_layers=4)
    adapter = ModelAdapter.from_model(model)
    layer_mods = adapter.get_layer_modules()
    assert len(layer_mods) == 4
    indices = [idx for idx, _ in layer_mods]
    assert indices == [0, 1, 2, 3]


def test_get_mlp_modules():
    model = _make_llama_model(num_layers=4)
    adapter = ModelAdapter.from_model(model)
    mlp_mods = adapter.get_mlp_modules()
    assert len(mlp_mods) == 4
    for idx, mod in mlp_mods:
        assert mod.name == "mlp"


def test_get_lm_head_llama():
    model = _make_llama_model()
    adapter = ModelAdapter.from_model(model)
    head = adapter.get_lm_head()
    assert head.name == "lm_head"


def test_get_norm_llama():
    model = _make_llama_model()
    adapter = ModelAdapter.from_model(model)
    norm = adapter.get_norm()
    assert norm.name == "final_norm"


def test_get_norm_gpt2():
    model = _make_gpt2_model()
    adapter = ModelAdapter.from_model(model)
    norm = adapter.get_norm()
    assert norm.name == "ln_f"


# ---------------------------------------------------------------
# Module accessors return copies
# ---------------------------------------------------------------


def test_get_attention_modules_returns_copy():
    model = _make_llama_model()
    adapter = ModelAdapter.from_model(model)
    mods1 = adapter.get_attention_modules()
    mods2 = adapter.get_attention_modules()
    assert mods1 is not mods2
    assert mods1 == mods2


# ---------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------


def test_missing_layers_container_raises():
    config = types.SimpleNamespace(
        num_hidden_layers=4,
        num_attention_heads=8,
        hidden_size=512,
        vocab_size=10000,
    )
    # Model with no recognized layer container path
    model = types.SimpleNamespace(
        config=config,
        lm_head=types.SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="Could not find layer container"):
        ModelAdapter.from_model(model)


def test_missing_lm_head_raises():
    layers = [_make_layer() for _ in range(2)]
    inner = types.SimpleNamespace(
        layers=layers,
        norm=types.SimpleNamespace(),
    )
    config = types.SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=8,
        hidden_size=512,
        vocab_size=10000,
    )
    model = types.SimpleNamespace(
        model=inner,
        config=config,
        # no lm_head, output, or embed_out
    )
    with pytest.raises(RuntimeError, match="language model head"):
        ModelAdapter.from_model(model)


def test_missing_final_norm_raises():
    layers = [_make_layer() for _ in range(2)]
    inner = types.SimpleNamespace(
        layers=layers,
        # no norm attribute
    )
    config = types.SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=8,
        hidden_size=512,
        vocab_size=10000,
    )
    model = types.SimpleNamespace(
        model=inner,
        config=config,
        lm_head=types.SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="final normalization"):
        ModelAdapter.from_model(model)


def test_layer_count_mismatch_raises():
    """If num_hidden_layers doesn't match the actual container length."""
    layers = [_make_layer() for _ in range(3)]
    inner = types.SimpleNamespace(
        layers=layers,
        norm=types.SimpleNamespace(),
    )
    config = types.SimpleNamespace(
        num_hidden_layers=5,  # mismatch: 5 vs 3
        num_attention_heads=8,
        hidden_size=512,
        vocab_size=10000,
    )
    model = types.SimpleNamespace(
        model=inner,
        config=config,
        lm_head=types.SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="Could not find layer container"):
        ModelAdapter.from_model(model)


def test_missing_attention_submodule_raises():
    """Layers without any recognized attention attribute."""
    # Create layers with no self_attn, attn, or attention
    layers = [types.SimpleNamespace(mlp=_make_mlp_module()) for _ in range(2)]
    inner = types.SimpleNamespace(
        layers=layers,
        norm=types.SimpleNamespace(),
    )
    config = types.SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=8,
        hidden_size=512,
        vocab_size=10000,
    )
    model = types.SimpleNamespace(
        model=inner,
        config=config,
        lm_head=types.SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="attention modules"):
        ModelAdapter.from_model(model)
