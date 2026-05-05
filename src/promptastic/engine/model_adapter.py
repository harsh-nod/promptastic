"""Auto-discovers model architecture from any HuggingFace transformer.

Probes model.config for layer count, head counts (query + KV), hidden size,
and vocab size. Walks the module tree to find layer containers, attention
submodules, MLP submodules, the LM head, and the final normalization layer.

Supports: Llama, Qwen, Mistral, Gemma, GPT-2, GPT-Neo, GPT-NeoX, Phi,
and any model following standard HuggingFace decoder-only conventions.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ModelAdapter:
    """Auto-discovers architecture details from a loaded HuggingFace model.

    Use the ``from_model`` factory to construct an adapter by inspecting a
    live model instance.  All downstream code should go through this adapter
    rather than probing the model directly, so architecture-specific paths
    stay in one place.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_query_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        vocab_size: int,
        attention_modules: list[tuple[int, Any]],
        layer_modules: list[tuple[int, Any]],
        mlp_modules: list[tuple[int, Any]],
        lm_head: Any,
        norm: Any,
        model_name: str = "unknown",
    ) -> None:
        self._num_layers = num_layers
        self._num_query_heads = num_query_heads
        self._num_kv_heads = num_kv_heads
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._attention_modules = attention_modules
        self._layer_modules = layer_modules
        self._mlp_modules = mlp_modules
        self._lm_head = lm_head
        self._norm = norm
        self._model_name = model_name

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_model(cls, model: Any, tokenizer: Any = None) -> ModelAdapter:
        """Construct adapter by inspecting a loaded model.

        Parameters
        ----------
        model:
            A ``transformers.PreTrainedModel`` (or any object exposing
            ``.config`` and a standard module tree).
        tokenizer:
            Optional tokenizer -- reserved for future use (e.g. vocab
            sanity checks).  Not consumed today.
        """
        config = model.config

        num_layers: int = config.num_hidden_layers
        num_query_heads: int = config.num_attention_heads
        num_kv_heads: int = getattr(config, "num_key_value_heads", num_query_heads)
        hidden_size: int = config.hidden_size
        vocab_size: int = config.vocab_size
        model_name: str = getattr(config, "_name_or_path", "unknown")

        logger.info(
            "Model config: %d layers, %d query heads, %d kv heads, "
            "hidden=%d, vocab=%d",
            num_layers,
            num_query_heads,
            num_kv_heads,
            hidden_size,
            vocab_size,
        )

        # --- Layer container ---
        layers_container = _find_layers_container(model)
        if layers_container is None or len(layers_container) != num_layers:
            raise RuntimeError(
                f"Could not find layer container with {num_layers} layers. "
                "Expected model.model.layers, model.transformer.h, or "
                "model.gpt_neox.layers."
            )

        # --- Per-layer submodules ---
        attention_modules: list[tuple[int, Any]] = []
        layer_modules: list[tuple[int, Any]] = []
        mlp_modules: list[tuple[int, Any]] = []

        for idx, layer in enumerate(layers_container):
            layer_modules.append((idx, layer))

            attn = _find_attention_submodule(layer)
            if attn is not None:
                attention_modules.append((idx, attn))

            mlp = _find_mlp_submodule(layer)
            if mlp is not None:
                mlp_modules.append((idx, mlp))

        if len(attention_modules) != num_layers:
            raise RuntimeError(
                f"Found {len(attention_modules)} attention modules, "
                f"expected {num_layers}"
            )

        logger.info(
            "Found %d layer modules, %d attention modules, %d MLP modules",
            len(layer_modules),
            len(attention_modules),
            len(mlp_modules),
        )

        if len(mlp_modules) != num_layers:
            logger.warning(
                "Found %d MLP modules (expected %d) -- MLP capture may not "
                "work for all layers",
                len(mlp_modules),
                num_layers,
            )

        # --- LM head ---
        lm_head = _find_lm_head(model)
        if lm_head is None:
            raise RuntimeError(
                "Could not find language model head "
                "(tried: lm_head, output, embed_out)"
            )

        # --- Final norm ---
        norm = _find_final_norm(model)
        if norm is None:
            raise RuntimeError(
                "Could not find final normalization layer "
                "(tried: model.norm, model.final_layernorm, "
                "transformer.ln_f, gpt_neox.final_layer_norm)"
            )

        logger.info("Model adapter ready: %s", model_name)

        return cls(
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            attention_modules=attention_modules,
            layer_modules=layer_modules,
            mlp_modules=mlp_modules,
            lm_head=lm_head,
            norm=norm,
            model_name=model_name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        """Total number of transformer decoder layers."""
        return self._num_layers

    @property
    def num_query_heads(self) -> int:
        """Number of query attention heads per layer."""
        return self._num_query_heads

    @property
    def num_kv_heads(self) -> int:
        """Number of key/value attention heads per layer (GQA-aware)."""
        return self._num_kv_heads

    @property
    def hidden_size(self) -> int:
        """Dimensionality of the residual stream."""
        return self._hidden_size

    @property
    def vocab_size(self) -> int:
        """Size of the token vocabulary."""
        return self._vocab_size

    @property
    def model_name(self) -> str:
        """Model identifier (``_name_or_path`` from config)."""
        return self._model_name

    # ------------------------------------------------------------------
    # Module accessors
    # ------------------------------------------------------------------

    def get_attention_modules(self) -> list[tuple[int, Any]]:
        """Return ``(layer_idx, attn_module)`` pairs for hook registration."""
        return list(self._attention_modules)

    def get_layer_modules(self) -> list[tuple[int, Any]]:
        """Return ``(layer_idx, layer_module)`` pairs for residual hooks."""
        return list(self._layer_modules)

    def get_mlp_modules(self) -> list[tuple[int, Any]]:
        """Return ``(layer_idx, mlp_module)`` pairs for MLP hooks."""
        return list(self._mlp_modules)

    def get_lm_head(self) -> Any:
        """Return the language model head for logit lens projection."""
        return self._lm_head

    def get_norm(self) -> Any:
        """Return the final normalization layer before the LM head."""
        return self._norm


# ======================================================================
# Private helpers -- architecture probing
# ======================================================================

# Candidate attribute paths for the sequential layer container.
_LAYERS_CONTAINER_PATHS: list[tuple[str, ...]] = [
    ("model", "layers"),       # Llama, Qwen, Mistral, Gemma, Phi-3
    ("transformer", "h"),      # GPT-2, GPT-Neo
    ("gpt_neox", "layers"),    # GPT-NeoX, Pythia
]

# Candidate attribute names for the self-attention submodule inside a
# single decoder layer.
_ATTENTION_ATTR_NAMES: list[str] = ["self_attn", "attention", "attn"]

# Candidate attribute names for the MLP / feed-forward submodule.
_MLP_ATTR_NAMES: list[str] = ["mlp", "feed_forward", "ffn"]

# Candidate attribute names for the LM head on the top-level model.
_LM_HEAD_ATTR_NAMES: list[str] = ["lm_head", "output", "embed_out"]

# Candidate attribute paths for the final normalization layer.
_FINAL_NORM_PATHS: list[tuple[str, ...]] = [
    ("model", "norm"),              # Llama, Qwen, Mistral, Gemma
    ("model", "final_layernorm"),   # Falcon
    ("transformer", "ln_f"),        # GPT-2, GPT-Neo
    ("gpt_neox", "final_layer_norm"),  # GPT-NeoX, Pythia
]


def _walk_path(root: Any, path: tuple[str, ...]) -> Any | None:
    """Traverse a dotted attribute path, returning ``None`` on any miss."""
    obj = root
    for attr in path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _find_layers_container(model: Any) -> Any | None:
    """Locate the sequential container holding all decoder layers."""
    for path in _LAYERS_CONTAINER_PATHS:
        container = _walk_path(model, path)
        if container is not None and hasattr(container, "__len__"):
            logger.info("Found layers at: model.%s", ".".join(path))
            return container
    return None


def _find_attention_submodule(layer: Any) -> Any | None:
    """Locate the self-attention submodule inside a single decoder layer."""
    for name in _ATTENTION_ATTR_NAMES:
        submodule = getattr(layer, name, None)
        if submodule is not None:
            return submodule
    return None


def _find_mlp_submodule(layer: Any) -> Any | None:
    """Locate the MLP / feed-forward submodule inside a single decoder layer."""
    for name in _MLP_ATTR_NAMES:
        submodule = getattr(layer, name, None)
        if submodule is not None:
            return submodule
    return None


def _find_lm_head(model: Any) -> Any | None:
    """Locate the language model head on the top-level model."""
    for name in _LM_HEAD_ATTR_NAMES:
        head = getattr(model, name, None)
        if head is not None:
            logger.info("Found LM head at: model.%s", name)
            return head
    return None


def _find_final_norm(model: Any) -> Any | None:
    """Locate the final normalization layer applied before the LM head."""
    for path in _FINAL_NORM_PATHS:
        norm = _walk_path(model, path)
        if norm is not None:
            logger.info("Found final norm at: model.%s", ".".join(path))
            return norm
    return None
