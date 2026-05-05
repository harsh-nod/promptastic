"""Composable hook system for capturing activations during a forward pass.

Provides cache classes that register PyTorch forward hooks on specific
submodules (decoder layers, self-attention, MLP) and store extracted
tensors keyed by ``(layer, position_name)``.  Each cache class follows
the same protocol:

- ``make_hook(...)`` returns a callable suitable for
  ``module.register_forward_hook()``.
- ``get(layer, position_name)`` retrieves stored data.
- ``clear()`` releases all cached tensors.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ======================================================================
# Base class
# ======================================================================

class BaseCache(ABC):
    """Abstract base for activation caches.

    Subclasses must implement ``make_hook`` (the signature varies by
    capture type, so it is left as a regular method rather than
    enforcing a fixed signature here).
    """

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def clear(self) -> None:
        """Release all cached data."""
        self.data.clear()

    @staticmethod
    def _key(prefix: str, layer: int, pos_name: str) -> str:
        return f"{prefix}_L{layer}_{pos_name}"

    @abstractmethod
    def make_hook(self, *args: Any, **kwargs: Any) -> Any:
        """Create and return a hook function (signature varies by subclass)."""
        ...


# ======================================================================
# Residual stream
# ======================================================================

class ResidualCache(BaseCache):
    """Hooks decoder layers and captures ``output[0]`` at query positions.

    The captured tensor is the post-layer residual stream vector --
    the hidden state *after* both attention and MLP within the layer.
    """

    def make_hook(
        self,
        layer_idx: int,
        query_positions: dict[str, int],
    ) -> Any:
        """Return a forward hook that stores residual vectors.

        Parameters
        ----------
        layer_idx:
            Index of the decoder layer being hooked.
        query_positions:
            ``{position_name: token_index}`` -- which sequence positions
            to capture.
        """
        data = self.data

        def _hook(module: Any, inputs: Any, output: Any) -> None:
            hidden = output[0]
            # Some models return (seq_len, hidden) without batch dim.
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

            for pos_name, pos_idx in query_positions.items():
                if pos_idx < hidden.shape[1]:
                    key = BaseCache._key("resid", layer_idx, pos_name)
                    data[key] = hidden[0, pos_idx, :].float().cpu()

        return _hook

    def get(self, layer: int, position_name: str) -> torch.Tensor | None:
        """Retrieve the residual vector at *(layer, position_name)*, or ``None``."""
        return self.data.get(BaseCache._key("resid", layer, position_name))


# ======================================================================
# Attention
# ======================================================================

class AttentionCache(BaseCache):
    """Hooks self-attention submodules and captures per-region attention weights.

    By default attention is averaged across heads (matching standard
    cooking-curve analysis).  Set ``per_head=True`` to also retain the
    full per-head weight vector for head specialisation analysis.

    The hook **replaces the attention weight tensor in the output with
    ``None``** to free the ~2 GB per-layer memory that would otherwise
    accumulate.
    """

    def __init__(
        self,
        *,
        per_head: bool = False,
        capture_per_token: bool = True,
    ) -> None:
        super().__init__()
        self._per_head = per_head
        self._capture_per_token = capture_per_token

    def make_hook(
        self,
        layer_idx: int,
        query_positions: dict[str, int],
        region_map: dict[str, dict[str, int]],
    ) -> Any:
        """Return a forward hook that extracts per-region attention.

        Parameters
        ----------
        layer_idx:
            Layer index of the attention module.
        query_positions:
            ``{position_name: token_index}``
        region_map:
            ``{region_name: {"tok_start": int, "tok_end": int, ...}}``
        """
        data = self.data
        per_head_flag = self._per_head
        per_token_flag = self._capture_per_token

        def _hook(module: Any, inputs: Any, output: Any) -> Any:
            # output = (attn_output, attn_weights, ...) -- attn_weights
            # has shape (batch, heads, seq, seq).
            if len(output) < 2 or output[1] is None:
                return output

            attn_weights = output[1]
            if attn_weights.dim() == 3:
                attn_weights = attn_weights.unsqueeze(0)

            for pos_name, pos_idx in query_positions.items():
                if pos_idx >= attn_weights.shape[2]:
                    continue

                # row: (heads, seq_len)
                row = attn_weights[0, :, pos_idx, :].float()

                # -- Per-region aggregation (head-mean) --
                region_weights: dict[str, float] = {}
                for region_name, region_info in region_map.items():
                    tok_start = region_info["tok_start"]
                    tok_end = region_info["tok_end"]
                    if tok_end <= attn_weights.shape[3]:
                        per_head_sum = row[:, tok_start:tok_end].sum(dim=1)
                        region_weights[region_name] = round(
                            per_head_sum.mean().item(), 6
                        )

                key = BaseCache._key("attn", layer_idx, pos_name)
                data[key] = region_weights

                # -- Per-head storage (optional) --
                if per_head_flag:
                    head_weights: dict[str, list[float]] = {}
                    for region_name, region_info in region_map.items():
                        tok_start = region_info["tok_start"]
                        tok_end = region_info["tok_end"]
                        if tok_end <= attn_weights.shape[3]:
                            per_head_sum = row[:, tok_start:tok_end].sum(dim=1)
                            head_weights[region_name] = per_head_sum.cpu().tolist()
                    ph_key = BaseCache._key("attn_ph", layer_idx, pos_name)
                    data[ph_key] = head_weights

                # -- Per-token storage (optional) --
                if per_token_flag:
                    per_tok = row.mean(dim=0).cpu().tolist()
                    pt_key = BaseCache._key("attn_pt", layer_idx, pos_name)
                    data[pt_key] = per_tok

            # CRITICAL: free the attention weight tensor.
            return (output[0], None) + output[2:]

        return _hook

    def get(self, layer: int, position_name: str) -> dict[str, float] | None:
        """Retrieve head-averaged per-region attention at *(layer, position)*."""
        return self.data.get(BaseCache._key("attn", layer, position_name))

    def get_per_head(
        self, layer: int, position_name: str
    ) -> dict[str, list[float]] | None:
        """Retrieve full per-head attention at *(layer, position)*, or ``None``.

        Only populated when ``per_head=True`` was set at construction.
        """
        return self.data.get(BaseCache._key("attn_ph", layer, position_name))

    def get_per_token(
        self, layer: int, position_name: str
    ) -> list[float] | None:
        """Retrieve per-token (head-averaged) attention, or ``None``.

        Only populated when ``capture_per_token=True`` (the default).
        """
        return self.data.get(BaseCache._key("attn_pt", layer, position_name))


# ======================================================================
# MLP
# ======================================================================

class MLPCache(BaseCache):
    """Hooks MLP submodules to capture pre- and post-MLP activations.

    Two hooks per layer:

    * **Input hook** (``register_forward_pre_hook``): stores the MLP
      *input* -- i.e. the post-attention residual stream.
    * **Output hook** (``register_forward_hook``): stores the MLP
      *output* -- i.e. the delta that the MLP adds to the residual.
    """

    def make_hook(self, *args: Any, **kwargs: Any) -> Any:
        """Not used directly -- call ``make_input_hook`` or ``make_output_hook``."""
        raise NotImplementedError(
            "Use make_input_hook() or make_output_hook() instead"
        )

    def make_input_hook(
        self,
        layer_idx: int,
        query_positions: dict[str, int],
    ) -> Any:
        """Return a *pre*-hook capturing the MLP input (post-attention residual).

        Register with ``module.register_forward_pre_hook(hook)``.
        """
        data = self.data

        def _pre_hook(module: Any, inputs: Any) -> None:
            # inputs is a tuple; first element is the hidden state tensor.
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

            for pos_name, pos_idx in query_positions.items():
                if pos_idx < hidden.shape[1]:
                    key = BaseCache._key("mlp_in", layer_idx, pos_name)
                    data[key] = hidden[0, pos_idx, :].float().cpu()

        return _pre_hook

    def make_output_hook(
        self,
        layer_idx: int,
        query_positions: dict[str, int],
    ) -> Any:
        """Return a *post*-hook capturing the MLP output (the MLP delta).

        Register with ``module.register_forward_hook(hook)``.
        """
        data = self.data

        def _hook(module: Any, inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

            for pos_name, pos_idx in query_positions.items():
                if pos_idx < hidden.shape[1]:
                    key = BaseCache._key("mlp_out", layer_idx, pos_name)
                    data[key] = hidden[0, pos_idx, :].float().cpu()

        return _hook

    def get_input(self, layer: int, position_name: str) -> torch.Tensor | None:
        """Retrieve MLP input (post-attention residual) at *(layer, position)*."""
        return self.data.get(BaseCache._key("mlp_in", layer, position_name))

    def get_output(self, layer: int, position_name: str) -> torch.Tensor | None:
        """Retrieve MLP output (delta) at *(layer, position)*."""
        return self.data.get(BaseCache._key("mlp_out", layer, position_name))


# ======================================================================
# Gradient capture
# ======================================================================

class GradientCache(BaseCache):
    """Captures gradient norms flowing through residual stream positions.

    The forward hook retains the hidden-state tensor and registers a
    backward hook on it to capture the gradient when ``.backward()`` is
    called.  The stored value is the L2 norm of the gradient vector at
    each query position.

    .. note::

       The forward pass must be run **with gradients enabled** (i.e.
       *not* inside ``torch.no_grad()``) for the backward hook to fire.
    """

    def make_hook(
        self,
        layer_idx: int,
        query_positions: dict[str, int],
    ) -> Any:
        """Return a forward hook that registers a backward hook for gradient capture.

        Register on the decoder layer (not the attention submodule).
        """
        data = self.data

        def _hook(module: Any, inputs: Any, output: Any) -> None:
            hidden = output[0]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

            # Retain grad on the hidden state so the backward hook fires.
            if not hidden.requires_grad:
                hidden.requires_grad_(True)
            hidden.retain_grad()

            def _backward_hook(grad: torch.Tensor) -> None:
                if grad is None:
                    return
                if grad.dim() == 2:
                    grad = grad.unsqueeze(0)
                for pos_name, pos_idx in query_positions.items():
                    if pos_idx < grad.shape[1]:
                        grad_vec = grad[0, pos_idx, :].float()
                        key = BaseCache._key("grad", layer_idx, pos_name)
                        data[key] = grad_vec.norm().item()

            hidden.register_hook(_backward_hook)

        return _hook

    def get_gradient_norm(
        self, layer: int, position_name: str
    ) -> float | None:
        """Retrieve the gradient L2 norm at *(layer, position)*, or ``None``."""
        return self.data.get(BaseCache._key("grad", layer, position_name))
