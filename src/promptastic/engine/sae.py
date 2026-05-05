"""Sparse autoencoder for residual stream decomposition.

Loads pre-trained SAE weights and decomposes residual stream states into
sparse feature activations.  This lets the pipeline identify which learned
features are active at each layer, providing a complementary view to
attention patterns and logit lens projections.

Supports two weight formats:
1. Raw PyTorch state dicts with keys ``encoder.weight``, ``encoder.bias``,
   ``decoder.weight``, ``decoder.bias``.
2. ``sae_lens`` format (if the library is installed).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .._types import SAEFeatureActivation

logger = logging.getLogger(__name__)


class SAEDecoder:
    """Sparse autoencoder that encodes residual stream vectors into sparse features.

    Loads pre-trained weights and provides encode/decode methods.  The
    encoder applies ``ReLU(W_enc @ h + b_enc)`` to produce a sparse feature
    vector, and the decoder reconstructs via ``W_dec @ features + b_dec``.
    """

    def __init__(self, weights_path: str | Path) -> None:
        """Load pre-trained SAE weights from disk.

        Attempts to load as a raw PyTorch state dict first.  If the expected
        keys are missing and ``sae_lens`` is installed, falls back to loading
        via that library.

        Args:
            weights_path: Path to a ``.pt`` / ``.bin`` file or a directory
                containing ``sae_lens``-format weights.
        """
        weights_path = Path(weights_path)
        self._encoder_weight: torch.Tensor
        self._encoder_bias: torch.Tensor
        self._decoder_weight: torch.Tensor
        self._decoder_bias: torch.Tensor

        loaded = False

        # Strategy 1: raw PyTorch dict
        if weights_path.is_file():
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "encoder.weight" in state:
                self._encoder_weight = state["encoder.weight"].float()
                self._encoder_bias = state["encoder.bias"].float()
                self._decoder_weight = state["decoder.weight"].float()
                self._decoder_bias = state["decoder.bias"].float()
                loaded = True
                logger.info(
                    "Loaded SAE from PyTorch dict: %s (features=%d, hidden=%d)",
                    weights_path,
                    self._encoder_weight.shape[0],
                    self._encoder_weight.shape[1],
                )

        # Strategy 2: sae_lens
        if not loaded:
            try:
                from sae_lens import SAE as SaeLensSAE

                sae_obj = SaeLensSAE.load_from_pretrained(str(weights_path))
                self._encoder_weight = sae_obj.W_enc.detach().float()
                self._encoder_bias = sae_obj.b_enc.detach().float()
                self._decoder_weight = sae_obj.W_dec.detach().float()
                self._decoder_bias = sae_obj.b_dec.detach().float()
                loaded = True
                logger.info(
                    "Loaded SAE via sae_lens: %s (features=%d, hidden=%d)",
                    weights_path,
                    self._encoder_weight.shape[0],
                    self._encoder_weight.shape[1],
                )
            except ImportError:
                pass
            except Exception as exc:
                logger.warning("sae_lens load failed: %s", exc)

        if not loaded:
            raise RuntimeError(
                f"Could not load SAE weights from {weights_path}. "
                "Expected a PyTorch state dict with encoder.weight/bias and "
                "decoder.weight/bias, or an sae_lens-compatible directory."
            )

    @property
    def n_features(self) -> int:
        """Number of sparse features (output dimension of encoder)."""
        return self._encoder_weight.shape[0]

    @property
    def hidden_size(self) -> int:
        """Model hidden size (input dimension of encoder)."""
        return self._encoder_weight.shape[1]

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Encode a hidden state into a sparse feature vector.

        Computes ``ReLU(W_enc @ h + b_enc)``.

        Args:
            h: Hidden state tensor, shape ``(..., hidden_size)``.

        Returns:
            Sparse feature activations, shape ``(..., n_features)``.
        """
        device = h.device
        w = self._encoder_weight.to(device)
        b = self._encoder_bias.to(device)
        return F.relu(F.linear(h.float(), w, b))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to hidden-state space.

        Computes ``W_dec @ features + b_dec``.

        Args:
            features: Sparse feature tensor, shape ``(..., n_features)``.

        Returns:
            Reconstructed hidden state, shape ``(..., hidden_size)``.
        """
        device = features.device
        w = self._decoder_weight.to(device)
        b = self._decoder_bias.to(device)
        return F.linear(features.float(), w, b)


class SAEAnalyzer:
    """Runs SAE feature analysis on cached residual stream states.

    At each layer, encodes the residual at the query positions and identifies
    the top-K most active features by magnitude.
    """

    def __init__(self, sae: SAEDecoder) -> None:
        """Initialize with a loaded SAE.

        Args:
            sae: A ``SAEDecoder`` instance with pre-trained weights.
        """
        self.sae = sae

    def analyze(
        self,
        residual_cache: object,
        query_positions: dict[str, int],
        num_layers: int,
        top_k: int = 20,
    ) -> dict:
        """Identify top-K active SAE features at each layer.

        Args:
            residual_cache: Object with a ``.get(layer, pos_name)`` method
                returning a hidden-state tensor, or ``None`` if not cached.
            query_positions: Named query positions (only the first is used).
            num_layers: Total number of model layers to iterate.
            top_k: Number of top features to report per layer.

        Returns:
            Dictionary with ``"per_layer"`` containing a list of dicts, each
            with ``"layer"`` (int) and ``"top_features"`` (list of dicts with
            ``"feature_idx"`` and ``"activation"``).
        """
        pos_name = next(iter(query_positions))
        per_layer: list[dict] = []

        for layer_idx in range(num_layers):
            h = residual_cache.get(layer_idx, pos_name)
            if h is None:
                continue

            features = self.sae.encode(h.unsqueeze(0)).squeeze(0)

            k = min(top_k, features.shape[-1])
            top_vals, top_idxs = torch.topk(features, k)

            top_features: list[dict] = []
            for feat_idx, act_val in zip(top_idxs.tolist(), top_vals.tolist()):
                top_features.append(
                    {
                        "feature_idx": feat_idx,
                        "activation": round(act_val, 6),
                    }
                )

            per_layer.append(
                {
                    "layer": layer_idx,
                    "top_features": top_features,
                }
            )

        return {"per_layer": per_layer}
