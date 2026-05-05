"""Gradient-based attribution for prompt regions.

Measures how much each region of the input contributes to a target output
token by computing gradient norms through the residual stream.  Supports
two methods:

- **Vanilla**: single backward pass from the target logit, measuring
  ``||dlogit/dh_layer||`` summed over each region's token positions.
- **Integrated gradients** (Sundararajan et al. 2017): interpolates the
  embedding from a zero baseline to the actual input over *n_steps*,
  averages the gradients, then sums per region.  More faithful to the
  model's computation but proportionally more expensive.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import torch

from .._types import RegionInfo

logger = logging.getLogger(__name__)


class GradientAnalyzer:
    """Compute gradient-based attribution scores for prompt regions.

    Registers hooks to capture per-layer gradients flowing through the
    residual stream, then aggregates them by region.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        adapter: object,
        method: str = "vanilla",
    ) -> None:
        """Initialize the gradient analyzer.

        Args:
            model: A loaded HuggingFace causal LM.
            adapter: A ``ModelAdapter`` instance.
            method: ``"vanilla"`` for a single backward pass, or
                ``"integrated"`` for integrated gradients.
        """
        if method not in ("vanilla", "integrated"):
            raise ValueError(f"Unknown gradient method: {method!r}. Use 'vanilla' or 'integrated'.")
        self.model = model
        self.adapter = adapter
        self.method = method

    # ------------------------------------------------------------------
    # Vanilla gradients
    # ------------------------------------------------------------------

    def compute_vanilla(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_idx: int,
        query_positions: dict[str, int],
        region_map: dict[str, RegionInfo],
    ) -> dict:
        """Single backward pass from target logit through the residual stream.

        At each layer, captures the gradient norm of the residual stream
        with respect to the target logit, then sums those norms over each
        region's token span.

        Args:
            input_ids: Token ids shaped ``(1, seq_len)``.
            attention_mask: Attention mask shaped ``(1, seq_len)``.
            target_token_idx: Vocabulary index of the target token.
            query_positions: Named query positions.
            region_map: Region name to ``RegionInfo`` mapping.

        Returns:
            Dict with ``"target_token"`` (str index), ``"per_layer"`` list of
            dicts each containing ``"layer"``, ``"per_region_gradient"``
            mapping region names to summed gradient norms, and
            ``"total_gradient_norm"``.
        """
        layer_modules = self.adapter.get_layer_modules()
        num_layers = self.adapter.num_layers

        # Storage for captured activations (need .grad after backward)
        layer_activations: dict[int, torch.Tensor] = {}
        handles: list = []

        for layer_idx, layer_mod in layer_modules:

            def _hook(module, inp, output, idx=layer_idx):
                h = output[0]
                if h.dim() == 2:
                    h = h.unsqueeze(0)
                h.retain_grad()
                layer_activations[idx] = h

            handles.append(layer_mod.register_forward_hook(_hook))

        # Forward with gradient tracking
        self.model.zero_grad()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            use_cache=False,
        )

        # Pick the query position for the backward target
        query_pos_name = next(iter(query_positions))
        query_pos_idx = query_positions[query_pos_name]

        logits = outputs.logits[0, query_pos_idx, :]
        target_logit = logits[target_token_idx]
        target_logit.backward(retain_graph=False)

        for h in handles:
            h.remove()

        # Collect per-layer, per-region gradient norms
        per_layer: list[dict] = []
        for layer_idx in range(num_layers):
            act = layer_activations.get(layer_idx)
            if act is None or act.grad is None:
                continue

            grad = act.grad[0]  # (seq_len, hidden_size)
            grad_norms = grad.norm(dim=-1)  # (seq_len,)

            per_region_gradient: dict[str, float] = {}
            for region_name, rinfo in region_map.items():
                tok_s = rinfo["tok_start"]
                tok_e = rinfo["tok_end"]
                region_grad = grad_norms[tok_s:tok_e].sum().item()
                per_region_gradient[region_name] = round(region_grad, 6)

            total_norm = grad_norms.sum().item()

            per_layer.append(
                {
                    "layer": layer_idx,
                    "per_region_gradient": per_region_gradient,
                    "total_gradient_norm": round(total_norm, 6),
                }
            )

        del layer_activations, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "target_token": str(target_token_idx),
            "per_layer": per_layer,
        }

    # ------------------------------------------------------------------
    # Integrated gradients
    # ------------------------------------------------------------------

    def compute_integrated(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_idx: int,
        query_positions: dict[str, int],
        region_map: dict[str, RegionInfo],
        n_steps: int = 50,
    ) -> dict:
        """Integrated gradients from a zero embedding baseline to actual input.

        Interpolates the input embeddings from zero to their true values
        over *n_steps*, computes the gradient at each interpolation point,
        and averages them.  The result approximates the path integral and
        satisfies the completeness axiom.

        Args:
            input_ids: Token ids shaped ``(1, seq_len)``.
            attention_mask: Attention mask shaped ``(1, seq_len)``.
            target_token_idx: Vocabulary index of the target token.
            query_positions: Named query positions.
            region_map: Region name to ``RegionInfo`` mapping.
            n_steps: Number of interpolation steps.

        Returns:
            Same format as ``compute_vanilla``.
        """
        num_layers = self.adapter.num_layers
        layer_modules = self.adapter.get_layer_modules()

        query_pos_name = next(iter(query_positions))
        query_pos_idx = query_positions[query_pos_name]

        # Get the embedding layer
        embed_module = None
        for name in ("embed_tokens", "wte"):
            parent = getattr(self.model, "model", self.model)
            embed_module = getattr(parent, name, None)
            if embed_module is not None:
                break
        if embed_module is None:
            # Fallback: use the model's get_input_embeddings
            embed_module = self.model.get_input_embeddings()

        with torch.no_grad():
            baseline_embeds = torch.zeros_like(embed_module(input_ids)).float()
            actual_embeds = embed_module(input_ids).float()

        # Accumulate gradients across steps
        accumulated_grads: dict[int, torch.Tensor] = {}

        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps  # midpoint rule
            interpolated = baseline_embeds + alpha * (actual_embeds - baseline_embeds)
            interpolated = interpolated.to(dtype=actual_embeds.dtype).requires_grad_(True)

            layer_activations: dict[int, torch.Tensor] = {}
            handles: list = []

            for layer_idx, layer_mod in layer_modules:

                def _hook(module, inp, output, idx=layer_idx):
                    h = output[0]
                    if h.dim() == 2:
                        h = h.unsqueeze(0)
                    h.retain_grad()
                    layer_activations[idx] = h

                handles.append(layer_mod.register_forward_hook(_hook))

            self.model.zero_grad()
            outputs = self.model(
                inputs_embeds=interpolated.to(next(self.model.parameters()).dtype),
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False,
            )

            logits = outputs.logits[0, query_pos_idx, :]
            target_logit = logits[target_token_idx]
            target_logit.backward(retain_graph=False)

            for h in handles:
                h.remove()

            for layer_idx in range(num_layers):
                act = layer_activations.get(layer_idx)
                if act is None or act.grad is None:
                    continue
                grad = act.grad[0].detach().float()
                if layer_idx in accumulated_grads:
                    accumulated_grads[layer_idx] += grad
                else:
                    accumulated_grads[layer_idx] = grad.clone()

            del layer_activations, outputs
            gc.collect()

        # Average and compute per-region norms
        per_layer: list[dict] = []
        for layer_idx in range(num_layers):
            grad = accumulated_grads.get(layer_idx)
            if grad is None:
                continue

            avg_grad = grad / n_steps
            grad_norms = avg_grad.norm(dim=-1)

            per_region_gradient: dict[str, float] = {}
            for region_name, rinfo in region_map.items():
                tok_s = rinfo["tok_start"]
                tok_e = rinfo["tok_end"]
                region_grad = grad_norms[tok_s:tok_e].sum().item()
                per_region_gradient[region_name] = round(region_grad, 6)

            total_norm = grad_norms.sum().item()
            per_layer.append(
                {
                    "layer": layer_idx,
                    "per_region_gradient": per_region_gradient,
                    "total_gradient_norm": round(total_norm, 6),
                }
            )

        del accumulated_grads
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "target_token": str(target_token_idx),
            "per_layer": per_layer,
        }

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_idx: int,
        query_positions: dict[str, int],
        region_map: dict[str, RegionInfo],
        tokenizer: object,
    ) -> dict:
        """Dispatch to the configured gradient method.

        Args:
            input_ids: Token ids shaped ``(1, seq_len)``.
            attention_mask: Attention mask shaped ``(1, seq_len)``.
            target_token_idx: Vocabulary index of the target token.
            query_positions: Named query positions.
            region_map: Region name to ``RegionInfo`` mapping.
            tokenizer: Tokenizer (used for logging the target token).

        Returns:
            Attribution result dict from the selected method.
        """
        target_str = tokenizer.decode([target_token_idx])
        logger.info(
            "Running %s gradient attribution for target token %r (id=%d)",
            self.method,
            target_str,
            target_token_idx,
        )

        if self.method == "vanilla":
            result = self.compute_vanilla(
                input_ids, attention_mask, target_token_idx,
                query_positions, region_map,
            )
        else:
            result = self.compute_integrated(
                input_ids, attention_mask, target_token_idx,
                query_positions, region_map,
            )

        result["target_token"] = target_str
        return result
