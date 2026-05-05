"""Activation patching -- zero or mean ablation of a region's contribution at a specific layer.

Measures the causal importance of a prompt region at a given layer by replacing
that region's residual stream contribution with zeros (zero ablation) or the
mean across all positions (mean ablation), then measuring how much the output
distribution changes.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from .._types import PatchingResult, RegionInfo

logger = logging.getLogger(__name__)


class PatchingEngine:
    """Runs activation patching experiments over (region, layer) pairs.

    Performs a single baseline forward pass to cache residual stream states,
    then reruns the forward pass with targeted ablations to measure each
    region's causal contribution at each layer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        adapter: object,
        tokenizer: object,
        method: str = "zero",
    ) -> None:
        """Initialize the patching engine.

        Args:
            model: A loaded HuggingFace causal LM.
            adapter: A ModelAdapter instance providing architecture accessors.
            tokenizer: The corresponding tokenizer.
            method: Ablation strategy -- ``"zero"`` or ``"mean"``.
        """
        if method not in ("zero", "mean"):
            raise ValueError(f"Unknown patching method: {method!r}. Use 'zero' or 'mean'.")
        self.model = model
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.method = method

    # ------------------------------------------------------------------
    # Baseline pass
    # ------------------------------------------------------------------

    def run_baseline(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        query_positions: dict[str, int],
    ) -> dict:
        """Forward pass that caches the full residual stream at every layer.

        Args:
            input_ids: Token ids shaped ``(1, seq_len)``.
            attention_mask: Attention mask shaped ``(1, seq_len)``.
            query_positions: Mapping of position name to token index.

        Returns:
            Dictionary with ``"logits"`` (the output logits tensor on CPU) and
            ``"residuals"`` mapping each layer index to the full residual
            tensor at that layer (detached, on CPU).
        """
        residuals: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHook] = []

        layer_modules = self.adapter.get_layer_modules()

        for layer_idx, layer_module in layer_modules:

            def _capture_hook(module, inp, output, idx=layer_idx):
                hidden = output[0]
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)
                residuals[idx] = hidden.detach().cpu()

            handles.append(layer_module.register_forward_hook(_capture_hook))

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False,
            )

        for h in handles:
            h.remove()

        logits = outputs.logits.detach().cpu()
        del outputs

        return {"logits": logits, "residuals": residuals}

    # ------------------------------------------------------------------
    # Patched pass
    # ------------------------------------------------------------------

    def run_patched(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        baseline_residuals: dict[int, torch.Tensor],
        region_name: str,
        region_info: RegionInfo,
        target_layer: int,
        query_positions: dict[str, int],
    ) -> dict:
        """Forward pass with one region ablated at one layer.

        A pre-hook on ``target_layer`` replaces the residual stream values at
        the region's token positions with either zeros or the positional mean,
        depending on ``self.method``.

        Args:
            input_ids: Token ids shaped ``(1, seq_len)``.
            attention_mask: Attention mask shaped ``(1, seq_len)``.
            baseline_residuals: Residual cache from ``run_baseline``.
            region_name: Name of the region to ablate (for logging).
            region_info: Token boundary dict with ``tok_start`` and ``tok_end``.
            target_layer: Layer index at which to apply the ablation.
            query_positions: Mapping of position name to token index.

        Returns:
            Dictionary with ``"logits"`` (patched output logits, CPU).
        """
        tok_start = region_info["tok_start"]
        tok_end = region_info["tok_end"]
        method = self.method

        layer_modules = self.adapter.get_layer_modules()
        target_module = None
        for idx, mod in layer_modules:
            if idx == target_layer:
                target_module = mod
                break

        if target_module is None:
            raise ValueError(f"Layer {target_layer} not found in adapter layer modules.")

        def _patch_pre_hook(module, args):
            hidden = args[0]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            patched = hidden.clone()
            if method == "zero":
                patched[:, tok_start:tok_end, :] = 0.0
            elif method == "mean":
                position_mean = hidden.mean(dim=1, keepdim=True)
                patched[:, tok_start:tok_end, :] = position_mean.expand_as(
                    patched[:, tok_start:tok_end, :]
                )
            if args[0].dim() == 2:
                patched = patched.squeeze(0)
            return (patched,) + args[1:]

        handle = target_module.register_forward_pre_hook(_patch_pre_hook)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False,
            )

        handle.remove()

        logits = outputs.logits.detach().cpu()
        del outputs

        return {"logits": logits}

    # ------------------------------------------------------------------
    # Effect computation
    # ------------------------------------------------------------------

    def compute_effect(
        self,
        baseline_logits: torch.Tensor,
        patched_logits: torch.Tensor,
        query_position_idx: int,
        tokenizer: object,
    ) -> PatchingResult:
        """Measure the distributional shift caused by patching.

        Computes KL divergence between the baseline and patched output
        distributions at the query position, plus the raw logit difference
        for the baseline's top token.

        Args:
            baseline_logits: Logits from the unmodified forward pass.
            patched_logits: Logits from the patched forward pass.
            query_position_idx: Sequence position to compare at.
            tokenizer: Tokenizer for decoding token ids.

        Returns:
            A ``PatchingResult`` dict.
        """
        base_logit_vec = baseline_logits[0, query_position_idx, :].float()
        patch_logit_vec = patched_logits[0, query_position_idx, :].float()

        base_log_probs = F.log_softmax(base_logit_vec, dim=-1)
        patch_log_probs = F.log_softmax(patch_logit_vec, dim=-1)
        base_probs = base_log_probs.exp()

        kl_div = F.kl_div(patch_log_probs, base_probs, reduction="sum", log_target=False).item()

        base_top_id = base_logit_vec.argmax().item()
        patch_top_id = patch_logit_vec.argmax().item()
        logit_diff = (base_logit_vec[base_top_id] - patch_logit_vec[base_top_id]).item()

        baseline_top_token = tokenizer.decode([base_top_id])
        top_token_change = tokenizer.decode([patch_top_id])

        return PatchingResult(
            region="",
            layer=0,
            kl_divergence=round(kl_div, 6),
            logit_diff=round(logit_diff, 4),
            top_token_change=top_token_change,
            baseline_top_token=baseline_top_token,
        )

    # ------------------------------------------------------------------
    # Full sweep
    # ------------------------------------------------------------------

    def run_full_sweep(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        query_positions: dict[str, int],
        region_map: dict[str, RegionInfo],
        target_layers: list[int],
        tokenizer: object,
    ) -> list[PatchingResult]:
        """Run patching over all (region, layer) combinations.

        Performs the baseline forward pass once, then iterates every
        combination of region and target layer, running a patched forward
        pass for each. Cleans up GPU memory between passes.

        Args:
            input_ids: Token ids shaped ``(1, seq_len)``.
            attention_mask: Attention mask shaped ``(1, seq_len)``.
            query_positions: Named query positions.
            region_map: Region name to ``RegionInfo`` mapping.
            target_layers: Layer indices to patch at.
            tokenizer: Tokenizer for decoding.

        Returns:
            List of ``PatchingResult`` dicts, one per (region, layer) pair.
        """
        total_pairs = len(region_map) * len(target_layers)
        logger.info(
            "Patching sweep: %d regions x %d layers = %d pairs",
            len(region_map), len(target_layers), total_pairs,
        )

        # Run baseline once
        print(f"  Running baseline forward pass...")
        baseline = self.run_baseline(input_ids, attention_mask, query_positions)
        baseline_logits = baseline["logits"]
        baseline_residuals = baseline["residuals"]

        # Pick the first query position for effect measurement
        query_pos_name = next(iter(query_positions))
        query_pos_idx = query_positions[query_pos_name]

        results: list[PatchingResult] = []
        completed = 0

        for region_name, region_info in region_map.items():
            for layer_idx in target_layers:
                completed += 1
                if completed % 10 == 0 or completed == total_pairs:
                    print(f"  Patching progress: {completed}/{total_pairs}")

                patched = self.run_patched(
                    input_ids,
                    attention_mask,
                    baseline_residuals,
                    region_name,
                    region_info,
                    layer_idx,
                    query_positions,
                )

                effect = self.compute_effect(
                    baseline_logits,
                    patched["logits"],
                    query_pos_idx,
                    tokenizer,
                )
                effect["region"] = region_name
                effect["layer"] = layer_idx

                results.append(effect)

                del patched
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del baseline_residuals
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Patching sweep complete: {len(results)} results")
        return results
