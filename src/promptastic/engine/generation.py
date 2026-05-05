"""Cross-step generation tracking with per-step attention and logit lens capture.

Implements autoregressive generation with hooks that capture, at each
decoding step, which prompt regions the model is attending to and what the
logit lens predicts at intermediate layers.  This reveals how the model's
attention distribution over the prompt evolves as it generates each new token
-- a perspective that single-forward-pass analysis cannot provide.

Only per-step summaries are retained; raw attention matrices and residual
tensors are discarded immediately to keep memory bounded.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from .._types import GenerationStepData, RegionInfo

logger = logging.getLogger(__name__)


class GenerationTracker:
    """Autoregressive generation with per-step attention and logit lens capture.

    At each decoding step the tracker:
    1. Registers attention hooks to capture per-region weights at the last
       (newly generated) position.
    2. Registers residual hooks for logit lens projection at the last position.
    3. Runs a single forward step with KV cache.
    4. Extracts summaries, removes hooks, and appends the next token.

    Only compact per-step summaries are stored -- never raw attention matrices
    or full residual tensors.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        adapter: object,
        tokenizer: object,
        max_new_tokens: int = 32,
    ) -> None:
        """Initialize the generation tracker.

        Args:
            model: A loaded HuggingFace causal LM.
            adapter: A ``ModelAdapter`` providing architecture accessors.
            tokenizer: The corresponding tokenizer.
            max_new_tokens: Maximum number of tokens to generate.
        """
        self.model = model
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def generate_with_tracking(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        region_map: dict[str, RegionInfo],
    ) -> dict:
        """Generate tokens autoregressively, capturing per-step diagnostics.

        Args:
            input_ids: Prompt token ids shaped ``(1, seq_len)``.
            attention_mask: Attention mask shaped ``(1, seq_len)``.
            region_map: Region name to ``RegionInfo`` mapping for the prompt.

        Returns:
            Dictionary with:
            - ``"generated_text"``: the full decoded generation.
            - ``"generated_tokens"``: list of individual token strings.
            - ``"steps"``: list of ``GenerationStepData`` dicts.
        """
        device = input_ids.device
        num_layers = self.adapter.num_layers
        attention_modules = self.adapter.get_attention_modules()
        layer_modules = self.adapter.get_layer_modules()
        norm = self.adapter.get_norm()
        lm_head = self.adapter.get_lm_head()
        model_dtype = next(self.model.parameters()).dtype

        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()
        past_key_values = None

        generated_token_ids: list[int] = []
        generated_tokens: list[str] = []
        steps: list[GenerationStepData] = []

        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        for step_idx in range(self.max_new_tokens):
            seq_len = current_ids.shape[1]
            query_pos = seq_len - 1

            # -- attention hooks ------------------------------------------
            step_region_attn: dict[str, float] = {}

            attn_handles: list = []
            for layer_idx, attn_mod in attention_modules:

                def _attn_hook(module, inp, output, _region_attn=step_region_attn):
                    if len(output) < 2 or output[1] is None:
                        return output
                    A = output[1]
                    if A.dim() == 3:
                        A = A.unsqueeze(0)
                    # Last query position, averaged across heads
                    last_pos = A.shape[2] - 1
                    row = A[0, :, last_pos, :].float().mean(dim=0)
                    for rname, rinfo in region_map.items():
                        ts = rinfo["tok_start"]
                        te = rinfo["tok_end"]
                        if te <= row.shape[0]:
                            w = row[ts:te].sum().item()
                            _region_attn[rname] = _region_attn.get(rname, 0.0) + w
                    return (output[0], None) + output[2:]

                attn_handles.append(attn_mod.register_forward_hook(_attn_hook))

            # -- residual hooks for logit lens ----------------------------
            last_layer_residual: dict[str, torch.Tensor] = {}

            resid_handles: list = []
            # Only capture a few layers for the logit lens summary to save memory
            sample_layers = _sample_layer_indices(num_layers, max_samples=8)
            for layer_idx, layer_mod in layer_modules:
                if layer_idx not in sample_layers:
                    continue

                def _resid_hook(module, inp, output, idx=layer_idx):
                    h = output[0]
                    if h.dim() == 2:
                        h = h.unsqueeze(0)
                    last_layer_residual[str(idx)] = h[0, -1, :].detach().float().cpu()

                resid_handles.append(layer_mod.register_forward_hook(_resid_hook))

            # -- forward step ---------------------------------------------
            with torch.no_grad():
                if past_key_values is not None:
                    # Only feed the last token when using KV cache
                    step_input = current_ids[:, -1:]
                    step_mask = current_mask
                else:
                    step_input = current_ids
                    step_mask = current_mask

                outputs = self.model(
                    input_ids=step_input,
                    attention_mask=step_mask,
                    output_attentions=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )

            past_key_values = outputs.past_key_values

            # -- remove hooks immediately ---------------------------------
            for h in attn_handles:
                h.remove()
            for h in resid_handles:
                h.remove()

            # -- greedy sample --------------------------------------------
            next_logits = outputs.logits[0, -1, :].float()
            next_token_id = next_logits.argmax().item()
            next_token_str = self.tokenizer.decode([next_token_id])

            # -- logit lens top-k from sampled layers ---------------------
            logit_lens_top: list[dict] = []
            for lidx_str in sorted(last_layer_residual.keys(), key=int):
                h = last_layer_residual[lidx_str]
                with torch.no_grad():
                    lm_device = next(lm_head.parameters()).device
                    h_proj = norm(h.unsqueeze(0).unsqueeze(0).to(device=lm_device, dtype=model_dtype))
                    proj_logits = lm_head(h_proj).squeeze().float()
                    probs = F.softmax(proj_logits, dim=-1)

                top_vals, top_idxs = torch.topk(proj_logits, 5)
                top_probs = probs[top_idxs]

                tokens_at_layer = []
                for tok_id, lval, pval in zip(
                    top_idxs.tolist(), top_vals.tolist(), top_probs.tolist()
                ):
                    tokens_at_layer.append(
                        {
                            "token": self.tokenizer.decode([tok_id]),
                            "token_id": tok_id,
                            "logit": round(lval, 4),
                            "prob": round(pval, 6),
                        }
                    )

                logit_lens_top.append(
                    {"layer": int(lidx_str), "top_tokens": tokens_at_layer}
                )

            # -- normalize region attention (average across layers) -------
            if attention_modules:
                n_attn_layers = len(attention_modules)
                for rname in step_region_attn:
                    step_region_attn[rname] = round(
                        step_region_attn[rname] / n_attn_layers, 6
                    )

            # -- record step ----------------------------------------------
            step_data = GenerationStepData(
                step=step_idx,
                generated_token=next_token_str,
                generated_token_id=next_token_id,
                region_attention=step_region_attn,
                logit_lens_top=logit_lens_top,
            )
            steps.append(step_data)

            generated_token_ids.append(next_token_id)
            generated_tokens.append(next_token_str)

            # -- update ids and mask for next step ------------------------
            next_id_tensor = torch.tensor([[next_token_id]], device=device)
            current_ids = torch.cat([current_ids, next_id_tensor], dim=1)
            current_mask = torch.cat(
                [current_mask, torch.ones(1, 1, device=device, dtype=current_mask.dtype)],
                dim=1,
            )

            # -- cleanup --------------------------------------------------
            del outputs, last_layer_residual, next_logits
            gc.collect()

            # -- early stop on EOS ----------------------------------------
            if eos_id is not None and next_token_id == eos_id:
                logger.info("EOS token generated at step %d", step_idx)
                break

        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        logger.info(
            "Generation complete: %d tokens, text=%r",
            len(generated_tokens),
            generated_text[:80],
        )

        return {
            "generated_text": generated_text,
            "generated_tokens": generated_tokens,
            "steps": steps,
        }


def _sample_layer_indices(num_layers: int, max_samples: int = 8) -> set[int]:
    """Pick evenly spaced layer indices plus the first and last layers.

    Keeps logit lens capture lightweight during generation by only
    sampling a handful of representative layers.
    """
    if num_layers <= max_samples:
        return set(range(num_layers))
    step = max(1, num_layers // (max_samples - 1))
    indices = set(range(0, num_layers, step))
    indices.add(num_layers - 1)  # always include final layer
    return indices
