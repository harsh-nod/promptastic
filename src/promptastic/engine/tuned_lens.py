"""Tuned lens -- per-layer affine probes for cleaner intermediate predictions.

Implements the tuned lens technique from Belrose et al. (2023).  Instead of
projecting raw residual stream states through the final layer norm and LM head
(standard logit lens), a learned affine transformation is applied at each
layer first.  This compensates for the fact that intermediate residuals were
never "meant" to be read by the final unembedding, producing sharper and more
interpretable per-layer predictions.

Training fits one ``nn.Linear(hidden_size, hidden_size)`` probe per layer,
minimizing cross-entropy between ``lm_head(probe(residual))`` and the true
next-token targets.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TunedLensProbe(nn.Module):
    """Single-layer affine probe mapping hidden states to a corrected space.

    Wraps a standard ``nn.Linear`` that transforms a residual stream vector
    before it is projected through the final norm and LM head.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize close to identity so untrained probes are near logit-lens
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the affine probe.

        Args:
            h: Hidden state tensor, shape ``(..., hidden_size)``.

        Returns:
            Transformed tensor of the same shape.
        """
        return self.linear(h)


class TunedLens:
    """Collection of per-layer probes plus training and projection routines.

    After training, each probe can be applied to a cached residual stream
    vector before the standard norm+lm_head projection, yielding per-layer
    token predictions that are significantly less noisy than vanilla logit
    lens.
    """

    def __init__(self, num_layers: int, hidden_size: int) -> None:
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.probes: dict[int, TunedLensProbe] = {
            layer: TunedLensProbe(hidden_size) for layer in range(num_layers)
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_probes(
        self,
        model: nn.Module,
        adapter: object,
        tokenizer: object,
        texts: list[str],
        epochs: int = 3,
        lr: float = 1e-3,
        batch_size: int = 4,
    ) -> dict:
        """Train all probes to minimize next-token prediction loss.

        For each text, runs a forward pass through the full model while
        caching residual stream outputs at every layer.  Then trains each
        layer's probe so that ``lm_head(norm(probe(residual)))`` predicts
        the correct next token.

        Args:
            model: The loaded HuggingFace model.
            adapter: A ``ModelAdapter`` providing architecture accessors.
            tokenizer: The corresponding tokenizer.
            texts: Training corpus as a list of plain strings.
            epochs: Number of training epochs over the corpus.
            lr: Learning rate for Adam.
            batch_size: Number of texts to process before each optimizer step.

        Returns:
            Dictionary with ``"per_layer_loss"`` -- a list of final training
            losses, one per layer.
        """
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        norm = adapter.get_norm()
        lm_head = adapter.get_lm_head()
        layer_modules = adapter.get_layer_modules()

        # Move probes to device
        for probe in self.probes.values():
            probe.to(device=device, dtype=torch.float32)
            probe.train()

        optimizers = {
            layer: torch.optim.Adam(self.probes[layer].parameters(), lr=lr)
            for layer in range(self.num_layers)
        }

        per_layer_losses: list[float] = [0.0] * self.num_layers

        for epoch in range(epochs):
            epoch_losses = [0.0] * self.num_layers
            n_batches = 0

            for text_idx in range(0, len(texts), batch_size):
                batch_texts = texts[text_idx : text_idx + batch_size]
                n_batches += 1

                for text in batch_texts:
                    tokens = tokenizer.encode(text, add_special_tokens=True)
                    if len(tokens) < 2:
                        continue

                    input_ids = torch.tensor([tokens], device=device)
                    target_ids = input_ids[:, 1:].contiguous()

                    # Collect residuals
                    residuals: dict[int, torch.Tensor] = {}
                    handles = []
                    for layer_idx, layer_mod in layer_modules:

                        def _hook(mod, inp, out, idx=layer_idx):
                            h = out[0]
                            if h.dim() == 2:
                                h = h.unsqueeze(0)
                            residuals[idx] = h[:, :-1, :].detach().float()

                        handles.append(layer_mod.register_forward_hook(_hook))

                    with torch.no_grad():
                        model(
                            input_ids=input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            output_attentions=False,
                            use_cache=False,
                        )

                    for h in handles:
                        h.remove()

                    # Train each probe
                    for layer_idx in range(self.num_layers):
                        resid = residuals.get(layer_idx)
                        if resid is None:
                            continue

                        probe = self.probes[layer_idx]
                        optimizer = optimizers[layer_idx]

                        probed = probe(resid.to(device))
                        with torch.no_grad():
                            normed = norm(probed.to(model_dtype))
                            logits = lm_head(normed).float()

                        # Recompute with grad for the probe path only
                        probed_grad = probe(resid.to(device))
                        normed_grad = norm(probed_grad.to(model_dtype))
                        logits_grad = lm_head(normed_grad).float()
                        logits_flat = logits_grad.view(-1, logits_grad.size(-1))
                        targets_flat = target_ids.view(-1)
                        loss = F.cross_entropy(logits_flat, targets_flat)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_losses[layer_idx] += loss.item()

                    del residuals, input_ids, target_ids
                    gc.collect()

            # Average losses
            for layer_idx in range(self.num_layers):
                avg = epoch_losses[layer_idx] / max(n_batches * batch_size, 1)
                per_layer_losses[layer_idx] = round(avg, 6)

            logger.info(
                "Epoch %d/%d -- mean loss across layers: %.4f",
                epoch + 1,
                epochs,
                sum(per_layer_losses) / max(len(per_layer_losses), 1),
            )

        # Move probes back to CPU and eval mode
        for probe in self.probes.values():
            probe.eval()
            probe.cpu()

        return {"per_layer_loss": per_layer_losses}

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(
        self,
        residual_cache: dict[int, torch.Tensor],
        norm: nn.Module,
        lm_head: nn.Module,
        query_positions: dict[str, int],
        num_layers: int,
        tokenizer: object,
        top_k: int = 50,
        tracked_token_ids: Optional[dict[str, int]] = None,
    ) -> dict[str, list[dict]]:
        """Project cached residuals through tuned probes, norm, and LM head.

        Produces the same output format as the standard logit lens so that
        downstream renderers and analysis can consume either interchangeably.

        Args:
            residual_cache: Mapping from ``(layer_idx, pos_name)`` to tensors,
                keyed as ``{layer_idx: tensor}`` per position, or a
                ``ResidualCache``-like object with a ``.get(layer, pos)``
                method.
            norm: The model's final layer norm.
            lm_head: The model's unembedding / LM head.
            query_positions: Named query positions.
            num_layers: Total number of model layers.
            tokenizer: Tokenizer for decoding token ids.
            top_k: Number of top tokens to return per layer.
            tracked_token_ids: Optional mapping of token string to id for
                tracked-token rank reporting.

        Returns:
            Dictionary mapping position name to a list of per-layer dicts,
            each containing ``"layer"``, ``"top_k"``, and ``"tracked"`` keys.
        """
        if tracked_token_ids is None:
            tracked_token_ids = {}

        device = next(lm_head.parameters()).device
        model_dtype = next(lm_head.parameters()).dtype

        results: dict[str, list[dict]] = {}

        for pos_name in query_positions:
            layer_results: list[dict] = []

            for layer_idx in range(num_layers):
                # Support both dict-of-tensors and ResidualCache objects
                if hasattr(residual_cache, "get") and callable(residual_cache.get):
                    h = residual_cache.get(layer_idx, pos_name)
                else:
                    h = None

                if h is None:
                    continue

                probe = self.probes.get(layer_idx)
                if probe is None:
                    continue

                with torch.no_grad():
                    probe_device = probe.to(device)
                    h_probed = probe_device(h.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32))
                    h_normed = norm(h_probed.to(model_dtype))
                    logits = lm_head(h_normed).squeeze().float()
                    probs = F.softmax(logits, dim=-1)
                    probe_device.cpu()

                topk_logits, topk_indices = torch.topk(logits, top_k)
                topk_probs = probs[topk_indices]

                top_k_list = []
                for rank, (tok_id, logit_val, prob_val) in enumerate(
                    zip(
                        topk_indices.tolist(),
                        topk_logits.tolist(),
                        topk_probs.tolist(),
                    )
                ):
                    tok_str = tokenizer.decode([tok_id])
                    top_k_list.append(
                        {
                            "token": tok_str,
                            "token_id": tok_id,
                            "logit": round(logit_val, 4),
                            "prob": round(prob_val, 6),
                            "rank": rank + 1,
                        }
                    )

                tracked: dict[str, dict] = {}
                for tok_str, tok_id in tracked_token_ids.items():
                    tok_logit = logits[tok_id].item()
                    tok_prob = probs[tok_id].item()
                    tok_rank = int((logits > tok_logit).sum().item()) + 1
                    tracked[tok_str] = {
                        "token_id": tok_id,
                        "logit": round(tok_logit, 4),
                        "prob": round(tok_prob, 6),
                        "rank": tok_rank,
                    }

                layer_results.append(
                    {
                        "layer": layer_idx,
                        "top_k": top_k_list,
                        "tracked": tracked,
                    }
                )

            results[pos_name] = layer_results

        return results

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, dir_path: str | Path) -> None:
        """Persist all probes to disk.

        Each probe is saved as ``probe_L{idx}.pt`` inside *dir_path*.
        A ``meta.json`` records ``num_layers`` and ``hidden_size``.

        Args:
            dir_path: Directory to write probe files into (created if needed).
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        meta = {"num_layers": self.num_layers, "hidden_size": self.hidden_size}
        with open(dir_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        for layer_idx, probe in self.probes.items():
            torch.save(probe.state_dict(), dir_path / f"probe_L{layer_idx}.pt")

        logger.info("Saved %d probes to %s", len(self.probes), dir_path)

    @classmethod
    def load(cls, dir_path: str | Path) -> "TunedLens":
        """Load probes from a directory previously written by ``save``.

        Args:
            dir_path: Directory containing ``meta.json`` and probe files.

        Returns:
            A ``TunedLens`` instance with all probes loaded.
        """
        dir_path = Path(dir_path)

        with open(dir_path / "meta.json") as f:
            meta = json.load(f)

        instance = cls(
            num_layers=meta["num_layers"],
            hidden_size=meta["hidden_size"],
        )

        for layer_idx in range(instance.num_layers):
            probe_path = dir_path / f"probe_L{layer_idx}.pt"
            if probe_path.exists():
                state = torch.load(probe_path, map_location="cpu", weights_only=True)
                instance.probes[layer_idx].load_state_dict(state)

        logger.info("Loaded %d probes from %s", len(instance.probes), dir_path)
        return instance


# ======================================================================
# CLI entry point
# ======================================================================


def main() -> None:
    """Command-line interface for training tuned lens probes."""
    parser = argparse.ArgumentParser(
        description="Train tuned lens probes for a HuggingFace model",
    )
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a JSON file containing a list of training strings",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for trained probe weights",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    if not args.train:
        parser.error("--train flag is required to run training")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Lazy import to keep module importable without model_adapter on path
    from ..engine import _resolve_adapter

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model.eval()

    adapter = _resolve_adapter(model)

    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset) as f:
        texts = json.load(f)
    if not isinstance(texts, list):
        raise ValueError("Dataset JSON must be a list of strings")

    num_layers = adapter.num_layers
    hidden_size = adapter.hidden_size

    print(f"Training tuned lens: {num_layers} layers, hidden_size={hidden_size}")
    lens = TunedLens(num_layers, hidden_size)
    stats = lens.train_probes(
        model,
        adapter,
        tokenizer,
        texts,
        epochs=args.epochs,
        lr=args.lr,
    )

    lens.save(args.output)
    print(f"Training complete. Per-layer losses: {stats['per_layer_loss']}")
    print(f"Probes saved to {args.output}")


if __name__ == "__main__":
    main()
