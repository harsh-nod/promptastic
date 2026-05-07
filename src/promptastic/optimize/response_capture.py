"""Optional response capture for the meta-rewriter.

Generates actual model responses for a set of conversations so the
meta-rewriter can assess qualitative output changes across prompt
iterations.  This is separate from
:class:`~promptastic.engine.generation.GenerationTracker` because it
does not need per-step attention/logit-lens tracking — just raw text.
"""

from __future__ import annotations

from typing import Any

from ._types import PromptSpec


def capture_responses(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    spec: PromptSpec,
    conversations: list[dict[str, Any]],
    max_new_tokens: int = 64,
    max_cases: int = 2,
) -> list[str]:
    """Generate model responses for a subset of conversations.

    Parameters
    ----------
    model:
        Loaded HuggingFace model.
    tokenizer:
        Corresponding tokenizer.
    adapter:
        ModelAdapter instance.
    spec:
        Current prompt specification (system prompt + optional prefix turns).
    conversations:
        Full list of conversation dicts (each with ``"user_message"``).
    max_new_tokens:
        Maximum tokens to generate per response.
    max_cases:
        Number of conversations to sample (from the front of the list).

    Returns
    -------
    list[str]
        Generated response strings, one per sampled conversation.
    """
    import torch

    if not conversations:
        return []

    cases = conversations[:max_cases]
    responses: list[str] = []

    for case in cases:
        user_message = case.get("user_message", "")
        if not user_message:
            continue

        messages = spec.to_messages(user_message)

        # Tokenize using the chat template
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0, input_len:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response_text.strip())

    return responses
