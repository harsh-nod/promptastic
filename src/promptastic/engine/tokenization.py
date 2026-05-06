"""Token sequence building and region mapping.

Handles chat template application, character-to-token boundary resolution,
and the construction of a complete token-level region map from character-level
annotations.  All boundary detection uses cumulative prefix decoding and
binary search to stay robust across BPE implementations (SentencePiece,
tiktoken, etc.).
"""

from __future__ import annotations

import bisect
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ======================================================================
# Chat token sequence
# ======================================================================

def build_chat_tokens(
    tokenizer: Any,
    system_prompt: str,
    user_message: str,
    response: str,
) -> tuple[list[int], dict[str, tuple[int, int]]]:
    """Build a token sequence via the model's chat template and locate pieces.

    Applies the chat template with system / user / assistant roles, then
    finds each piece's character span in the decoded text and maps those
    spans back to token indices.

    If the tokenizer does not support a ``system`` role, the system prompt
    is prepended to the user message automatically.

    Returns
    -------
    token_ids : list[int]
        Full token sequence.
    piece_boundaries : dict[str, tuple[int, int]]
        Mapping from piece name (``system_prompt``, ``user_message``,
        ``response``, ``chat_template``) to ``(tok_start, tok_end)``
        half-open intervals.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response},
    ]

    try:
        token_ids = _apply_chat_template(tokenizer, messages)
    except Exception as exc:
        if "system" in str(exc).lower():
            logger.info(
                "Model does not support system role -- merging into user message"
            )
            merged_user = system_prompt + "\n" + user_message
            messages = [
                {"role": "user", "content": merged_user},
                {"role": "assistant", "content": response},
            ]
            token_ids = _apply_chat_template(tokenizer, messages)
        else:
            raise

    # Decode the *full* sequence once.  Per-token decode + concatenation
    # breaks SentencePiece leading-space markers; full-sequence decode
    # does not.
    full_decoded: str = tokenizer.decode(token_ids, skip_special_tokens=False)

    # Locate each content piece via plain string search.
    boundaries: dict[str, tuple[int, int]] = {}
    search_from = 0

    for piece_name, piece_text in [
        ("system_prompt", system_prompt),
        ("user_message", user_message),
        ("response", response),
    ]:
        char_pos = full_decoded.find(piece_text, search_from)
        if char_pos < 0:
            logger.warning("Could not locate '%s' in decoded text", piece_name)
            continue

        char_end = char_pos + len(piece_text)
        tok_start = char_to_token_bisect(tokenizer, token_ids, char_pos)
        tok_end = char_to_token_bisect(tokenizer, token_ids, char_end - 1) + 1
        boundaries[piece_name] = (tok_start, tok_end)
        search_from = char_end

    # chat_template = all token positions not covered by content pieces.
    content_indices: set[int] = set()
    for start, end in boundaries.values():
        content_indices.update(range(start, end))

    template_indices = set(range(len(token_ids))) - content_indices
    if template_indices:
        boundaries["chat_template"] = (
            min(template_indices),
            max(template_indices) + 1,
        )

    return token_ids, boundaries


def build_chat_tokens_multi(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> tuple[list[int], dict[str, tuple[int, int]]]:
    """Build a token sequence from an arbitrary message list.

    Generalises :func:`build_chat_tokens` to support prefix conversation
    turns (e.g. few-shot examples extracted from the system prompt).

    The returned *piece_boundaries* use the following key convention:

    - ``"system_prompt"`` for the system message
    - ``"prefix_turn_0"``, ``"prefix_turn_1"``, ... for prefix turns
    - ``"user_message"`` for the final user message
    - ``"response"`` for the final assistant message
    - ``"chat_template"`` for all non-content token positions
    """
    if not messages:
        return [], {}

    try:
        token_ids = _apply_chat_template(tokenizer, messages)
    except Exception as exc:
        if "system" in str(exc).lower():
            logger.info(
                "Model does not support system role -- merging into first user message"
            )
            merged = list(messages)
            if merged and merged[0]["role"] == "system":
                sys_content = merged.pop(0)["content"]
                if merged and merged[0]["role"] == "user":
                    merged[0] = {
                        "role": "user",
                        "content": sys_content + "\n" + merged[0]["content"],
                    }
                else:
                    merged.insert(0, {"role": "user", "content": sys_content})
            token_ids = _apply_chat_template(tokenizer, merged)
        else:
            raise

    full_decoded: str = tokenizer.decode(token_ids, skip_special_tokens=False)

    # Assign piece names to each message.
    piece_names = _assign_piece_names(messages)

    # Locate each content piece via sequential string search.
    boundaries: dict[str, tuple[int, int]] = {}
    search_from = 0

    for piece_name, msg in zip(piece_names, messages):
        piece_text = msg["content"]
        if not piece_text:
            continue

        char_pos = full_decoded.find(piece_text, search_from)
        if char_pos < 0:
            logger.warning("Could not locate '%s' in decoded text", piece_name)
            continue

        char_end = char_pos + len(piece_text)
        tok_start = char_to_token_bisect(tokenizer, token_ids, char_pos)
        tok_end = char_to_token_bisect(tokenizer, token_ids, char_end - 1) + 1
        boundaries[piece_name] = (tok_start, tok_end)
        search_from = char_end

    # chat_template = positions not covered by any content piece.
    content_indices: set[int] = set()
    for start, end in boundaries.values():
        content_indices.update(range(start, end))

    template_indices = set(range(len(token_ids))) - content_indices
    if template_indices:
        boundaries["chat_template"] = (
            min(template_indices),
            max(template_indices) + 1,
        )

    return token_ids, boundaries


def _assign_piece_names(messages: list[dict[str, str]]) -> list[str]:
    """Assign stable piece names to each message in the list.

    Convention: system → ``system_prompt``, final user → ``user_message``,
    final assistant → ``response``, intermediate → ``prefix_turn_N``.
    """
    names: list[str] = []
    prefix_idx = 0

    # Find the last user and last assistant indices.
    last_user_idx = -1
    last_assistant_idx = -1
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            last_user_idx = i
        elif msg["role"] == "assistant":
            last_assistant_idx = i

    for i, msg in enumerate(messages):
        if msg["role"] == "system":
            names.append("system_prompt")
        elif i == last_user_idx:
            names.append("user_message")
        elif i == last_assistant_idx:
            names.append("response")
        else:
            names.append(f"prefix_turn_{prefix_idx}")
            prefix_idx += 1

    return names


# ======================================================================
# Character-to-token mapping
# ======================================================================

def char_to_token_bisect(
    tokenizer: Any,
    token_ids: list[int],
    target_char: int,
) -> int:
    """Binary search for the token index whose decoded span contains *target_char*.

    Uses progressive prefix decoding: ``tokenizer.decode(token_ids[:n])``
    gives the text through token ``n - 1``.  The search finds the smallest
    ``n`` where the decoded prefix length exceeds ``target_char``.
    """
    lo, hi = 0, len(token_ids) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        prefix_len = len(
            tokenizer.decode(token_ids[: mid + 1], skip_special_tokens=False)
        )
        if prefix_len <= target_char:
            lo = mid + 1
        else:
            hi = mid
    return lo


def resolve_char_regions_to_tokens(
    tokenizer: Any,
    token_ids: list[int],
    char_regions: dict[str, dict[str, int]],
    global_char_offset: int,
) -> dict[str, dict[str, int]]:
    """Map character-level region boundaries to token positions.

    Builds a cumulative character-length table by decoding each token
    individually, then uses ``bisect`` to translate character offsets
    into token indices.  This avoids BPE boundary issues that plague
    subsequence-matching approaches on long regions.

    Parameters
    ----------
    tokenizer:
        HuggingFace tokenizer.
    token_ids:
        Token IDs for the *piece* (not the full sequence).
    char_regions:
        ``{region_name: {"char_start": int, "char_end": int}}``.
    global_char_offset:
        Token index offset of this piece within the full sequence.

    Returns
    -------
    Resolved regions as ``{name: {"tok_start", "tok_end", "n_tokens"}}``.
    """
    # Build cumulative character lengths.
    cum_chars: list[int] = [0]
    for tid in token_ids:
        tok_text = tokenizer.decode([tid])
        cum_chars.append(cum_chars[-1] + len(tok_text))

    total_decoded = cum_chars[-1]

    def _char_to_tok(char_pos: int) -> int:
        idx = bisect.bisect_right(cum_chars, char_pos) - 1
        return max(0, min(idx, len(token_ids) - 1))

    resolved: dict[str, dict[str, int]] = {}

    for region_name, bounds in char_regions.items():
        char_start: int = bounds["char_start"]
        char_end: int = bounds["char_end"]

        tok_start_local = _char_to_tok(char_start)
        tok_end_local = (
            _char_to_tok(char_end - 1) + 1 if char_end > char_start else tok_start_local
        )

        tok_start_global = global_char_offset + tok_start_local
        tok_end_global = global_char_offset + tok_end_local
        n_tokens = tok_end_local - tok_start_local

        if n_tokens <= 0:
            logger.warning("Region '%s' resolved to 0 tokens -- skipping", region_name)
            continue

        resolved[region_name] = {
            "tok_start": tok_start_global,
            "tok_end": tok_end_global,
            "n_tokens": n_tokens,
        }

    if total_decoded > 0:
        logger.debug("Cumulative decode length: %d chars", total_decoded)

    return resolved


# ======================================================================
# Full region map
# ======================================================================

def build_full_region_map(
    tokenizer: Any,
    token_ids: list[int],
    piece_boundaries: dict[str, tuple[int, int]],
    system_prompt: str,
    user_message: str,
    response: str,
    system_char_regions: dict[str, dict[str, int]] | None = None,
    user_char_regions: dict[str, dict[str, int]] | None = None,
    response_char_regions: dict[str, dict[str, int]] | None = None,
) -> dict[str, dict[str, int]]:
    """Assemble a complete token-level region map from character annotations.

    Includes top-level pieces (``system_prompt``, ``user_message``,
    ``response``, ``chat_template``, and any ``prefix_turn_*`` pieces)
    and all character-level sub-regions within each piece.

    Returns
    -------
    ``{region_name: {"tok_start": int, "tok_end": int, "n_tokens": int}}``
    """
    region_map: dict[str, dict[str, int]] = {}

    # -- Top-level pieces (including any prefix_turn_* boundaries) --
    for piece_name, (start, end) in piece_boundaries.items():
        region_map[piece_name] = {
            "tok_start": start,
            "tok_end": end,
            "n_tokens": end - start,
        }

    # -- Sub-regions within each piece --
    _resolve_piece_subregions(
        tokenizer, token_ids, piece_boundaries,
        "system_prompt", system_char_regions, region_map,
    )
    _resolve_piece_subregions(
        tokenizer, token_ids, piece_boundaries,
        "user_message", user_char_regions, region_map,
    )
    _resolve_piece_subregions(
        tokenizer, token_ids, piece_boundaries,
        "response", response_char_regions, region_map,
    )

    return region_map


# ======================================================================
# Query position resolution
# ======================================================================

def resolve_query_positions(
    tokenizer: Any,
    token_ids: list[int],
    piece_boundaries: dict[str, tuple[int, int]],
    position_defs: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Resolve named query positions within the token sequence.

    Built-in positions
    ------------------
    ``terminal``
        Last token of the user message.

    Extra positions from *position_defs*
    -------------------------------------
    ``"last_token"``
        Alias for the last user-message token.
    ``{"after_text": "..."}``
        First non-whitespace token after the given text in the response.
    ``{"at_text": "..."}``
        Token *at* the given text in the response.
    """
    positions: dict[str, int] = {}

    usr_piece = piece_boundaries.get("user_message")
    if usr_piece is not None:
        positions["terminal"] = usr_piece[1] - 1

    if not position_defs:
        return positions

    resp_piece = piece_boundaries.get("response")
    resp_start = resp_piece[0] if resp_piece else 0
    resp_end = resp_piece[1] if resp_piece else len(token_ids)

    for pos_name, pos_def in position_defs.items():
        if pos_name == "terminal":
            continue  # already handled above

        if pos_def == "last_token":
            if usr_piece is not None:
                positions[pos_name] = usr_piece[1] - 1
            continue

        if not isinstance(pos_def, dict):
            continue

        if "after_text" in pos_def:
            text: str = pos_def["after_text"]
            text_tokens = tokenizer.encode(text, add_special_tokens=False)
            resp_tokens = token_ids[resp_start:resp_end]
            idx = _find_subsequence(resp_tokens, text_tokens)
            if idx < 0:
                logger.warning(
                    "Could not find '%s' tokens in response for position '%s'",
                    text,
                    pos_name,
                )
                continue
            content_start = resp_start + idx + len(text_tokens)
            # Skip whitespace-only tokens.
            while content_start < resp_end:
                tok_text = tokenizer.decode([token_ids[content_start]]).strip()
                if tok_text:
                    break
                content_start += 1
            if content_start < resp_end:
                positions[pos_name] = content_start

        elif "at_text" in pos_def:
            text = pos_def["at_text"]
            text_tokens = tokenizer.encode(text, add_special_tokens=False)
            resp_tokens = token_ids[resp_start:resp_end]
            idx = _find_subsequence(resp_tokens, text_tokens)
            if idx >= 0:
                positions[pos_name] = resp_start + idx
            else:
                logger.warning(
                    "Could not find '%s' tokens in response for position '%s'",
                    text,
                    pos_name,
                )

    return positions


# ======================================================================
# Private helpers
# ======================================================================

def _apply_chat_template(tokenizer: Any, messages: list[dict]) -> list[int]:
    """Apply the tokenizer's chat template and return a plain list of ints."""
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    # Some tokenizers return a dict with an ``input_ids`` key.
    if hasattr(result, "keys"):
        ids = result["input_ids"]
        # Handle batched output (list of lists) or Encoding objects.
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        elif not isinstance(ids, list):
            ids = list(ids)
    else:
        ids = list(result)

    # Ensure plain ints (some tokenizers return numpy int64).
    if ids and not isinstance(ids[0], int):
        ids = [int(x) for x in ids]

    return ids


def _resolve_piece_subregions(
    tokenizer: Any,
    token_ids: list[int],
    piece_boundaries: dict[str, tuple[int, int]],
    piece_name: str,
    char_regions: dict[str, dict[str, int]] | None,
    region_map: dict[str, dict[str, int]],
) -> None:
    """Resolve character-level sub-regions for a single piece into *region_map*."""
    piece = piece_boundaries.get(piece_name)
    if piece is None or not char_regions:
        return

    piece_start, piece_end = piece
    piece_tokens = token_ids[piece_start:piece_end]

    resolved = resolve_char_regions_to_tokens(
        tokenizer, piece_tokens, char_regions, piece_start
    )
    region_map.update(resolved)


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Return the index of *needle* in *haystack*, or -1 if not found.

    Simple linear scan -- token sequences are short enough that this is
    not a bottleneck.
    """
    if not needle:
        return 0
    needle_len = len(needle)
    limit = len(haystack) - needle_len + 1
    for i in range(limit):
        if haystack[i : i + needle_len] == needle:
            return i
    return -1
