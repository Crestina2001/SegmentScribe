"""Text classification rules local to ``slide_LLM``."""

from __future__ import annotations

import unicodedata
from typing import Sequence

from slicing_utils.asr import CharToken


NON_PUNCT_BRACKETS = frozenset("<>()\u300a\u300b\uff08\uff09")


def is_punct_or_space(ch: str) -> bool:
    """True for whitespace and punctuation, excluding slide_LLM text brackets."""
    if ch.isspace():
        return True
    if ch in NON_PUNCT_BRACKETS:
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def canonical_nonpunct(text: str) -> str:
    return "".join(ch for ch in text if not is_punct_or_space(ch))


def token_to_text_positions(
    full_text: str,
    tokens: Sequence[CharToken],
) -> dict[int, list[int]]:
    if not full_text or not tokens:
        return {}

    text_units = [
        (pos, ch)
        for pos, ch in enumerate(full_text)
        if not is_punct_or_space(ch)
    ]
    token_units: list[tuple[int, str]] = []
    for token_idx, token in enumerate(tokens):
        for ch in str(token.char):
            if not is_punct_or_space(ch):
                token_units.append((token_idx, ch))

    mapping: dict[int, list[int]] = {}
    for unit_idx, (token_idx, _ch) in enumerate(token_units):
        if unit_idx >= len(text_units):
            break
        mapping.setdefault(token_idx, []).append(text_units[unit_idx][0])
    return mapping
