"""Shared helpers for ``slide_workflow`` phases.

Centralizes the text/char projection logic that is reused across phases so
the individual phase modules stay focused on their own LLM prompting and
decision making.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Optional, Sequence

from .asr import CharToken

STRONG_STOPS = frozenset("。？！.?!")
WEAK_STOPS = frozenset("，；、,;:")


def is_punct_or_space(ch: str) -> bool:
    """True for whitespace and any Unicode punctuation / symbol."""
    if ch.isspace():
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def canonical_nonpunct(text: str) -> str:
    return "".join(ch for ch in text if not is_punct_or_space(ch))


def token_to_text_positions(
    full_text: str,
    tokens: Sequence[CharToken],
) -> dict[int, list[int]]:
    """Map aligner token idx -> list of positions in ``full_text``.

    Punctuation/whitespace positions in ``full_text`` are skipped because
    aligner tokens only carry spoken characters. The mapping is order-preserving:
    the N-th non-punct text char is paired with the N-th non-punct token char.
    """
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


def pause_category_after_token(
    full_text: str,
    token_to_positions: dict[int, list[int]],
    token_idx: int,
) -> str:
    """Return ``'strong' | 'weak' | 'no_punc'`` for the pause after ``token_idx``."""
    positions = token_to_positions.get(token_idx)
    if not positions:
        return "no_punc"
    cursor = max(positions) + 1
    punctuation_run: list[str] = []
    while cursor < len(full_text):
        ch = full_text[cursor]
        if is_punct_or_space(ch):
            punctuation_run.append(ch)
            cursor += 1
            continue
        break
    for ch in punctuation_run:
        if ch in STRONG_STOPS:
            return "strong"
        if ch in WEAK_STOPS:
            return "weak"
    return "no_punc"


@dataclass(frozen=True)
class FirstMismatch:
    """Describe the first diverging character between two canonical strings."""

    index: int
    original_char: Optional[str] = None
    revised_char: Optional[str] = None
    context_original: Optional[str] = None
    context_revised: Optional[str] = None
    length_original: Optional[int] = None
    length_revised: Optional[int] = None
    extra_tail: Optional[str] = None


def first_mismatch(a: str, b: str) -> FirstMismatch:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return FirstMismatch(
                index=i,
                original_char=a[i],
                revised_char=b[i],
                context_original=a[max(0, i - 5) : i + 6],
                context_revised=b[max(0, i - 5) : i + 6],
            )
    if len(a) != len(b):
        return FirstMismatch(
            index=n,
            length_original=len(a),
            length_revised=len(b),
            extra_tail=(a if len(a) > len(b) else b)[n:],
        )
    return FirstMismatch(index=-1)
