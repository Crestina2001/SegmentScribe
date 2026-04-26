"""Rule-based punctuation correction for ``slide_rule``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from slicing_utils.asr import CharToken
from slicing_utils.prepass import FullPrepass
from slicing_utils.shared import (
    canonical_nonpunct,
    is_punct_or_space,
    pause_category_after_token,
    token_to_text_positions,
)


@dataclass
class RulePunctuationEdit:
    token_idx: int
    left_char: str
    right_char: Optional[str]
    text_pos: int
    category: str
    pause_ms: float
    threshold_ms: float
    old_punctuation: str
    new_punctuation: str
    action: str
    reason: str


@dataclass
class RulePunctuationResult:
    corrected_full_text: str
    applied_windows: list[RulePunctuationEdit] = field(default_factory=list)
    call_traces: list[dict[str, Any]] = field(default_factory=list)
    corrected_token_to_positions: dict[int, list[int]] = field(default_factory=dict)


def run_rule_punctuation_phase(prepass: FullPrepass) -> RulePunctuationResult:
    """Apply punctuation-only edits from pause statistics.

    The returned object intentionally matches the attributes consumed from
    ``slide_workflow.punctuation.Phase2Result`` by later phases.
    """
    full_text = prepass.full_text or ""
    chars = prepass.global_chars
    token_to_positions = token_to_text_positions(full_text, chars)
    if not full_text or len(chars) < 2:
        return RulePunctuationResult(
            corrected_full_text=full_text,
            corrected_token_to_positions=token_to_positions,
        )

    edits: list[tuple[int, int, str, RulePunctuationEdit]] = []
    strong_short_ms = max(float(prepass.strong.p5_ms), float(prepass.weak.p20_ms))
    weak_short_ms = max(float(prepass.weak.p5_ms), float(prepass.no_punc.p60_ms))
    no_punc_long_ms = max(float(prepass.no_punc.p95_ms), 180.0)

    for token_idx, (left, right) in enumerate(zip(chars, chars[1:])):
        positions = token_to_positions.get(token_idx)
        if not positions:
            continue
        run_start, run_end, punct_run = _punctuation_run_after(full_text, max(positions))
        pause_ms = max(0.0, (float(right.start_sec) - float(left.end_sec)) * 1000.0)
        category = pause_category_after_token(full_text, token_to_positions, token_idx)

        if category == "strong" and pause_ms <= strong_short_ms and punct_run:
            edit = _make_edit(
                token_idx=token_idx,
                left=left,
                right=right,
                text_pos=max(positions),
                category=category,
                pause_ms=pause_ms,
                threshold_ms=strong_short_ms,
                old_punctuation=punct_run,
                new_punctuation="",
                action="remove_punctuation",
                reason="strong punctuation after short silence",
            )
            edits.append((run_start, run_end, "", edit))
        elif category == "weak" and pause_ms <= weak_short_ms and punct_run:
            edit = _make_edit(
                token_idx=token_idx,
                left=left,
                right=right,
                text_pos=max(positions),
                category=category,
                pause_ms=pause_ms,
                threshold_ms=weak_short_ms,
                old_punctuation=punct_run,
                new_punctuation="",
                action="remove_punctuation",
                reason="weak punctuation after short silence",
            )
            edits.append((run_start, run_end, "", edit))
        elif category == "no_punc" and pause_ms >= no_punc_long_ms:
            edit = _make_edit(
                token_idx=token_idx,
                left=left,
                right=right,
                text_pos=max(positions),
                category=category,
                pause_ms=pause_ms,
                threshold_ms=no_punc_long_ms,
                old_punctuation="",
                new_punctuation="，",
                action="add_comma",
                reason="long silence without punctuation",
            )
            edits.append((run_start, run_start, "，", edit))

    corrected_text, applied = _apply_text_edits(full_text, edits)
    if canonical_nonpunct(corrected_text) != canonical_nonpunct(full_text):
        # Defensive guard: rules are punctuation-only, so reject all edits if a
        # future change accidentally alters spoken content.
        corrected_text = full_text
        applied = []

    corrected_positions = token_to_text_positions(corrected_text, chars)
    return RulePunctuationResult(
        corrected_full_text=corrected_text,
        applied_windows=applied,
        call_traces=[_edit_to_trace(edit) for edit in applied],
        corrected_token_to_positions=corrected_positions,
    )


def run_identity_punctuation_phase(prepass: FullPrepass) -> RulePunctuationResult:
    """Return prepass punctuation unchanged."""
    full_text = prepass.full_text or ""
    return RulePunctuationResult(
        corrected_full_text=full_text,
        applied_windows=[],
        call_traces=[],
        corrected_token_to_positions=token_to_text_positions(full_text, prepass.global_chars),
    )


def _punctuation_run_after(full_text: str, text_pos: int) -> tuple[int, int, str]:
    cursor = text_pos + 1
    start = cursor
    while cursor < len(full_text) and is_punct_or_space(full_text[cursor]):
        cursor += 1
    punct_run = full_text[start:cursor]
    if not any(is_punct_or_space(ch) and not ch.isspace() for ch in punct_run):
        punct_run = ""
    return start, cursor, punct_run


def _make_edit(
    *,
    token_idx: int,
    left: CharToken,
    right: Optional[CharToken],
    text_pos: int,
    category: str,
    pause_ms: float,
    threshold_ms: float,
    old_punctuation: str,
    new_punctuation: str,
    action: str,
    reason: str,
) -> RulePunctuationEdit:
    return RulePunctuationEdit(
        token_idx=int(token_idx),
        left_char=str(left.char),
        right_char=str(right.char) if right is not None else None,
        text_pos=int(text_pos),
        category=category,
        pause_ms=round(float(pause_ms), 1),
        threshold_ms=round(float(threshold_ms), 1),
        old_punctuation=old_punctuation,
        new_punctuation=new_punctuation,
        action=action,
        reason=reason,
    )


def _apply_text_edits(
    full_text: str,
    edits: Sequence[tuple[int, int, str, RulePunctuationEdit]],
) -> tuple[str, list[RulePunctuationEdit]]:
    if not edits:
        return full_text, []
    ordered = sorted(edits, key=lambda item: (item[0], item[1]))
    parts: list[str] = []
    applied: list[RulePunctuationEdit] = []
    cursor = 0
    for start, end, replacement, edit in ordered:
        if start < cursor:
            continue
        parts.append(full_text[cursor:start])
        parts.append(replacement)
        cursor = end
        applied.append(edit)
    parts.append(full_text[cursor:])
    return "".join(parts), applied


def _edit_to_trace(edit: RulePunctuationEdit) -> dict[str, Any]:
    return {
        "token_idx": edit.token_idx,
        "left_char": edit.left_char,
        "right_char": edit.right_char,
        "text_pos": edit.text_pos,
        "category": edit.category,
        "pause_ms": edit.pause_ms,
        "threshold_ms": edit.threshold_ms,
        "old_punctuation": edit.old_punctuation,
        "new_punctuation": edit.new_punctuation,
        "action": edit.action,
        "reason": edit.reason,
    }
