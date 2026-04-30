"""Phase 2 - LLM punctuation correction for ``slide_LLM``."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from llm_gateway import MemoryManager, UnifiedClient
from llm_gateway.models import ProviderName
from slicing_utils.prepass import FullPrepass, compact_pause_stats
from slicing_utils.shared import (
    STRONG_STOPS,
    first_mismatch,
)
from slicing_utils.waveform_viz import WarningSpan

from .local_prompts import PUNCTUATION_SYSTEM_PROMPT, build_punctuation_prompt, build_punctuation_retry_prompt
from .text_rules import canonical_nonpunct, is_punct_or_space, token_to_text_positions


PUNCT_RETRY_LIMIT = 3
EXTRA_STRONG_STOPS = frozenset(".!?。？！")


@dataclass
class LLMPunctuationWindow:
    window_index: int
    token_start_idx: int
    token_end_idx: int
    text_start: int
    text_end: int
    original_text: str
    revised_text: str
    warnings: list[dict[str, Any]]
    attempt_count: int


@dataclass
class LLMPunctuationResult:
    corrected_full_text: str
    applied_windows: list[LLMPunctuationWindow] = field(default_factory=list)
    call_traces: list[dict[str, Any]] = field(default_factory=list)
    corrected_token_to_positions: dict[int, list[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class SentenceSpan:
    token_start_idx: int
    token_end_idx: int
    text_start: int
    text_end: int


class PunctuationValidationError(ValueError):
    pass


async def run_llm_punctuation_phase(
    prepass: FullPrepass,
    *,
    llm_client: UnifiedClient,
    model: str,
    provider: ProviderName | None = None,
) -> LLMPunctuationResult:
    full_text = prepass.full_text or ""
    base_mapping = token_to_text_positions(full_text, prepass.global_chars)
    if not full_text or not prepass.global_chars or not prepass.warnings:
        return LLMPunctuationResult(
            corrected_full_text=full_text,
            corrected_token_to_positions=base_mapping,
        )

    windows = group_warning_windows(prepass)
    replacements: list[tuple[int, int, str, LLMPunctuationWindow]] = []
    call_traces: list[dict[str, Any]] = []
    for index, span in enumerate(windows, start=1):
        original_text = full_text[span.text_start:span.text_end]
        token_to_positions_map = token_to_text_positions(full_text, prepass.global_chars)
        warning_dicts = [
            _warning_to_dict(w, prepass=prepass, token_to_positions_map=token_to_positions_map)
            for w in prepass.warnings
            if span.token_start_idx <= int(w.left_idx) <= span.token_end_idx
            or span.token_start_idx <= int(w.right_idx) <= span.token_end_idx
        ]
        marked_text = _insert_warning_markers(
            original_text,
            text_start=span.text_start,
            warnings=warning_dicts,
        )
        revised_text, attempts = await _request_revision(
            llm_client=llm_client,
            model=model,
            provider=provider,
            original_text=original_text,
            marked_text=marked_text,
            stats=_punctuation_pause_stats(prepass),
            window_index=index,
        )
        window = LLMPunctuationWindow(
            window_index=index,
            token_start_idx=span.token_start_idx,
            token_end_idx=span.token_end_idx,
            text_start=span.text_start,
            text_end=span.text_end,
            original_text=original_text,
            revised_text=revised_text,
            warnings=warning_dicts,
            attempt_count=attempts,
        )
        replacements.append((span.text_start, span.text_end, revised_text, window))
        call_traces.append(dataclasses.asdict(window))

    corrected_text, applied = _apply_replacements(full_text, replacements)
    corrected_mapping = token_to_text_positions(corrected_text, prepass.global_chars)
    return LLMPunctuationResult(
        corrected_full_text=corrected_text,
        applied_windows=applied,
        call_traces=call_traces,
        corrected_token_to_positions=corrected_mapping,
    )


def group_warning_windows(prepass: FullPrepass) -> list[SentenceSpan]:
    full_text = prepass.full_text or ""
    chars = prepass.global_chars
    if not full_text or not chars or not prepass.warnings:
        return []
    mapping = token_to_text_positions(full_text, chars)
    sentence_spans = _sentence_spans(full_text, mapping, len(chars))
    if not sentence_spans:
        return [_text_span_for_tokens(full_text, mapping, 0, len(chars) - 1)]

    ranges: list[tuple[int, int]] = []
    for warning in sorted(prepass.warnings, key=lambda w: (w.left_idx, w.right_idx)):
        sent_idx = _sentence_index_for_warning(sentence_spans, warning)
        if sent_idx is None:
            continue
        start_idx = sent_idx
        end_idx = sent_idx
        while start_idx > 0 and _has_start_edge_warning(sentence_spans[start_idx], prepass.warnings):
            start_idx -= 1
        while end_idx + 1 < len(sentence_spans) and _has_end_edge_warning(sentence_spans[end_idx], prepass.warnings):
            end_idx += 1
        ranges.append((start_idx, end_idx))

    merged_sentence_ranges = _merge_ranges(ranges)
    return [
        _text_span_for_tokens(
            full_text,
            mapping,
            sentence_spans[start].token_start_idx,
            sentence_spans[end].token_end_idx,
        )
        for start, end in merged_sentence_ranges
    ]


def validate_revision(original_text: str, revised_text: str) -> None:
    original = canonical_nonpunct(original_text)
    revised = canonical_nonpunct(revised_text)
    if original == revised:
        return
    mismatch = first_mismatch(original, revised)
    raise PunctuationValidationError(
        "LLM punctuation correction changed spoken text: "
        + json.dumps(dataclasses.asdict(mismatch), ensure_ascii=False)
    )


async def _request_revision(
    *,
    llm_client: UnifiedClient,
    model: str,
    provider: ProviderName | None,
    original_text: str,
    marked_text: str,
    stats: dict[str, Any],
    window_index: int,
) -> tuple[str, int]:
    memory_manager = MemoryManager(system_prompt=PUNCTUATION_SYSTEM_PROMPT, stateful=True)
    error_feedback = ""
    for attempt in range(1, PUNCT_RETRY_LIMIT + 1):
        if attempt == 1:
            prompt = build_punctuation_prompt(
                window_index=window_index,
                marked_text=marked_text,
                stats=stats,
            )
        else:
            prompt = build_punctuation_retry_prompt(validation_error=error_feedback)
        response = await llm_client.send_prompt(
            memory_manager,
            json.dumps(prompt, ensure_ascii=False, indent=2),
            provider=provider,
            model=model,
            temperature=0,
            trace_name="slide_LLM.punctuation",
            metadata={"phase": "punctuation", "window_index": window_index, "attempt": attempt},
        )
        revised = (response.text or "").strip("\r\n")
        try:
            validate_revision(original_text, revised)
            return revised, attempt
        except PunctuationValidationError as exc:
            error_feedback = str(exc)
    raise PunctuationValidationError(error_feedback or "LLM punctuation correction failed validation.")


def _sentence_spans(
    full_text: str,
    token_to_positions_map: dict[int, list[int]],
    token_count: int,
) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    start_token = 0
    for token_idx in range(token_count):
        positions = token_to_positions_map.get(token_idx)
        if not positions:
            continue
        cursor = max(positions) + 1
        punct_run: list[str] = []
        while cursor < len(full_text) and is_punct_or_space(full_text[cursor]):
            punct_run.append(full_text[cursor])
            cursor += 1
        if any(ch in STRONG_STOPS or ch in EXTRA_STRONG_STOPS for ch in punct_run):
            spans.append(_text_span_for_tokens(full_text, token_to_positions_map, start_token, token_idx))
            start_token = token_idx + 1
    if start_token < token_count:
        spans.append(_text_span_for_tokens(full_text, token_to_positions_map, start_token, token_count - 1))
    return [span for span in spans if span.token_start_idx <= span.token_end_idx]


def _text_span_for_tokens(
    full_text: str,
    token_to_positions_map: dict[int, list[int]],
    token_start_idx: int,
    token_end_idx: int,
) -> SentenceSpan:
    positions: list[int] = []
    for token_idx in range(token_start_idx, token_end_idx + 1):
        positions.extend(token_to_positions_map.get(token_idx, []))
    if not positions:
        return SentenceSpan(token_start_idx, token_end_idx, 0, len(full_text))
    text_start = min(positions)
    text_end = max(positions) + 1
    while text_end < len(full_text) and is_punct_or_space(full_text[text_end]):
        text_end += 1
    return SentenceSpan(token_start_idx, token_end_idx, text_start, text_end)


def _sentence_index_for_warning(spans: Sequence[SentenceSpan], warning: WarningSpan) -> Optional[int]:
    for idx, span in enumerate(spans):
        if span.token_start_idx <= int(warning.left_idx) <= span.token_end_idx:
            return idx
        if span.token_start_idx <= int(warning.right_idx) <= span.token_end_idx:
            return idx
    return None


def _has_start_edge_warning(span: SentenceSpan, warnings: Sequence[WarningSpan]) -> bool:
    return any(int(w.right_idx) == span.token_start_idx for w in warnings)


def _has_end_edge_warning(span: SentenceSpan, warnings: Sequence[WarningSpan]) -> bool:
    return any(int(w.left_idx) == span.token_end_idx for w in warnings)


def _merge_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        merged[-1] = (prev_start, max(prev_end, end))
    return merged


def _apply_replacements(
    full_text: str,
    replacements: Sequence[tuple[int, int, str, LLMPunctuationWindow]],
) -> tuple[str, list[LLMPunctuationWindow]]:
    parts: list[str] = []
    applied: list[LLMPunctuationWindow] = []
    cursor = 0
    for start, end, revised, window in sorted(replacements, key=lambda item: (item[0], item[1])):
        if start < cursor:
            continue
        parts.append(full_text[cursor:start])
        parts.append(revised)
        cursor = end
        applied.append(window)
    parts.append(full_text[cursor:])
    return "".join(parts), applied


def _warning_to_dict(
    warning: WarningSpan,
    *,
    prepass: FullPrepass,
    token_to_positions_map: dict[int, list[int]],
) -> dict[str, Any]:
    right_positions = token_to_positions_map.get(int(warning.right_idx), [])
    left_positions = token_to_positions_map.get(int(warning.left_idx), [])
    pause_ms = round(float(warning.gap_ms), 1)
    return {
        "left_idx": int(warning.left_idx),
        "right_idx": int(warning.right_idx),
        "left_char": warning.left_char,
        "right_char": warning.right_char,
        "pause_ms": pause_ms,
        "category": warning.category,
        "marker_kind": _warning_marker_kind(warning),
        "insert_text_pos": min(right_positions) if right_positions else (max(left_positions) + 1 if left_positions else None),
        "reason": warning.reason,
    }


def _warning_marker_kind(warning: WarningSpan | dict[str, Any]) -> str:
    category = warning.category if isinstance(warning, WarningSpan) else str(warning.get("category", ""))
    return "too long" if category == "no_punc" else "too short"


def _insert_warning_markers(
    text: str,
    *,
    text_start: int,
    warnings: Sequence[dict[str, Any]],
) -> str:
    insertions: list[tuple[int, str]] = []
    for warning in warnings:
        insert_text_pos = warning.get("insert_text_pos")
        if insert_text_pos is None:
            continue
        relative_pos = int(insert_text_pos) - int(text_start)
        if not 0 <= relative_pos <= len(text):
            continue
        pause_ms = int(round(float(warning.get("pause_ms", 0.0))))
        marker = f"{{{warning.get('marker_kind', 'too long')}: {pause_ms} ms}}"
        insertions.append((relative_pos, marker))

    marked = text
    for relative_pos, marker in sorted(insertions, key=lambda item: item[0], reverse=True):
        marked = marked[:relative_pos] + marker + marked[relative_pos:]
    return marked


def _punctuation_pause_stats(prepass: FullPrepass) -> dict[str, dict[str, Any] | str]:
    raw = compact_pause_stats(prepass)

    def _pick(name: str) -> dict[str, Any]:
        bucket = raw[name]
        return {
            "p20_ms": bucket["p20_ms"],
            "p50_ms": bucket["p50_ms"],
            "p80_ms": bucket["p80_ms"],
        }

    return {
        "explanation": (
            "Pause-ms distribution for this timbre/source. strong_stops are pauses after sentence-ending punctuation; "
            "weak_stops are pauses after comma-like punctuation; no_pause means pauses after boundaries without recognized strong/weak punctuation, "
            "including plain character-to-character flow and punctuation/symbols outside the strong/weak sets. "
        ),
        "strong_stops": _pick("strong_stops"),
        "weak_stops": _pick("weak_stops"),
        "no_pause": _pick("no_punc"),
    }
