"""Phase 3 - deterministic punctuation-boundary rough cut planning.

Shared preprocessing:

1. Extract candidate cut boundaries only at punctuation runs in the corrected
   transcript (plus a synthetic final boundary).
2. Label each boundary with rule-based ``bad`` / ``ok`` / ``perfect`` quality.
3. Estimate candidate segment length with ``librosa.effects.trim(top_db=60)``.

Planner behavior:

- pick a planning anchor in the 15-30s range when possible
- prefer strong-stop boundaries, then weak-stop boundaries, then largest pause
- use dynamic programming inside each anchor window
- optimize for valid trimmed length first, then boundary quality
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Optional, Sequence

import numpy as np

from .asr import CharToken

from .prepass import FullPrepass, pause_ms_after_token
from .shared import STRONG_STOPS, WEAK_STOPS, is_punct_or_space


_TRIM_TOP_DB = 60
_MAYBE_SLACK_SEC = 0.35
_MAX_SEARCH_OVERSHOOT_SEC = 8.0
_ANCHOR_MIN_SEC = 15.0
_ANCHOR_MAX_SEC = 30.0
_MAX_BOUNDARY_PADDING_SEC = 0.1


@dataclass
class RoughSegment:
    """One candidate segment produced by the rough-cut phase."""

    char_start_idx: int
    char_end_idx: int  # inclusive
    start_sec: float
    end_sec: float
    drop: bool
    reason: str


@dataclass
class BlockCallTrace:
    block_index: int
    char_start_idx: int
    char_end_idx: int
    block_start_sec: float
    block_end_sec: float
    prompt_summary: dict[str, Any]
    response_cuts: list[dict[str, Any]]
    applied_cuts: list[dict[str, Any]]
    carry_over_from_idx: Optional[int]
    error: Optional[str] = None


@dataclass
class Phase3Result:
    segments: list[RoughSegment] = field(default_factory=list)
    block_traces: list[BlockCallTrace] = field(default_factory=list)


@dataclass(frozen=True)
class RoughBoundary:
    candidate_id: int
    token_idx: int
    text_pos: int
    boundary_text: str
    boundary_kind: str  # "strong" | "weak" | "final"
    quality: str  # "bad" | "ok" | "perfect"
    cut_policy: str  # "must_cut" | "no_cut" | "normal"
    pause_ms: float
    cut_sec: float
    next_token_start_sec: float
    reason: str


@dataclass(frozen=True)
class _PrioritySpan:
    start_token_idx: int
    end_boundary: RoughBoundary
    phase: str
    valid: bool
    raw_duration_sec: float
    trimmed_duration_sec: float


@dataclass(frozen=True)
class _PriorityRegion:
    start_token_idx: int
    end_boundary: RoughBoundary


def _quality_rank(label: str) -> int:
    if label == "perfect":
        return 0
    if label == "ok":
        return 1
    return 2


def _boundary_penalty(label: str) -> tuple[int, int]:
    if label == "perfect":
        return (0, 0)
    if label == "ok":
        return (0, 1)
    return (1, 0)


def _boundary_cut_policy(
    *,
    boundary_kind: str,
    pause_ms: float,
    prepass: FullPrepass,
) -> str:
    if boundary_kind == "final":
        return "must_cut"
    if boundary_kind == "strong" and pause_ms >= float(prepass.strong.p80_ms):
        return "must_cut"
    if boundary_kind == "weak" and pause_ms >= min(float(prepass.strong.p80_ms), float(prepass.weak.p95_ms)):
        return "must_cut"
    if boundary_kind == "weak" and pause_ms <= float(prepass.weak.p20_ms):
        return "no_cut"
    return "normal"


def _length_tier(
    *,
    raw_sec: float,
    trimmed_sec: float,
    min_seg_sec: float,
    max_seg_sec: float,
) -> tuple[int, int]:
    if min_seg_sec <= trimmed_sec <= max_seg_sec:
        return (0, 0)
    relaxed_min = max(0.0, min_seg_sec - _MAYBE_SLACK_SEC)
    relaxed_max = max_seg_sec + _MAYBE_SLACK_SEC
    raw_ok = min_seg_sec <= raw_sec <= max_seg_sec
    trimmed_relaxed = relaxed_min <= trimmed_sec <= relaxed_max
    if raw_ok or trimmed_relaxed:
        return (0, 1)
    return (1, 0)


def _deviation_ms(trimmed_sec: float, min_seg_sec: float, max_seg_sec: float) -> int:
    target = (min_seg_sec + max_seg_sec) * 0.5
    return int(round(abs(trimmed_sec - target) * 1000.0))


def _segment_start_sec(chars: Sequence[CharToken], token_idx: int) -> float:
    current_start = float(chars[token_idx].start_sec)
    if token_idx <= 0:
        return current_start
    previous_end = float(chars[token_idx - 1].end_sec)
    midpoint = previous_end + (current_start - previous_end) * 0.5
    return max(midpoint, current_start - _MAX_BOUNDARY_PADDING_SEC)


def _segment_end_sec(chars: Sequence[CharToken], token_idx: int) -> float:
    current_end = float(chars[token_idx].end_sec)
    if token_idx + 1 >= len(chars):
        return current_end
    next_start = float(chars[token_idx + 1].start_sec)
    midpoint = current_end + (next_start - current_end) * 0.5
    return min(midpoint, current_end + _MAX_BOUNDARY_PADDING_SEC)


def _estimate_trimmed_duration_sec(
    audio: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    cache: dict[tuple[int, int], float],
) -> float:
    start_sample = max(0, int(round(start_sec * sample_rate)))
    end_sample = min(len(audio), int(round(end_sec * sample_rate)))
    if end_sample <= start_sample:
        return 0.0
    key = (start_sample, end_sample)
    cached = cache.get(key)
    if cached is not None:
        return cached

    seg_audio = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
    if seg_audio.size == 0:
        cache[key] = 0.0
        return 0.0

    import librosa

    trimmed, _index = librosa.effects.trim(seg_audio, top_db=_TRIM_TOP_DB)
    trimmed_sec = float(len(trimmed) / sample_rate) if trimmed.size else 0.0
    cache[key] = trimmed_sec
    return trimmed_sec


def _extract_punctuation_boundaries(
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    chars: Sequence[CharToken],
    prepass: FullPrepass,
) -> list[RoughBoundary]:
    boundaries: list[RoughBoundary] = []
    if not corrected_full_text or not chars:
        return boundaries

    for token_idx, left_char in enumerate(chars[:-1]):
        positions = corrected_token_to_positions.get(token_idx)
        if not positions:
            continue
        cursor = max(positions) + 1
        punctuation_run: list[str] = []
        while cursor < len(corrected_full_text):
            ch = corrected_full_text[cursor]
            if not is_punct_or_space(ch):
                break
            punctuation_run.append(ch)
            cursor += 1
        visible_punct = [ch for ch in punctuation_run if not ch.isspace()]
        if not visible_punct:
            continue

        punct_text = "".join(punctuation_run)
        pause_ms = pause_ms_after_token(prepass, token_idx)
        if any(ch in STRONG_STOPS for ch in visible_punct):
            kind = "strong"
            quality, reason = _classify_strong_boundary(pause_ms, prepass)
        elif any(ch in WEAK_STOPS for ch in visible_punct):
            kind = "weak"
            quality, reason = _classify_weak_boundary(pause_ms, prepass)
        else:
            kind = "weak"
            quality, reason = _classify_weak_boundary(pause_ms, prepass)

        boundaries.append(
            RoughBoundary(
                candidate_id=len(boundaries),
                token_idx=token_idx,
                text_pos=max(positions),
                boundary_text=punct_text,
                boundary_kind=kind,
                quality=quality,
                cut_policy=_boundary_cut_policy(
                    boundary_kind=kind,
                    pause_ms=pause_ms,
                    prepass=prepass,
                ),
                pause_ms=round(pause_ms, 1),
                cut_sec=float(left_char.end_sec),
                next_token_start_sec=float(chars[token_idx + 1].start_sec),
                reason=reason,
            )
        )

    last_idx = len(chars) - 1
    boundaries.append(
        RoughBoundary(
            candidate_id=len(boundaries),
            token_idx=last_idx,
            text_pos=max(corrected_token_to_positions.get(last_idx, [len(corrected_full_text) - 1])),
            boundary_text="<final>",
            boundary_kind="final",
            quality="perfect",
            cut_policy="must_cut",
            pause_ms=0.0,
            cut_sec=float(chars[last_idx].end_sec),
            next_token_start_sec=float(chars[last_idx].end_sec),
            reason="synthetic final boundary",
        )
    )
    return boundaries


def _classify_strong_boundary(pause_ms: float, prepass: FullPrepass) -> tuple[str, str]:
    perfect_ms = max(float(prepass.strong.p50_ms), float(prepass.weak.p80_ms), 180.0)
    ok_ms = max(float(prepass.strong.p20_ms), float(prepass.weak.p50_ms), 90.0)
    if pause_ms >= perfect_ms:
        return ("perfect", f"strong stop with {pause_ms:.0f}ms pause")
    if pause_ms >= ok_ms:
        return ("ok", f"strong stop with usable {pause_ms:.0f}ms pause")
    return ("bad", f"strong stop but only {pause_ms:.0f}ms pause")


def _classify_weak_boundary(pause_ms: float, prepass: FullPrepass) -> tuple[str, str]:
    perfect_ms = max(float(prepass.weak.p80_ms), float(prepass.no_punc.p95_ms), 220.0)
    ok_ms = max(float(prepass.weak.p50_ms), float(prepass.no_punc.p80_ms), 120.0)
    if pause_ms >= perfect_ms:
        return ("perfect", f"weak stop with standout {pause_ms:.0f}ms pause")
    if pause_ms >= ok_ms:
        return ("ok", f"weak stop with usable {pause_ms:.0f}ms pause")
    return ("bad", f"weak stop but only {pause_ms:.0f}ms pause")


def _extract_span_text(
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    first_token_idx: int,
    last_token_idx: int,
) -> str:
    if not corrected_full_text or last_token_idx < first_token_idx:
        return ""
    positions: list[int] = []
    for token_idx in range(first_token_idx, last_token_idx + 1):
        positions.extend(corrected_token_to_positions.get(token_idx, []))
    if not positions:
        return ""
    start_pos = min(positions)
    end_pos = max(positions) + 1
    while end_pos < len(corrected_full_text) and is_punct_or_space(corrected_full_text[end_pos]):
        end_pos += 1
    return corrected_full_text[start_pos:end_pos]


def _boundary_sort_key(boundary: RoughBoundary) -> tuple[int, int, int, int]:
    policy_rank = 0 if boundary.cut_policy == "must_cut" else 1
    return (policy_rank, _quality_rank(boundary.quality), -int(round(boundary.pause_ms)), boundary.token_idx)


def _choose_anchor_boundary(
    *,
    start_token_idx: int,
    boundaries: Sequence[RoughBoundary],
    chars: Sequence[CharToken],
) -> Optional[RoughBoundary]:
    start_sec = _segment_start_sec(chars, start_token_idx)
    in_window: list[RoughBoundary] = []
    before_max: list[RoughBoundary] = []
    for boundary in boundaries:
        if boundary.token_idx < start_token_idx:
            continue
        if boundary.cut_policy == "no_cut":
            continue
        raw_end_sec = _segment_end_sec(chars, boundary.token_idx)
        duration_sec = raw_end_sec - start_sec
        if duration_sec <= _ANCHOR_MAX_SEC + 1e-6:
            before_max.append(boundary)
        if _ANCHOR_MIN_SEC <= duration_sec <= _ANCHOR_MAX_SEC + 1e-6:
            in_window.append(boundary)

    if not before_max:
        return None

    tail_boundary = before_max[-1]
    tail_end_sec = _segment_end_sec(chars, tail_boundary.token_idx)
    if tail_end_sec - start_sec <= _ANCHOR_MAX_SEC + 1e-6 and tail_boundary.boundary_kind == "final":
        return tail_boundary

    stopping = [b for b in in_window if b.boundary_kind in {"strong", "final"}]
    if stopping:
        return min(stopping, key=_boundary_sort_key)

    non_stopping = [b for b in in_window if b.boundary_kind == "weak"]
    if non_stopping:
        return min(non_stopping, key=_boundary_sort_key)

    if in_window:
        return max(in_window, key=lambda b: (b.pause_ms, -_quality_rank(b.quality), -b.token_idx))

    if before_max:
        return max(before_max, key=lambda b: b.token_idx)

    later = [b for b in boundaries if b.token_idx >= start_token_idx]
    if later:
        return min(later, key=lambda b: b.token_idx)
    return None


def _plan_segments_legacy_dp(
    *,
    audio: np.ndarray,
    sample_rate: int,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    chars: Sequence[CharToken],
    boundaries: Sequence[RoughBoundary],
    min_seg_sec: float,
    max_seg_sec: float,
) -> tuple[list[RoughSegment], list[dict[str, Any]], Optional[str]]:
    if not boundaries:
        return [], [], "no candidate punctuation boundaries found"

    trim_cache: dict[tuple[int, int], float] = {}
    segments: list[RoughSegment] = []
    chosen_cuts: list[dict[str, Any]] = []
    start_token_idx = 0
    while start_token_idx < len(chars):
        anchor_boundary = _choose_anchor_boundary(
            start_token_idx=start_token_idx,
            boundaries=boundaries,
            chars=chars,
        )
        if anchor_boundary is None:
            return segments, chosen_cuts, "unable to choose planning anchor"

        window_boundaries = [
            boundary
            for boundary in boundaries
            if start_token_idx <= boundary.token_idx <= anchor_boundary.token_idx
        ]
        if not window_boundaries:
            return segments, chosen_cuts, "planning window had no candidate boundaries"

        best_scores: dict[int, tuple[int, int, int, int, int]] = {}
        back_ptr: dict[int, Optional[int]] = {}
        cut_meta: dict[int, dict[str, Any]] = {}
        must_cut_prefix: list[int] = []
        must_cut_seen = 0
        for boundary in window_boundaries:
            if boundary.cut_policy == "must_cut":
                must_cut_seen += 1
            must_cut_prefix.append(must_cut_seen)
        best_scores[-1] = (0, 0, 0, 0, 0)
        back_ptr[-1] = None

        for end_idx, boundary in enumerate(window_boundaries):
            if boundary.cut_policy == "no_cut":
                continue
            best_score: Optional[tuple[int, int, int, int, int]] = None
            best_prev: Optional[int] = None
            best_meta: Optional[dict[str, Any]] = None

            for prev_idx in range(-1, end_idx):
                prev_score = best_scores.get(prev_idx)
                if prev_score is None:
                    continue
                prev_must_cuts = 0 if prev_idx < 0 else must_cut_prefix[prev_idx]
                skipped_must_cuts = must_cut_prefix[end_idx - 1] - prev_must_cuts if end_idx > 0 else 0
                if skipped_must_cuts > 0:
                    continue

                seg_token_start = (
                    start_token_idx
                    if prev_idx < 0
                    else window_boundaries[prev_idx].token_idx + 1
                )
                seg_token_end = boundary.token_idx
                if seg_token_end < seg_token_start or seg_token_start >= len(chars):
                    continue

                seg_start_sec = _segment_start_sec(chars, seg_token_start)
                raw_end_sec = _segment_end_sec(chars, seg_token_end)
                raw_sec = max(0.0, raw_end_sec - seg_start_sec)
                if raw_sec > max_seg_sec + _MAX_SEARCH_OVERSHOOT_SEC:
                    continue

                trimmed_sec = _estimate_trimmed_duration_sec(
                    audio,
                    sample_rate,
                    seg_start_sec,
                    raw_end_sec,
                    trim_cache,
                )
                invalid_penalty, maybe_penalty = _length_tier(
                    raw_sec=raw_sec,
                    trimmed_sec=trimmed_sec,
                    min_seg_sec=min_seg_sec,
                    max_seg_sec=max_seg_sec,
                )
                bad_penalty, ok_penalty = _boundary_penalty(boundary.quality)
                score = (
                    prev_score[0] + invalid_penalty,
                    prev_score[1] + maybe_penalty,
                    prev_score[2] + bad_penalty,
                    prev_score[3] + ok_penalty,
                    prev_score[4] + _deviation_ms(trimmed_sec, min_seg_sec, max_seg_sec),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_prev = prev_idx
                    best_meta = {
                        "char_start_idx": seg_token_start,
                        "char_end_idx": seg_token_end,
                        "start_sec": seg_start_sec,
                        "end_sec": raw_end_sec,
                        "raw_duration_sec": round(raw_sec, 3),
                        "trimmed_duration_sec": round(trimmed_sec, 3),
                        "length_tier": (
                            "must-valid"
                            if invalid_penalty == 0 and maybe_penalty == 0
                            else "maybe-valid"
                            if invalid_penalty == 0
                            else "invalid"
                        ),
                        "boundary_quality": boundary.quality,
                        "boundary_kind": boundary.boundary_kind,
                        "cut_policy": boundary.cut_policy,
                        "boundary_text": boundary.boundary_text,
                        "pause_ms": boundary.pause_ms,
                        "anchor_candidate_id": anchor_boundary.candidate_id,
                        "anchor_window_start_sec": round(_segment_start_sec(chars, start_token_idx), 3),
                        "anchor_window_end_sec": round(raw_end_sec, 3),
                        "reason": _segment_reason(
                            boundary=boundary,
                            raw_sec=raw_sec,
                            trimmed_sec=trimmed_sec,
                            invalid_penalty=invalid_penalty,
                            maybe_penalty=maybe_penalty,
                        ),
                    }

            if best_score is not None and best_prev is not None and best_meta is not None:
                best_scores[end_idx] = best_score
                back_ptr[end_idx] = best_prev
                cut_meta[end_idx] = best_meta

        anchor_local_idx = len(window_boundaries) - 1
        if anchor_local_idx not in best_scores:
            return segments, chosen_cuts, "unable to plan a cut path to the selected anchor"

        chosen_local_indices: list[int] = []
        cursor = anchor_local_idx
        while cursor >= 0:
            chosen_local_indices.append(cursor)
            prev_cursor = back_ptr[cursor]
            if prev_cursor is None or prev_cursor < 0:
                break
            cursor = prev_cursor
        chosen_local_indices.reverse()

        for cut_idx in chosen_local_indices:
            meta = cut_meta[cut_idx]
            segments.append(
                RoughSegment(
                    char_start_idx=int(meta["char_start_idx"]),
                    char_end_idx=int(meta["char_end_idx"]),
                    start_sec=float(meta["start_sec"]),
                    end_sec=float(meta["end_sec"]),
                    drop=False,
                    reason=str(meta["reason"]),
                )
            )
            chosen_cuts.append(
                {
                    "cut_char_idx": int(meta["char_end_idx"]),
                    "boundary_quality": meta["boundary_quality"],
                    "boundary_kind": meta["boundary_kind"],
                    "cut_policy": meta["cut_policy"],
                    "boundary_text": meta["boundary_text"],
                    "pause_ms": meta["pause_ms"],
                    "raw_duration_sec": meta["raw_duration_sec"],
                    "trimmed_duration_sec": meta["trimmed_duration_sec"],
                    "length_tier": meta["length_tier"],
                    "anchor_candidate_id": meta["anchor_candidate_id"],
                    "anchor_window_start_sec": meta["anchor_window_start_sec"],
                    "anchor_window_end_sec": meta["anchor_window_end_sec"],
                    "reason": meta["reason"],
                }
            )

        start_token_idx = anchor_boundary.token_idx + 1
        if anchor_boundary.boundary_kind == "final":
            break

    return segments, chosen_cuts, None


def _priority_must_cut(boundary: RoughBoundary, prepass: FullPrepass) -> bool:
    if boundary.boundary_kind == "final":
        return True
    return (
        boundary.boundary_kind == "strong"
        and boundary.pause_ms >= max(float(prepass.strong.p80_ms), 220.0)
    )


def _priority_good_cut(boundary: RoughBoundary, prepass: FullPrepass) -> bool:
    return (
        boundary.boundary_kind == "strong"
        and boundary.pause_ms >= max(float(prepass.weak.p60_ms), 120.0)
    )


def _priority_v2_good_cut(boundary: RoughBoundary, prepass: FullPrepass) -> bool:
    return (
        boundary.boundary_kind == "strong"
        and boundary.pause_ms >= float(prepass.weak.p60_ms)
    )


def _priority_valid_cut(boundary: RoughBoundary, prepass: FullPrepass) -> bool:
    return boundary.pause_ms >= max(float(prepass.weak.p60_ms), 120.0)


def _priority_v2_legal_cut(boundary: RoughBoundary, prepass: FullPrepass) -> bool:
    return (
        boundary.boundary_kind in {"strong", "weak"}
        and boundary.pause_ms >= float(prepass.weak.p60_ms)
    )


def _dp_strategy_2_must_cut(boundary: RoughBoundary, prepass: FullPrepass) -> bool:
    if boundary.boundary_kind == "final":
        return True
    return boundary.boundary_kind == "strong" and boundary.pause_ms >= float(prepass.strong.p80_ms)


def _dp_strategy_2_pause_score(boundary: RoughBoundary, prepass: FullPrepass) -> Optional[int]:
    pause_ms = float(boundary.pause_ms)
    if pause_ms <= float(prepass.weak.p20_ms):
        return None
    medium_floor = max(float(prepass.strong.p60_ms), float(prepass.weak.p60_ms))
    if pause_ms >= float(prepass.strong.p80_ms):
        return 3
    if pause_ms >= medium_floor:
        return 2
    if pause_ms >= float(prepass.weak.p60_ms):
        return 1
    return -1


def _dp_strategy_2_cut_score(boundary: RoughBoundary, prepass: FullPrepass) -> Optional[int]:
    if boundary.boundary_kind == "final":
        return 0
    if boundary.boundary_kind not in {"strong", "weak"}:
        return None
    pause_score = _dp_strategy_2_pause_score(boundary, prepass)
    if pause_score is None:
        return None
    punctuation_score = 2 if boundary.boundary_kind == "strong" else -1
    return punctuation_score + pause_score


def _dp_strategy_2_legal_cut(boundary: RoughBoundary, prepass: FullPrepass) -> bool:
    return _dp_strategy_2_cut_score(boundary, prepass) is not None


def _dp_strategy_2_internal_penalty(
    boundaries: Sequence[RoughBoundary],
    *,
    start_token_idx: int,
    end_token_idx: int,
    prepass: FullPrepass,
) -> int:
    penalty = 0
    medium_floor = max(float(prepass.strong.p60_ms), float(prepass.weak.p60_ms))
    for boundary in boundaries:
        if not (start_token_idx <= boundary.token_idx < end_token_idx):
            continue
        if boundary.boundary_kind != "strong":
            continue
        penalty -= 1
        if medium_floor < boundary.pause_ms < float(prepass.strong.p80_ms):
            penalty -= 1
    return penalty


def _priority_boundary_policy(boundary: RoughBoundary, prepass: FullPrepass) -> str:
    if _priority_must_cut(boundary, prepass):
        return "must_cut"
    if _priority_good_cut(boundary, prepass):
        return "good_cut"
    if _priority_valid_cut(boundary, prepass):
        return "valid_cut"
    return "not_candidate"


def _make_priority_span(
    *,
    audio: np.ndarray,
    sample_rate: int,
    chars: Sequence[CharToken],
    start_token_idx: int,
    boundary: RoughBoundary,
    phase: str,
    min_seg_sec: float,
    max_seg_sec: float,
    trim_cache: dict[tuple[int, int], float],
) -> _PrioritySpan:
    seg_start_sec = _segment_start_sec(chars, start_token_idx)
    raw_end_sec = _segment_end_sec(chars, boundary.token_idx)
    raw_sec = max(0.0, raw_end_sec - seg_start_sec)
    trimmed_sec = _estimate_trimmed_duration_sec(
        audio,
        sample_rate,
        seg_start_sec,
        raw_end_sec,
        trim_cache,
    )
    return _PrioritySpan(
        start_token_idx=start_token_idx,
        end_boundary=boundary,
        phase=phase,
        valid=min_seg_sec <= trimmed_sec <= max_seg_sec,
        raw_duration_sec=round(raw_sec, 3),
        trimmed_duration_sec=round(trimmed_sec, 3),
    )


def _priority_span_score(
    span: _PrioritySpan,
    *,
    split_priority: str,
) -> tuple[float, int, float]:
    if not span.valid:
        return (0.0, 0, 0.0)
    segment_count = 1
    if split_priority == "segments_before_silence":
        return (span.trimmed_duration_sec, segment_count, span.end_boundary.pause_ms)
    return (span.trimmed_duration_sec, int(round(span.end_boundary.pause_ms)), segment_count)


def _score_better(
    left: tuple[float, int, float],
    right: tuple[float, int, float],
) -> bool:
    return left > right


def _subtract_priority_spans(
    regions: Sequence[_PriorityRegion],
    spans: Sequence[_PrioritySpan],
    chars: Sequence[CharToken],
) -> list[_PriorityRegion]:
    remaining: list[_PriorityRegion] = []
    spans_by_start = sorted(spans, key=lambda span: (span.start_token_idx, span.end_boundary.token_idx))
    for region in regions:
        cursor = region.start_token_idx
        for span in spans_by_start:
            if span.end_boundary.token_idx < cursor or span.start_token_idx > region.end_boundary.token_idx:
                continue
            if cursor < span.start_token_idx:
                remaining.append(
                    _PriorityRegion(
                        start_token_idx=cursor,
                        end_boundary=_synthetic_boundary_before(
                            chars,
                            span.start_token_idx,
                            reason="gap before solved priority slice",
                        ),
                    )
                )
            cursor = max(cursor, span.end_boundary.token_idx + 1)
        if cursor <= region.end_boundary.token_idx:
            remaining.append(_PriorityRegion(start_token_idx=cursor, end_boundary=region.end_boundary))
    return remaining


def _candidate_in_regions(boundary: RoughBoundary, regions: Sequence[_PriorityRegion]) -> bool:
    return any(
        region.start_token_idx <= boundary.token_idx < region.end_boundary.token_idx
        for region in regions
    )


def _priority_v2_legal_candidates(
    boundaries: Sequence[RoughBoundary],
    regions: Sequence[_PriorityRegion],
    prepass: FullPrepass,
) -> list[RoughBoundary]:
    return [
        boundary
        for boundary in boundaries
        if _candidate_in_regions(boundary, regions)
        and _priority_v2_legal_cut(boundary, prepass)
        and not _priority_must_cut(boundary, prepass)
    ]


def _priority_v2_partition_candidates(
    candidates: Sequence[RoughBoundary],
) -> tuple[list[RoughBoundary], list[RoughBoundary]]:
    target_count = ceil(len(candidates) / 2)
    strong = sorted(
        (boundary for boundary in candidates if boundary.boundary_kind == "strong"),
        key=lambda boundary: (-boundary.pause_ms, boundary.token_idx, boundary.candidate_id),
    )
    weak = sorted(
        (boundary for boundary in candidates if boundary.boundary_kind == "weak"),
        key=lambda boundary: (-boundary.pause_ms, boundary.token_idx, boundary.candidate_id),
    )
    if len(strong) >= target_count:
        top = strong[:target_count]
    else:
        top = strong + weak[: max(0, target_count - len(strong))]
    top_ids = {boundary.candidate_id for boundary in top}
    remaining = [
        boundary
        for boundary in sorted(
            candidates,
            key=lambda candidate: (-candidate.pause_ms, candidate.token_idx, candidate.candidate_id),
        )
        if boundary.candidate_id not in top_ids
    ]
    return top, remaining


def _choose_priority_valid_spans(
    *,
    audio: np.ndarray,
    sample_rate: int,
    chars: Sequence[CharToken],
    start_token_idx: int,
    end_boundary: RoughBoundary,
    candidate_boundaries: Sequence[RoughBoundary],
    phase: str,
    split_priority: str,
    min_seg_sec: float,
    max_seg_sec: float,
    trim_cache: dict[tuple[int, int], float],
    require_internal_candidate: bool = False,
) -> list[_PrioritySpan]:
    endpoints_by_token = {
        boundary.token_idx: boundary
        for boundary in candidate_boundaries
        if start_token_idx <= boundary.token_idx < end_boundary.token_idx
    }
    endpoints_by_token[end_boundary.token_idx] = end_boundary
    endpoints = [
        boundary
        for _token_idx, boundary in sorted(endpoints_by_token.items())
    ]

    best_scores: dict[int, tuple[float, int, float]] = {-1: (0.0, 0, 0.0)}
    back_ptr: dict[int, Optional[int]] = {-1: None}
    best_span: dict[int, _PrioritySpan] = {}

    for end_idx, boundary in enumerate(endpoints):
        local_best_score: Optional[tuple[float, int, float]] = None
        local_best_prev: Optional[int] = None
        local_best_span: Optional[_PrioritySpan] = None
        for prev_idx in range(-1, end_idx):
            prev_score = best_scores.get(prev_idx)
            if prev_score is None:
                continue
            seg_start = start_token_idx if prev_idx < 0 else endpoints[prev_idx].token_idx + 1
            if seg_start > boundary.token_idx:
                continue
            span = _make_priority_span(
                audio=audio,
                sample_rate=sample_rate,
                chars=chars,
                start_token_idx=seg_start,
                boundary=boundary,
                phase=phase,
                min_seg_sec=min_seg_sec,
                max_seg_sec=max_seg_sec,
                trim_cache=trim_cache,
            )
            span_score = _priority_span_score(span, split_priority=split_priority)
            if span_score[0] <= 0:
                continue
            score = (
                prev_score[0] + span_score[0],
                prev_score[1] + span_score[1],
                prev_score[2] + span_score[2],
            )
            if local_best_score is None or _score_better(score, local_best_score):
                local_best_score = score
                local_best_prev = prev_idx
                local_best_span = span
        if local_best_score is not None and local_best_prev is not None and local_best_span is not None:
            best_scores[end_idx] = local_best_score
            back_ptr[end_idx] = local_best_prev
            best_span[end_idx] = local_best_span

    if not best_scores:
        return []

    cursor = max(best_scores, key=lambda idx: best_scores[idx])
    chosen: list[_PrioritySpan] = []
    while cursor >= 0:
        span = best_span.get(cursor)
        if span is None:
            break
        chosen.append(span)
        prev = back_ptr.get(cursor)
        if prev is None:
            break
        cursor = prev
    chosen.reverse()
    if require_internal_candidate and not any(
        span.end_boundary.token_idx < end_boundary.token_idx for span in chosen
    ):
        return []
    return chosen


def _synthetic_boundary_before(chars: Sequence[CharToken], token_idx: int, *, reason: str) -> RoughBoundary:
    boundary_token_idx = max(0, token_idx - 1)
    next_token_idx = min(token_idx, len(chars) - 1)
    return RoughBoundary(
        candidate_id=-1,
        token_idx=boundary_token_idx,
        text_pos=boundary_token_idx,
        boundary_text="<leftover>",
        boundary_kind="synthetic",
        quality="bad",
        cut_policy="normal",
        pause_ms=0.0,
        cut_sec=float(chars[boundary_token_idx].end_sec),
        next_token_start_sec=float(chars[next_token_idx].start_sec),
        reason=reason,
    )


def _fill_priority_gaps(
    *,
    audio: np.ndarray,
    sample_rate: int,
    chars: Sequence[CharToken],
    region_start_token_idx: int,
    region_end_boundary: RoughBoundary,
    locked_spans: Sequence[_PrioritySpan],
    valid_candidates: Sequence[RoughBoundary],
    min_seg_sec: float,
    max_seg_sec: float,
    trim_cache: dict[tuple[int, int], float],
) -> list[_PrioritySpan]:
    filled: list[_PrioritySpan] = []
    cursor = region_start_token_idx
    for locked in locked_spans:
        if cursor < locked.start_token_idx:
            gap_end_boundary = _synthetic_boundary_before(
                chars,
                locked.start_token_idx,
                reason="gap before locked priority cut",
            )
            filled.extend(
                _choose_priority_valid_spans(
                    audio=audio,
                    sample_rate=sample_rate,
                    chars=chars,
                    start_token_idx=cursor,
                    end_boundary=gap_end_boundary,
                    candidate_boundaries=valid_candidates,
                    min_seg_sec=min_seg_sec,
                    max_seg_sec=max_seg_sec,
                    trim_cache=trim_cache,
                    phase="valid_cut",
                    split_priority="silence_before_segments",
                    require_internal_candidate=False,
                )
            )
        filled.append(locked)
        cursor = locked.end_boundary.token_idx + 1

    if cursor <= region_end_boundary.token_idx:
        filled.extend(
            _choose_priority_valid_spans(
                audio=audio,
                sample_rate=sample_rate,
                chars=chars,
                start_token_idx=cursor,
                end_boundary=region_end_boundary,
                candidate_boundaries=valid_candidates,
                phase="valid_cut",
                split_priority="silence_before_segments",
                require_internal_candidate=False,
                min_seg_sec=min_seg_sec,
                max_seg_sec=max_seg_sec,
                trim_cache=trim_cache,
            )
        )
    return sorted(filled, key=lambda span: (span.start_token_idx, span.end_boundary.token_idx))


def _priority_leftover_spans(
    *,
    audio: np.ndarray,
    sample_rate: int,
    chars: Sequence[CharToken],
    region_start_token_idx: int,
    region_end_boundary: RoughBoundary,
    chosen_spans: Sequence[_PrioritySpan],
    min_seg_sec: float,
    max_seg_sec: float,
    trim_cache: dict[tuple[int, int], float],
) -> list[_PrioritySpan]:
    spans: list[_PrioritySpan] = []
    cursor = region_start_token_idx
    for chosen in sorted(chosen_spans, key=lambda span: (span.start_token_idx, span.end_boundary.token_idx)):
        if cursor < chosen.start_token_idx:
            spans.append(
                _make_priority_span(
                    audio=audio,
                    sample_rate=sample_rate,
                    chars=chars,
                    start_token_idx=cursor,
                    boundary=_synthetic_boundary_before(
                        chars,
                        chosen.start_token_idx,
                        reason="leftover before locked priority cut",
                    ),
                    phase="leftover",
                    min_seg_sec=min_seg_sec,
                    max_seg_sec=max_seg_sec,
                    trim_cache=trim_cache,
                )
            )
        spans.append(chosen)
        cursor = chosen.end_boundary.token_idx + 1
    if cursor <= region_end_boundary.token_idx:
        spans.append(
            _make_priority_span(
                audio=audio,
                sample_rate=sample_rate,
                chars=chars,
                start_token_idx=cursor,
                boundary=region_end_boundary,
                phase="leftover",
                min_seg_sec=min_seg_sec,
                max_seg_sec=max_seg_sec,
                trim_cache=trim_cache,
            )
        )
    return sorted(spans, key=lambda span: (span.start_token_idx, span.end_boundary.token_idx))


def _priority_span_to_segment(
    span: _PrioritySpan,
    chars: Sequence[CharToken],
    *,
    strategy: str = "priority_silence_v1",
) -> RoughSegment:
    boundary = span.end_boundary
    start_sec = _segment_start_sec(chars, span.start_token_idx)
    end_sec = _segment_end_sec(chars, boundary.token_idx)
    validity = "valid" if span.valid else "invalid"
    drop = span.phase == "leftover"
    return RoughSegment(
        char_start_idx=span.start_token_idx,
        char_end_idx=boundary.token_idx,
        start_sec=start_sec,
        end_sec=end_sec,
        drop=drop,
        reason=(
            f"{strategy} {span.phase} / {boundary.boundary_kind} pause at "
            f"'{boundary.boundary_text}' ({boundary.pause_ms:.0f}ms); {validity} segment"
            f"{'; drop=true unresolved leftover' if drop else ''} "
            f"(raw {span.raw_duration_sec:.2f}s, trimmed {span.trimmed_duration_sec:.2f}s)"
        ),
    )


def _priority_span_to_cut(
    span: _PrioritySpan,
    *,
    strategy: str = "priority_silence_v1",
) -> dict[str, Any]:
    boundary = span.end_boundary
    return {
        "cut_char_idx": int(boundary.token_idx),
        "boundary_quality": boundary.quality,
        "boundary_kind": boundary.boundary_kind,
        "cut_policy": boundary.cut_policy,
        "strategy_cut_policy": span.phase,
        "strategy_phase": span.phase,
        "boundary_text": boundary.boundary_text,
        "pause_ms": boundary.pause_ms,
        "raw_duration_sec": span.raw_duration_sec,
        "trimmed_duration_sec": span.trimmed_duration_sec,
        "drop": span.phase == "leftover",
        "length_tier": "must-valid" if span.valid else "invalid",
        "reason": (
            f"{strategy} {span.phase}; "
            f"{'drop unresolved leftover' if span.phase == 'leftover' else 'valid' if span.valid else 'invalid'} "
            "trimmed duration"
        ),
    }


def _plan_segments_priority_silence_v1(
    *,
    audio: np.ndarray,
    sample_rate: int,
    prepass: FullPrepass,
    chars: Sequence[CharToken],
    boundaries: Sequence[RoughBoundary],
    min_seg_sec: float,
    max_seg_sec: float,
) -> tuple[list[RoughSegment], list[dict[str, Any]], Optional[str]]:
    if not boundaries:
        return [], [], "no candidate punctuation boundaries found"

    trim_cache: dict[tuple[int, int], float] = {}
    must_boundaries = [boundary for boundary in boundaries if _priority_must_cut(boundary, prepass)]
    if not must_boundaries or must_boundaries[-1].boundary_kind != "final":
        return [], [], "priority_silence_v1 found no final must_cut boundary"

    segments: list[RoughSegment] = []
    chosen_cuts: list[dict[str, Any]] = []
    region_start = 0
    for region_end in must_boundaries:
        if region_start > region_end.token_idx:
            continue
        region_boundaries = [
            boundary
            for boundary in boundaries
            if region_start <= boundary.token_idx <= region_end.token_idx
        ]
        good_candidates = [
            boundary
            for boundary in region_boundaries
            if _priority_good_cut(boundary, prepass)
        ]
        valid_candidates = [
            boundary
            for boundary in region_boundaries
            if _priority_valid_cut(boundary, prepass)
        ]

        locked_good_spans = _choose_priority_valid_spans(
            audio=audio,
            sample_rate=sample_rate,
            chars=chars,
            start_token_idx=region_start,
            end_boundary=region_end,
            candidate_boundaries=good_candidates,
            phase="good_cut",
            split_priority="segments_before_silence",
            require_internal_candidate=True,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            trim_cache=trim_cache,
        )
        filled_spans = _fill_priority_gaps(
            audio=audio,
            sample_rate=sample_rate,
            chars=chars,
            region_start_token_idx=region_start,
            region_end_boundary=region_end,
            locked_spans=locked_good_spans,
            valid_candidates=valid_candidates,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            trim_cache=trim_cache,
        )
        all_spans = _priority_leftover_spans(
            audio=audio,
            sample_rate=sample_rate,
            chars=chars,
            region_start_token_idx=region_start,
            region_end_boundary=region_end,
            chosen_spans=filled_spans,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            trim_cache=trim_cache,
        )
        for span in all_spans:
            segments.append(_priority_span_to_segment(span, chars))
            chosen_cuts.append(_priority_span_to_cut(span))
        region_start = region_end.token_idx + 1

    return segments, chosen_cuts, None


def _solve_priority_regions(
    *,
    audio: np.ndarray,
    sample_rate: int,
    chars: Sequence[CharToken],
    regions: Sequence[_PriorityRegion],
    candidate_boundaries: Sequence[RoughBoundary],
    phase: str,
    split_priority: str,
    min_seg_sec: float,
    max_seg_sec: float,
    trim_cache: dict[tuple[int, int], float],
    require_internal_candidate: bool = False,
) -> list[_PrioritySpan]:
    solved: list[_PrioritySpan] = []
    for region in regions:
        spans = _choose_priority_valid_spans(
            audio=audio,
            sample_rate=sample_rate,
            chars=chars,
            start_token_idx=region.start_token_idx,
            end_boundary=region.end_boundary,
            candidate_boundaries=candidate_boundaries,
            phase=phase,
            split_priority=split_priority,
            require_internal_candidate=require_internal_candidate,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            trim_cache=trim_cache,
        )
        solved.extend(spans)
    return sorted(solved, key=lambda span: (span.start_token_idx, span.end_boundary.token_idx))


def _plan_segments_priority_silence_v2(
    *,
    audio: np.ndarray,
    sample_rate: int,
    prepass: FullPrepass,
    chars: Sequence[CharToken],
    boundaries: Sequence[RoughBoundary],
    min_seg_sec: float,
    max_seg_sec: float,
) -> tuple[list[RoughSegment], list[dict[str, Any]], Optional[str], dict[str, Any]]:
    if not boundaries:
        return [], [], "no candidate punctuation boundaries found", {}

    trim_cache: dict[tuple[int, int], float] = {}
    must_boundaries = [boundary for boundary in boundaries if _priority_must_cut(boundary, prepass)]
    if not must_boundaries or must_boundaries[-1].boundary_kind != "final":
        return [], [], "priority_silence_v2 found no final must_cut boundary", {}

    must_regions: list[_PriorityRegion] = []
    region_start = 0
    for region_end in must_boundaries:
        if region_start <= region_end.token_idx:
            must_regions.append(_PriorityRegion(start_token_idx=region_start, end_boundary=region_end))
        region_start = region_end.token_idx + 1

    good_spans: list[_PrioritySpan] = []
    for region in must_regions:
        region_boundaries = [
            boundary
            for boundary in boundaries
            if region.start_token_idx <= boundary.token_idx <= region.end_boundary.token_idx
        ]
        good_candidates = [
            boundary
            for boundary in region_boundaries
            if _priority_v2_good_cut(boundary, prepass)
        ]
        good_spans.extend(
            _choose_priority_valid_spans(
                audio=audio,
                sample_rate=sample_rate,
                chars=chars,
                start_token_idx=region.start_token_idx,
                end_boundary=region.end_boundary,
                candidate_boundaries=good_candidates,
                phase="good_cut",
                split_priority="segments_before_silence",
                require_internal_candidate=True,
                min_seg_sec=min_seg_sec,
                max_seg_sec=max_seg_sec,
                trim_cache=trim_cache,
            )
        )
    good_spans = sorted(good_spans, key=lambda span: (span.start_token_idx, span.end_boundary.token_idx))

    after_good_regions = _subtract_priority_spans(must_regions, good_spans, chars)
    legal_candidates = _priority_v2_legal_candidates(boundaries, after_good_regions, prepass)
    top50_candidates, partition_remaining50_candidates = _priority_v2_partition_candidates(legal_candidates)

    top50_spans = _solve_priority_regions(
        audio=audio,
        sample_rate=sample_rate,
        chars=chars,
        regions=after_good_regions,
        candidate_boundaries=top50_candidates,
        phase="top50_cut",
        split_priority="silence_before_segments",
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
        trim_cache=trim_cache,
    )

    after_top50_regions = _subtract_priority_spans(after_good_regions, top50_spans, chars)
    step4_candidates = _priority_v2_legal_candidates(boundaries, after_top50_regions, prepass)
    remaining50_spans = _solve_priority_regions(
        audio=audio,
        sample_rate=sample_rate,
        chars=chars,
        regions=after_top50_regions,
        candidate_boundaries=step4_candidates,
        phase="remaining50_cut",
        split_priority="silence_before_segments",
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
        trim_cache=trim_cache,
    )

    solved_spans = sorted(
        [*good_spans, *top50_spans, *remaining50_spans],
        key=lambda span: (span.start_token_idx, span.end_boundary.token_idx),
    )
    all_spans: list[_PrioritySpan] = []
    for region in must_regions:
        region_spans = [
            span
            for span in solved_spans
            if region.start_token_idx <= span.start_token_idx <= span.end_boundary.token_idx <= region.end_boundary.token_idx
        ]
        all_spans.extend(
            _priority_leftover_spans(
                audio=audio,
                sample_rate=sample_rate,
                chars=chars,
                region_start_token_idx=region.start_token_idx,
                region_end_boundary=region.end_boundary,
                chosen_spans=region_spans,
                min_seg_sec=min_seg_sec,
                max_seg_sec=max_seg_sec,
                trim_cache=trim_cache,
            )
        )

    segments = [
        _priority_span_to_segment(span, chars, strategy="priority_silence_v2")
        for span in all_spans
    ]
    chosen_cuts = [
        _priority_span_to_cut(span, strategy="priority_silence_v2")
        for span in all_spans
    ]
    meta = {
        "priority_v2_total_legal_candidate_count": len(legal_candidates),
        "priority_v2_top50_candidate_count": len(top50_candidates),
        "priority_v2_remaining50_candidate_count": len(step4_candidates),
        "priority_v2_partition_remaining50_candidate_count": len(partition_remaining50_candidates),
        "priority_v2_top50_strong_candidate_count": sum(
            1 for boundary in top50_candidates if boundary.boundary_kind == "strong"
        ),
        "priority_v2_top50_weak_candidate_count": sum(
            1 for boundary in top50_candidates if boundary.boundary_kind == "weak"
        ),
    }
    return segments, chosen_cuts, None, meta


def _choose_dp_strategy_2_spans(
    *,
    audio: np.ndarray,
    sample_rate: int,
    chars: Sequence[CharToken],
    region_boundaries: Sequence[RoughBoundary],
    start_token_idx: int,
    end_boundary: RoughBoundary,
    candidate_boundaries: Sequence[RoughBoundary],
    prepass: FullPrepass,
    min_seg_sec: float,
    max_seg_sec: float,
    trim_cache: dict[tuple[int, int], float],
) -> list[_PrioritySpan]:
    endpoints_by_token = {
        boundary.token_idx: boundary
        for boundary in candidate_boundaries
        if start_token_idx <= boundary.token_idx < end_boundary.token_idx
        and _dp_strategy_2_legal_cut(boundary, prepass)
    }
    endpoints_by_token[end_boundary.token_idx] = end_boundary
    endpoints = [boundary for _idx, boundary in sorted(endpoints_by_token.items())]

    best_scores: dict[int, tuple[int, int, int, int, int]] = {-1: (0, 0, 0, 0, 0)}
    back_ptr: dict[int, Optional[int]] = {-1: None}
    best_span: dict[int, _PrioritySpan] = {}
    midpoint_sec = (min_seg_sec + max_seg_sec) / 2.0

    for end_idx, boundary in enumerate(endpoints):
        local_best_score: Optional[tuple[int, int, int, int, int]] = None
        local_best_prev: Optional[int] = None
        local_best_span: Optional[_PrioritySpan] = None
        boundary_cut_score = _dp_strategy_2_cut_score(boundary, prepass)
        if boundary_cut_score is None and boundary.token_idx != end_boundary.token_idx:
            continue
        if boundary.token_idx == end_boundary.token_idx and boundary_cut_score is None:
            boundary_cut_score = 0
        for prev_idx in range(-1, end_idx):
            prev_score = best_scores.get(prev_idx)
            if prev_score is None:
                continue
            seg_start = start_token_idx if prev_idx < 0 else endpoints[prev_idx].token_idx + 1
            if seg_start > boundary.token_idx:
                continue
            span = _make_priority_span(
                audio=audio,
                sample_rate=sample_rate,
                chars=chars,
                start_token_idx=seg_start,
                boundary=boundary,
                phase="dp_strategy_2",
                min_seg_sec=min_seg_sec,
                max_seg_sec=max_seg_sec,
                trim_cache=trim_cache,
            )
            if not span.valid:
                continue
            length_score = 1 if span.trimmed_duration_sec <= midpoint_sec else 0
            internal_penalty = _dp_strategy_2_internal_penalty(
                region_boundaries,
                start_token_idx=seg_start,
                end_token_idx=boundary.token_idx,
                prepass=prepass,
            )
            quality_score = int(boundary_cut_score) + length_score + internal_penalty
            coverage_ms = int(round(span.trimmed_duration_sec * 1000.0))
            max_duration_ms = max(-prev_score[2], coverage_ms)
            score = (
                prev_score[0] + coverage_ms,
                prev_score[1] + quality_score,
                -max_duration_ms,
                prev_score[3] + 1,
                prev_score[4] + int(round(boundary.pause_ms)),
            )
            if local_best_score is None or score > local_best_score:
                local_best_score = score
                local_best_prev = prev_idx
                local_best_span = span
        if local_best_score is not None and local_best_prev is not None and local_best_span is not None:
            best_scores[end_idx] = local_best_score
            back_ptr[end_idx] = local_best_prev
            best_span[end_idx] = local_best_span

    final_idx = len(endpoints) - 1
    if final_idx not in best_scores:
        return []

    cursor = final_idx
    chosen: list[_PrioritySpan] = []
    while cursor >= 0:
        span = best_span.get(cursor)
        if span is None:
            break
        chosen.append(span)
        prev = back_ptr.get(cursor)
        if prev is None:
            break
        cursor = prev
    chosen.reverse()
    return chosen


def _plan_segments_dp_strategy_2(
    *,
    audio: np.ndarray,
    sample_rate: int,
    prepass: FullPrepass,
    chars: Sequence[CharToken],
    boundaries: Sequence[RoughBoundary],
    min_seg_sec: float,
    max_seg_sec: float,
) -> tuple[list[RoughSegment], list[dict[str, Any]], Optional[str], dict[str, Any]]:
    if not boundaries:
        return [], [], "no candidate punctuation boundaries found", {}

    trim_cache: dict[tuple[int, int], float] = {}
    must_boundaries = [boundary for boundary in boundaries if _dp_strategy_2_must_cut(boundary, prepass)]
    if not must_boundaries or must_boundaries[-1].boundary_kind != "final":
        return [], [], "dp_strategy_2 found no final must_cut boundary", {}

    all_spans: list[_PrioritySpan] = []
    region_start = 0
    legal_candidate_count = 0
    must_candidate_count = len(must_boundaries)
    for region_end in must_boundaries:
        if region_start > region_end.token_idx:
            continue
        region_boundaries = [
            boundary
            for boundary in boundaries
            if region_start <= boundary.token_idx <= region_end.token_idx
        ]
        candidate_boundaries = [
            boundary
            for boundary in region_boundaries
            if boundary.token_idx < region_end.token_idx
            and _dp_strategy_2_legal_cut(boundary, prepass)
            and not _dp_strategy_2_must_cut(boundary, prepass)
        ]
        legal_candidate_count += len(candidate_boundaries)
        chosen = _choose_dp_strategy_2_spans(
            audio=audio,
            sample_rate=sample_rate,
            chars=chars,
            region_boundaries=region_boundaries,
            start_token_idx=region_start,
            end_boundary=region_end,
            candidate_boundaries=candidate_boundaries,
            prepass=prepass,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            trim_cache=trim_cache,
        )
        if chosen:
            all_spans.extend(chosen)
        else:
            all_spans.append(
                _make_priority_span(
                    audio=audio,
                    sample_rate=sample_rate,
                    chars=chars,
                    start_token_idx=region_start,
                    boundary=region_end,
                    phase="leftover",
                    min_seg_sec=min_seg_sec,
                    max_seg_sec=max_seg_sec,
                    trim_cache=trim_cache,
                )
            )
        region_start = region_end.token_idx + 1

    segments = [
        _priority_span_to_segment(span, chars, strategy="dp_strategy_2")
        for span in all_spans
    ]
    chosen_cuts = [
        _priority_span_to_cut(span, strategy="dp_strategy_2")
        for span in all_spans
    ]
    meta = {
        "dp_strategy_2_must_cut_boundary_count": must_candidate_count,
        "dp_strategy_2_legal_candidate_count": legal_candidate_count,
        "dp_strategy_2_weak_p20_ms": float(prepass.weak.p20_ms),
        "dp_strategy_2_weak_p60_ms": float(prepass.weak.p60_ms),
        "dp_strategy_2_strong_p60_ms": float(prepass.strong.p60_ms),
        "dp_strategy_2_strong_p80_ms": float(prepass.strong.p80_ms),
    }
    return segments, chosen_cuts, None, meta


def _plan_segments(
    *,
    strategy: str,
    audio: np.ndarray,
    sample_rate: int,
    prepass: FullPrepass,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    chars: Sequence[CharToken],
    boundaries: Sequence[RoughBoundary],
    min_seg_sec: float,
    max_seg_sec: float,
) -> tuple[list[RoughSegment], list[dict[str, Any]], Optional[str], dict[str, Any]]:
    if strategy == "legacy_dp":
        segments, chosen_cuts, error = _plan_segments_legacy_dp(
            audio=audio,
            sample_rate=sample_rate,
            corrected_full_text=corrected_full_text,
            corrected_token_to_positions=corrected_token_to_positions,
            chars=chars,
            boundaries=boundaries,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
        )
        return segments, chosen_cuts, error, {}
    if strategy == "priority_silence_v1":
        segments, chosen_cuts, error = _plan_segments_priority_silence_v1(
            audio=audio,
            sample_rate=sample_rate,
            prepass=prepass,
            chars=chars,
            boundaries=boundaries,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
        )
        return segments, chosen_cuts, error, {}
    if strategy == "priority_silence_v2":
        return _plan_segments_priority_silence_v2(
            audio=audio,
            sample_rate=sample_rate,
            prepass=prepass,
            chars=chars,
            boundaries=boundaries,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
        )
    if strategy == "dp_strategy_2":
        return _plan_segments_dp_strategy_2(
            audio=audio,
            sample_rate=sample_rate,
            prepass=prepass,
            chars=chars,
            boundaries=boundaries,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
        )
    return [], [], f"unknown rough cut strategy: {strategy}", {}


def _segment_reason(
    *,
    boundary: RoughBoundary,
    raw_sec: float,
    trimmed_sec: float,
    invalid_penalty: int,
    maybe_penalty: int,
) -> str:
    tier = (
        "must-valid"
        if invalid_penalty == 0 and maybe_penalty == 0
        else "maybe-valid"
        if invalid_penalty == 0
        else "invalid"
    )
    if boundary.boundary_kind == "final":
        return (
            f"final boundary; {tier} segment "
            f"(raw {raw_sec:.2f}s, trimmed {trimmed_sec:.2f}s)"
        )
    return (
        f"{boundary.cut_policy} / {boundary.quality} {boundary.boundary_kind} pause at '{boundary.boundary_text}' "
        f"({boundary.pause_ms:.0f}ms); {tier} segment "
        f"(raw {raw_sec:.2f}s, trimmed {trimmed_sec:.2f}s)"
    )


async def run_rough_cut_phase(
    *,
    audio: np.ndarray,
    sample_rate: int,
    prepass: FullPrepass,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    min_seg_sec: float,
    max_seg_sec: float,
    strategy: str = "legacy_dp",
) -> Phase3Result:
    result = Phase3Result()
    chars = prepass.global_chars
    if not chars:
        return result

    boundaries = _extract_punctuation_boundaries(
        corrected_full_text,
        corrected_token_to_positions,
        chars,
        prepass,
    )
    segments, chosen_cuts, error, planner_meta = _plan_segments(
        strategy=strategy,
        audio=audio,
        sample_rate=sample_rate,
        prepass=prepass,
        corrected_full_text=corrected_full_text,
        corrected_token_to_positions=corrected_token_to_positions,
        chars=chars,
        boundaries=boundaries,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
    )
    result.segments.extend(segments)

    response_cuts = []
    for boundary in boundaries:
        cut = {
            "candidate_id": int(boundary.candidate_id),
            "cut_char_idx": int(boundary.token_idx),
            "boundary_kind": boundary.boundary_kind,
            "boundary_quality": boundary.quality,
            "cut_policy": boundary.cut_policy,
            "boundary_text": boundary.boundary_text,
            "pause_ms": boundary.pause_ms,
            "reason": boundary.reason,
        }
        if strategy in {"priority_silence_v1", "priority_silence_v2"}:
            cut["strategy_cut_policy"] = _priority_boundary_policy(boundary, prepass)
            if strategy == "priority_silence_v2":
                cut["strategy_v2_legal_cut"] = _priority_v2_legal_cut(boundary, prepass)
        elif strategy == "dp_strategy_2":
            cut["strategy_cut_policy"] = (
                "must_cut"
                if _dp_strategy_2_must_cut(boundary, prepass)
                else "legal_cut"
                if _dp_strategy_2_legal_cut(boundary, prepass)
                else "not_candidate"
            )
            cut["strategy_cut_score"] = _dp_strategy_2_cut_score(boundary, prepass)
        response_cuts.append(cut)
    result.block_traces.append(
        BlockCallTrace(
            block_index=1,
            char_start_idx=0,
            char_end_idx=len(chars) - 1,
            block_start_sec=float(chars[0].start_sec),
            block_end_sec=float(chars[-1].end_sec),
            prompt_summary={
                "planner": "deterministic_punctuation_dp" if strategy == "legacy_dp" else strategy,
                "strategy": strategy,
                "candidate_boundary_count": len(boundaries),
                "perfect_boundary_count": sum(1 for b in boundaries if b.quality == "perfect"),
                "ok_boundary_count": sum(1 for b in boundaries if b.quality == "ok"),
                "bad_boundary_count": sum(1 for b in boundaries if b.quality == "bad"),
                "must_cut_boundary_count": sum(1 for b in boundaries if b.cut_policy == "must_cut"),
                "no_cut_boundary_count": sum(1 for b in boundaries if b.cut_policy == "no_cut"),
                "text_preview": _extract_span_text(
                    corrected_full_text,
                    corrected_token_to_positions,
                    0,
                    len(chars) - 1,
                )[:160],
                "trim_top_db": _TRIM_TOP_DB,
                "classification_mode": "rule_based",
                **(
                    {
                        "priority_must_cut_boundary_count": sum(
                            1 for b in boundaries if _priority_must_cut(b, prepass)
                        ),
                        "priority_good_cut_boundary_count": sum(
                            1
                            for b in boundaries
                            if (
                                _priority_v2_good_cut(b, prepass)
                                if strategy == "priority_silence_v2"
                                else _priority_good_cut(b, prepass)
                            )
                        ),
                        "priority_valid_cut_boundary_count": sum(
                            1 for b in boundaries if _priority_valid_cut(b, prepass)
                        ),
                    }
                    if strategy in {"priority_silence_v1", "priority_silence_v2"}
                    else {}
                ),
                **planner_meta,
            },
            response_cuts=response_cuts,
            applied_cuts=chosen_cuts,
            carry_over_from_idx=None,
            error=error,
        )
    )
    return result
