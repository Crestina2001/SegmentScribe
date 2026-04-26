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
from typing import Any, Optional, Sequence

import numpy as np

from .asr import CharToken

from .prepass import FullPrepass
from .shared import STRONG_STOPS, WEAK_STOPS, is_punct_or_space


_TRIM_TOP_DB = 60
_MAYBE_SLACK_SEC = 0.35
_MAX_SEARCH_OVERSHOOT_SEC = 8.0
_ANCHOR_MIN_SEC = 15.0
_ANCHOR_MAX_SEC = 30.0


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


def _pause_ms(left: CharToken, right: CharToken) -> float:
    return max(0.0, (float(right.start_sec) - float(left.end_sec)) * 1000.0)


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
        pause_ms = _pause_ms(left_char, chars[token_idx + 1])
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
    start_sec = float(chars[start_token_idx].start_sec)
    in_window: list[RoughBoundary] = []
    before_max: list[RoughBoundary] = []
    for boundary in boundaries:
        if boundary.token_idx < start_token_idx:
            continue
        if boundary.cut_policy == "no_cut":
            continue
        raw_end_sec = (
            boundary.next_token_start_sec
            if boundary.boundary_kind != "final"
            else boundary.cut_sec
        )
        duration_sec = raw_end_sec - start_sec
        if duration_sec <= _ANCHOR_MAX_SEC + 1e-6:
            before_max.append(boundary)
        if _ANCHOR_MIN_SEC <= duration_sec <= _ANCHOR_MAX_SEC + 1e-6:
            in_window.append(boundary)

    if not before_max:
        return None

    tail_boundary = before_max[-1]
    tail_end_sec = (
        tail_boundary.next_token_start_sec
        if tail_boundary.boundary_kind != "final"
        else tail_boundary.cut_sec
    )
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


def _plan_segments(
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

                seg_start_sec = float(chars[seg_token_start].start_sec)
                raw_end_sec = (
                    boundary.next_token_start_sec
                    if boundary.boundary_kind != "final"
                    else boundary.cut_sec
                )
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
                        "end_sec": float(chars[seg_token_end].end_sec),
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
                        "anchor_window_start_sec": round(float(chars[start_token_idx].start_sec), 3),
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
    segments, chosen_cuts, error = _plan_segments(
        audio=audio,
        sample_rate=sample_rate,
        corrected_full_text=corrected_full_text,
        corrected_token_to_positions=corrected_token_to_positions,
        chars=chars,
        boundaries=boundaries,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
    )
    result.segments.extend(segments)

    response_cuts = [
        {
            "candidate_id": int(boundary.candidate_id),
            "cut_char_idx": int(boundary.token_idx),
            "boundary_kind": boundary.boundary_kind,
            "boundary_quality": boundary.quality,
            "cut_policy": boundary.cut_policy,
            "boundary_text": boundary.boundary_text,
            "pause_ms": boundary.pause_ms,
            "reason": boundary.reason,
        }
        for boundary in boundaries
    ]
    result.block_traces.append(
        BlockCallTrace(
            block_index=1,
            char_start_idx=0,
            char_end_idx=len(chars) - 1,
            block_start_sec=float(chars[0].start_sec),
            block_end_sec=float(chars[-1].end_sec),
            prompt_summary={
                "planner": "deterministic_punctuation_dp",
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
            },
            response_cuts=response_cuts,
            applied_cuts=chosen_cuts,
            carry_over_from_idx=None,
            error=error,
        )
    )
    return result
