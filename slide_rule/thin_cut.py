"""Rule-based thin cut using ``librosa.effects.trim``."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from slicing_utils.rough_cut import RoughSegment

from .config import DEFAULT_THIN_CUT_PADDING_SEC

TRIM_TOP_DB = 60


@dataclass
class RefinedSegment:
    rough: RoughSegment
    cut_start_sec: float
    cut_end_sec: float
    duration_sec: float
    zoom_reason: Optional[str] = None
    zoom_error: Optional[str] = None
    zoom_id: Optional[str] = None


@dataclass
class ThinCutCallTrace:
    segment_index: int
    char_idx_a: int
    char_idx_b: Optional[int]
    zoom_start_sec: float
    zoom_end_sec: float
    candidates: list[dict[str, float]]
    response: Optional[dict[str, int]]
    applied_cut_start_sec: float
    applied_cut_end_sec: float
    zoom_id: Optional[str]
    png_path: Optional[str]
    error: Optional[str] = None


@dataclass
class Phase4Result:
    refined: list[RefinedSegment] = field(default_factory=list)
    traces: list[ThinCutCallTrace] = field(default_factory=list)


@dataclass
class ThinCutTrace:
    segment_index: int
    original_start_sec: float
    original_end_sec: float
    trimmed_start_sec: float
    trimmed_end_sec: float
    trim_start_sample: int
    trim_end_sample: int
    top_db: int = TRIM_TOP_DB
    error: Optional[str] = None


def run_thin_cut_phase(
    *,
    audio: np.ndarray,
    sample_rate: int,
    segments: Sequence[RoughSegment],
    top_db: int = TRIM_TOP_DB,
    padding_sec: float = DEFAULT_THIN_CUT_PADDING_SEC,
) -> Phase4Result:
    result = Phase4Result()
    for seg_idx, rough in enumerate(segments, start=1):
        refined, trace = _trim_one_segment(
            audio=audio,
            sample_rate=sample_rate,
            rough=rough,
            segment_index=seg_idx,
            top_db=top_db,
            padding_sec=padding_sec,
        )
        result.refined.append(refined)
        result.traces.append(
            ThinCutCallTrace(
                segment_index=trace.segment_index,
                char_idx_a=int(refined.rough.char_end_idx),
                char_idx_b=None,
                zoom_start_sec=trace.original_start_sec,
                zoom_end_sec=trace.original_end_sec,
                candidates=[],
                response={
                    "trim_start_sample": trace.trim_start_sample,
                    "trim_end_sample": trace.trim_end_sample,
                    "top_db": trace.top_db,
                    "padding_sec": padding_sec,
                },
                applied_cut_start_sec=refined.cut_start_sec,
                applied_cut_end_sec=refined.cut_end_sec,
                zoom_id=f"thin_{seg_idx:06d}",
                png_path=None,
                error=trace.error,
            )
        )
    return result


def _trim_one_segment(
    *,
    audio: np.ndarray,
    sample_rate: int,
    rough: RoughSegment,
    segment_index: int,
    top_db: int,
    padding_sec: float,
) -> tuple[RefinedSegment, ThinCutTrace]:
    start_sample = max(0, int(round(float(rough.start_sec) * sample_rate)))
    end_sample = min(len(audio), max(start_sample, int(round(float(rough.end_sec) * sample_rate))))
    seg_audio = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
    if seg_audio.size == 0:
        return _fallback_refined(
            rough=rough,
            segment_index=segment_index,
            top_db=top_db,
            error="trim_empty",
        )

    import librosa

    trimmed, index = librosa.effects.trim(seg_audio, top_db=top_db)
    if trimmed.size == 0 or len(index) != 2 or int(index[1]) <= int(index[0]):
        return _fallback_refined(
            rough=rough,
            segment_index=segment_index,
            top_db=top_db,
            error="trim_empty",
        )

    trim_start = int(index[0])
    trim_end = int(index[1])
    padding_samples = max(0, int(round(float(padding_sec) * sample_rate)))
    padded_start = max(0, trim_start - padding_samples)
    padded_end = min(len(seg_audio), trim_end + padding_samples)
    padded_start_sec = (start_sample + padded_start) / float(sample_rate)
    padded_end_sec = (start_sample + padded_end) / float(sample_rate)
    trimmed_rough = replace(rough, start_sec=padded_start_sec)
    refined = RefinedSegment(
        rough=trimmed_rough,
        cut_start_sec=padded_end_sec,
        cut_end_sec=padded_end_sec,
        duration_sec=max(0.0, padded_end_sec - padded_start_sec),
        zoom_reason=f"librosa.trim(top_db={top_db}, padding_sec={padding_sec:g})",
        zoom_error=None,
        zoom_id=f"thin_{segment_index:06d}",
    )
    trace = ThinCutTrace(
        segment_index=segment_index,
        original_start_sec=float(rough.start_sec),
        original_end_sec=float(rough.end_sec),
        trimmed_start_sec=padded_start_sec,
        trimmed_end_sec=padded_end_sec,
        trim_start_sample=padded_start,
        trim_end_sample=padded_end,
        top_db=top_db,
        error=None,
    )
    return refined, trace


def _fallback_refined(
    *,
    rough: RoughSegment,
    segment_index: int,
    top_db: int,
    error: str,
) -> tuple[RefinedSegment, ThinCutTrace]:
    refined = RefinedSegment(
        rough=rough,
        cut_start_sec=float(rough.end_sec),
        cut_end_sec=float(rough.end_sec),
        duration_sec=max(0.0, float(rough.end_sec) - float(rough.start_sec)),
        zoom_reason=f"librosa.trim(top_db={top_db}) fallback",
        zoom_error=error,
        zoom_id=f"thin_{segment_index:06d}",
    )
    trace = ThinCutTrace(
        segment_index=segment_index,
        original_start_sec=float(rough.start_sec),
        original_end_sec=float(rough.end_sec),
        trimmed_start_sec=float(rough.start_sec),
        trimmed_end_sec=float(rough.end_sec),
        trim_start_sample=0,
        trim_end_sample=0,
        top_db=top_db,
        error=error,
    )
    return refined, trace
