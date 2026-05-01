"""Phase 1 - global VAD-packed ASR + pause quantiles + warnings.

Produces a :class:`FullPrepass` for the whole source:

- Strong / weak / no_punc pause quantiles computed over the full stream.
- A single stitched :class:`CharToken` list with absolute timestamps.
- A single ``full_text`` string concatenated from per-chunk ASR texts.
- A list of :class:`WarningSpan` describing every suspicious inter-char pause.

Later phases operate on this global structure, so the source is only ASR'd
once (vs. slide_LLM which re-runs ASR on every sliding window).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .asr import AsrBackend, CharToken
from .preprocess import (
    MAX_PAUSE_MS,
    PausePercentiles,
    VadConfig,
    _choose_vad_backend,
    _librosa_regions,
    _pack_regions_into_chunks,
    _quantiles,
    _silero_regions,
    empty_prepass,
)
from .waveform_viz import WarningSpan

from .shared import (
    pause_category_after_token,
    token_to_text_positions,
)


logger = logging.getLogger(__name__)


# When concatenating chunk texts into one document we insert a whitespace
# delimiter; punctuation/whitespace are ignored by every downstream mapping so
# this does not affect char<->text alignment.
CHUNK_TEXT_DELIMITER = " "


@dataclass
class FullPrepass:
    """Result of the full-source pre-pass."""

    strong: PausePercentiles
    weak: PausePercentiles
    no_punc: PausePercentiles
    global_chars: list[CharToken]
    full_text: str
    warnings: list[WarningSpan]
    chunk_spans: list[dict[str, Any]] = field(default_factory=list)
    total_speech_sec: float = 0.0
    purged_count: int = 0
    pause_ms_by_token: dict[int, float] = field(default_factory=dict)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "strong": self.strong.to_dict(label="pause after strong stop", chars="。？！.?!"),
            "weak": self.weak.to_dict(label="pause after weak stop", chars="，；、,;:"),
            "no_punc": self.no_punc.to_dict(label="pause after a non-punctuation char"),
            "chunk_count": len(self.chunk_spans),
            "global_char_count": len(self.global_chars),
            "warning_count": len(self.warnings),
            "total_speech_sec": round(self.total_speech_sec, 3),
            "purged_count": self.purged_count,
        }

    def to_stream_dump(self) -> dict[str, Any]:
        return {
            "summary": self.to_summary_dict(),
            "full_text": self.full_text,
            "chunks": self.chunk_spans,
            "global_chars": [
                {
                    "idx": c.idx,
                    "char": c.char,
                    "start_sec": round(c.start_sec, 3),
                    "end_sec": round(c.end_sec, 3),
                }
                for c in self.global_chars
            ],
            "warnings": [
                {
                    "left_idx": w.left_idx,
                    "right_idx": w.right_idx,
                    "left_char": w.left_char,
                    "right_char": w.right_char,
                    "gap_ms": w.gap_ms,
                    "category": w.category,
                    "reason": w.reason,
                }
                for w in self.warnings
            ],
            "pause_ms_by_token": {
                str(token_idx): round(float(pause_ms), 1)
                for token_idx, pause_ms in sorted(self.pause_ms_by_token.items())
            },
        }


@dataclass(frozen=True)
class FullPrepassPlan:
    chunks: list[tuple[int, int]]


def compact_pause_stats(prepass: FullPrepass) -> dict[str, dict[str, Any]]:
    """Return a compact dict summary of strong/weak/no_punc pause quantiles."""
    keys = ("p5_ms", "p20_ms", "p40_ms", "p50_ms", "p60_ms", "p80_ms", "p95_ms")

    def _pick(pp: PausePercentiles) -> dict[str, Any]:
        out: dict[str, Any] = {"n": pp.count, "is_fallback": pp.is_fallback}
        for key in keys:
            out[key] = round(float(getattr(pp, key)), 1)
        return out

    return {
        "strong_stops": _pick(prepass.strong),
        "weak_stops": _pick(prepass.weak),
        "no_punc": _pick(prepass.no_punc),
    }


def pause_ms_after_token(prepass: FullPrepass, token_idx: int) -> float:
    """Return the best available pause duration after ``token_idx`` in ms."""
    cached = prepass.pause_ms_by_token.get(int(token_idx))
    if cached is not None:
        return float(cached)
    chars = prepass.global_chars
    if token_idx < 0 or token_idx + 1 >= len(chars):
        return 0.0
    left = chars[token_idx]
    right = chars[token_idx + 1]
    return max(0.0, (float(right.start_sec) - float(left.end_sec)) * 1000.0)


def _run_vad(
    audio: np.ndarray,
    sample_rate: int,
    vad_cfg: VadConfig,
) -> list[tuple[int, int]]:
    backend = _choose_vad_backend(vad_cfg.backend)
    try:
        if backend == "silero":
            return _silero_regions(audio, sample_rate, vad_cfg)
        return _librosa_regions(audio, sample_rate, vad_cfg)
    except Exception as exc:
        logger.warning(
            "VAD failed (%s): %s. Falling back to pure 30s hard splits for the global pre-pass.",
            backend,
            exc,
        )
        return []


def _transcribe_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunks: Sequence[tuple[int, int]],
    asr_backend: AsrBackend,
    *,
    max_inference_batch_size: Optional[int],
    source_label: str = "source",
) -> list[tuple[str, list[CharToken], tuple[int, int]]]:
    batch_size = max(1, int(max_inference_batch_size or 1))
    results: list[tuple[str, list[CharToken], tuple[int, int]]] = []
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        batch_start = i + 1
        batch_end = i + len(batch)
        logger.info(
            "Qwen3-ASR processing %s chunk(s) %d-%d/%d",
            source_label,
            batch_start,
            batch_end,
            total,
        )
        windows: list[tuple[np.ndarray, int]] = [
            (np.ascontiguousarray(audio[start:end], dtype=np.float32), sample_rate)
            for start, end in batch
        ]
        try:
            transcripts = asr_backend.transcribe_windows(windows)
        except Exception as exc:
            logger.warning(
                "Pre-pass ASR failed on %s batch starting at chunk %d: %s",
                source_label,
                i,
                exc,
            )
            transcripts = []
        if len(transcripts) != len(batch):
            for span in batch[len(transcripts):]:
                results.append(("", [], span))
        pairs = list(zip(batch[: len(transcripts)], transcripts))
        for (start, end), tr in pairs:
            offset = start / float(sample_rate)
            shifted = [
                CharToken(
                    idx=c.idx,
                    char=c.char,
                    start_sec=float(c.start_sec) + offset,
                    end_sec=float(c.end_sec) + offset,
                )
                for c in tr.chars
            ]
            results.append((str(tr.text or ""), shifted, (start, end)))
    return results


def transcribe_chunk_windows(
    windows: Sequence[tuple[np.ndarray, int, tuple[int, int]]],
    asr_backend: AsrBackend,
) -> list[tuple[str, list[CharToken], tuple[int, int]]]:
    """Transcribe an arbitrary batch of source-local chunks.

    This is intentionally batch-shaped so callers can fill one ASR inference
    call with chunks from multiple source audios.
    """
    prepared: list[tuple[np.ndarray, int]] = [
        (np.ascontiguousarray(audio, dtype=np.float32), sample_rate)
        for audio, sample_rate, _span in windows
    ]
    transcripts = asr_backend.transcribe_windows(prepared)
    results: list[tuple[str, list[CharToken], tuple[int, int]]] = []
    for (_audio, sample_rate, (start, end)), tr in zip(windows[: len(transcripts)], transcripts):
        offset = start / float(sample_rate)
        shifted = [
            CharToken(
                idx=c.idx,
                char=c.char,
                start_sec=float(c.start_sec) + offset,
                end_sec=float(c.end_sec) + offset,
            )
            for c in tr.chars
        ]
        results.append((str(tr.text or ""), shifted, (start, end)))
    return results


def _timestamp_gap_ms(left: CharToken, right: CharToken) -> float:
    return max(0.0, (float(right.start_sec) - float(left.end_sec)) * 1000.0)


def _measure_punctuation_silence_ms(
    audio: np.ndarray,
    sample_rate: int,
    left: CharToken,
    right: CharToken,
    *,
    scan_sec: float = 0.55,
) -> float:
    """Estimate actual quiet span around a punctuation boundary.

    Aligner timestamps can put neighboring characters almost adjacent even
    when the waveform has a short audible pause. We scan backward from the
    right token and forward from the left token, then measure the contiguous
    low-energy region around the boundary.
    """
    if audio.size == 0 or sample_rate <= 0:
        return _timestamp_gap_ms(left, right)

    left_end = float(left.end_sec)
    right_start = float(right.start_sec)
    anchor_sec = (left_end + right_start) * 0.5 if right_start >= left_end else left_end
    win_start_sec = max(0.0, min(left_end, anchor_sec) - scan_sec)
    win_end_sec = min(len(audio) / float(sample_rate), max(right_start, anchor_sec) + scan_sec)
    start_sample = max(0, int(round(win_start_sec * sample_rate)))
    end_sample = min(len(audio), int(round(win_end_sec * sample_rate)))
    if end_sample <= start_sample:
        return _timestamp_gap_ms(left, right)

    window = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
    frame = max(1, int(round(0.02 * sample_rate)))
    hop = max(1, int(round(0.005 * sample_rate)))
    if window.size < frame:
        return _timestamp_gap_ms(left, right)

    rms: list[float] = []
    centers: list[float] = []
    for offset in range(0, window.size - frame + 1, hop):
        chunk = window[offset : offset + frame]
        rms.append(float(np.sqrt(np.mean(np.square(chunk)))))
        centers.append((start_sample + offset + frame * 0.5) / float(sample_rate))
    if not rms:
        return _timestamp_gap_ms(left, right)

    rms_arr = np.asarray(rms, dtype=np.float32)
    floor = float(np.percentile(rms_arr, 20))
    median = float(np.percentile(rms_arr, 50))
    peak = float(np.max(rms_arr))
    threshold = max(floor * 1.8, median * 0.45, peak * 0.025, 1e-5)
    silent = rms_arr <= threshold
    if not bool(np.any(silent)):
        return _timestamp_gap_ms(left, right)

    anchor_index = min(range(len(centers)), key=lambda i: abs(centers[i] - anchor_sec))
    search_radius = max(1, int(round(0.12 / 0.005)))
    if not silent[anchor_index]:
        lo = max(0, anchor_index - search_radius)
        hi = min(len(silent), anchor_index + search_radius + 1)
        candidates = [i for i in range(lo, hi) if silent[i]]
        if not candidates:
            return _timestamp_gap_ms(left, right)
        anchor_index = min(candidates, key=lambda i: abs(centers[i] - anchor_sec))

    left_i = anchor_index
    while left_i > 0 and silent[left_i - 1]:
        left_i -= 1
    right_i = anchor_index
    while right_i + 1 < len(silent) and silent[right_i + 1]:
        right_i += 1

    silence_start = centers[left_i] - frame * 0.5 / float(sample_rate)
    silence_end = centers[right_i] + frame * 0.5 / float(sample_rate)
    raw_gap_ms = _timestamp_gap_ms(left, right)
    measured_ms = max(0.0, (silence_end - silence_start) * 1000.0)
    return max(raw_gap_ms, measured_ms)


def _compute_pause_ms_by_token(
    audio: np.ndarray,
    sample_rate: int,
    full_text: str,
    global_chars: Sequence[CharToken],
) -> dict[int, float]:
    if len(global_chars) < 2:
        return {}
    token_to_positions = token_to_text_positions(full_text, global_chars)
    pause_ms_by_token: dict[int, float] = {}
    for token_idx, (left, right) in enumerate(zip(global_chars, global_chars[1:])):
        category = pause_category_after_token(full_text, token_to_positions, token_idx)
        if category in {"strong", "weak"}:
            pause_ms = _measure_punctuation_silence_ms(audio, sample_rate, left, right)
        else:
            pause_ms = _timestamp_gap_ms(left, right)
        pause_ms_by_token[token_idx] = round(float(pause_ms), 1)
    return pause_ms_by_token


def _collect_pauses_with_overrides(
    full_text: str,
    chars: Sequence[CharToken],
    pause_ms_by_token: dict[int, float],
) -> tuple[list[float], list[float], list[float], int]:
    token_to_positions = token_to_text_positions(full_text, chars)
    strong_ms: list[float] = []
    weak_ms: list[float] = []
    no_punc_ms: list[float] = []
    purged = 0
    for token_idx in range(max(0, len(chars) - 1)):
        gap_ms = float(pause_ms_by_token.get(token_idx, 0.0))
        if gap_ms < 0:
            continue
        if gap_ms > MAX_PAUSE_MS:
            purged += 1
            continue
        category = pause_category_after_token(full_text, token_to_positions, token_idx)
        if category == "strong":
            strong_ms.append(gap_ms)
        elif category == "weak":
            weak_ms.append(gap_ms)
        else:
            no_punc_ms.append(gap_ms)
    return strong_ms, weak_ms, no_punc_ms, purged


def _detect_warnings(
    full_text: str,
    global_chars: Sequence[CharToken],
    strong: PausePercentiles,
    weak: PausePercentiles,
    no_punc: PausePercentiles,
    pause_ms_by_token: dict[int, float],
) -> list[WarningSpan]:
    """Flag every suspicious inter-char pause on the whole stream."""
    if len(global_chars) < 2:
        return []

    token_to_positions = token_to_text_positions(full_text, global_chars)

    no_punc_threshold = max(float(no_punc.p95_ms), 180.0)
    weak_threshold = max(float(weak.p5_ms), float(no_punc.p60_ms))
    strong_threshold = max(float(strong.p5_ms), float(weak.p20_ms))

    warnings: list[WarningSpan] = []
    for token_idx, (left, right) in enumerate(zip(global_chars, global_chars[1:])):
        gap_ms = float(pause_ms_by_token.get(token_idx, _timestamp_gap_ms(left, right)))
        if gap_ms < 0:
            continue
        category = pause_category_after_token(full_text, token_to_positions, token_idx)
        is_suspicious = (
            (category == "no_punc" and gap_ms >= no_punc_threshold)
            or (category == "weak" and gap_ms <= weak_threshold)
            or (category == "strong" and gap_ms <= strong_threshold)
        )
        if not is_suspicious:
            continue
        warnings.append(
            WarningSpan(
                left_idx=int(left.idx),
                right_idx=int(right.idx),
                gap_ms=round(gap_ms, 1),
                category=category,
                reason="suspicious pause relative to source-wide pause statistics",
                left_char=str(left.char),
                right_char=str(right.char),
            )
        )
    return warnings


def compute_full_prepass(
    audio: np.ndarray,
    sample_rate: int,
    asr_backend: AsrBackend,
    *,
    chunk_sec: float = 30.0,
    vad_cfg: VadConfig = VadConfig(),
    max_inference_batch_size: Optional[int] = None,
    source_label: str = "source",
) -> FullPrepass:
    """Run VAD + 30s chunking + ASR and stitch into a single char stream."""
    plan = prepare_full_prepass_plan(
        audio,
        sample_rate,
        chunk_sec=chunk_sec,
        vad_cfg=vad_cfg,
    )
    if not plan.chunks:
        return _empty_full_prepass()

    transcribed = _transcribe_chunks(
        audio,
        sample_rate,
        plan.chunks,
        asr_backend,
        max_inference_batch_size=max_inference_batch_size,
        source_label=source_label,
    )
    return assemble_full_prepass(audio, sample_rate, transcribed)


def prepare_full_prepass_plan(
    audio: np.ndarray,
    sample_rate: int,
    *,
    chunk_sec: float = 30.0,
    vad_cfg: VadConfig = VadConfig(),
) -> FullPrepassPlan:
    """Run non-ASR pre-pass planning and return source-local audio chunks."""
    if audio.size == 0:
        return FullPrepassPlan(chunks=[])

    regions = _run_vad(audio, sample_rate, vad_cfg)
    chunks = _pack_regions_into_chunks(
        regions,
        sample_rate,
        chunk_sec=chunk_sec,
        total_samples=len(audio),
        audio=audio,
    )
    return FullPrepassPlan(chunks=list(chunks))


def assemble_full_prepass(
    audio: np.ndarray,
    sample_rate: int,
    transcribed: Sequence[tuple[str, list[CharToken], tuple[int, int]]],
) -> FullPrepass:
    """Stitch transcribed chunks into the same structure as compute_full_prepass."""
    if not transcribed:
        return _empty_full_prepass()

    # Stitch chunks into one global stream. Each chunk has its own token idx
    # numbering; re-index globally so downstream phases can trust chars[i].idx == i.
    global_chars: list[CharToken] = []
    text_parts: list[str] = []
    chunk_spans: list[dict[str, Any]] = []
    # Preserve original per-chunk texts so we can also emit warnings mapping
    # back to the right chunk for the dump. Easy approach: classify pauses
    # using the full concatenated text and token_to_positions mapping.
    for chunk_idx, (text, chars, (start_sample, end_sample)) in enumerate(transcribed):
        start_sec = start_sample / float(sample_rate)
        end_sec = end_sample / float(sample_rate)
        # Re-index tokens into the global sequence.
        global_idx_start = len(global_chars)
        for c in chars:
            global_chars.append(
                CharToken(
                    idx=len(global_chars),
                    char=c.char,
                    start_sec=float(c.start_sec),
                    end_sec=float(c.end_sec),
                )
            )
        if text_parts:
            text_parts.append(CHUNK_TEXT_DELIMITER)
        text_parts.append(text)
        chunk_spans.append(
            {
                "chunk_index": chunk_idx,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "text": text,
                "char_idx_range": [global_idx_start, len(global_chars)],
            }
        )

    full_text = "".join(text_parts)

    pause_ms_by_token = _compute_pause_ms_by_token(audio, sample_rate, full_text, global_chars)

    # Compute pause quantiles on the stitched stream. For punctuation-adjacent
    # gaps, use waveform silence around the punctuation rather than trusting
    # only the forced-aligner timestamps.
    strong_ms, weak_ms, no_punc_ms, purged = _collect_pauses_with_overrides(
        full_text,
        global_chars,
        pause_ms_by_token,
    )
    strong = _quantiles(strong_ms)
    weak = _quantiles(weak_ms)
    no_punc = _quantiles(no_punc_ms)

    warnings = _detect_warnings(full_text, global_chars, strong, weak, no_punc, pause_ms_by_token)

    total_speech_samples = sum(end - start for _text, _chars, (start, end) in transcribed)
    return FullPrepass(
        strong=strong,
        weak=weak,
        no_punc=no_punc,
        global_chars=global_chars,
        full_text=full_text,
        warnings=warnings,
        chunk_spans=chunk_spans,
        total_speech_sec=float(total_speech_samples) / float(sample_rate),
        purged_count=purged,
        pause_ms_by_token=pause_ms_by_token,
    )


def _empty_full_prepass() -> FullPrepass:
    base = empty_prepass()
    return FullPrepass(
        strong=base.strong,
        weak=base.weak,
        no_punc=base.no_punc,
        global_chars=[],
        full_text="",
        warnings=[],
        chunk_spans=[],
        total_speech_sec=0.0,
        purged_count=0,
        pause_ms_by_token={},
    )
