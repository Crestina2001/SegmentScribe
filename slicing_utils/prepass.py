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
    PausePercentiles,
    VadConfig,
    _choose_vad_backend,
    _collect_pauses,
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
        }


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
) -> list[tuple[str, list[CharToken], tuple[int, int]]]:
    batch_size = max(1, int(max_inference_batch_size or 1))
    results: list[tuple[str, list[CharToken], tuple[int, int]]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        windows: list[tuple[np.ndarray, int]] = [
            (np.ascontiguousarray(audio[start:end], dtype=np.float32), sample_rate)
            for start, end in batch
        ]
        try:
            transcripts = asr_backend.transcribe_windows(windows)
        except Exception as exc:
            logger.warning(
                "Pre-pass ASR failed on batch starting at chunk %d: %s",
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


def _detect_warnings(
    full_text: str,
    global_chars: Sequence[CharToken],
    strong: PausePercentiles,
    weak: PausePercentiles,
    no_punc: PausePercentiles,
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
        gap_ms = (float(right.start_sec) - float(left.end_sec)) * 1000.0
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
) -> FullPrepass:
    """Run VAD + 30s chunking + ASR and stitch into a single char stream."""
    if audio.size == 0:
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
        )

    regions = _run_vad(audio, sample_rate, vad_cfg)
    chunks = _pack_regions_into_chunks(
        regions,
        sample_rate,
        chunk_sec=chunk_sec,
        total_samples=len(audio),
        audio=audio,
    )
    if not chunks:
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
        )

    transcribed = _transcribe_chunks(
        audio,
        sample_rate,
        chunks,
        asr_backend,
        max_inference_batch_size=max_inference_batch_size,
    )

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

    # Compute pause quantiles on the stitched stream. Reuse the shared helper
    # which accepts (full_text, chars) tuples and classifies via punctuation.
    strong_ms, weak_ms, no_punc_ms, purged = _collect_pauses([(full_text, global_chars)])
    strong = _quantiles(strong_ms)
    weak = _quantiles(weak_ms)
    no_punc = _quantiles(no_punc_ms)

    warnings = _detect_warnings(full_text, global_chars, strong, weak, no_punc)

    total_speech_samples = sum(end - start for start, end in chunks)
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
    )
