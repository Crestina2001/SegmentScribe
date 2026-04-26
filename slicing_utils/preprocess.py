"""Per-source pre-pass: VAD + 30s chunking + ASR -> pause percentiles.

Before the sliding-window loop runs, this module transcribes the full source
once in 30s chunks aligned with speech activity, then computes source-wide
quantile distributions of the inter-character pause durations in three
categories:

- ``strong``  : pauses following a strong stopping punctuation (``。？！.?!``)
- ``weak``    : pauses following a weak  stopping punctuation (``，；、,;:``)
- ``no_punc`` : pauses following a non-punctuation character

Extremely long pauses (> ``MAX_PAUSE_MS``) are purged before computing
percentiles so they don't dominate the tail of the distribution.

These statistics are injected into every per-window LLM prompt so the model
can spot two kinds of anomalies *and* visually verify them via
``zoom_between_chars``:

- a no-punc pause that is unusually long relative to the no-punc distribution
  (speaker stalled mid-phrase, candidate for keep=false);
- a weak/strong-punc pause that is unusually short relative to the punc
  distribution, especially when a nearby no-punc pause is longer (the ASR may
  have mispredicted the punctuation, and the true semantic boundary is
  actually at the longer no-punc pause).
"""

from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .asr import AsrBackend, CharToken


STRONG_STOPS = frozenset("。？！.?!")
WEAK_STOPS = frozenset("，；、,;:")
ALL_STOPS = STRONG_STOPS | WEAK_STOPS

# Purge pauses longer than 3 seconds before computing quantiles. Such gaps are
# almost always track boundaries, long silences between sentences, or ASR
# alignment errors; they would otherwise drag the p95 so high that nothing
# ever looks "unusually long" to the LLM.
MAX_PAUSE_MS: float = 3000.0

# Quantile levels surfaced in the prompt: 5th / 20th / 40th / 50th (median) /
# 60th / 80th / 95th percentiles.
QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.20, 0.40, 0.50, 0.60, 0.80, 0.95)
QUANTILE_NAMES: tuple[str, ...] = ("p5", "p20", "p40", "p50", "p60", "p80", "p95")

# Conservative defaults used when a bucket has too few samples (min 5) to
# produce meaningful quantiles. Roughly interpolates 150ms -> 1500ms.
_FALLBACK_VALUES: tuple[float, ...] = (150.0, 280.0, 420.0, 500.0, 600.0, 900.0, 1500.0)


@dataclass(frozen=True)
class PausePercentiles:
    """Quantiles of inter-char pause durations around one pause class."""

    count: int
    p5_ms: float
    p20_ms: float
    p40_ms: float
    p50_ms: float  # median
    p60_ms: float
    p80_ms: float
    p95_ms: float
    is_fallback: bool = False

    def to_dict(self, *, label: str, chars: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": label,
            "n": self.count,
            "p5_ms": round(self.p5_ms, 1),
            "p20_ms": round(self.p20_ms, 1),
            "p40_ms": round(self.p40_ms, 1),
            "p50_ms": round(self.p50_ms, 1),
            "p60_ms": round(self.p60_ms, 1),
            "p80_ms": round(self.p80_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "is_fallback": self.is_fallback,
        }
        if chars:
            payload["chars"] = chars
        return payload


@dataclass(frozen=True)
class SourcePrepass:
    """Result of running the pre-pass over a single source file."""

    strong: PausePercentiles
    weak: PausePercentiles
    no_punc: PausePercentiles
    chunk_count: int
    total_speech_sec: float
    purged_count: int = 0
    debug_chunks: list[dict[str, Any]] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "strong": self.strong.to_dict(
                label="pause after strong stop",
                chars="。？！.?!",
            ),
            "weak": self.weak.to_dict(
                label="pause after weak stop",
                chars="，；、,;:",
            ),
            "no_punc": self.no_punc.to_dict(
                label="pause after a non-punctuation char",
            ),
            "chunk_count": self.chunk_count,
            "total_speech_sec": round(self.total_speech_sec, 3),
            "purged_count": self.purged_count,
            "max_pause_ms_used_for_purge": MAX_PAUSE_MS,
        }


@dataclass(frozen=True)
class VadConfig:
    backend: str = "auto"  # "silero" | "librosa" | "auto"
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 300
    speech_pad_ms: int = 200


def _fallback_percentiles(count: int = 0) -> PausePercentiles:
    return PausePercentiles(
        count=count,
        p5_ms=_FALLBACK_VALUES[0],
        p20_ms=_FALLBACK_VALUES[1],
        p40_ms=_FALLBACK_VALUES[2],
        p50_ms=_FALLBACK_VALUES[3],
        p60_ms=_FALLBACK_VALUES[4],
        p80_ms=_FALLBACK_VALUES[5],
        p95_ms=_FALLBACK_VALUES[6],
        is_fallback=True,
    )


def empty_prepass() -> SourcePrepass:
    """Prepass filled with fallback percentiles (used when pre-pass is disabled)."""
    return SourcePrepass(
        strong=_fallback_percentiles(),
        weak=_fallback_percentiles(),
        no_punc=_fallback_percentiles(),
        chunk_count=0,
        total_speech_sec=0.0,
        purged_count=0,
        debug_chunks=[],
    )


def _choose_vad_backend(requested: str) -> str:
    if requested == "silero":
        try:
            import silero_vad  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "silero-vad is not installed. Install with: pip install silero-vad"
            ) from exc
        return "silero"
    if requested == "librosa":
        return "librosa"
    try:
        import silero_vad  # noqa: F401
        import torch  # noqa: F401
        return "silero"
    except ImportError:
        return "librosa"


def _silero_regions(
    audio: np.ndarray,
    sample_rate: int,
    cfg: VadConfig,
) -> list[tuple[int, int]]:
    from silero_vad import get_speech_timestamps, load_silero_vad
    import torch

    model = load_silero_vad()
    speech = torch.from_numpy(audio)
    raw = get_speech_timestamps(
        speech,
        model,
        threshold=cfg.threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=cfg.min_speech_ms,
        min_silence_duration_ms=cfg.min_silence_ms,
        speech_pad_ms=cfg.speech_pad_ms,
        return_seconds=False,
    )
    regions: list[tuple[int, int]] = []
    for entry in raw:
        start = int(entry["start"])
        end = int(entry["end"])
        if end > start:
            regions.append((start, end))
    return regions


def _librosa_regions(
    audio: np.ndarray,
    sample_rate: int,
    cfg: VadConfig,
) -> list[tuple[int, int]]:
    import librosa

    top_db = max(10, int(round((1.0 - cfg.threshold) * 60)))
    intervals = librosa.effects.split(
        audio,
        top_db=top_db,
        frame_length=2048,
        hop_length=512,
    )
    pad = int(sample_rate * (cfg.speech_pad_ms / 1000.0))
    min_samples = int(sample_rate * (cfg.min_speech_ms / 1000.0))
    min_silence_samples = int(sample_rate * (cfg.min_silence_ms / 1000.0))
    regions: list[tuple[int, int]] = []
    for start, end in intervals.tolist():
        start = max(0, int(start) - pad)
        end = min(len(audio), int(end) + pad)
        if end - start < min_samples:
            continue
        if regions:
            silence = start - regions[-1][1]
            if silence < min_silence_samples:
                regions[-1] = (regions[-1][0], end)
                continue
        regions.append((start, end))
    return regions


def _find_quiet_boundary(
    audio: Optional[np.ndarray],
    sample_rate: int,
    start_sample: int,
    max_end_sample: int,
    *,
    min_cut_sec: float = 10.0,
    window_ms: float = 100.0,
) -> int:
    """Pick the lowest-energy point between ``start + min_cut_sec`` and ``max_end``.

    This avoids exact hard seams inside fluent speech when a VAD region is
    longer than the pre-pass chunk budget. If audio is unavailable or the
    search window is invalid, fall back to ``max_end_sample``.
    """
    if audio is None or sample_rate <= 0:
        return max_end_sample

    audio_1d = np.asarray(audio)
    if audio_1d.ndim > 1:
        audio_1d = np.mean(audio_1d, axis=-1)

    lo = int(start_sample + round(min_cut_sec * sample_rate))
    hi = int(max_end_sample)
    lo = max(int(start_sample) + 1, min(lo, len(audio_1d)))
    hi = max(lo, min(hi, len(audio_1d)))
    if hi <= lo:
        return max_end_sample

    win = max(4, int(round((window_ms / 1000.0) * sample_rate)))
    search = np.abs(np.asarray(audio_1d[lo:hi], dtype=np.float32))
    if search.size <= win:
        return lo + int(np.argmin(search)) if search.size else max_end_sample

    window_sums = np.convolve(search, np.ones(win, dtype=np.float32), mode="valid")
    if float(np.max(window_sums) - np.min(window_sums)) <= 1e-6:
        return max_end_sample
    quiet_window_start = int(np.argmin(window_sums))
    quiet_window = search[quiet_window_start : quiet_window_start + win]
    quiet_inner = int(np.argmin(quiet_window)) if quiet_window.size else 0
    return lo + quiet_window_start + quiet_inner


def _pack_regions_into_chunks(
    regions: Sequence[tuple[int, int]],
    sample_rate: int,
    chunk_sec: float,
    total_samples: int,
    audio: Optional[np.ndarray] = None,
) -> list[tuple[int, int]]:
    """Pack VAD regions into chunks no longer than ``chunk_sec`` seconds.

    When a single region is longer than the chunk budget, split near the
    quietest point between 10s and ``chunk_sec`` from the chunk start. When
    there are no regions, split the entire audio the same way so we still get
    some stats.
    """
    if chunk_sec <= 0:
        return [(0, total_samples)] if total_samples > 0 else []
    budget_samples = int(round(chunk_sec * sample_rate))
    if budget_samples <= 0:
        return [(0, total_samples)] if total_samples > 0 else []

    if not regions:
        chunks: list[tuple[int, int]] = []
        cur = 0
        while cur < total_samples:
            hard_nxt = min(cur + budget_samples, total_samples)
            nxt = _find_quiet_boundary(audio, sample_rate, cur, hard_nxt)
            if nxt > cur:
                chunks.append((cur, nxt))
            cur = nxt
        return chunks

    chunks: list[tuple[int, int]] = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None
    for r_start, r_end in regions:
        while r_end - r_start > budget_samples:
            if cur_start is None:
                cur_start = r_start
                cur_end = _find_quiet_boundary(
                    audio,
                    sample_rate,
                    r_start,
                    r_start + budget_samples,
                )
                chunks.append((cur_start, cur_end))
                r_start = cur_end
                cur_start = cur_end = None
            else:
                chunks.append((cur_start, cur_end or cur_start))
                cur_start = cur_end = None
        if cur_start is None:
            cur_start = r_start
            cur_end = r_end
            continue
        assert cur_end is not None
        if r_end - cur_start <= budget_samples:
            cur_end = r_end
        else:
            chunks.append((cur_start, cur_end))
            cur_start = r_start
            cur_end = r_end
    if cur_start is not None and cur_end is not None:
        chunks.append((cur_start, cur_end))
    return chunks


def _quantiles(samples_ms: Sequence[float]) -> PausePercentiles:
    arr = np.asarray(samples_ms, dtype=np.float64)
    if arr.size < 5:
        return _fallback_percentiles(count=int(arr.size))
    values = np.quantile(arr, QUANTILE_LEVELS)
    return PausePercentiles(
        count=int(arr.size),
        p5_ms=float(values[0]),
        p20_ms=float(values[1]),
        p40_ms=float(values[2]),
        p50_ms=float(values[3]),
        p60_ms=float(values[4]),
        p80_ms=float(values[5]),
        p95_ms=float(values[6]),
        is_fallback=False,
    )


def _is_punctuation_or_space(ch: str) -> bool:
    if ch.isspace():
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def _classify_text_fragment(text: str) -> str:
    """Classify the pause following ``text`` into 'strong' / 'weak' / 'no_punc'."""
    for ch in text:
        if ch in STRONG_STOPS:
            return "strong"
        if ch in WEAK_STOPS:
            return "weak"
    return "no_punc"


def _token_to_text_positions(full_text: str, tokens: Sequence[CharToken]) -> dict[int, list[int]]:
    text_units = [(pos, ch) for pos, ch in enumerate(full_text) if not _is_punctuation_or_space(ch)]
    token_units: list[tuple[int, str]] = []
    for token_idx, token in enumerate(tokens):
        for ch in str(token.char):
            if not _is_punctuation_or_space(ch):
                token_units.append((token_idx, ch))

    token_to_positions: dict[int, list[int]] = {}
    for unit_idx, (token_idx, _ch) in enumerate(token_units):
        if unit_idx >= len(text_units):
            break
        token_to_positions.setdefault(token_idx, []).append(text_units[unit_idx][0])
    return token_to_positions


def _classify_pause_from_text(
    full_text: str,
    token_to_positions: dict[int, list[int]],
    token_idx: int,
) -> str:
    """Classify the pause after ``tokens[token_idx]`` using punctuation in ``full_text``."""
    positions = token_to_positions.get(token_idx)
    if not positions:
        return "no_punc"
    cursor = max(positions) + 1
    punctuation_run: list[str] = []
    while cursor < len(full_text):
        ch = full_text[cursor]
        if _is_punctuation_or_space(ch):
            punctuation_run.append(ch)
            cursor += 1
            continue
        break
    return _classify_text_fragment("".join(punctuation_run))


def _collect_pauses(
    chunks: Sequence[Any],
) -> tuple[list[float], list[float], list[float], int]:
    """Return ``(strong_ms, weak_ms, no_punc_ms, purged_count)``.

    Pauses longer than :data:`MAX_PAUSE_MS` are purged; ``purged_count`` tracks
    how many gaps were removed (across all categories) so the prompt and
    summary can be transparent about it.
    """
    strong_ms: list[float] = []
    weak_ms: list[float] = []
    no_punc_ms: list[float] = []
    purged = 0
    for item in chunks:
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
        ):
            full_text, chars = item
            token_to_positions = _token_to_text_positions(full_text, chars)
            classify = lambda idx: _classify_pause_from_text(full_text, token_to_positions, idx)
        else:
            chars = item
            classify = lambda idx: _classify_text_fragment(str(chars[idx].char))
        for i in range(len(chars) - 1):
            gap_ms = (chars[i + 1].start_sec - chars[i].end_sec) * 1000.0
            if gap_ms < 0:
                continue
            if gap_ms > MAX_PAUSE_MS:
                purged += 1
                continue
            category = classify(i)
            if category == "strong":
                strong_ms.append(gap_ms)
            elif category == "weak":
                weak_ms.append(gap_ms)
            else:
                no_punc_ms.append(gap_ms)
    return strong_ms, weak_ms, no_punc_ms, purged


def compute_source_prepass(
    audio: np.ndarray,
    sample_rate: int,
    asr_backend: AsrBackend,
    *,
    chunk_sec: float = 30.0,
    vad: VadConfig = VadConfig(),
    max_inference_batch_size: Optional[int] = None,
) -> SourcePrepass:
    """Run VAD + 30s chunking + ASR over the full source and compute quantiles.

    ``asr_backend.transcribe_windows`` is called directly (bypassing the async
    scheduler) so the pre-pass can reuse the same underlying model without
    interfering with the sliding-loop microbatcher.
    """
    if audio.size == 0:
        return empty_prepass()

    backend_name = _choose_vad_backend(vad.backend)
    try:
        if backend_name == "silero":
            regions = _silero_regions(audio, sample_rate, vad)
        else:
            regions = _librosa_regions(audio, sample_rate, vad)
    except Exception as exc:
        print(
            f"[slide_LLM] VAD failed ({backend_name}): {exc}. "
            f"Falling back to pure 30s hard splits for pause-stat pre-pass.",
            file=sys.stderr,
        )
        regions = []

    chunks = _pack_regions_into_chunks(
        regions,
        sample_rate,
        chunk_sec=chunk_sec,
        total_samples=len(audio),
        audio=audio,
    )
    if not chunks:
        return empty_prepass()

    batch_size = max(1, int(max_inference_batch_size or 1))
    total_speech_samples = sum(end - start for start, end in chunks)
    text_and_chars_per_chunk: list[tuple[str, list[CharToken]]] = []
    debug_chunks: list[dict[str, Any]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        windows: list[tuple[np.ndarray, int]] = [
            (np.ascontiguousarray(audio[start:end], dtype=np.float32), sample_rate)
            for start, end in batch
        ]
        try:
            transcripts = asr_backend.transcribe_windows(windows)
        except Exception as exc:
            print(
                f"[slide_LLM] Pre-pass ASR failed on batch starting at chunk {i}: {exc}",
                file=sys.stderr,
            )
            transcripts = []
        if len(transcripts) != len(batch):
            for _ in batch[len(transcripts):]:
                text_and_chars_per_chunk.append(("", []))
                debug_chunks.append({})
            for (start, end), tr in zip(batch[: len(transcripts)], transcripts):
                shifted = [
                    CharToken(
                        idx=c.idx,
                        char=c.char,
                        start_sec=float(c.start_sec) + start / float(sample_rate),
                        end_sec=float(c.end_sec) + start / float(sample_rate),
                    )
                    for c in tr.chars
                ]
                text_and_chars_per_chunk.append((str(tr.text or ""), shifted))
                debug_chunks.append(
                    {
                        "start_sec": round(start / float(sample_rate), 3),
                        "end_sec": round(end / float(sample_rate), 3),
                        "text": str(tr.text or ""),
                        "chars": [
                            {
                                "idx": c.idx,
                                "char": c.char,
                                "start_sec": round(c.start_sec, 3),
                                "end_sec": round(c.end_sec, 3),
                            }
                            for c in shifted
                        ],
                    }
                )
            continue
        for (start, _end), tr in zip(batch, transcripts):
            offset = start / float(sample_rate)
            shifted: list[CharToken] = [
                CharToken(
                    idx=c.idx,
                    char=c.char,
                    start_sec=float(c.start_sec) + offset,
                    end_sec=float(c.end_sec) + offset,
                )
                for c in tr.chars
            ]
            text_and_chars_per_chunk.append((str(tr.text or ""), shifted))
            debug_chunks.append(
                {
                    "start_sec": round(start / float(sample_rate), 3),
                    "end_sec": round(_end / float(sample_rate), 3),
                    "text": str(tr.text or ""),
                    "chars": [
                        {
                            "idx": c.idx,
                            "char": c.char,
                            "start_sec": round(c.start_sec, 3),
                            "end_sec": round(c.end_sec, 3),
                        }
                        for c in shifted
                    ],
                }
            )

    strong_ms, weak_ms, no_punc_ms, purged = _collect_pauses(text_and_chars_per_chunk)
    return SourcePrepass(
        strong=_quantiles(strong_ms),
        weak=_quantiles(weak_ms),
        no_punc=_quantiles(no_punc_ms),
        chunk_count=len(chunks),
        total_speech_sec=float(total_speech_samples) / float(sample_rate),
        purged_count=purged,
        debug_chunks=debug_chunks,
    )
