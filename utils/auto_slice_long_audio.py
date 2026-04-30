#!/usr/bin/env python
"""Split long denoised audio into roughly even silence-aware WAV pieces.

This utility is intended to run after denoising and before slide_LLM. It uses
local RMS energy, not VAD, to find the longest low-energy span in a search
window around each ideal cut.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma")
TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SilenceSpan:
    start_sample: int
    end_sample: int

    @property
    def duration_samples(self) -> int:
        return max(0, self.end_sample - self.start_sample)


@dataclass(frozen=True)
class CutChoice:
    sample: int
    method: str
    silence_samples: int
    window_start_sample: int
    window_end_sample: int


@dataclass(frozen=True)
class OutputRecord:
    source_path: str
    output_path: str
    part_index: int
    part_count: int
    start_sec: float
    end_sec: float
    duration_sec: float
    cut_method: str
    silence_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split denoised long audio into roughly 5-minute, silence-aware "
            "16 kHz mono WAV pieces before running slide_LLM."
        )
    )
    parser.add_argument("--input", required=True, help="Input denoised audio file or folder.")
    parser.add_argument("--output", required=True, help="Separate flat output folder.")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan folders recursively. Default: true.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--target-piece-seconds",
        type=float,
        default=300.0,
        help="Target source-piece duration. Default: 300 seconds.",
    )
    parser.add_argument(
        "--search-window-seconds",
        type=float,
        default=60.0,
        help="Window around each ideal cut to search for silence. Default: 60 seconds.",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=20.0,
        help="RMS frame length in milliseconds. Default: 20.",
    )
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=5.0,
        help="RMS hop length in milliseconds. Default: 5.",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=float,
        default=30.0,
        help="Minimum continuous low-energy span to use as silence. Default: 30.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=list(DEFAULT_EXTENSIONS),
        help="Audio extensions to scan when --input is a folder.",
    )
    return parser.parse_args()


def normalize_extensions(extensions: Sequence[str]) -> set[str]:
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}


def iter_audio_files(input_path: Path, extensions: Sequence[str], recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    allowed = normalize_extensions(extensions)
    paths = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in allowed)


def load_audio_mono(path: Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    except Exception:
        import librosa

        audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
        audio = np.asarray(audio, dtype=np.float32)

    if sample_rate != target_sample_rate:
        import librosa

        audio = librosa.resample(
            np.asarray(audio, dtype=np.float32),
            orig_sr=int(sample_rate),
            target_sr=int(target_sample_rate),
        )
        sample_rate = target_sample_rate
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def piece_count_for_duration(duration_sec: float, target_piece_seconds: float) -> int:
    if target_piece_seconds <= 0:
        raise ValueError("target_piece_seconds must be greater than 0")
    if duration_sec <= target_piece_seconds:
        return 1
    return max(2, int(math.floor(duration_sec / target_piece_seconds)) + 1)


def detect_low_energy_spans(
    audio: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    *,
    frame_ms: float,
    hop_ms: float,
    min_silence_ms: float,
) -> tuple[list[SilenceSpan], int]:
    """Return low-energy spans plus the quietest-frame center as a fallback."""
    start_sample = max(0, min(int(start_sample), len(audio)))
    end_sample = max(start_sample, min(int(end_sample), len(audio)))
    window = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
    frame = max(1, int(round(frame_ms / 1000.0 * sample_rate)))
    hop = max(1, int(round(hop_ms / 1000.0 * sample_rate)))
    if window.size == 0:
        return [], start_sample
    if window.size < frame:
        quietest = start_sample + int(np.argmin(np.abs(window)))
        return [SilenceSpan(start_sample, end_sample)] if window.size else [], quietest

    rms: list[float] = []
    centers: list[int] = []
    for offset in range(0, window.size - frame + 1, hop):
        chunk = window[offset : offset + frame]
        rms.append(float(np.sqrt(np.mean(np.square(chunk)))))
        centers.append(start_sample + offset + frame // 2)
    if not rms:
        quietest = start_sample + int(np.argmin(np.abs(window)))
        return [], quietest

    rms_arr = np.asarray(rms, dtype=np.float32)
    quietest_center = centers[int(np.argmin(rms_arr))]
    if float(np.max(rms_arr) - np.min(rms_arr)) <= 1e-8:
        return [], quietest_center
    floor = float(np.percentile(rms_arr, 20))
    median = float(np.percentile(rms_arr, 50))
    peak = float(np.max(rms_arr))
    threshold = max(floor * 1.8, median * 0.45, peak * 0.025, 1e-5)
    if median > 1e-7:
        threshold = min(threshold, median * 0.95)
    silent = rms_arr <= threshold
    min_silence_samples = max(1, int(round(min_silence_ms / 1000.0 * sample_rate)))

    spans: list[SilenceSpan] = []
    idx = 0
    while idx < len(silent):
        if not bool(silent[idx]):
            idx += 1
            continue
        run_start = idx
        while idx + 1 < len(silent) and bool(silent[idx + 1]):
            idx += 1
        run_end = idx
        span_start = max(start_sample, centers[run_start] - frame // 2)
        span_end = min(end_sample, centers[run_end] + frame // 2)
        span = SilenceSpan(span_start, span_end)
        if span.duration_samples >= min_silence_samples:
            spans.append(span)
        idx += 1

    return spans, quietest_center


def choose_cut_in_window(
    audio: np.ndarray,
    sample_rate: int,
    *,
    ideal_sample: int,
    window_start_sample: int,
    window_end_sample: int,
    frame_ms: float,
    hop_ms: float,
    min_silence_ms: float,
) -> CutChoice:
    spans, quietest_center = detect_low_energy_spans(
        audio,
        sample_rate,
        window_start_sample,
        window_end_sample,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        min_silence_ms=min_silence_ms,
    )
    if spans:
        best = max(
            spans,
            key=lambda span: (
                span.duration_samples,
                -abs((span.start_sample + span.end_sample) // 2 - ideal_sample),
            ),
        )
        return CutChoice(
            sample=(best.start_sample + best.end_sample) // 2,
            method="longest_low_energy_silence",
            silence_samples=best.duration_samples,
            window_start_sample=int(window_start_sample),
            window_end_sample=int(window_end_sample),
        )

    return CutChoice(
        sample=int(quietest_center),
        method="quietest_point_no_min_silence",
        silence_samples=0,
        window_start_sample=int(window_start_sample),
        window_end_sample=int(window_end_sample),
    )


def plan_cuts(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_piece_seconds: float,
    search_window_seconds: float,
    frame_ms: float,
    hop_ms: float,
    min_silence_ms: float,
) -> list[CutChoice]:
    duration_sec = len(audio) / float(sample_rate) if sample_rate > 0 else 0.0
    part_count = piece_count_for_duration(duration_sec, target_piece_seconds)
    if part_count <= 1:
        return []

    cuts: list[CutChoice] = []
    half_window_samples = max(1, int(round(search_window_seconds * sample_rate / 2.0)))
    previous_cut = 0
    total_samples = len(audio)
    for cut_index in range(1, part_count):
        ideal_sample = int(round(total_samples * cut_index / part_count))
        remaining_parts = part_count - cut_index
        min_cut = previous_cut + 1
        max_cut = total_samples - remaining_parts
        if min_cut > max_cut:
            cut_sample = min(max(previous_cut + 1, ideal_sample), total_samples)
            cuts.append(
                CutChoice(
                    sample=cut_sample,
                    method="clamped_target_no_room",
                    silence_samples=0,
                    window_start_sample=cut_sample,
                    window_end_sample=cut_sample,
                )
            )
            previous_cut = cut_sample
            continue

        window_start = max(min_cut, ideal_sample - half_window_samples)
        window_end = min(max_cut, ideal_sample + half_window_samples)
        if window_end <= window_start:
            window_start = min_cut
            window_end = max_cut

        choice = choose_cut_in_window(
            audio,
            sample_rate,
            ideal_sample=ideal_sample,
            window_start_sample=window_start,
            window_end_sample=window_end,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            min_silence_ms=min_silence_ms,
        )
        cut_sample = min(max(choice.sample, min_cut), max_cut)
        method = choice.method if cut_sample == choice.sample else f"{choice.method}_clamped"
        choice = CutChoice(
            sample=cut_sample,
            method=method,
            silence_samples=choice.silence_samples,
            window_start_sample=choice.window_start_sample,
            window_end_sample=choice.window_end_sample,
        )
        cuts.append(choice)
        previous_cut = cut_sample

    return cuts


def sanitize_stem(path: Path, input_root: Path) -> str:
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        rel = path.name
    rel_path = Path(rel)
    stem_parts = list(rel_path.with_suffix("").parts)
    raw = "__".join(stem_parts) if stem_parts else path.stem
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return sanitized or "audio"


def unique_base_name(base: str, seen: dict[str, int]) -> str:
    count = seen.get(base, 0)
    seen[base] = count + 1
    if count == 0:
        return base
    return f"{base}__dup{count + 1}"


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")


def process_file(
    source_path: Path,
    *,
    input_root: Path,
    output_dir: Path,
    base_name: str,
    overwrite: bool,
    target_piece_seconds: float,
    search_window_seconds: float,
    frame_ms: float,
    hop_ms: float,
    min_silence_ms: float,
) -> list[OutputRecord]:
    audio, sample_rate = load_audio_mono(source_path, TARGET_SAMPLE_RATE)
    duration_sec = len(audio) / float(sample_rate) if sample_rate > 0 else 0.0
    part_count = piece_count_for_duration(duration_sec, target_piece_seconds)
    cuts = plan_cuts(
        audio,
        sample_rate,
        target_piece_seconds=target_piece_seconds,
        search_window_seconds=search_window_seconds,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        min_silence_ms=min_silence_ms,
    )
    boundaries = [0, *(cut.sample for cut in cuts), len(audio)]
    records: list[OutputRecord] = []

    for part_idx, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        stem = base_name if part_count == 1 else f"{base_name}__part{part_idx:03d}"
        output_path = output_dir / f"{stem}.wav"
        segment = np.ascontiguousarray(audio[start:end], dtype=np.float32)
        write_wav(output_path, segment, sample_rate, overwrite=overwrite)
        cut = cuts[part_idx - 1] if part_idx <= len(cuts) else None
        records.append(
            OutputRecord(
                source_path=str(source_path),
                output_path=str(output_path),
                part_index=part_idx,
                part_count=part_count,
                start_sec=start / float(sample_rate),
                end_sec=end / float(sample_rate),
                duration_sec=(end - start) / float(sample_rate),
                cut_method=cut.method if cut is not None else "final",
                silence_sec=(cut.silence_samples / float(sample_rate)) if cut is not None else 0.0,
            )
        )
    return records


def write_mapping(output_dir: Path, records: Sequence[OutputRecord], *, overwrite: bool) -> Path:
    path = output_dir / "mapping.csv"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Mapping already exists: {path}. Pass --overwrite to replace it.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source_path",
                "output_path",
                "part_index",
                "part_count",
                "start_sec",
                "end_sec",
                "duration_sec",
                "cut_method",
                "silence_sec",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_path": record.source_path,
                    "output_path": record.output_path,
                    "part_index": record.part_index,
                    "part_count": record.part_count,
                    "start_sec": f"{record.start_sec:.3f}",
                    "end_sec": f"{record.end_sec:.3f}",
                    "duration_sec": f"{record.duration_sec:.3f}",
                    "cut_method": record.cut_method,
                    "silence_sec": f"{record.silence_sec:.3f}",
                }
            )
    return path


def validate_args(args: argparse.Namespace) -> None:
    if args.target_piece_seconds <= 0:
        raise SystemExit("--target-piece-seconds must be greater than 0.")
    if args.search_window_seconds <= 0:
        raise SystemExit("--search-window-seconds must be greater than 0.")
    if args.frame_ms <= 0:
        raise SystemExit("--frame-ms must be greater than 0.")
    if args.hop_ms <= 0:
        raise SystemExit("--hop-ms must be greater than 0.")
    if args.min_silence_ms <= 0:
        raise SystemExit("--min-silence-ms must be greater than 0.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise SystemExit(f"Output exists but is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "mapping.csv"
    if mapping_path.exists() and not args.overwrite:
        raise SystemExit(f"Mapping already exists: {mapping_path}. Pass --overwrite to replace it.")

    source_files = iter_audio_files(input_path, args.extensions, args.recursive)
    if not source_files:
        raise SystemExit(f"No audio files found under: {input_path}")

    input_root = input_path if input_path.is_dir() else input_path.parent
    seen_bases: dict[str, int] = {}
    all_records: list[OutputRecord] = []
    for index, source_path in enumerate(source_files, start=1):
        base = unique_base_name(sanitize_stem(source_path, input_root), seen_bases)
        print(f"[{index}/{len(source_files)}] {source_path}")
        records = process_file(
            source_path,
            input_root=input_root,
            output_dir=output_dir,
            base_name=base,
            overwrite=args.overwrite,
            target_piece_seconds=args.target_piece_seconds,
            search_window_seconds=args.search_window_seconds,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            min_silence_ms=args.min_silence_ms,
        )
        all_records.extend(records)
        print(f"  wrote {len(records)} piece(s)")

    mapping_path = write_mapping(output_dir, all_records, overwrite=args.overwrite)
    print(f"\nDone. Wrote {len(all_records)} WAV file(s) to: {output_dir}")
    print(f"Mapping CSV: {mapping_path}")


if __name__ == "__main__":
    main()
