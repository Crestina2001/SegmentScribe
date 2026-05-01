#!/usr/bin/env python3
"""Experiment with RMS-silence cuts in a 5-15s segment window.

The policy under test is intentionally waveform-only:

1. Start at the current cursor.
2. Search only candidate cut points whose segment duration is within
   [min_seg_sec, max_seg_sec].
3. Compute short-frame RMS and mark low-energy frames as silence.
4. Pick the longest contiguous silent run in that window.
5. Cut at the center of that silent run.

This script does not replace the production punctuation-aware cutter. It is a
fast measurement tool for deciding whether an RMS-silence fallback is worth
integrating.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np


SUPPORTED_AUDIO_EXTS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
}


@dataclass(frozen=True)
class Segment:
    source: str
    segment_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    cut_sec: float
    silence_start_sec: float
    silence_end_sec: float
    silence_duration_ms: float
    silence_rms: float
    threshold: float
    fallback: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure longest low-RMS silence cuts inside 5-15s windows.",
    )
    parser.add_argument("--input", required=True, help="Audio file or directory.")
    parser.add_argument("--output-dir", default="", help="Directory for JSONL/CSV outputs.")
    parser.add_argument("--recursive", action="store_true", help="Scan input directories recursively.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Load audio at this sample rate.")
    parser.add_argument("--min-seg-sec", type=float, default=5.0)
    parser.add_argument("--max-seg-sec", type=float, default=15.0)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=5.0)
    parser.add_argument(
        "--silence-percentile",
        type=float,
        default=25.0,
        help="Per-file RMS percentile used as the adaptive silence floor.",
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=1.8,
        help="Multiplier applied to the percentile floor.",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=float,
        default=80.0,
        help="Minimum silent run worth using as a cut.",
    )
    parser.add_argument(
        "--fallback",
        choices=("max", "midpoint"),
        default="max",
        help="Cut location when no qualifying silence exists in the search window.",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files; 0 means all.")
    parser.add_argument("--max-duration-sec", type=float, default=0.0, help="Trim each file for faster tests.")
    return parser.parse_args()


def iter_audio_files(root: Path, *, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_AUDIO_EXTS else []
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS)


def frame_rms(audio: np.ndarray, sample_rate: int, frame_ms: float, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if audio.size < frame:
        padded = np.zeros(frame, dtype=np.float32)
        padded[: audio.size] = audio
        audio = padded
    offsets = np.arange(0, audio.size - frame + 1, hop, dtype=np.int64)
    values = np.empty(len(offsets), dtype=np.float32)
    for i, offset in enumerate(offsets):
        chunk = audio[offset : offset + frame]
        values[i] = float(np.sqrt(np.mean(np.square(chunk))))
    centers = (offsets + frame * 0.5) / float(sample_rate)
    return values, centers.astype(np.float32)


def silent_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    start: int | None = None
    for i, value in enumerate(mask):
        if bool(value) and start is None:
            start = i
        elif not bool(value) and start is not None:
            yield start, i - 1
            start = None
    if start is not None:
        yield start, len(mask) - 1


def choose_cut(
    *,
    source: str,
    segment_index: int,
    cursor_sec: float,
    audio_duration_sec: float,
    rms: np.ndarray,
    centers: np.ndarray,
    threshold: float,
    min_seg_sec: float,
    max_seg_sec: float,
    min_silence_ms: float,
    fallback: str,
) -> Segment:
    earliest = cursor_sec + min_seg_sec
    latest = min(cursor_sec + max_seg_sec, audio_duration_sec)
    if latest <= cursor_sec:
        return Segment(
            source=source,
            segment_index=segment_index,
            start_sec=round(cursor_sec, 6),
            end_sec=round(audio_duration_sec, 6),
            duration_sec=round(max(0.0, audio_duration_sec - cursor_sec), 6),
            cut_sec=round(audio_duration_sec, 6),
            silence_start_sec=round(audio_duration_sec, 6),
            silence_end_sec=round(audio_duration_sec, 6),
            silence_duration_ms=0.0,
            silence_rms=0.0,
            threshold=round(threshold, 8),
            fallback="tail",
        )

    in_window = (centers >= earliest) & (centers <= latest)
    silent = in_window & (rms <= threshold)
    min_frames = max(1, int(math.ceil(min_silence_ms / 1000.0 / max(float(centers[1] - centers[0]) if len(centers) > 1 else 0.005, 1e-6))))

    best: tuple[float, float, float, float, int, int] | None = None
    for start_i, end_i in silent_runs(silent):
        if end_i - start_i + 1 < min_frames:
            continue
        start_sec = float(centers[start_i])
        end_sec = float(centers[end_i])
        duration_sec = max(0.0, end_sec - start_sec)
        mean_rms = float(np.mean(rms[start_i : end_i + 1]))
        score = (duration_sec, -mean_rms, -abs(((start_sec + end_sec) * 0.5) - (cursor_sec + (min_seg_sec + max_seg_sec) * 0.5)))
        candidate = (score[0], score[1], score[2], mean_rms, start_i, end_i)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        cut_sec = latest if fallback == "max" else cursor_sec + (latest - cursor_sec) * 0.5
        return Segment(
            source=source,
            segment_index=segment_index,
            start_sec=round(cursor_sec, 6),
            end_sec=round(cut_sec, 6),
            duration_sec=round(cut_sec - cursor_sec, 6),
            cut_sec=round(cut_sec, 6),
            silence_start_sec=round(cut_sec, 6),
            silence_end_sec=round(cut_sec, 6),
            silence_duration_ms=0.0,
            silence_rms=0.0,
            threshold=round(threshold, 8),
            fallback=f"no_silence_{fallback}",
        )

    _, _, _, mean_rms, start_i, end_i = best
    silence_start = float(centers[start_i])
    silence_end = float(centers[end_i])
    cut_sec = (silence_start + silence_end) * 0.5
    return Segment(
        source=source,
        segment_index=segment_index,
        start_sec=round(cursor_sec, 6),
        end_sec=round(cut_sec, 6),
        duration_sec=round(cut_sec - cursor_sec, 6),
        cut_sec=round(cut_sec, 6),
        silence_start_sec=round(silence_start, 6),
        silence_end_sec=round(silence_end, 6),
        silence_duration_ms=round((silence_end - silence_start) * 1000.0, 3),
        silence_rms=round(mean_rms, 8),
        threshold=round(threshold, 8),
        fallback="",
    )


def segment_file(path: Path, args: argparse.Namespace) -> list[Segment]:
    audio, sample_rate = librosa.load(path, sr=args.sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if args.max_duration_sec > 0:
        audio = audio[: int(round(args.max_duration_sec * sample_rate))]
    duration_sec = len(audio) / float(sample_rate)
    if duration_sec <= 0:
        return []

    rms, centers = frame_rms(audio, sample_rate, args.frame_ms, args.hop_ms)
    floor = float(np.percentile(rms, args.silence_percentile)) if rms.size else 0.0
    median = float(np.percentile(rms, 50)) if rms.size else 0.0
    peak = float(np.max(rms)) if rms.size else 0.0
    threshold = max(floor * args.threshold_multiplier, median * 0.45, peak * 0.025, 1e-6)

    segments: list[Segment] = []
    cursor = 0.0
    index = 1
    while cursor < duration_sec - 1e-6:
        remaining = duration_sec - cursor
        if remaining <= args.max_seg_sec:
            segments.append(
                Segment(
                    source=str(path),
                    segment_index=index,
                    start_sec=round(cursor, 6),
                    end_sec=round(duration_sec, 6),
                    duration_sec=round(remaining, 6),
                    cut_sec=round(duration_sec, 6),
                    silence_start_sec=round(duration_sec, 6),
                    silence_end_sec=round(duration_sec, 6),
                    silence_duration_ms=0.0,
                    silence_rms=0.0,
                    threshold=round(threshold, 8),
                    fallback="tail",
                )
            )
            break
        segment = choose_cut(
            source=str(path),
            segment_index=index,
            cursor_sec=cursor,
            audio_duration_sec=duration_sec,
            rms=rms,
            centers=centers,
            threshold=threshold,
            min_seg_sec=args.min_seg_sec,
            max_seg_sec=args.max_seg_sec,
            min_silence_ms=args.min_silence_ms,
            fallback=args.fallback,
        )
        segments.append(segment)
        next_cursor = max(segment.end_sec, cursor + 0.1)
        cursor = next_cursor
        index += 1
    return segments


def summarize(segments: list[Segment]) -> dict[str, object]:
    durations = [s.duration_sec for s in segments]
    silence_ms = [s.silence_duration_ms for s in segments if s.silence_duration_ms > 0]
    fallbacks = [s for s in segments if s.fallback and s.fallback != "tail"]
    return {
        "segments": len(segments),
        "duration_mean_sec": round(statistics.mean(durations), 3) if durations else 0.0,
        "duration_median_sec": round(statistics.median(durations), 3) if durations else 0.0,
        "duration_min_sec": round(min(durations), 3) if durations else 0.0,
        "duration_max_sec": round(max(durations), 3) if durations else 0.0,
        "silence_hit_rate": round(len(silence_ms) / len(segments), 4) if segments else 0.0,
        "silence_median_ms": round(statistics.median(silence_ms), 1) if silence_ms else 0.0,
        "silence_p90_ms": round(float(np.percentile(np.asarray(silence_ms), 90)), 1) if silence_ms else 0.0,
        "fallback_segments": len(fallbacks),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.input)
    files = iter_audio_files(root, recursive=args.recursive)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No supported audio files found under {root}")

    all_segments: list[Segment] = []
    for path in files:
        segments = segment_file(path, args)
        all_segments.extend(segments)
        file_summary = summarize(segments)
        print(
            f"{path}: segments={file_summary['segments']} "
            f"median={file_summary['duration_median_sec']}s "
            f"silence_hit={file_summary['silence_hit_rate']} "
            f"fallbacks={file_summary['fallback_segments']}"
        )

    summary = {
        "input": str(root),
        "files": len(files),
        "min_seg_sec": args.min_seg_sec,
        "max_seg_sec": args.max_seg_sec,
        "frame_ms": args.frame_ms,
        "hop_ms": args.hop_ms,
        "silence_percentile": args.silence_percentile,
        "threshold_multiplier": args.threshold_multiplier,
        "min_silence_ms": args.min_silence_ms,
        **summarize(all_segments),
    }
    print("SUMMARY_JSON " + json.dumps(summary, ensure_ascii=False, sort_keys=True))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "rms_silence_5_15_segments.jsonl"
        csv_path = output_dir / "rms_silence_5_15_segments.csv"
        summary_path = output_dir / "rms_silence_5_15_summary.json"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for segment in all_segments:
                f.write(json.dumps(asdict(segment), ensure_ascii=False) + "\n")
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_segments[0]).keys()) if all_segments else [])
            if all_segments:
                writer.writeheader()
                writer.writerows(asdict(segment) for segment in all_segments)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote: {jsonl_path}")
        print(f"Wrote: {csv_path}")
        print(f"Wrote: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
