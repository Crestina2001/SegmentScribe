#!/usr/bin/env python
"""Rank audio files by likely abrupt starts and ends.

The score is relative to all audio files in the input folder. It looks for
edges that are unusually energetic or non-faded compared with the corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    sf = None


AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}
EPS = 1e-12


@dataclass
class EdgeFeatures:
    path: str
    sample_rate: int
    duration_sec: float
    side: str
    edge_rms: float
    edge_peak: float
    boundary_abs: float
    sustain_ratio: float
    zero_crossing_rate: float
    score: float = 0.0
    likelihood: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank audio files by likely abrupt start/end using folder-wide robust statistics."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="pipeline_runs/ju_strategy_2/04_sliced/audios",
        help="Folder containing audio files. Searched recursively.",
    )
    parser.add_argument(
        "--edge-ms",
        type=float,
        default=50.0,
        help="Window at each edge used for energy checks.",
    )
    parser.add_argument(
        "--peak-ms",
        type=float,
        default=10.0,
        help="Window closest to the boundary used for peak checks.",
    )
    parser.add_argument(
        "--interior-start-ms",
        type=float,
        default=50.0,
        help="Offset from edge for the nearby interior comparison window.",
    )
    parser.add_argument(
        "--interior-ms",
        type=float,
        default=200.0,
        help="Nearby interior window used to estimate expected edge energy.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Rows to print per ranking.",
    )
    parser.add_argument(
        "--output-dir",
        default="test_scripts/logs/abrupt_edges",
        help="Directory for CSV and JSONL outputs.",
    )
    return parser.parse_args()


def audio_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS
    )


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    if sf is not None:
        audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32), sample_rate
        mono = np.mean(audio, axis=1, dtype=np.float32)
        return mono, sample_rate
    if path.suffix.lower() != ".wav":
        raise RuntimeError("soundfile is not installed; standard-library fallback only supports .wav")
    return read_wav_mono(path)


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if not frames:
        return np.zeros(0, dtype=np.float32), sample_rate

    if sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        data = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return data.astype(np.float32, copy=False), sample_rate


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def peak(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def zcr(audio: np.ndarray) -> float:
    if audio.size < 2:
        return 0.0
    signs = np.signbit(audio)
    return float(np.count_nonzero(signs[1:] != signs[:-1]) / (audio.size - 1))


def window(audio: np.ndarray, start: int, stop: int) -> np.ndarray:
    start = max(0, min(start, audio.size))
    stop = max(start, min(stop, audio.size))
    return audio[start:stop]


def edge_features(
    path: Path,
    root: Path,
    audio: np.ndarray,
    sample_rate: int,
    edge_ms: float,
    peak_ms: float,
    interior_start_ms: float,
    interior_ms: float,
) -> list[EdgeFeatures]:
    edge_n = max(1, round(sample_rate * edge_ms / 1000.0))
    peak_n = max(1, round(sample_rate * peak_ms / 1000.0))
    interior_offset_n = max(0, round(sample_rate * interior_start_ms / 1000.0))
    interior_n = max(1, round(sample_rate * interior_ms / 1000.0))
    duration_sec = audio.size / sample_rate if sample_rate else 0.0
    rel_path = path.relative_to(root).as_posix()

    start_edge = window(audio, 0, edge_n)
    start_peak = window(audio, 0, peak_n)
    start_interior = window(audio, interior_offset_n, interior_offset_n + interior_n)

    end_edge = window(audio, audio.size - edge_n, audio.size)
    end_peak = window(audio, audio.size - peak_n, audio.size)
    end_interior = window(
        audio,
        audio.size - interior_offset_n - interior_n,
        audio.size - interior_offset_n,
    )

    start = EdgeFeatures(
        path=rel_path,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        side="start",
        edge_rms=rms(start_edge),
        edge_peak=peak(start_peak),
        boundary_abs=float(abs(audio[0])) if audio.size else 0.0,
        sustain_ratio=rms(start_edge) / (rms(start_interior) + EPS),
        zero_crossing_rate=zcr(start_edge),
    )
    end = EdgeFeatures(
        path=rel_path,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        side="end",
        edge_rms=rms(end_edge),
        edge_peak=peak(end_peak),
        boundary_abs=float(abs(audio[-1])) if audio.size else 0.0,
        sustain_ratio=rms(end_edge) / (rms(end_interior) + EPS),
        zero_crossing_rate=zcr(end_edge),
    )
    return [start, end]


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-9:
        std = np.std(values)
        scale = std if std > 1e-9 else 1.0
        return (values - median) / scale
    return 0.67448975 * (values - median) / mad


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def score_edges(edges: list[EdgeFeatures]) -> None:
    for side in ("start", "end"):
        side_edges = [edge for edge in edges if edge.side == side]
        if not side_edges:
            continue

        feature_matrix = np.array(
            [
                [
                    math.log10(edge.edge_rms + EPS),
                    math.log10(edge.edge_peak + EPS),
                    math.log10(edge.boundary_abs + EPS),
                    math.log10(edge.sustain_ratio + EPS),
                    edge.zero_crossing_rate,
                ]
                for edge in side_edges
            ],
            dtype=np.float64,
        )
        z = np.column_stack([robust_z(feature_matrix[:, i]) for i in range(feature_matrix.shape[1])])

        for edge, row in zip(side_edges, z):
            positive = np.maximum(row, 0.0)
            edge.score = float(
                0.30 * positive[0]
                + 0.20 * positive[1]
                + 0.30 * positive[2]
                + 0.15 * positive[3]
                + 0.05 * positive[4]
            )
            edge.likelihood = sigmoid(edge.score - 1.5)


def write_outputs(edges: list[EdgeFeatures], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "abrupt_edge_ranking.csv"
    jsonl_path = output_dir / "abrupt_edge_ranking.jsonl"

    rows = [asdict(edge) for edge in sorted(edges, key=lambda item: item.score, reverse=True)]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def print_ranking(edges: list[EdgeFeatures], side: str, top: int) -> None:
    ranked = sorted((edge for edge in edges if edge.side == side), key=lambda item: item.score, reverse=True)
    print(f"\nTop {min(top, len(ranked))} likely abrupt {side}s")
    print("rank score  prob  edge_rms peak   boundary sustain zcr   path")
    for idx, edge in enumerate(ranked[:top], start=1):
        print(
            f"{idx:>4} {edge.score:>5.2f} {edge.likelihood:>5.2f} "
            f"{edge.edge_rms:>8.5f} {edge.edge_peak:>6.3f} "
            f"{edge.boundary_abs:>8.5f} {edge.sustain_ratio:>7.2f} "
            f"{edge.zero_crossing_rate:>5.2f} {edge.path}"
        )


def print_file_ranking(edges: list[EdgeFeatures], top: int) -> None:
    by_path: dict[str, dict[str, EdgeFeatures]] = {}
    for edge in edges:
        by_path.setdefault(edge.path, {})[edge.side] = edge

    rows = []
    for path, sides in by_path.items():
        start = sides.get("start")
        end = sides.get("end")
        start_score = start.score if start else 0.0
        end_score = end.score if end else 0.0
        worst_side = "start" if start_score >= end_score else "end"
        worst_score = max(start_score, end_score)
        worst_likelihood = max(start.likelihood if start else 0.0, end.likelihood if end else 0.0)
        rows.append((worst_score, worst_likelihood, worst_side, start_score, end_score, path))

    rows.sort(reverse=True)
    print(f"\nTop {min(top, len(rows))} audio files to inspect first")
    print("rank score  prob  side   start end   path")
    for idx, (score, likelihood, side, start_score, end_score, path) in enumerate(rows[:top], start=1):
        print(
            f"{idx:>4} {score:>5.2f} {likelihood:>5.2f} "
            f"{side:<6} {start_score:>5.2f} {end_score:>5.2f} {path}"
        )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    paths = audio_paths(input_dir)
    if not paths:
        raise SystemExit(f"No audio files found under {input_dir}")

    edges: list[EdgeFeatures] = []
    failed: list[tuple[Path, str]] = []
    for path in paths:
        try:
            audio, sample_rate = read_mono(path)
            edges.extend(
                edge_features(
                    path=path,
                    root=input_dir,
                    audio=audio,
                    sample_rate=sample_rate,
                    edge_ms=args.edge_ms,
                    peak_ms=args.peak_ms,
                    interior_start_ms=args.interior_start_ms,
                    interior_ms=args.interior_ms,
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic script
            failed.append((path, str(exc)))

    if not edges:
        raise SystemExit("No readable audio files were analyzed.")

    score_edges(edges)
    csv_path, jsonl_path = write_outputs(edges, Path(args.output_dir))

    print(f"Analyzed {len(edges) // 2} audio files under {input_dir}")
    print(f"CSV:   {csv_path}")
    print(f"JSONL: {jsonl_path}")
    if failed:
        print(f"Failed to read {len(failed)} files:")
        for path, reason in failed[:10]:
            print(f"  {path}: {reason}")
        if len(failed) > 10:
            print(f"  ... {len(failed) - 10} more")

    print_ranking(edges, "start", args.top)
    print_ranking(edges, "end", args.top)
    print_file_ranking(edges, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
