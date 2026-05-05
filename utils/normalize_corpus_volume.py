#!/usr/bin/env python
"""Normalize a sliced VoxCPM-style dataset to a fixed LUFS target.

The tool is intended as the final postprocess after ``slide_rule`` has written
``audios/.../*.wav`` and ``*_voxcpm.jsonl``. It reads the JSONL, measures each
listed segment, prunes risky audio, writes normalized WAV files into a new
dataset folder, and rewrites JSONL paths to point at the normalized audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm


EPSILON = 1e-12


@dataclass(frozen=True)
class JsonlRecord:
    index: int
    payload: dict[str, Any]
    audio_relpath: str
    source_audio_path: Path
    output_audio_relpath: str


@dataclass(frozen=True)
class AudioStats:
    record: JsonlRecord
    sample_rate: int
    duration_sec: float
    lufs: float
    peak_dbfs: float
    peak_headroom_db: float
    active_ratio: float
    dynamic_range_db: float


@dataclass(frozen=True)
class PruneDecision:
    prune: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class NormalizeResult:
    requested_gain_db: float
    applied_gain_db: float
    peak_limited: bool
    output_sample_rate: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize segment audio from a completed slide_rule output folder "
            "to a fixed LUFS target and write a new VoxCPM JSONL dataset."
        )
    )
    parser.add_argument("--input", required=True, help="Completed slide_rule output directory.")
    parser.add_argument("--output", required=True, help="New output dataset directory.")
    parser.add_argument(
        "--jsonl",
        default=None,
        help=(
            "Optional JSONL path or filename. If omitted, exactly one *_voxcpm.jsonl "
            "must exist under --input."
        ),
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-20.0,
        help="Integrated loudness target. Default: -20 LUFS.",
    )
    parser.add_argument(
        "--max-volume-change-db",
        type=float,
        default=12.0,
        help=(
            "Prune segments that need more upward gain than this to reach "
            "--target-lufs. Downward attenuation is always allowed. Default: 12 dB."
        ),
    )
    parser.add_argument(
        "--max-dynamic-range-db",
        type=float,
        default=24.0,
        help=(
            "Prune segments whose active-frame RMS P95-P10 range is larger than this. "
            "Default: 24 dB."
        ),
    )
    parser.add_argument(
        "--peak-margin-db",
        type=float,
        default=1.0,
        help="Peak margin kept below 0 dBFS after normalization. Default: 1 dB.",
    )
    parser.add_argument(
        "--silence-top-db",
        type=float,
        default=40.0,
        help="Frames quieter than peak RMS minus this value are ignored. Default: 40 dB.",
    )
    parser.add_argument(
        "--min-active-ratio",
        type=float,
        default=0.03,
        help="Prune segments with too little active audio. Default: 0.03.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Optional output sample rate. Default: keep each segment's source rate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and write reports, but do not write normalized audio or JSONL.",
    )
    parser.add_argument(
        "--allow-empty-output",
        action="store_true",
        help=(
            "Exit successfully when every row is pruned. Analysis and pruned-row reports "
            "are still written, but no normalized JSONL/audio is produced."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def require_loudnorm():
    try:
        import pyloudnorm as pyln
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyloudnorm. Install dependencies with:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return pyln


def db_to_gain(db_value: float) -> float:
    return float(10.0 ** (db_value / 20.0))


def amplitude_to_dbfs(value: float) -> float:
    return float(20.0 * math.log10(max(value, EPSILON)))


def load_audio(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32, copy=False), int(sr)


def frame_rms_db(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_length = min(2048, max(256, int(sample_rate * 0.050)))
    hop_length = max(1, frame_length // 4)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return 20.0 * np.log10(np.maximum(rms, EPSILON))


def resolve_jsonl(input_dir: Path, jsonl_arg: str | None) -> Path:
    if jsonl_arg:
        candidate = Path(jsonl_arg).expanduser()
        if not candidate.is_absolute():
            candidate = input_dir / candidate
        if not candidate.exists() or not candidate.is_file():
            raise SystemExit(f"JSONL file does not exist: {candidate}")
        return candidate

    matches = sorted(input_dir.glob("*_voxcpm.jsonl"))
    if not matches:
        raise SystemExit(f"No *_voxcpm.jsonl file found in: {input_dir}")
    if len(matches) > 1:
        choices = "\n".join(f"  - {path.name}" for path in matches)
        raise SystemExit(f"Multiple *_voxcpm.jsonl files found. Pass --jsonl.\n{choices}")
    return matches[0]


def parse_jsonl_records(input_dir: Path, jsonl_path: Path) -> list[JsonlRecord]:
    records: list[JsonlRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {jsonl_path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise SystemExit(f"JSONL row {line_number} must be an object.")

            audio_value = payload.get("audio")
            if not isinstance(audio_value, str) or not audio_value:
                raise SystemExit(f"JSONL row {line_number} is missing a string 'audio' field.")
            audio_rel = Path(audio_value)
            if audio_rel.is_absolute():
                raise SystemExit(f"JSONL row {line_number} uses an absolute audio path: {audio_value}")
            if any(part == ".." for part in audio_rel.parts):
                raise SystemExit(f"JSONL row {line_number} uses an unsafe audio path: {audio_value}")

            source_audio_path = input_dir / audio_value
            if not source_audio_path.exists() or not source_audio_path.is_file():
                raise SystemExit(
                    f"JSONL row {line_number} references missing audio: {source_audio_path}"
                )
            records.append(
                JsonlRecord(
                    index=line_number,
                    payload=payload,
                    audio_relpath=audio_value,
                    source_audio_path=source_audio_path,
                    output_audio_relpath=str(Path(audio_value).with_suffix(".wav")).replace("\\", "/"),
                )
            )
    if not records:
        raise SystemExit(f"No usable rows found in JSONL: {jsonl_path}")
    return records


def measure_record(record: JsonlRecord, pyln: Any, silence_top_db: float) -> AudioStats:
    audio, sr = load_audio(record.source_audio_path)
    if audio.size == 0:
        raise ValueError("empty audio")

    duration_sec = float(audio.size / sr)
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(audio))
    if not math.isfinite(lufs):
        raise ValueError("could not compute finite LUFS")

    rms_db = frame_rms_db(audio, sr)
    peak_rms_db = float(np.max(rms_db)) if rms_db.size else -math.inf
    active = rms_db[rms_db >= peak_rms_db - silence_top_db]
    active_ratio = float(active.size / max(rms_db.size, 1))
    if active.size:
        dynamic_range_db = float(np.percentile(active, 95) - np.percentile(active, 10))
    else:
        dynamic_range_db = 0.0

    peak = float(np.max(np.abs(audio)))
    peak_dbfs = amplitude_to_dbfs(peak)
    return AudioStats(
        record=record,
        sample_rate=sr,
        duration_sec=duration_sec,
        lufs=lufs,
        peak_dbfs=peak_dbfs,
        peak_headroom_db=-peak_dbfs,
        active_ratio=active_ratio,
        dynamic_range_db=dynamic_range_db,
    )


def classify_segment(
    item: AudioStats,
    *,
    target_lufs: float,
    max_volume_change_db: float,
    peak_margin_db: float,
    min_active_ratio: float,
    max_dynamic_range_db: float,
) -> PruneDecision:
    reasons: list[str] = []
    needed_gain = target_lufs - item.lufs
    allowed_gain = item.peak_headroom_db - peak_margin_db

    if needed_gain > max_volume_change_db:
        reasons.append(f"too quiet; needs +{needed_gain:.1f} dB to reach {target_lufs:.1f} LUFS")
    if needed_gain > allowed_gain:
        reasons.append(
            f"would clip before target; needs +{needed_gain:.1f} dB but has {allowed_gain:.1f} dB safe headroom"
        )
    if item.active_ratio < min_active_ratio:
        reasons.append(f"only {item.active_ratio:.1%} active audio")
    if item.dynamic_range_db > max_dynamic_range_db:
        reasons.append(f"dynamic range {item.dynamic_range_db:.1f} dB > {max_dynamic_range_db:.1f} dB")

    return PruneDecision(prune=bool(reasons), reasons=tuple(reasons))


def normalized_payload(record: JsonlRecord) -> dict[str, Any]:
    payload = dict(record.payload)
    payload["audio"] = record.output_audio_relpath
    return payload


def pruned_payload(record: JsonlRecord, decision: PruneDecision) -> dict[str, Any]:
    payload = dict(record.payload)
    payload["normalization_pruned"] = True
    payload["normalization_reasons"] = list(decision.reasons)
    return payload


def ensure_no_existing_outputs(
    output_dir: Path,
    output_jsonl_path: Path,
    kept_stats: Sequence[AudioStats],
    pruned_stats: Sequence[AudioStats],
    *,
    dry_run: bool,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    paths = [
        output_dir / "volume_analysis.csv",
        output_dir / "normalized_manifest.csv",
    ]
    if not dry_run:
        paths.append(output_jsonl_path)
        if pruned_stats:
            paths.append(output_dir / "pruned_volume_segments.jsonl")
        paths.extend(output_dir / item.record.output_audio_relpath for item in kept_stats)
    existing = [path for path in paths if path.exists()]
    if existing:
        rendered = "\n".join(f"  - {path}" for path in existing[:20])
        if len(existing) > 20:
            rendered += f"\n  ... and {len(existing) - 20} more"
        raise SystemExit(f"Output files already exist. Pass --overwrite to replace them.\n{rendered}")


def write_analysis_report(
    report_path: Path,
    stats: Sequence[AudioStats],
    decisions: dict[int, PruneDecision],
    target_lufs: float,
    pruned_indices: set[int],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = (
            "jsonl_row",
            "audio",
            "source_audio_path",
            "duration_sec",
            "sample_rate",
            "lufs",
            "target_lufs",
            "needed_gain_db",
            "peak_dbfs",
            "peak_headroom_db",
            "active_ratio",
            "dynamic_range_db",
            "pruned",
            "reasons",
        )
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in stats:
            decision = decisions[item.record.index]
            writer.writerow(
                {
                    "jsonl_row": item.record.index,
                    "audio": item.record.audio_relpath,
                    "source_audio_path": str(item.record.source_audio_path),
                    "duration_sec": f"{item.duration_sec:.3f}",
                    "sample_rate": item.sample_rate,
                    "lufs": f"{item.lufs:.3f}",
                    "target_lufs": f"{target_lufs:.3f}",
                    "needed_gain_db": f"{target_lufs - item.lufs:.3f}",
                    "peak_dbfs": f"{item.peak_dbfs:.3f}",
                    "peak_headroom_db": f"{item.peak_headroom_db:.3f}",
                    "active_ratio": f"{item.active_ratio:.5f}",
                    "dynamic_range_db": f"{item.dynamic_range_db:.3f}",
                    "pruned": item.record.index in pruned_indices,
                    "reasons": "; ".join(decision.reasons),
                }
            )


def write_manifest(
    manifest_path: Path,
    rows: Sequence[tuple[AudioStats, Path, NormalizeResult]],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = (
            "jsonl_row",
            "source_audio",
            "output_audio",
            "duration_sec",
            "source_sample_rate",
            "output_sample_rate",
            "source_lufs",
            "requested_gain_db",
            "applied_gain_db",
            "peak_limited",
        )
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item, output_path, result in rows:
            writer.writerow(
                {
                    "jsonl_row": item.record.index,
                    "source_audio": str(item.record.source_audio_path),
                    "output_audio": str(output_path),
                    "duration_sec": f"{item.duration_sec:.3f}",
                    "source_sample_rate": item.sample_rate,
                    "output_sample_rate": result.output_sample_rate,
                    "source_lufs": f"{item.lufs:.3f}",
                    "requested_gain_db": f"{result.requested_gain_db:.3f}",
                    "applied_gain_db": f"{result.applied_gain_db:.3f}",
                    "peak_limited": result.peak_limited,
                }
            )


def write_jsonl(path: Path, payloads: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_file(
    source_path: Path,
    output_path: Path,
    target_lufs: float,
    source_lufs: float,
    sample_rate: int | None,
    peak_margin_db: float,
) -> NormalizeResult:
    audio, sr = load_audio(source_path, sample_rate=sample_rate)
    requested_gain_db = target_lufs - source_lufs
    normalized = audio * db_to_gain(requested_gain_db)

    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    peak_limit = db_to_gain(-peak_margin_db)
    applied_gain_db = requested_gain_db
    peak_limited = False
    if peak > peak_limit:
        limiter_gain = peak_limit / max(peak, EPSILON)
        normalized *= limiter_gain
        applied_gain_db += amplitude_to_dbfs(limiter_gain)
        peak_limited = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, normalized, sr, subtype="PCM_16")
    return NormalizeResult(
        requested_gain_db=requested_gain_db,
        applied_gain_db=applied_gain_db,
        peak_limited=peak_limited,
        output_sample_rate=sr,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_rate is not None and args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be greater than 0.")
    if args.max_volume_change_db < 0:
        raise SystemExit("--max-volume-change-db must be non-negative.")
    if args.peak_margin_db < 0:
        raise SystemExit("--peak-margin-db must be non-negative.")
    if args.min_active_ratio < 0 or args.min_active_ratio > 1:
        raise SystemExit("--min-active-ratio must be between 0 and 1.")
    if args.max_dynamic_range_db < 0:
        raise SystemExit("--max-dynamic-range-db must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    pyln = require_loudnorm()

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input must be a completed slide_rule output directory: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise SystemExit("--output must be a new directory, not the same directory as --input.")

    jsonl_path = resolve_jsonl(input_dir, args.jsonl)
    records = parse_jsonl_records(input_dir, jsonl_path)
    output_jsonl_path = output_dir / jsonl_path.name
    report_path = output_dir / "volume_analysis.csv"
    manifest_path = output_dir / "normalized_manifest.csv"
    pruned_jsonl_path = output_dir / "pruned_volume_segments.jsonl"

    print(f"Using JSONL: {jsonl_path}")
    print(f"Target loudness: {args.target_lufs:.1f} LUFS")
    print(f"Analyzing {len(records)} JSONL-listed segment audio file(s)...")
    stats: list[AudioStats] = []
    analysis_progress = tqdm(records, desc="Volume analysis", unit="seg", dynamic_ncols=True)
    for record in analysis_progress:
        try:
            item = measure_record(record, pyln, args.silence_top_db)
        except Exception as exc:  # noqa: BLE001 - report the exact row and stop.
            raise SystemExit(
                f"Failed to measure JSONL row {record.index} ({record.audio_relpath}): {exc}"
            ) from exc
        stats.append(item)
        analysis_progress.set_postfix_str(
            f"row {record.index} {item.lufs:.1f} LUFS peak {item.peak_dbfs:.1f} dBFS",
            refresh=False,
        )

    decisions = {
        item.record.index: classify_segment(
            item,
            target_lufs=args.target_lufs,
            max_volume_change_db=args.max_volume_change_db,
            peak_margin_db=args.peak_margin_db,
            min_active_ratio=args.min_active_ratio,
            max_dynamic_range_db=args.max_dynamic_range_db,
        )
        for item in stats
    }
    pruned_indices = {
        item.record.index for item in stats if decisions[item.record.index].prune
    }
    kept_stats = [item for item in stats if item.record.index not in pruned_indices]
    pruned_stats = [item for item in stats if item.record.index in pruned_indices]
    ensure_no_existing_outputs(
        output_dir=output_dir,
        output_jsonl_path=output_jsonl_path,
        kept_stats=kept_stats,
        pruned_stats=pruned_stats,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    write_analysis_report(report_path, stats, decisions, args.target_lufs, pruned_indices)
    if not kept_stats:
        if not args.dry_run:
            write_jsonl(
                pruned_jsonl_path,
                [pruned_payload(item.record, decisions[item.record.index]) for item in pruned_stats],
            )
        print("\nSegment loudness summary:")
        print(f"  JSONL rows analyzed: {len(stats)}")
        print(f"  Rows pruned: {len(pruned_indices)}")
        print("  Rows kept: 0")
        print(f"  Target loudness: {args.target_lufs:.1f} LUFS")
        print(f"  Analysis report: {report_path}")
        if not args.dry_run:
            print(f"  Pruned rows JSONL: {pruned_jsonl_path}")
        message = "All JSONL rows were pruned; no normalized audio was written."
        print(f"\n{message}")
        if args.allow_empty_output:
            return
        raise SystemExit(message)

    print("\nSegment loudness summary:")
    print(f"  JSONL rows analyzed: {len(stats)}")
    print(f"  Rows pruned: {len(pruned_indices)}")
    print(f"  Rows kept: {len(kept_stats)}")
    print(f"  Target loudness: {args.target_lufs:.1f} LUFS")
    print(f"  Max upward gain: {args.max_volume_change_db:.1f} dB")
    print(f"  Max dynamic range: {args.max_dynamic_range_db:.1f} dB")
    print(f"  Analysis report: {report_path}")

    if args.dry_run:
        print("\nDry run complete. Reports were written, but no normalized audio or JSONL was written.")
        return

    manifest_rows: list[tuple[AudioStats, Path, NormalizeResult]] = []
    normalize_progress = tqdm(kept_stats, desc="Volume normalize", unit="seg", dynamic_ncols=True)
    for item in normalize_progress:
        output_audio_path = output_dir / item.record.output_audio_relpath
        result = normalize_file(
            source_path=item.record.source_audio_path,
            output_path=output_audio_path,
            target_lufs=args.target_lufs,
            source_lufs=item.lufs,
            sample_rate=args.sample_rate,
            peak_margin_db=args.peak_margin_db,
        )
        manifest_rows.append((item, output_audio_path, result))
        suffix = " peak-limited" if result.peak_limited else ""
        normalize_progress.set_postfix_str(
            f"row {item.record.index} gain {result.applied_gain_db:+.1f} dB{suffix}",
            refresh=False,
        )

    write_jsonl(output_jsonl_path, [normalized_payload(item.record) for item in kept_stats])
    if pruned_stats:
        write_jsonl(
            pruned_jsonl_path,
            [pruned_payload(item.record, decisions[item.record.index]) for item in pruned_stats],
        )
    write_manifest(manifest_path, manifest_rows)

    print(f"\nDone. Wrote {len(manifest_rows)} normalized segment WAV file(s) to: {output_dir}")
    print(f"Updated JSONL: {output_jsonl_path}")
    print(f"Manifest: {manifest_path}")
    if pruned_stats:
        print(f"Pruned rows JSONL: {pruned_jsonl_path}")


if __name__ == "__main__":
    main()
