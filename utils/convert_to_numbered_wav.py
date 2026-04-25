#!/usr/bin/env python
"""Convert audio files to numbered 16 kHz mono WAV files.

This avoids shell quoting issues with long non-ASCII filenames by passing paths
directly to ffmpeg through subprocess.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


DEFAULT_EXTENSIONS = (".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert audio files to numbered 16 kHz mono PCM WAV files."
    )
    parser.add_argument("--input", required=True, help="Input audio directory.")
    parser.add_argument("--output", required=True, help="Output WAV directory.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First output number. Default: 1.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of digits in output names, e.g. 3 gives 001.wav. Default: 3.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Output sample rate. Default: 16000.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=list(DEFAULT_EXTENSIONS),
        help="Input extensions to include. Default: common audio extensions.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scan input directory recursively. Default: false.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing numbered WAV files and mapping CSV.",
    )
    return parser.parse_args()


def normalize_extensions(extensions: Sequence[str]) -> set[str]:
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}


def iter_audio_files(input_dir: Path, extensions: Sequence[str], recursive: bool) -> list[Path]:
    allowed = normalize_extensions(extensions)
    paths = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in allowed)


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg was not found on PATH. Install it first, then rerun this script.")
    return ffmpeg


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative.")
    if args.digits <= 0:
        raise SystemExit("--digits must be greater than 0.")
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be greater than 0.")

    ffmpeg = require_ffmpeg()
    files = iter_audio_files(input_dir, args.extensions, args.recursive)
    if not files:
        raise SystemExit(f"No matching audio files found under: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "mapping.csv"
    if mapping_path.exists() and not args.overwrite:
        raise SystemExit(f"Mapping already exists: {mapping_path}. Pass --overwrite to replace it.")

    rows: list[dict[str, str]] = []
    for offset, source_path in enumerate(files):
        index = args.start_index + offset
        output_name = f"{index:0{args.digits}d}.wav"
        output_path = output_dir / output_name
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

        command = [
            ffmpeg,
            "-y" if args.overwrite else "-n",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-ar",
            str(args.sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        print(f"[{offset + 1}/{len(files)}] {source_path} -> {output_path}")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"ffmpeg failed for {source_path}") from exc

        rows.append(
            {
                "numbered_file": output_name,
                "source_file": str(source_path),
            }
        )

    with mapping_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("numbered_file", "source_file"))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Wrote {len(rows)} WAV files to: {output_dir}")
    print(f"Mapping CSV: {mapping_path}")


if __name__ == "__main__":
    main()
