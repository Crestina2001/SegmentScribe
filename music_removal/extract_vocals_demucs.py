#!/usr/bin/env python
"""Extract vocal stems from audio files with Demucs.

This is intended as a preprocessing step for music-bed speech audio:

    original audio -> vocal stem -> enhance_audio.py

Demucs is not vendored here. Install it separately with:

    pip install -U demucs soundfile
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract vocal stems with Demucs so music-heavy speech can be fed "
            "into audio_enhancement/enhance_audio.py."
        )
    )
    parser.add_argument("--input", required=True, help="Input audio file or directory.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output WAV path for one input file, or output directory for directory input.",
    )
    parser.add_argument(
        "--model",
        default="htdemucs",
        help=(
            "Demucs model name. Try htdemucs_ft for better quality if you can "
            "afford slower processing. Default: htdemucs."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Demucs device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="Demucs random shifts. Higher can improve quality but is slower. Default: 1.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Demucs parallel jobs. Default: 0.",
    )
    parser.add_argument(
        "--segment",
        type=float,
        default=None,
        help="Optional Demucs segment length in seconds. Lower values use less VRAM.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --input is a directory, scan audio files recursively. Default: true.",
    )
    parser.add_argument(
        "--allowed-extensions",
        nargs="*",
        default=sorted(AUDIO_EXTENSIONS),
        help="Audio extensions to process when --input is a directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output WAVs.",
    )
    parser.add_argument(
        "--keep-demucs-output",
        action="store_true",
        help="Keep Demucs' full output folder next to each copied vocal WAV.",
    )
    return parser.parse_args()


def normalize_extensions(extensions: Sequence[str]) -> set[str]:
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}


def iter_audio_files(input_dir: Path, allowed_extensions: Sequence[str], recursive: bool) -> list[Path]:
    exts = normalize_extensions(allowed_extensions)
    paths = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in exts)


def output_path_for_input(input_file: Path, input_root: Path, output_root: Path) -> Path:
    relpath = input_file.relative_to(input_root)
    return output_root / relpath.with_suffix(".wav")


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def demucs_command(
    input_path: Path,
    demucs_output_dir: Path,
    model: str,
    device: str | None,
    shifts: int,
    jobs: int,
    segment: float | None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("demucs_soundfile_runner.py")),
        "--two-stems",
        "vocals",
        "--name",
        model,
        "--out",
        str(demucs_output_dir),
        "--shifts",
        str(shifts),
        "--jobs",
        str(jobs),
    ]
    if device:
        command.extend(["--device", device])
    if segment is not None:
        command.extend(["--segment", str(segment)])
    command.append(str(input_path))
    return command


def find_vocals_wav(demucs_output_dir: Path, model: str, input_path: Path) -> Path:
    expected = demucs_output_dir / model / input_path.stem / "vocals.wav"
    if expected.exists():
        return expected

    matches = sorted(demucs_output_dir.rglob("vocals.wav"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(f"Demucs did not produce vocals.wav for {input_path}")
    raise RuntimeError(
        f"Found multiple vocals.wav files for {input_path}; expected one under {expected}"
    )


def copy_or_move_vocals(vocals_path: Path, output_path: Path, overwrite: bool) -> None:
    ensure_output_path(output_path, overwrite=overwrite)
    if output_path.exists():
        output_path.unlink()
    shutil.copy2(vocals_path, output_path)


def require_runtime_dependencies() -> None:
    if (
        importlib.util.find_spec("demucs") is None
        or importlib.util.find_spec("demucs.separate") is None
    ):
        raise SystemExit(
            "Missing dependency 'demucs'. Install it with: pip install -U demucs soundfile"
        )

    if importlib.util.find_spec("soundfile") is None:
        raise SystemExit("Missing dependency 'soundfile'. Install it with: pip install -U soundfile")


def extract_one(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> bool:
    try:
        ensure_output_path(output_path, overwrite=args.overwrite)
        if args.keep_demucs_output:
            demucs_root = output_path.parent / f"{output_path.stem}_demucs"
            if demucs_root.exists() and args.overwrite:
                shutil.rmtree(demucs_root)
            demucs_root.mkdir(parents=True, exist_ok=True)
            command = demucs_command(
                input_path=input_path,
                demucs_output_dir=demucs_root,
                model=args.model,
                device=args.device,
                shifts=args.shifts,
                jobs=args.jobs,
                segment=args.segment,
            )
            subprocess.run(command, check=True)
            vocals_path = find_vocals_wav(demucs_root, args.model, input_path)
            copy_or_move_vocals(vocals_path, output_path, overwrite=args.overwrite)
        else:
            with tempfile.TemporaryDirectory(prefix="demucs_vocals_") as temp_root:
                demucs_root = Path(temp_root)
                command = demucs_command(
                    input_path=input_path,
                    demucs_output_dir=demucs_root,
                    model=args.model,
                    device=args.device,
                    shifts=args.shifts,
                    jobs=args.jobs,
                    segment=args.segment,
                )
                subprocess.run(command, check=True)
                vocals_path = find_vocals_wav(demucs_root, args.model, input_path)
                copy_or_move_vocals(vocals_path, output_path, overwrite=args.overwrite)

        print(f"Wrote vocal stem: {output_path}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Failed: Demucs exited with {exc.returncode} for {input_path}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"Failed: {input_path}: {exc}", file=sys.stderr)
        return False


def main() -> None:
    args = parse_args()

    require_runtime_dependencies()

    if args.shifts < 0:
        raise SystemExit("--shifts must be greater than or equal to 0.")
    if args.jobs < 0:
        raise SystemExit("--jobs must be greater than or equal to 0.")
    if args.segment is not None and args.segment <= 0:
        raise SystemExit("--segment must be greater than 0.")

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    if not input_path.exists():
        raise SystemExit(f"Input audio does not exist: {input_path}")

    if input_path.is_dir():
        source_files = iter_audio_files(input_path, args.allowed_extensions, args.recursive)
        if not source_files:
            raise SystemExit(f"No audio files found under: {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Found {len(source_files)} audio files under {input_path}")

        succeeded = 0
        failed = 0
        for index, source_file in enumerate(source_files, start=1):
            destination = output_path_for_input(source_file, input_path, output_path)
            print(f"\n[{index}/{len(source_files)}] {source_file} -> {destination}")
            if extract_one(source_file, destination, args):
                succeeded += 1
            else:
                failed += 1

        print(f"\nDone. Succeeded: {succeeded}; failed: {failed}; output: {output_path}")
        if failed:
            raise SystemExit(1)
        return

    if output_path.exists() and output_path.is_dir():
        output_path = output_path / input_path.with_suffix(".wav").name
    if not extract_one(input_path, output_path, args):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
