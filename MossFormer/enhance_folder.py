"""Enhance every audio file in a folder with MossFormer2_SE_48K.

This script is intentionally portable: copy the whole MossFormer folder into
another project and run this file from there. Relative checkpoint paths are
resolved from the parent directory that contains MossFormer.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


MODEL_NAME = "MossFormer2_SE_48K"
DEFAULT_AUDIO_EXTENSIONS = (
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".aac",
    ".aiff",
    ".m4a",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance a folder of audios with the bundled MossFormer2 speech enhancement code."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing noisy input audio files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where enhanced audio files will be written.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=4.0,
        help="Decode window in seconds. Larger values can be faster but use more memory.",
    )
    parser.add_argument(
        "--one-time-decode-seconds",
        type=float,
        default=20.0,
        help="Maximum length decoded in one pass before segmented decoding is used.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help=(
            "Number of background workers for audio loading and resampling. "
            "GPU inference remains single-threaded. Default: 1."
        ),
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help=(
            "Number of equal-length MossFormer windows to run per GPU forward "
            "pass for long audio. Higher values use more VRAM. Default: 1."
        ),
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even when CUDA is available.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help=(
            "Directory containing last_best_checkpoint and checkpoint weights. "
            "Relative paths are resolved from the parent directory of MossFormer."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files with the same names.",
    )
    return parser.parse_args()


def configure_imports(script_dir: Path) -> None:
    sys.path.insert(0, str(script_dir))


def resolve_model_path(model_path: Path, project_root: Path) -> Path:
    if model_path.is_absolute():
        return model_path
    return project_root / model_path


def validate_model_path(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint folder does not exist: {model_path}. "
            "Download it first with download_checkpoints.py --model-path <path>."
        )
    if not model_path.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a folder: {model_path}")

    required_files = ("last_best_checkpoint", "last_best_checkpoint.pt")
    missing = [name for name in required_files if not (model_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint folder is missing required file(s): {', '.join(missing)}. "
            f"Path: {model_path}"
        )


def validate_input_dir(input_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    has_audio = any(
        path.is_file() and path.suffix.lower() in DEFAULT_AUDIO_EXTENSIONS
        for path in input_dir.iterdir()
    )
    if not has_audio:
        extensions = ", ".join(DEFAULT_AUDIO_EXTENSIONS)
        raise FileNotFoundError(f"No supported audio files found in {input_dir}. Supported: {extensions}")


def flatten_model_output(staging_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if not staging_dir.exists():
        raise FileNotFoundError(f"Expected model output folder was not created: {staging_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for source in staging_dir.iterdir():
        target = output_dir / source.name
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"Output already exists: {target}. Re-run with --overwrite to replace it.")
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(source), str(target))

    try:
        staging_dir.rmdir()
    except OSError:
        pass


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_path = resolve_model_path(args.model_path, project_root)

    validate_input_dir(input_dir)
    validate_model_path(model_path)
    if args.preprocess_workers <= 0:
        raise ValueError("--preprocess-workers must be greater than 0.")
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be greater than 0.")
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MOSSFORMER_CHECKPOINT_DIR"] = str(model_path.resolve())
    configure_imports(script_dir)

    previous_cwd = Path.cwd()
    os.chdir(project_root)
    try:
        try:
            from clearvoice import ClearVoice
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f"Missing Python dependency: {exc.name}\n"
                f"Install dependencies with: {sys.executable} -m pip install -r "
                f"\"{script_dir / 'requirements.txt'}\""
            ) from exc

        print(f"Loading {MODEL_NAME}...")
        clearvoice = ClearVoice(task="speech_enhancement", model_names=[MODEL_NAME])
        for model in clearvoice.models:
            model.args.decode_window = args.chunk_seconds
            model.args.one_time_decode_length = args.one_time_decode_seconds
            model.args.preprocess_workers = args.preprocess_workers
            model.args.inference_batch_size = args.inference_batch_size
            if hasattr(model.model, "eval"):
                model.model.eval()
            if args.cpu:
                model.args.use_cuda = 0

        output_dir.mkdir(parents=True, exist_ok=True)
        devices = ", ".join(sorted({str(model.device) for model in clearvoice.models}))
        print(f"Input : {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Device: {devices}")
        print(f"Model : {model_path.resolve()}")
        print(f"Preprocess workers: {args.preprocess_workers}")
        print(f"Inference batch size: {args.inference_batch_size}")

        model_output_dir = output_dir / MODEL_NAME
        if model_output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Temporary model output folder already exists: {model_output_dir}. "
                    "Re-run with --overwrite or remove it first."
                )
            shutil.rmtree(model_output_dir)

        clearvoice(input_path=str(input_dir), online_write=True, output_path=str(output_dir))

        flatten_model_output(model_output_dir, output_dir, args.overwrite)
        print("Done.")
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
