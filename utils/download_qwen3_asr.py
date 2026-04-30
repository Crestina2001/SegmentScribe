#!/usr/bin/env python
"""Download Qwen3-ASR and Qwen3-ForcedAligner checkpoints.

Relative output paths are resolved from the project root.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


MODELS = {
    "asr": "Qwen/Qwen3-ASR-1.7B",
    "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Qwen3-ASR and Qwen3-ForcedAligner checkpoints."
    )
    parser.add_argument(
        "--provider",
        choices=("modelscope", "huggingface", "hf"),
        default="modelscope",
        help="Model hub to use. Default: modelscope.",
    )
    parser.add_argument(
        "--output-root",
        "--download-path",
        "--download_path",
        dest="output_root",
        type=Path,
        default=Path("checkpoints"),
        help="Directory that will contain both model folders. Default: checkpoints.",
    )
    parser.add_argument(
        "--asr-dir",
        type=Path,
        default=Path("Qwen3-ASR-1.7B"),
        help="ASR model directory name or path under --output-root.",
    )
    parser.add_argument(
        "--aligner-dir",
        type=Path,
        default=Path("Qwen3-ForcedAligner-0.6B"),
        help="Forced aligner directory name or path under --output-root.",
    )
    parser.add_argument(
        "--model",
        choices=("all", "asr", "aligner"),
        default="all",
        help="Which checkpoint to download. Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if the destination directory is not empty.",
    )
    return parser.parse_args()


def resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root / path


def resolve_model_dir(output_root: Path, model_dir: Path, project_root: Path) -> Path:
    if model_dir.is_absolute():
        return model_dir
    return resolve_path(output_root, project_root) / model_dir


def ensure_download_allowed(destination: Path, force: bool) -> None:
    if destination.exists() and any(destination.iterdir()) and not force:
        raise SystemExit(f"Destination is not empty: {destination}. Pass --force to reuse it.")
    destination.mkdir(parents=True, exist_ok=True)


def download_with_modelscope(model_id: str, destination: Path) -> None:
    modelscope = shutil.which("modelscope")
    if not modelscope:
        raise SystemExit(
            "Missing ModelScope CLI. Install dependencies with: "
            f"{sys.executable} -m pip install -r requirements.txt"
        )

    command = [
        modelscope,
        "download",
        "--model",
        model_id,
        "--local_dir",
        str(destination),
    ]
    subprocess.run(command, check=True)


def download_with_hf(model_id: str, destination: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency: huggingface_hub\n"
            f"Install dependencies with: {sys.executable} -m pip install -r requirements.txt"
        ) from exc

    snapshot_download(repo_id=model_id, local_dir=str(destination))


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    targets = ("asr", "aligner") if args.model == "all" else (args.model,)
    destinations = {
        "asr": resolve_model_dir(args.output_root, args.asr_dir, project_root).resolve(),
        "aligner": resolve_model_dir(args.output_root, args.aligner_dir, project_root).resolve(),
    }

    for target in targets:
        model_id = MODELS[target]
        destination = destinations[target]
        ensure_download_allowed(destination, args.force)
        print(f"Downloading {model_id}")
        print(f"Destination: {destination}")
        provider = "huggingface" if args.provider == "hf" else args.provider
        if provider == "modelscope":
            download_with_modelscope(model_id, destination)
        else:
            download_with_hf(model_id, destination)

    print("Done.")


if __name__ == "__main__":
    main()
