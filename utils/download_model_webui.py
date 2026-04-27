#!/usr/bin/env python
"""Download the model checkpoints expected by webui.py.

The WebUI defaults point at:

- checkpoints/MossFormer2_SE_48K
- checkpoints/Qwen3-ASR-1.7B
- checkpoints/Qwen3-ForcedAligner-0.6B

This script prepares those folders in one command and skips existing models
unless --force is passed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


QWEN_MODELS = {
    "asr": "Qwen/Qwen3-ASR-1.7B",
    "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
}
MOSSFORMER_REPO = "alibabasglab/MossFormer2_SE_48K"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download checkpoints into the default folders used by webui.py."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all", "qwen3", "asr", "aligner", "mossformer"),
        default=["all"],
        help="Models to download. Default: all.",
    )
    parser.add_argument(
        "--provider",
        choices=("modelscope", "hf"),
        default="modelscope",
        help="Provider for Qwen3 ASR/aligner downloads. Default: modelscope.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("checkpoints"),
        help="Root checkpoint folder used by the WebUI. Default: checkpoints.",
    )
    parser.add_argument(
        "--asr-dir",
        type=Path,
        default=Path("Qwen3-ASR-1.7B"),
        help="ASR model folder name/path. Default: Qwen3-ASR-1.7B.",
    )
    parser.add_argument(
        "--aligner-dir",
        type=Path,
        default=Path("Qwen3-ForcedAligner-0.6B"),
        help="Forced-aligner model folder name/path. Default: Qwen3-ForcedAligner-0.6B.",
    )
    parser.add_argument(
        "--mossformer-path",
        type=Path,
        default=Path("checkpoints/MossFormer2_SE_48K"),
        help="MossFormer checkpoint folder used by the WebUI.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if the destination already contains files.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def resolve_model_dir(output_root: Path, model_dir: Path, root: Path) -> Path:
    if model_dir.is_absolute():
        return model_dir
    return resolve_path(output_root, root) / model_dir


def selected_models(raw_models: list[str]) -> set[str]:
    if "all" in raw_models:
        return {"asr", "aligner", "mossformer"}
    selected: set[str] = set()
    for model in raw_models:
        if model == "qwen3":
            selected.update(("asr", "aligner"))
        else:
            selected.add(model)
    return selected


def destination_has_files(destination: Path) -> bool:
    return destination.exists() and any(destination.iterdir())


def skip_existing(destination: Path, force: bool, marker: Path | None = None) -> bool:
    if force:
        return False
    if marker is not None and marker.exists():
        print(f"Already exists: {destination}")
        return True
    if destination_has_files(destination):
        print(f"Already exists: {destination}")
        return True
    return False


def download_with_modelscope(model_id: str, destination: Path) -> None:
    modelscope = shutil.which("modelscope")
    if not modelscope:
        raise SystemExit(
            "Missing ModelScope CLI. Install dependencies with: "
            f"{sys.executable} -m pip install -r requirements.txt"
        )
    subprocess.run(
        [modelscope, "download", "--model", model_id, "--local_dir", str(destination)],
        check=True,
    )


def download_with_hf(model_id: str, destination: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency: huggingface_hub\n"
            f"Install dependencies with: {sys.executable} -m pip install -r requirements.txt"
        ) from exc
    snapshot_download(repo_id=model_id, local_dir=str(destination))


def download_qwen(target: str, provider: str, destination: Path, force: bool) -> None:
    if skip_existing(destination, force):
        return
    destination.mkdir(parents=True, exist_ok=True)
    model_id = QWEN_MODELS[target]
    print(f"Downloading {model_id}")
    print(f"Destination: {destination}")
    if provider == "modelscope":
        download_with_modelscope(model_id, destination)
    else:
        download_with_hf(model_id, destination)


def download_mossformer(destination: Path, force: bool) -> None:
    marker = destination / "last_best_checkpoint"
    if skip_existing(destination, force, marker=marker):
        return
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MOSSFORMER_REPO}")
    print(f"Destination: {destination}")
    download_with_hf(MOSSFORMER_REPO, destination)


def print_webui_paths(asr_dir: Path, aligner_dir: Path, mossformer_dir: Path) -> None:
    print("\nWebUI model paths:")
    print(f"  ASR model path           : {asr_dir}")
    print(f"  Forced aligner path      : {aligner_dir}")
    print(f"  MossFormer checkpoint    : {mossformer_dir}")


def main() -> None:
    args = parse_args()
    root = project_root()
    output_root = resolve_path(args.output_root, root)
    asr_dir = resolve_model_dir(args.output_root, args.asr_dir, root).resolve()
    aligner_dir = resolve_model_dir(args.output_root, args.aligner_dir, root).resolve()
    mossformer_dir = resolve_path(args.mossformer_path, root).resolve()

    targets = selected_models(args.models)
    if "asr" in targets:
        download_qwen("asr", args.provider, asr_dir, args.force)
    if "aligner" in targets:
        download_qwen("aligner", args.provider, aligner_dir, args.force)
    if "mossformer" in targets:
        download_mossformer(mossformer_dir, args.force)

    print_webui_paths(asr_dir, aligner_dir, mossformer_dir)
    print(f"\nCheckpoint root: {output_root.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
