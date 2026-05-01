#!/usr/bin/env python
"""Download the model checkpoints expected by webui.py.

The WebUI defaults point at:

- checkpoints/MossFormer2_SE_48K
- checkpoints/Qwen3-ASR-1.7B
- checkpoints/Qwen3-ForcedAligner-0.6B
- checkpoints/ZipEnhancer
- checkpoints/spkrec-ecapa-voxceleb
- Demucs package cache for htdemucs and htdemucs_ft
- Silero VAD package cache

This script prepares those folders in one command and skips existing models
unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MODEL_REPOS = {
    "huggingface": {
        "asr": "Qwen/Qwen3-ASR-1.7B",
        "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
        "mossformer": "alibabasglab/MossFormer2_SE_48K",
        "speaker": "speechbrain/spkrec-ecapa-voxceleb",
    },
    "modelscope": {
        "asr": "Qwen/Qwen3-ASR-1.7B",
        "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
        "mossformer": "alibabasglab/MossFormer2_SE_48K",
        "zipenhancer": "iic/speech_zipenhancer_ans_multiloss_16k_base",
        "speaker": "speechbrain/spkrec-ecapa-voxceleb",
    },
}
DEFAULT_DOWNLOAD_PATH = Path("checkpoints")
DEFAULT_MOSSFORMER_DIR = Path("MossFormer2_SE_48K")
DEFAULT_ZIPENHANCER_DIR = Path("ZipEnhancer")
DEFAULT_SPEAKER_DIR = Path("spkrec-ecapa-voxceleb")
DEFAULT_DEMUCS_MODELS = ("htdemucs", "htdemucs_ft")
ALL_MODELS = (
    "asr",
    "aligner",
    "mossformer",
    "zipenhancer",
    "speaker",
    "demucs",
    "silero",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download checkpoints into the default folders used by webui.py."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all", "qwen3", *ALL_MODELS),
        default=["all"],
        help="Models to download. Default: all.",
    )
    parser.add_argument(
        "--demucs-models",
        nargs="+",
        default=list(DEFAULT_DEMUCS_MODELS),
        help="Demucs model names to prefetch when --models includes demucs. Default: htdemucs htdemucs_ft.",
    )
    parser.add_argument(
        "--provider",
        choices=("modelscope", "huggingface", "hf"),
        default="modelscope",
        help="Download source for all selected models. Default: modelscope.",
    )
    parser.add_argument(
        "--download-path",
        "--download_path",
        type=Path,
        default=DEFAULT_DOWNLOAD_PATH,
        help="Root folder for Hugging Face and ModelScope downloads. Default: checkpoints.",
    )
    parser.add_argument(
        "--output-root",
        dest="download_path",
        type=Path,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
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
        default=None,
        help="MossFormer checkpoint folder. Default: <download-path>/MossFormer2_SE_48K.",
    )
    parser.add_argument(
        "--zipenhancer-dir",
        type=Path,
        default=DEFAULT_ZIPENHANCER_DIR,
        help="ZipEnhancer folder name/path. Default: ZipEnhancer.",
    )
    parser.add_argument(
        "--speaker-dir",
        type=Path,
        default=DEFAULT_SPEAKER_DIR,
        help="SpeechBrain speaker embedding folder name/path. Default: spkrec-ecapa-voxceleb.",
    )
    parser.add_argument(
        "--asr-model-id",
        default=None,
        help="Override the provider model id for Qwen3 ASR.",
    )
    parser.add_argument(
        "--aligner-model-id",
        default=None,
        help="Override the provider model id for the Qwen3 forced aligner.",
    )
    parser.add_argument(
        "--mossformer-model-id",
        default=None,
        help="Override the provider model id for MossFormer2.",
    )
    parser.add_argument(
        "--zipenhancer-model-id",
        default=None,
        help="Override the provider model id for ZipEnhancer.",
    )
    parser.add_argument(
        "--speaker-model-id",
        default=None,
        help="Override the provider model id for the SpeechBrain speaker embedding model.",
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


def resolve_model_dir(download_path: Path, model_dir: Path, root: Path) -> Path:
    if model_dir.is_absolute():
        return model_dir
    return resolve_path(download_path, root) / model_dir


def selected_models(raw_models: list[str]) -> set[str]:
    if "all" in raw_models:
        return set(ALL_MODELS)
    selected: set[str] = set()
    for model in raw_models:
        if model == "qwen3":
            selected.update(("asr", "aligner"))
        else:
            selected.add(model)
    return selected


def _missing_indexed_weight_files(destination: Path) -> list[Path] | None:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = destination / index_name
        if not index_path.exists():
            continue
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_files = set(index_data.get("weight_map", {}).values())
        return sorted(destination / filename for filename in weight_files if not (destination / filename).exists())
    return None


def destination_is_complete(destination: Path, marker: Path | None = None) -> bool:
    if not destination.exists():
        return False
    if marker is not None and marker.exists():
        return True

    missing_weight_files = _missing_indexed_weight_files(destination)
    if missing_weight_files is not None:
        if missing_weight_files:
            missing = "\n".join(f"  - {path.name}" for path in missing_weight_files[:8])
            print(f"Incomplete checkpoint: {destination}\nMissing indexed weight files:\n{missing}")
            return False
        return True

    complete_weight_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "last_best_checkpoint",
    )
    if any((destination / filename).exists() for filename in complete_weight_names):
        return True

    return False


def skip_existing(destination: Path, force: bool, marker: Path | None = None) -> bool:
    if force:
        return False
    if destination_is_complete(destination, marker):
        print(f"Already exists: {destination}")
        return True
    return False


def download_with_modelscope(model_id: str, destination: Path) -> None:
    try:
        from modelscope import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency: modelscope\n"
            f"{sys.executable} -m pip install -r requirements.txt"
        ) from exc
    snapshot_download(model_id=model_id, local_dir=str(destination))


def download_with_hf(model_id: str, destination: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency: huggingface_hub\n"
            f"Install dependencies with: {sys.executable} -m pip install -r requirements.txt"
        ) from exc
    snapshot_download(repo_id=model_id, local_dir=str(destination))


def download_demucs_models(model_names: list[str], force: bool) -> None:
    try:
        from demucs.pretrained import get_model
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency: demucs\n"
            f"Install dependencies with: {sys.executable} -m pip install -r requirements.txt"
        ) from exc

    for model_name in model_names:
        print(f"Downloading Demucs model: {model_name}")
        if force:
            print("Note: Demucs manages its own cache; force does not clear existing cached files.")
        get_model(model_name)


def download_silero_vad(force: bool) -> None:
    try:
        from silero_vad import load_silero_vad
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency: silero-vad\n"
            f"Install dependencies with: {sys.executable} -m pip install -r requirements.txt"
        ) from exc

    print("Downloading Silero VAD model")
    if force:
        print("Note: Silero VAD manages its own cache; force does not clear existing cached files.")
    load_silero_vad()


def download_model(
    model_id: str,
    provider: str,
    destination: Path,
    force: bool,
    *,
    marker: Path | None = None,
) -> None:
    if skip_existing(destination, force, marker=marker):
        return
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id}")
    print(f"Provider: {provider}")
    print(f"Destination: {destination}")
    if provider == "modelscope":
        download_with_modelscope(model_id, destination)
    else:
        download_with_hf(model_id, destination)


def model_id_for(args: argparse.Namespace, provider: str, target: str) -> str:
    overrides = {
        "asr": args.asr_model_id,
        "aligner": args.aligner_model_id,
        "mossformer": args.mossformer_model_id,
        "zipenhancer": args.zipenhancer_model_id,
        "speaker": args.speaker_model_id,
    }
    if overrides[target]:
        return overrides[target]
    if target not in MODEL_REPOS[provider]:
        raise SystemExit(
            f"No default {provider} model id is configured for '{target}'. "
            f"Pass --{target}-model-id if you have a compatible repo, or use --provider modelscope."
        )
    return MODEL_REPOS[provider][target]


def print_webui_paths(
    asr_dir: Path,
    aligner_dir: Path,
    mossformer_dir: Path,
    zipenhancer_dir: Path,
    speaker_dir: Path,
) -> None:
    print("\nDefault local model paths:")
    print(f"  ASR model path           : {asr_dir}")
    print(f"  Forced aligner path      : {aligner_dir}")
    print(f"  MossFormer checkpoint    : {mossformer_dir}")
    print(f"  ZipEnhancer model        : {zipenhancer_dir}")
    print(f"  Speaker embedding model  : {speaker_dir}")


def main() -> None:
    args = parse_args()
    root = project_root()
    download_path = resolve_path(args.download_path, root)
    asr_dir = resolve_model_dir(args.download_path, args.asr_dir, root).resolve()
    aligner_dir = resolve_model_dir(args.download_path, args.aligner_dir, root).resolve()
    mossformer_path = args.mossformer_path or args.download_path / DEFAULT_MOSSFORMER_DIR
    mossformer_dir = resolve_path(mossformer_path, root).resolve()
    zipenhancer_dir = resolve_model_dir(args.download_path, args.zipenhancer_dir, root).resolve()
    speaker_dir = resolve_model_dir(args.download_path, args.speaker_dir, root).resolve()
    provider = "huggingface" if args.provider == "hf" else args.provider

    targets = selected_models(args.models)
    if "asr" in targets:
        download_model(model_id_for(args, provider, "asr"), provider, asr_dir, args.force)
    if "aligner" in targets:
        download_model(model_id_for(args, provider, "aligner"), provider, aligner_dir, args.force)
    if "mossformer" in targets:
        download_model(
            model_id_for(args, provider, "mossformer"),
            provider,
            mossformer_dir,
            args.force,
            marker=mossformer_dir / "last_best_checkpoint",
        )
    if "zipenhancer" in targets:
        download_model(model_id_for(args, provider, "zipenhancer"), provider, zipenhancer_dir, args.force)
    if "speaker" in targets:
        download_model(
            model_id_for(args, provider, "speaker"),
            provider,
            speaker_dir,
            args.force,
            marker=speaker_dir / "hyperparams.yaml",
        )
    if "demucs" in targets:
        download_demucs_models(args.demucs_models, args.force)
    if "silero" in targets:
        download_silero_vad(args.force)

    print_webui_paths(asr_dir, aligner_dir, mossformer_dir, zipenhancer_dir, speaker_dir)
    print(f"\nDownload path: {download_path.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
