"""Download MossFormer2_SE_48K checkpoints.

Relative --model-path values are resolved from the parent directory that
contains this MossFormer folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


MODEL_NAME = "MossFormer2_SE_48K"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Download {MODEL_NAME} checkpoints from Hugging Face.")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help=(
            "Destination checkpoint directory. Relative paths are resolved from "
            "the parent directory of MossFormer."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if last_best_checkpoint already exists.",
    )
    return parser.parse_args()


def resolve_model_path(model_path: Path, project_root: Path) -> Path:
    if model_path.is_absolute():
        return model_path
    return project_root / model_path


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    model_path = resolve_model_path(args.model_path, project_root).resolve()
    marker = model_path / "last_best_checkpoint"

    if marker.exists() and not args.force:
        print(f"Checkpoint already exists: {model_path}")
        print("Use --force to download again.")
        return

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Python dependency: {exc.name}\n"
            f"Install dependencies with: {sys.executable} -m pip install -r "
            f"\"{script_dir / 'requirements.txt'}\""
        ) from exc

    model_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading alibabasglab/{MODEL_NAME}")
    print(f"Destination: {model_path}")
    snapshot_download(repo_id=f"alibabasglab/{MODEL_NAME}", local_dir=str(model_path))
    print("Done.")


if __name__ == "__main__":
    main()
