"""Path and validation helpers for the WebUI."""

from __future__ import annotations

from pathlib import Path

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = PROJECT_ROOT / "webui_runs"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma")


def resolve_path(raw_path: str | None) -> Path:
    path = Path((raw_path or "").strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def script_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def ensure_input(path: Path, *, must_be_dir: bool = False) -> None:
    if not path.exists():
        raise gr.Error(f"Input path does not exist: {path}")
    if must_be_dir and not path.is_dir():
        raise gr.Error(f"This stage expects a folder: {path}")

