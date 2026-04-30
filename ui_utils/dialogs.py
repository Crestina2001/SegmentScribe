"""Native browse dialogs used by the local WebUI."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from .paths import PROJECT_ROOT, resolve_path


def browse_folder(current_path: str | None = None) -> str:
    """Open a native folder chooser on the machine running the WebUI."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise gr.Error(f"Native folder dialog is unavailable: {exc}") from exc

    initial_dir = resolve_path(current_path or str(PROJECT_ROOT))
    if initial_dir.is_file():
        initial_dir = initial_dir.parent
    if not initial_dir.exists():
        initial_dir = PROJECT_ROOT

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(
            initialdir=str(initial_dir),
            title="Choose a folder",
            mustexist=False,
        )
    finally:
        root.destroy()
    return str(Path(selected).resolve()) if selected else (current_path or "")


def browse_jsonl_file(current_path: str | None = None) -> str:
    """Open a native JSONL file chooser on the machine running the WebUI."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise gr.Error(f"Native file dialog is unavailable: {exc}") from exc

    initial_dir = resolve_path(current_path or str(PROJECT_ROOT))
    if initial_dir.is_file():
        initial_dir = initial_dir.parent
    if not initial_dir.exists():
        initial_dir = PROJECT_ROOT

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            title="Choose a JSONL file",
            filetypes=(("JSONL files", "*.jsonl"), ("All files", "*.*")),
        )
    finally:
        root.destroy()
    return str(Path(selected).resolve()) if selected else (current_path or "")


def browse_jsonl_save(current_path: str | None = None) -> str:
    """Open a native JSONL save dialog on the machine running the WebUI."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise gr.Error(f"Native file dialog is unavailable: {exc}") from exc

    initial = resolve_path(current_path or str(PROJECT_ROOT / "filtered.jsonl"))
    initial_dir = initial.parent if initial.suffix else initial
    if not initial_dir.exists():
        initial_dir = PROJECT_ROOT

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.asksaveasfilename(
            initialdir=str(initial_dir),
            initialfile=initial.name if initial.suffix else "filtered.jsonl",
            title="Choose output JSONL",
            defaultextension=".jsonl",
            filetypes=(("JSONL files", "*.jsonl"), ("All files", "*.*")),
        )
    finally:
        root.destroy()
    return str(Path(selected).resolve()) if selected else (current_path or "")

