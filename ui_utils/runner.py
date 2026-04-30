"""Subprocess runner helpers for WebUI actions."""

from __future__ import annotations

import subprocess
from typing import Iterable

import gradio as gr

from .paths import PROJECT_ROOT


def format_command(command: Iterable[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run_command(command: list[str]) -> str:
    lines = [f"$ {format_command(command)}", ""]
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
    return_code = process.wait()
    lines.append("")
    lines.append(f"Exit code: {return_code}")
    if return_code != 0:
        raise gr.Error("\n".join(lines[-40:]))
    return "\n".join(lines)

