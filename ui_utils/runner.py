"""Subprocess runner helpers for WebUI actions."""

from __future__ import annotations

import os
import subprocess
from typing import Iterable, Iterator

import gradio as gr

from .paths import PROJECT_ROOT


def format_command(command: Iterable[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run_command(command: list[str]) -> str:
    lines: list[str] = []
    for output in run_command_stream(command):
        lines = output.splitlines()
    return "\n".join(lines)


def run_command_stream(command: list[str]) -> Iterator[str]:
    lines = [f"$ {format_command(command)}", ""]
    yield "\n".join(lines)
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
        yield "\n".join(lines)
    return_code = process.wait()
    lines.append("")
    lines.append(f"Exit code: {return_code}")
    yield "\n".join(lines)
    if return_code != 0:
        raise gr.Error("\n".join(lines[-40:]))

