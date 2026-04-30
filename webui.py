#!/usr/bin/env python
"""Gradio WebUI for the SegmentScribe audio preparation pipeline."""

from __future__ import annotations

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'gradio'. Install it with:\n\n"
        "  python -m pip install gradio\n\n"
        "Then rerun: python webui.py"
    ) from exc

from ui_utils.tabs import build_tabs


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SegmentScribe WebUI") as app:
        build_tabs()
    return app


if __name__ == "__main__":
    build_app().queue().launch()
