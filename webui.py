#!/usr/bin/env python
"""Gradio WebUI for the SegmentScribe audio preparation pipeline."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'gradio'. Install it with:\n\n"
        "  python -m pip install gradio\n\n"
        "Then rerun: python webui.py"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = PROJECT_ROOT / "webui_runs"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma")


def resolve_path(raw_path: str | None) -> Path:
    path = Path((raw_path or "").strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


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


def script_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


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


def ensure_input(path: Path, *, must_be_dir: bool = False) -> None:
    if not path.exists():
        raise gr.Error(f"Input path does not exist: {path}")
    if must_be_dir and not path.is_dir():
        raise gr.Error(f"This stage expects a folder: {path}")


def conversion_command(
    input_path: str,
    output_path: str,
    recursive: bool,
    overwrite: bool,
    sample_rate: int,
    start_index: int,
    digits: int,
) -> list[str]:
    source = resolve_path(input_path)
    output = resolve_path(output_path)
    ensure_input(source, must_be_dir=True)
    command = [
        sys.executable,
        script_path("utils", "convert_to_numbered_wav.py"),
        "--input",
        str(source),
        "--output",
        str(output),
        "--sample-rate",
        str(int(sample_rate)),
        "--start-index",
        str(int(start_index)),
        "--digits",
        str(int(digits)),
    ]
    if recursive:
        command.append("--recursive")
    if overwrite:
        command.append("--overwrite")
    return command


def music_command(
    input_path: str,
    output_path: str,
    device: str,
    model: str,
    shifts: int,
    jobs: int,
    segment: float | None,
    recursive: bool,
    overwrite: bool,
) -> list[str]:
    source = resolve_path(input_path)
    output = resolve_path(output_path)
    ensure_input(source)
    command = [
        sys.executable,
        script_path("music_removal", "extract_vocals_demucs.py"),
        "--input",
        str(source),
        "--output",
        str(output),
        "--model",
        model,
        "--shifts",
        str(int(shifts)),
        "--jobs",
        str(int(jobs)),
    ]
    if device.strip():
        command.extend(["--device", device.strip()])
    if segment and float(segment) > 0:
        command.extend(["--segment", str(segment)])
    command.append("--recursive" if recursive else "--no-recursive")
    if overwrite:
        command.append("--overwrite")
    return command


def denoise_command(
    method: str,
    input_path: str,
    output_path: str,
    device: str,
    mossformer_model_path: str,
    mossformer_cpu: bool,
    preprocess_workers: int,
    inference_batch_size: int,
    zip_model: str,
    zip_normalize: str,
    zip_alignment_metric: str,
    overwrite: bool,
) -> list[str]:
    source = resolve_path(input_path)
    output = resolve_path(output_path)
    ensure_input(source, must_be_dir=True)
    if method == "MossFormer2":
        command = [
            sys.executable,
            script_path("MossFormer", "enhance_folder.py"),
            "--input-dir",
            str(source),
            "--output-dir",
            str(output),
            "--model-path",
            str(resolve_path(mossformer_model_path)),
            "--preprocess-workers",
            str(int(preprocess_workers)),
            "--inference-batch-size",
            str(int(inference_batch_size)),
        ]
        if mossformer_cpu:
            command.append("--cpu")
    else:
        command = [
            sys.executable,
            script_path("zip_enhancer", "enhance_audio.py"),
            "--input",
            str(source),
            "--output",
            str(output),
            "--model",
            zip_model,
            "--preprocess-workers",
            str(int(preprocess_workers)),
            "--normalize",
            zip_normalize,
            "--alignment-metric",
            zip_alignment_metric,
            "--silence-aware-splits",
        ]
        if device.strip():
            command.extend(["--device", device.strip()])
    if overwrite:
        command.append("--overwrite")
    return command


def slice_command(
    input_path: str,
    output_path: str,
    model_path: str,
    aligner_path: str,
    backend: str,
    device: str,
    dtype: str,
    batch_size: int,
    min_seg_sec: float,
    max_seg_sec: float,
    language: str,
    vad_backend: str,
    punctuation: bool,
    overwrite: bool,
    dry_run: bool,
) -> list[str]:
    source = resolve_path(input_path)
    output = resolve_path(output_path)
    ensure_input(source)
    command = [
        sys.executable,
        "-m",
        "slide_rule",
        "--input",
        str(source),
        "--output-dir",
        str(output),
        "--model-path",
        str(resolve_path(model_path)),
        "--aligner-path",
        str(resolve_path(aligner_path)),
        "--asr-backend",
        backend,
        "--device",
        device,
        "--dtype",
        dtype,
        "--asr-max-batch-size",
        str(int(batch_size)),
        "--min-seg-sec",
        str(min_seg_sec),
        "--max-seg-sec",
        str(max_seg_sec),
        "--vad-backend",
        vad_backend,
    ]
    if language.strip():
        command.extend(["--language", language.strip()])
    if punctuation:
        command.append("--enable-punctuation-correction")
    if overwrite:
        command.append("--overwrite")
    if dry_run:
        command.append("--dry-run")
    return command


def run_convert(*args) -> str:
    return run_command(conversion_command(*args))


def run_music(*args) -> str:
    return run_command(music_command(*args))


def run_denoise(*args) -> str:
    return run_command(denoise_command(*args))


def run_slice(*args) -> str:
    return run_command(slice_command(*args))


def speaker_filter_command(
    input_jsonl: str,
    output_jsonl: str,
    dataset_root: str,
    device: str,
    mad_multiplier: float,
    max_prune_ratio: float,
    min_rows: int,
    dry_run: bool,
    overwrite: bool,
) -> list[str]:
    input_path = resolve_path(input_jsonl)
    output_path = resolve_path(output_jsonl)
    ensure_input(input_path)
    command = [
        sys.executable,
        script_path("speaker_outlier_filter", "filter_speaker_outliers.py"),
        "--input-jsonl",
        str(input_path),
        "--output-jsonl",
        str(output_path),
        "--device",
        device.strip() or "cuda:0",
        "--mad-multiplier",
        str(mad_multiplier),
        "--max-prune-ratio",
        str(max_prune_ratio),
        "--min-rows",
        str(int(min_rows)),
    ]
    if dataset_root.strip():
        command.extend(["--dataset-root", str(resolve_path(dataset_root))])
    if dry_run:
        command.append("--dry-run")
    if overwrite:
        command.append("--overwrite")
    return command


def volume_normalize_command(
    input_path: str,
    output_path: str,
    jsonl_path: str,
    target_lufs: float,
    max_volume_change_db: float,
    max_dynamic_range_db: float,
    peak_margin_db: float,
    min_active_ratio: float,
    sample_rate: int | None,
    dry_run: bool,
    overwrite: bool,
) -> list[str]:
    source = resolve_path(input_path)
    output = resolve_path(output_path)
    ensure_input(source, must_be_dir=True)
    command = [
        sys.executable,
        script_path("utils", "normalize_corpus_volume.py"),
        "--input",
        str(source),
        "--output",
        str(output),
        "--target-lufs",
        str(target_lufs),
        "--max-volume-change-db",
        str(max_volume_change_db),
        "--max-dynamic-range-db",
        str(max_dynamic_range_db),
        "--peak-margin-db",
        str(peak_margin_db),
        "--min-active-ratio",
        str(min_active_ratio),
    ]
    if jsonl_path.strip():
        command.extend(["--jsonl", jsonl_path.strip()])
    if sample_rate and int(sample_rate) > 0:
        command.extend(["--sample-rate", str(int(sample_rate))])
    if dry_run:
        command.append("--dry-run")
    if overwrite:
        command.append("--overwrite")
    return command


def run_speaker_filter(*args) -> str:
    return run_command(speaker_filter_command(*args))


def run_volume_normalize(*args) -> str:
    return run_command(volume_normalize_command(*args))


def run_model_download(
    models: list[str],
    provider: str,
    output_root: str,
    asr_dir: str,
    aligner_dir: str,
    mossformer_path: str,
    force: bool,
) -> str:
    selected = models or ["all"]
    command = [
        sys.executable,
        script_path("utils", "download_model_webui.py"),
        "--models",
        *selected,
        "--provider",
        provider,
        "--output-root",
        str(resolve_path(output_root)),
        "--asr-dir",
        asr_dir,
        "--aligner-dir",
        aligner_dir,
        "--mossformer-path",
        str(resolve_path(mossformer_path)),
    ]
    if force:
        command.append("--force")
    return run_command(command)


def run_full_pipeline(
    source_path: str,
    workspace: str,
    overwrite: bool,
    do_convert: bool,
    do_music: bool,
    do_denoise: bool,
    do_slice: bool,
    recursive: bool,
    sample_rate: int,
    device: str,
    demucs_model: str,
    demucs_shifts: int,
    demucs_jobs: int,
    demucs_segment: float | None,
    denoise_method: str,
    mossformer_model_path: str,
    mossformer_cpu: bool,
    preprocess_workers: int,
    inference_batch_size: int,
    zip_model: str,
    zip_normalize: str,
    zip_alignment_metric: str,
    asr_model_path: str,
    aligner_path: str,
    backend: str,
    dtype: str,
    batch_size: int,
    min_seg_sec: float,
    max_seg_sec: float,
    language: str,
    vad_backend: str,
    punctuation: bool,
) -> str:
    work_dir = resolve_path(workspace)
    work_dir.mkdir(parents=True, exist_ok=True)
    current = str(resolve_path(source_path))
    logs: list[str] = [f"Workspace: {work_dir}"]

    if do_convert:
        output = str(work_dir / "01_numbered_wavs")
        logs.append(run_command(conversion_command(current, output, recursive, overwrite, sample_rate, 1, 3)))
        current = output
    if do_music:
        output = str(work_dir / "02_vocals")
        logs.append(
            run_command(
                music_command(
                    current,
                    output,
                    device,
                    demucs_model,
                    demucs_shifts,
                    demucs_jobs,
                    demucs_segment,
                    recursive,
                    overwrite,
                )
            )
        )
        current = output
    if do_denoise:
        output = str(work_dir / "03_denoised")
        logs.append(
            run_command(
                denoise_command(
                    denoise_method,
                    current,
                    output,
                    device,
                    mossformer_model_path,
                    mossformer_cpu,
                    preprocess_workers,
                    inference_batch_size,
                    zip_model,
                    zip_normalize,
                    zip_alignment_metric,
                    overwrite,
                )
            )
        )
        current = output
    if do_slice:
        output = str(work_dir / "04_sliced")
        logs.append(
            run_command(
                slice_command(
                    current,
                    output,
                    asr_model_path,
                    aligner_path,
                    backend,
                    device,
                    dtype,
                    batch_size,
                    min_seg_sec,
                    max_seg_sec,
                    language,
                    vad_backend,
                    punctuation,
                    overwrite,
                    False,
                )
            )
        )

    return "\n\n" + ("-" * 80 + "\n\n").join(logs)


def read_summary(output_path: str) -> str:
    summary_path = resolve_path(output_path) / "summary.json"
    if not summary_path.exists():
        return f"No summary.json found in {summary_path.parent}"
    return json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)


def read_manifest(output_path: str) -> tuple[list[list[str]], list[str]]:
    output_dir = resolve_path(output_path)
    manifest_path = output_dir / "manifest.tsv"
    if not manifest_path.exists():
        return [], []
    rows: list[list[str]] = []
    choices: list[str] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            display = (
                f"{row['audio_relpath']} | {row['duration_sec']}s | "
                f"{row['transcript'][:80]}"
            )
            choices.append(display)
            rows.append(
                [
                    row["audio_relpath"],
                    row["duration_sec"],
                    row["sample_rate"],
                    row["transcript"],
                ]
            )
    return rows, choices


def refresh_results(output_path: str):
    summary = read_summary(output_path)
    rows, choices = read_manifest(output_path)
    value = choices[0] if choices else None
    return summary, rows, gr.update(choices=choices, value=value)


def preview_segment(output_path: str, selection: str | None):
    if not selection:
        return None, ""
    output_dir = resolve_path(output_path)
    audio_relpath = selection.split(" | ", 1)[0]
    audio_path = output_dir / audio_relpath
    transcript_path = output_dir / audio_relpath.replace("audios/", "transcripts/").replace(".wav", ".txt")
    transcript = transcript_path.read_text(encoding="utf-8").strip() if transcript_path.exists() else ""
    return str(audio_path), transcript


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SegmentScribe WebUI") as app:
        gr.Markdown("# SegmentScribe WebUI")
        gr.Markdown("Convert audio, remove music, denoise speech, slice segments, and review the result.")

        with gr.Tab("Full pipeline"):
            source_path = gr.Textbox(label="Source audio folder or file", value=str(PROJECT_ROOT / "audios"))
            workspace = gr.Textbox(label="Workspace", value=str(DEFAULT_WORK_DIR))
            with gr.Row():
                browse_full_source = gr.Button("Browse source...")
                browse_full_workspace = gr.Button("Browse workspace...")
            with gr.Row():
                overwrite = gr.Checkbox(label="Overwrite outputs", value=True)
                recursive = gr.Checkbox(label="Scan folders recursively", value=True)
            with gr.Row():
                do_convert = gr.Checkbox(label="1. Transform format", value=True)
                do_music = gr.Checkbox(label="2. Eliminate music", value=False)
                do_denoise = gr.Checkbox(label="3. Eliminate noise", value=True)
                do_slice = gr.Checkbox(label="4. Slice audios", value=True)

            with gr.Accordion("Conversion settings", open=False):
                sample_rate = gr.Number(label="Sample rate", value=16000, precision=0)

            with gr.Accordion("Music removal settings", open=False):
                demucs_model = gr.Dropdown(["htdemucs", "htdemucs_ft"], value="htdemucs", label="Demucs model")
                with gr.Row():
                    device = gr.Textbox(label="Device", value="cuda:0")
                    demucs_shifts = gr.Number(label="Demucs shifts", value=1, precision=0)
                    demucs_jobs = gr.Number(label="Demucs jobs", value=0, precision=0)
                    demucs_segment = gr.Number(label="Demucs segment seconds", value=0, precision=1)

            with gr.Accordion("Denoise settings", open=False):
                denoise_method = gr.Dropdown(["MossFormer2", "ZipEnhancer"], value="MossFormer2", label="Method")
                mossformer_model_path = gr.Textbox(
                    label="MossFormer checkpoint path",
                    value=str(PROJECT_ROOT / "checkpoints" / "MossFormer2_SE_48K"),
                )
                browse_full_mossformer = gr.Button("Browse MossFormer checkpoint...")
                mossformer_cpu = gr.Checkbox(label="Force MossFormer CPU", value=False)
                with gr.Row():
                    preprocess_workers = gr.Number(label="Preprocess workers", value=1, precision=0)
                    inference_batch_size = gr.Number(label="Inference batch size", value=1, precision=0)
                zip_model = gr.Textbox(label="ZipEnhancer model", value="iic/speech_zipenhancer_ans_multiloss_16k_base")
                with gr.Row():
                    zip_normalize = gr.Dropdown(
                        ["match_original", "none", "library_median"],
                        value="match_original",
                        label="Zip normalize",
                    )
                    zip_alignment_metric = gr.Dropdown(["rms", "peak"], value="rms", label="Zip alignment metric")

            with gr.Accordion("Slicing settings", open=False):
                asr_model_path = gr.Textbox(
                    label="ASR model path",
                    value=str(PROJECT_ROOT / "checkpoints" / "Qwen3-ASR-1.7B"),
                )
                browse_full_asr = gr.Button("Browse ASR model...")
                aligner_path = gr.Textbox(
                    label="Forced aligner path",
                    value=str(PROJECT_ROOT / "checkpoints" / "Qwen3-ForcedAligner-0.6B"),
                )
                browse_full_aligner = gr.Button("Browse aligner...")
                with gr.Row():
                    backend = gr.Dropdown(["transformers", "vllm"], value="transformers", label="ASR backend")
                    dtype = gr.Dropdown(["bfloat16", "float16", "float32"], value="bfloat16", label="dtype")
                    batch_size = gr.Number(label="ASR batch size", value=1, precision=0)
                with gr.Row():
                    min_seg_sec = gr.Number(label="Min segment seconds", value=3.0)
                    max_seg_sec = gr.Number(label="Max segment seconds", value=10.0)
                    vad_backend = gr.Dropdown(["auto", "silero", "librosa"], value="auto", label="VAD backend")
                language = gr.Textbox(label="Language hint", value="")
                punctuation = gr.Checkbox(label="Enable punctuation correction", value=False)

            browse_full_source.click(browse_folder, inputs=source_path, outputs=source_path)
            browse_full_workspace.click(browse_folder, inputs=workspace, outputs=workspace)
            browse_full_mossformer.click(browse_folder, inputs=mossformer_model_path, outputs=mossformer_model_path)
            browse_full_asr.click(browse_folder, inputs=asr_model_path, outputs=asr_model_path)
            browse_full_aligner.click(browse_folder, inputs=aligner_path, outputs=aligner_path)

            run_pipeline = gr.Button("Run selected pipeline", variant="primary")
            pipeline_log = gr.Textbox(label="Log", lines=24)
            run_pipeline.click(
                run_full_pipeline,
                inputs=[
                    source_path,
                    workspace,
                    overwrite,
                    do_convert,
                    do_music,
                    do_denoise,
                    do_slice,
                    recursive,
                    sample_rate,
                    device,
                    demucs_model,
                    demucs_shifts,
                    demucs_jobs,
                    demucs_segment,
                    denoise_method,
                    mossformer_model_path,
                    mossformer_cpu,
                    preprocess_workers,
                    inference_batch_size,
                    zip_model,
                    zip_normalize,
                    zip_alignment_metric,
                    asr_model_path,
                    aligner_path,
                    backend,
                    dtype,
                    batch_size,
                    min_seg_sec,
                    max_seg_sec,
                    language,
                    vad_backend,
                    punctuation,
                ],
                outputs=pipeline_log,
            )

        with gr.Tab("Single stages"):
            with gr.Accordion("1. Transform format", open=True):
                c_input = gr.Textbox(label="Input folder")
                c_output = gr.Textbox(label="Output WAV folder", value=str(DEFAULT_WORK_DIR / "01_numbered_wavs"))
                with gr.Row():
                    c_browse_input = gr.Button("Browse input...")
                    c_browse_output = gr.Button("Browse output...")
                with gr.Row():
                    c_recursive = gr.Checkbox(label="Recursive", value=True)
                    c_overwrite = gr.Checkbox(label="Overwrite", value=True)
                    c_sample_rate = gr.Number(label="Sample rate", value=16000, precision=0)
                    c_start = gr.Number(label="Start index", value=1, precision=0)
                    c_digits = gr.Number(label="Digits", value=3, precision=0)
                c_run = gr.Button("Run conversion")
                c_log = gr.Textbox(label="Conversion log", lines=10)
                c_run.click(
                    run_convert,
                    inputs=[c_input, c_output, c_recursive, c_overwrite, c_sample_rate, c_start, c_digits],
                    outputs=c_log,
                )

            with gr.Accordion("2. Eliminate music", open=False):
                m_input = gr.Textbox(label="Input file or folder")
                m_output = gr.Textbox(label="Output vocal WAV/folder", value=str(DEFAULT_WORK_DIR / "02_vocals"))
                with gr.Row():
                    m_browse_input = gr.Button("Browse input...")
                    m_browse_output = gr.Button("Browse output...")
                with gr.Row():
                    m_device = gr.Textbox(label="Device", value="cuda:0")
                    m_model = gr.Dropdown(["htdemucs", "htdemucs_ft"], value="htdemucs", label="Model")
                    m_shifts = gr.Number(label="Shifts", value=1, precision=0)
                    m_jobs = gr.Number(label="Jobs", value=0, precision=0)
                    m_segment = gr.Number(label="Segment seconds", value=0, precision=1)
                with gr.Row():
                    m_recursive = gr.Checkbox(label="Recursive", value=True)
                    m_overwrite = gr.Checkbox(label="Overwrite", value=True)
                m_run = gr.Button("Run music removal")
                m_log = gr.Textbox(label="Music removal log", lines=10)
                m_run.click(
                    run_music,
                    inputs=[m_input, m_output, m_device, m_model, m_shifts, m_jobs, m_segment, m_recursive, m_overwrite],
                    outputs=m_log,
                )

            with gr.Accordion("3. Eliminate noise", open=False):
                d_method = gr.Dropdown(["MossFormer2", "ZipEnhancer"], value="MossFormer2", label="Method")
                d_input = gr.Textbox(label="Input folder")
                d_output = gr.Textbox(label="Output folder", value=str(DEFAULT_WORK_DIR / "03_denoised"))
                with gr.Row():
                    d_browse_input = gr.Button("Browse input...")
                    d_browse_output = gr.Button("Browse output...")
                    d_browse_checkpoint = gr.Button("Browse checkpoint...")
                d_device = gr.Textbox(label="Device", value="cuda:0")
                d_mossformer_path = gr.Textbox(
                    label="MossFormer checkpoint path",
                    value=str(PROJECT_ROOT / "checkpoints" / "MossFormer2_SE_48K"),
                )
                d_cpu = gr.Checkbox(label="Force MossFormer CPU", value=False)
                with gr.Row():
                    d_workers = gr.Number(label="Preprocess workers", value=1, precision=0)
                    d_batch = gr.Number(label="Inference batch size", value=1, precision=0)
                d_zip_model = gr.Textbox(label="ZipEnhancer model", value="iic/speech_zipenhancer_ans_multiloss_16k_base")
                with gr.Row():
                    d_zip_normalize = gr.Dropdown(["match_original", "none", "library_median"], value="match_original", label="Normalize")
                    d_zip_metric = gr.Dropdown(["rms", "peak"], value="rms", label="Alignment metric")
                    d_overwrite = gr.Checkbox(label="Overwrite", value=True)
                d_run = gr.Button("Run denoise")
                d_log = gr.Textbox(label="Denoise log", lines=10)
                d_run.click(
                    run_denoise,
                    inputs=[
                        d_method,
                        d_input,
                        d_output,
                        d_device,
                        d_mossformer_path,
                        d_cpu,
                        d_workers,
                        d_batch,
                        d_zip_model,
                        d_zip_normalize,
                        d_zip_metric,
                        d_overwrite,
                    ],
                    outputs=d_log,
                )

            with gr.Accordion("4. Slice audios", open=False):
                s_input = gr.Textbox(label="Input file or folder")
                s_output = gr.Textbox(label="Output sliced folder", value=str(DEFAULT_WORK_DIR / "04_sliced"))
                s_model = gr.Textbox(label="ASR model path", value=str(PROJECT_ROOT / "checkpoints" / "Qwen3-ASR-1.7B"))
                s_aligner = gr.Textbox(
                    label="Forced aligner path",
                    value=str(PROJECT_ROOT / "checkpoints" / "Qwen3-ForcedAligner-0.6B"),
                )
                with gr.Row():
                    s_browse_input = gr.Button("Browse input...")
                    s_browse_output = gr.Button("Browse output...")
                    s_browse_model = gr.Button("Browse ASR model...")
                    s_browse_aligner = gr.Button("Browse aligner...")
                with gr.Row():
                    s_backend = gr.Dropdown(["transformers", "vllm"], value="transformers", label="Backend")
                    s_device = gr.Textbox(label="Device", value="cuda:0")
                    s_dtype = gr.Dropdown(["bfloat16", "float16", "float32"], value="bfloat16", label="dtype")
                    s_batch = gr.Number(label="Batch size", value=1, precision=0)
                with gr.Row():
                    s_min = gr.Number(label="Min seconds", value=3.0)
                    s_max = gr.Number(label="Max seconds", value=10.0)
                    s_vad = gr.Dropdown(["auto", "silero", "librosa"], value="auto", label="VAD")
                with gr.Row():
                    s_language = gr.Textbox(label="Language hint", value="")
                    s_punctuation = gr.Checkbox(label="Punctuation correction", value=False)
                    s_overwrite = gr.Checkbox(label="Overwrite", value=True)
                    s_dry = gr.Checkbox(label="Dry run", value=False)
                s_run = gr.Button("Run slicing")
                s_log = gr.Textbox(label="Slicing log", lines=10)
                s_run.click(
                    run_slice,
                    inputs=[
                        s_input,
                        s_output,
                        s_model,
                        s_aligner,
                        s_backend,
                        s_device,
                        s_dtype,
                        s_batch,
                        s_min,
                        s_max,
                        s_language,
                        s_vad,
                        s_punctuation,
                        s_overwrite,
                        s_dry,
                    ],
                    outputs=s_log,
                )

            c_browse_input.click(browse_folder, inputs=c_input, outputs=c_input)
            c_browse_output.click(browse_folder, inputs=c_output, outputs=c_output)
            m_browse_input.click(browse_folder, inputs=m_input, outputs=m_input)
            m_browse_output.click(browse_folder, inputs=m_output, outputs=m_output)
            d_browse_input.click(browse_folder, inputs=d_input, outputs=d_input)
            d_browse_output.click(browse_folder, inputs=d_output, outputs=d_output)
            d_browse_checkpoint.click(browse_folder, inputs=d_mossformer_path, outputs=d_mossformer_path)
            s_browse_input.click(browse_folder, inputs=s_input, outputs=s_input)
            s_browse_output.click(browse_folder, inputs=s_output, outputs=s_output)
            s_browse_model.click(browse_folder, inputs=s_model, outputs=s_model)
            s_browse_aligner.click(browse_folder, inputs=s_aligner, outputs=s_aligner)

        with gr.Tab("5. Check result"):
            result_dir = gr.Textbox(label="Sliced output folder", value=str(DEFAULT_WORK_DIR / "04_sliced"))
            result_browse_folder = gr.Button("Browse result folder...")
            refresh = gr.Button("Refresh results", variant="primary")
            summary = gr.Textbox(label="summary.json", lines=12)
            manifest = gr.Dataframe(
                headers=["audio_relpath", "duration_sec", "sample_rate", "transcript"],
                label="manifest.tsv",
                interactive=False,
                wrap=True,
            )
            segment = gr.Dropdown(label="Segment")
            with gr.Row():
                audio = gr.Audio(label="Audio", type="filepath")
                transcript = gr.Textbox(label="Transcript", lines=4)
            refresh.click(refresh_results, inputs=result_dir, outputs=[summary, manifest, segment])
            result_browse_folder.click(browse_folder, inputs=result_dir, outputs=result_dir)
            segment.change(preview_segment, inputs=[result_dir, segment], outputs=[audio, transcript])

        with gr.Tab("6. Filter speaker outliers"):
            speaker_input = gr.Textbox(label="Input VoxCPM JSONL")
            speaker_output = gr.Textbox(label="Filtered output JSONL")
            speaker_root = gr.Textbox(label="Dataset root (optional)")
            with gr.Row():
                speaker_browse_input = gr.Button("Browse input JSONL...")
                speaker_browse_output = gr.Button("Browse output JSONL...")
                speaker_browse_root = gr.Button("Browse dataset root...")
            with gr.Row():
                speaker_device = gr.Textbox(label="Device", value="cuda:0")
                speaker_mad = gr.Number(label="MAD multiplier", value=3.0)
                speaker_max_prune = gr.Number(label="Max prune ratio", value=0.10)
                speaker_min_rows = gr.Number(label="Min rows", value=8, precision=0)
            with gr.Row():
                speaker_dry_run = gr.Checkbox(label="Dry run", value=False)
                speaker_overwrite = gr.Checkbox(label="Overwrite", value=True)
            speaker_run = gr.Button("Run speaker filter", variant="primary")
            speaker_log = gr.Textbox(label="Speaker filter log", lines=16)
            speaker_browse_input.click(browse_jsonl_file, inputs=speaker_input, outputs=speaker_input)
            speaker_browse_output.click(browse_jsonl_save, inputs=speaker_output, outputs=speaker_output)
            speaker_browse_root.click(browse_folder, inputs=speaker_root, outputs=speaker_root)
            speaker_run.click(
                run_speaker_filter,
                inputs=[
                    speaker_input,
                    speaker_output,
                    speaker_root,
                    speaker_device,
                    speaker_mad,
                    speaker_max_prune,
                    speaker_min_rows,
                    speaker_dry_run,
                    speaker_overwrite,
                ],
                outputs=speaker_log,
            )

        with gr.Tab("7. Normalize volume"):
            volume_input = gr.Textbox(label="Sliced input folder", value=str(DEFAULT_WORK_DIR / "04_sliced"))
            volume_output = gr.Textbox(label="Normalized output folder", value=str(DEFAULT_WORK_DIR / "05_normalized"))
            volume_jsonl = gr.Textbox(label="JSONL filename/path (optional)", value="")
            with gr.Row():
                volume_browse_input = gr.Button("Browse input...")
                volume_browse_output = gr.Button("Browse output...")
            with gr.Row():
                volume_target = gr.Number(label="Target LUFS", value=-20.0)
                volume_max_gain = gr.Number(label="Max upward gain dB", value=12.0)
                volume_dynamic = gr.Number(label="Max dynamic range dB", value=24.0)
            with gr.Row():
                volume_peak_margin = gr.Number(label="Peak margin dB", value=1.0)
                volume_active_ratio = gr.Number(label="Min active ratio", value=0.03)
                volume_sample_rate = gr.Number(label="Output sample rate (0 keeps source)", value=0, precision=0)
            with gr.Row():
                volume_dry_run = gr.Checkbox(label="Dry run", value=False)
                volume_overwrite = gr.Checkbox(label="Overwrite", value=True)
            volume_run = gr.Button("Run volume normalization", variant="primary")
            volume_log = gr.Textbox(label="Volume normalization log", lines=16)
            volume_browse_input.click(browse_folder, inputs=volume_input, outputs=volume_input)
            volume_browse_output.click(browse_folder, inputs=volume_output, outputs=volume_output)
            volume_run.click(
                run_volume_normalize,
                inputs=[
                    volume_input,
                    volume_output,
                    volume_jsonl,
                    volume_target,
                    volume_max_gain,
                    volume_dynamic,
                    volume_peak_margin,
                    volume_active_ratio,
                    volume_sample_rate,
                    volume_dry_run,
                    volume_overwrite,
                ],
                outputs=volume_log,
            )

        with gr.Tab("Models"):
            gr.Markdown("Download checkpoints into the default folders used by the WebUI.")
            download_models = gr.CheckboxGroup(
                ["all", "qwen3", "asr", "aligner", "mossformer"],
                value=["all"],
                label="Models",
            )
            download_provider = gr.Dropdown(
                ["modelscope", "hf"],
                value="modelscope",
                label="Qwen3 download provider",
            )
            download_output_root = gr.Textbox(label="Checkpoint root", value=str(PROJECT_ROOT / "checkpoints"))
            download_asr_dir = gr.Textbox(label="ASR directory name", value="Qwen3-ASR-1.7B")
            download_aligner_dir = gr.Textbox(label="Aligner directory name", value="Qwen3-ForcedAligner-0.6B")
            download_mossformer_path = gr.Textbox(
                label="MossFormer checkpoint path",
                value=str(PROJECT_ROOT / "checkpoints" / "MossFormer2_SE_48K"),
            )
            with gr.Row():
                browse_download_root = gr.Button("Browse checkpoint root...")
                browse_download_mossformer = gr.Button("Browse MossFormer path...")
            download_force = gr.Checkbox(label="Force re-download", value=False)
            download_button = gr.Button("Download models", variant="primary")
            download_log = gr.Textbox(label="Download log", lines=18)
            browse_download_root.click(browse_folder, inputs=download_output_root, outputs=download_output_root)
            browse_download_mossformer.click(
                browse_folder,
                inputs=download_mossformer_path,
                outputs=download_mossformer_path,
            )
            download_button.click(
                run_model_download,
                inputs=[
                    download_models,
                    download_provider,
                    download_output_root,
                    download_asr_dir,
                    download_aligner_dir,
                    download_mossformer_path,
                    download_force,
                ],
                outputs=download_log,
            )

    return app


if __name__ == "__main__":
    build_app().queue().launch()
