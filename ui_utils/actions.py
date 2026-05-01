"""WebUI actions that bind Gradio events to CLI command builders."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from typing import Iterator

import gradio as gr

from .commands import (
    conversion_command,
    denoise_command,
    long_split_command,
    model_download_command,
    music_command,
    slice_command,
    speaker_filter_command,
    volume_normalize_command,
)
from .paths import resolve_path
from .runner import run_command_stream


LOG_SEPARATOR = "\n\n" + "-" * 80 + "\n\n"


def _stream_single_command(command: list[str]) -> Iterator[str]:
    yield from run_command_stream(command)


def run_convert(*args) -> Iterator[str]:
    return _stream_single_command(conversion_command(*args))


def run_music(*args) -> Iterator[str]:
    return _stream_single_command(music_command(*args))


def run_denoise(*args) -> Iterator[str]:
    return _stream_single_command(denoise_command(*args))


def run_long_split(*args) -> Iterator[str]:
    return _stream_single_command(long_split_command(*args))


def run_slice(*args) -> Iterator[str]:
    return _stream_single_command(slice_command(*args))


def run_speaker_filter(*args) -> Iterator[str]:
    return _stream_single_command(speaker_filter_command(*args))


def run_volume_normalize(*args) -> Iterator[str]:
    return _stream_single_command(volume_normalize_command(*args))


def run_model_download(*args) -> Iterator[str]:
    return _stream_single_command(model_download_command(*args))


def build_full_pipeline_commands(
    source_path: str,
    workspace: str,
    overwrite: bool,
    do_convert: bool,
    do_music: bool,
    do_denoise: bool,
    do_presplit: bool,
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
    presplit_target_piece_seconds: float,
    presplit_search_window_seconds: float,
    presplit_frame_ms: float,
    presplit_hop_ms: float,
    presplit_min_silence_ms: float,
    slice_mode: str,
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
    llm_model: str,
    punct_llm_model: str,
    rough_llm_model: str,
    llm_provider: str,
    env_path: str,
    llm_concurrency: int,
    llm_max_rounds: int,
) -> tuple[str, list[list[str]]]:
    work_dir = resolve_path(workspace)
    current = str(resolve_path(source_path))
    commands: list[list[str]] = []
    validate_current = True

    if do_convert:
        output = str(work_dir / "01_numbered_wavs")
        commands.append(
            conversion_command(
                current,
                output,
                recursive,
                overwrite,
                sample_rate,
                1,
                3,
                validate_input=validate_current,
            )
        )
        current = output
        validate_current = False
    if do_music:
        output = str(work_dir / "02_vocals")
        commands.append(
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
                validate_input=validate_current,
            )
        )
        current = output
        validate_current = False
    if do_denoise:
        output = str(work_dir / "03_denoised")
        commands.append(
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
                validate_input=validate_current,
            )
        )
        current = output
        validate_current = False
    if do_presplit:
        output = str(work_dir / "03_presliced")
        commands.append(
            long_split_command(
                current,
                output,
                recursive,
                overwrite,
                presplit_target_piece_seconds,
                presplit_search_window_seconds,
                presplit_frame_ms,
                presplit_hop_ms,
                presplit_min_silence_ms,
                validate_input=validate_current,
            )
        )
        current = output
        validate_current = False
    if do_slice:
        output = str(work_dir / "04_sliced")
        commands.append(
            slice_command(
                slice_mode,
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
                llm_model,
                punct_llm_model,
                rough_llm_model,
                llm_provider,
                env_path,
                llm_concurrency,
                llm_max_rounds,
                validate_input=validate_current,
            )
        )

    return str(work_dir), commands


def _find_single_voxcpm_jsonl(dataset_dir: Path) -> Path:
    matches = sorted(dataset_dir.glob("*_voxcpm.jsonl"))
    if not matches:
        raise gr.Error(f"No *_voxcpm.jsonl file found in {dataset_dir}")
    if len(matches) > 1:
        choices = "\n".join(f"  - {path.name}" for path in matches)
        raise gr.Error(f"Multiple *_voxcpm.jsonl files found. Use single-stage tools or keep one JSONL.\n{choices}")
    return matches[0]


def _stream_pipeline_command(command: list[str], completed_logs: list[str]) -> Iterator[str]:
    latest = ""
    for latest in run_command_stream(command):
        yield LOG_SEPARATOR.join([*completed_logs, latest])
    completed_logs.append(latest)


def run_full_pipeline(
    *args,
    do_speaker_filter: bool = False,
    speaker_device: str = "cuda:0",
    speaker_mad_multiplier: float = 3.0,
    speaker_max_prune_ratio: float = 0.10,
    speaker_min_rows: int = 8,
    do_volume_normalize: bool = False,
    volume_target_lufs: float = -20.0,
    volume_max_volume_change_db: float = 12.0,
    volume_max_dynamic_range_db: float = 24.0,
    volume_peak_margin_db: float = 1.0,
    volume_min_active_ratio: float = 0.03,
    volume_sample_rate: int = 0,
) -> Iterator[str]:
    command_arg_count = len(inspect.signature(build_full_pipeline_commands).parameters)
    command_args = args[:command_arg_count]
    post_args = args[command_arg_count:]
    post_defaults = (
        do_speaker_filter,
        speaker_device,
        speaker_mad_multiplier,
        speaker_max_prune_ratio,
        speaker_min_rows,
        do_volume_normalize,
        volume_target_lufs,
        volume_max_volume_change_db,
        volume_max_dynamic_range_db,
        volume_peak_margin_db,
        volume_min_active_ratio,
        volume_sample_rate,
    )
    (
        do_speaker_filter,
        speaker_device,
        speaker_mad_multiplier,
        speaker_max_prune_ratio,
        speaker_min_rows,
        do_volume_normalize,
        volume_target_lufs,
        volume_max_volume_change_db,
        volume_max_dynamic_range_db,
        volume_peak_margin_db,
        volume_min_active_ratio,
        volume_sample_rate,
    ) = (*post_args, *post_defaults[len(post_args) :])

    work_dir, commands = build_full_pipeline_commands(*command_args)
    resolve_path(work_dir).mkdir(parents=True, exist_ok=True)
    logs: list[str] = [f"Workspace: {work_dir}"]
    yield LOG_SEPARATOR.join(logs)
    for command in commands:
        yield from _stream_pipeline_command(command, logs)

    sliced_dir = resolve_path(work_dir) / "04_sliced"
    active_jsonl: Path | None = None
    active_dataset_root: Path | None = None

    if do_speaker_filter:
        active_jsonl = _find_single_voxcpm_jsonl(sliced_dir)
        active_dataset_root = sliced_dir
        filtered_jsonl = sliced_dir / f"{active_jsonl.stem}_speaker_filtered.jsonl"
        command = speaker_filter_command(
            str(active_jsonl),
            str(filtered_jsonl),
            str(active_dataset_root),
            speaker_device,
            speaker_mad_multiplier,
            speaker_max_prune_ratio,
            speaker_min_rows,
            False,
            True,
        )
        yield from _stream_pipeline_command(command, logs)
        active_jsonl = filtered_jsonl

    if do_volume_normalize:
        active_dataset_root = active_dataset_root or sliced_dir
        active_jsonl = active_jsonl or _find_single_voxcpm_jsonl(sliced_dir)
        normalized_dir = resolve_path(work_dir) / "05_normalized"
        command = volume_normalize_command(
            str(active_dataset_root),
            str(normalized_dir),
            str(active_jsonl),
            volume_target_lufs,
            volume_max_volume_change_db,
            volume_max_dynamic_range_db,
            volume_peak_margin_db,
            volume_min_active_ratio,
            volume_sample_rate,
            False,
            True,
        )
        yield from _stream_pipeline_command(command, logs)


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
            display = f"{row['audio_relpath']} | {row['duration_sec']}s | {row['transcript'][:80]}"
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
