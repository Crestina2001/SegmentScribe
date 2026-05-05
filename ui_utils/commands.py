"""CLI command builders for the WebUI."""

from __future__ import annotations

import sys

import gradio as gr

from .paths import ensure_input, resolve_path, script_path


RULE_SLICE_MODE = "Rule-based"
LLM_SLICE_MODE = "LLM-assisted"
SLICE_MODES = (RULE_SLICE_MODE, LLM_SLICE_MODE)
LLM_PROVIDERS = ("", "openai", "minimax", "anthropic", "gemini", "deepseek")
MODEL_REPO_PREFIXES = ("Qwen/", "alibabasglab/")
RULE_ROUGH_CUT_STRATEGIES = ("legacy_dp", "priority_silence_v1", "priority_silence_v2", "priority_silence_v3", "dp_strategy_2")
LLM_ROUGH_CUT_STRATEGIES = (
    "llm_pause_priority_silence_v2",
    "llm_tool",
    "llm_slice_v1",
    *RULE_ROUGH_CUT_STRATEGIES,
)


def resolve_model_arg(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if value.startswith(MODEL_REPO_PREFIXES):
        return value
    return str(resolve_path(value))


def conversion_command(
    input_path: str,
    output_path: str,
    recursive: bool,
    overwrite: bool,
    sample_rate: int,
    start_index: int,
    digits: int,
    *,
    validate_input: bool = True,
) -> list[str]:
    source = resolve_path(input_path)
    output = resolve_path(output_path)
    if validate_input:
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
    *,
    validate_input: bool = True,
) -> list[str]:
    source = resolve_path(input_path)
    output = resolve_path(output_path)
    if validate_input:
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
    *,
    validate_input: bool = True,
) -> list[str]:
    source = resolve_path(input_path)
    output = resolve_path(output_path)
    if validate_input:
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


def long_split_command(
    input_path: str,
    output_path: str,
    recursive: bool,
    overwrite: bool,
    target_piece_seconds: float,
    search_window_seconds: float,
    frame_ms: float,
    hop_ms: float,
    min_silence_ms: float,
    *,
    validate_input: bool = True,
) -> list[str]:
    source = resolve_path(input_path)
    output = resolve_path(output_path)
    if validate_input:
        ensure_input(source)
    command = [
        sys.executable,
        script_path("utils", "auto_slice_long_audio.py"),
        "--input",
        str(source),
        "--output",
        str(output),
        "--target-piece-seconds",
        str(target_piece_seconds),
        "--search-window-seconds",
        str(search_window_seconds),
        "--frame-ms",
        str(frame_ms),
        "--hop-ms",
        str(hop_ms),
        "--min-silence-ms",
        str(min_silence_ms),
        "--recursive" if recursive else "--no-recursive",
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def slice_command(
    mode: str,
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
    llm_model: str,
    punct_llm_model: str,
    rough_llm_model: str,
    llm_provider: str,
    env_path: str,
    llm_concurrency: int,
    llm_max_rounds: int,
    asr_backend_kwargs: str = "{}",
    forced_aligner_kwargs: str = "{}",
    aligner_max_batch_size: int = 1,
    aligner_concurrency: int = 1,
    target_sample_rate: int = 16000,
    max_source_seconds: float = 1800.0,
    allowed_extensions: str = "",
    preprocess_chunk_mode: str = "rms_silence",
    preprocess_chunk_sec: float = 30.0,
    preprocess_min_chunk_sec: float = 5.0,
    preprocess_max_chunk_sec: float = 15.0,
    rms_silence_frame_ms: float = 25.0,
    rms_silence_hop_ms: float = 5.0,
    rms_silence_percentile: float = 25.0,
    rms_silence_threshold_multiplier: float = 1.8,
    rms_min_silence_ms: float = 80.0,
    vad_threshold: float = 0.5,
    vad_min_speech_ms: int = 250,
    vad_min_silence_ms: int = 300,
    vad_speech_pad_ms: int = 200,
    rough_cut_strategy: str = "priority_silence_v3",
    llm_rough_cut_strategy: str = "llm_slice_v1",
    *,
    validate_input: bool = True,
) -> list[str]:
    if mode == LLM_SLICE_MODE:
        return llm_slice_command(
            input_path,
            output_path,
            model_path,
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
            dry_run,
            llm_model,
            punct_llm_model,
            rough_llm_model,
            llm_provider,
            env_path,
            llm_concurrency,
            llm_max_rounds,
            asr_backend_kwargs,
            forced_aligner_kwargs,
            aligner_max_batch_size,
            aligner_concurrency,
            target_sample_rate,
            max_source_seconds,
            allowed_extensions,
            preprocess_chunk_mode,
            preprocess_chunk_sec,
            preprocess_min_chunk_sec,
            preprocess_max_chunk_sec,
            rms_silence_frame_ms,
            rms_silence_hop_ms,
            rms_silence_percentile,
            rms_silence_threshold_multiplier,
            rms_min_silence_ms,
            vad_threshold,
            vad_min_speech_ms,
            vad_min_silence_ms,
            vad_speech_pad_ms,
            llm_rough_cut_strategy,
            validate_input=validate_input,
        )
    return rule_slice_command(
        input_path,
        output_path,
        model_path,
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
        dry_run,
        asr_backend_kwargs,
        forced_aligner_kwargs,
        aligner_max_batch_size,
        aligner_concurrency,
        target_sample_rate,
        max_source_seconds,
        allowed_extensions,
        preprocess_chunk_mode,
        preprocess_chunk_sec,
        preprocess_min_chunk_sec,
        preprocess_max_chunk_sec,
        rms_silence_frame_ms,
        rms_silence_hop_ms,
        rms_silence_percentile,
        rms_silence_threshold_multiplier,
        rms_min_silence_ms,
        vad_threshold,
        vad_min_speech_ms,
        vad_min_silence_ms,
        vad_speech_pad_ms,
        rough_cut_strategy,
        validate_input=validate_input,
    )


def _base_slice_command(
    module_name: str,
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
    asr_backend_kwargs: str,
    forced_aligner_kwargs: str,
    aligner_max_batch_size: int,
    aligner_concurrency: int,
    target_sample_rate: int,
    max_source_seconds: float,
    allowed_extensions: str,
    preprocess_chunk_mode: str,
    preprocess_chunk_sec: float,
    preprocess_min_chunk_sec: float,
    preprocess_max_chunk_sec: float,
    rms_silence_frame_ms: float,
    rms_silence_hop_ms: float,
    rms_silence_percentile: float,
    rms_silence_threshold_multiplier: float,
    rms_min_silence_ms: float,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    *,
    validate_input: bool = True,
) -> list[str]:
    source = resolve_path(input_path)
    output = resolve_path(output_path)
    if validate_input:
        ensure_input(source)
    command = [
        sys.executable,
        "-m",
        module_name,
        "--input",
        str(source),
        "--output-dir",
        str(output),
        "--model-path",
        resolve_model_arg(model_path),
        "--aligner-path",
        resolve_model_arg(aligner_path),
        "--asr-backend",
        backend,
        "--asr-backend-kwargs",
        asr_backend_kwargs.strip() or "{}",
        "--forced-aligner-kwargs",
        forced_aligner_kwargs.strip() or "{}",
        "--device",
        device,
        "--dtype",
        dtype,
        "--asr-max-batch-size",
        str(int(batch_size)),
        "--aligner-max-batch-size",
        str(int(aligner_max_batch_size)),
        "--aligner-concurrency",
        str(int(aligner_concurrency)),
        "--target-sample-rate",
        str(int(target_sample_rate)),
        "--max-source-seconds",
        str(max_source_seconds),
        "--preprocess-chunk-sec",
        str(preprocess_chunk_sec),
        "--preprocess-chunk-mode",
        preprocess_chunk_mode,
        "--preprocess-min-chunk-sec",
        str(preprocess_min_chunk_sec),
        "--preprocess-max-chunk-sec",
        str(preprocess_max_chunk_sec),
        "--rms-silence-frame-ms",
        str(rms_silence_frame_ms),
        "--rms-silence-hop-ms",
        str(rms_silence_hop_ms),
        "--rms-silence-percentile",
        str(rms_silence_percentile),
        "--rms-silence-threshold-multiplier",
        str(rms_silence_threshold_multiplier),
        "--rms-min-silence-ms",
        str(rms_min_silence_ms),
        "--min-seg-sec",
        str(min_seg_sec),
        "--max-seg-sec",
        str(max_seg_sec),
        "--vad-backend",
        vad_backend,
        "--vad-threshold",
        str(vad_threshold),
        "--vad-min-speech-ms",
        str(int(vad_min_speech_ms)),
        "--vad-min-silence-ms",
        str(int(vad_min_silence_ms)),
        "--vad-speech-pad-ms",
        str(int(vad_speech_pad_ms)),
    ]
    if language.strip():
        command.extend(["--language", language.strip()])
    extensions = [item.strip() for item in allowed_extensions.split(",") if item.strip()]
    if extensions:
        command.extend(["--allowed-extensions", *extensions])
    return command


def rule_slice_command(
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
    asr_backend_kwargs: str = "{}",
    forced_aligner_kwargs: str = "{}",
    aligner_max_batch_size: int = 1,
    aligner_concurrency: int = 1,
    target_sample_rate: int = 16000,
    max_source_seconds: float = 1800.0,
    allowed_extensions: str = "",
    preprocess_chunk_mode: str = "rms_silence",
    preprocess_chunk_sec: float = 30.0,
    preprocess_min_chunk_sec: float = 5.0,
    preprocess_max_chunk_sec: float = 15.0,
    rms_silence_frame_ms: float = 25.0,
    rms_silence_hop_ms: float = 5.0,
    rms_silence_percentile: float = 25.0,
    rms_silence_threshold_multiplier: float = 1.8,
    rms_min_silence_ms: float = 80.0,
    vad_threshold: float = 0.5,
    vad_min_speech_ms: int = 250,
    vad_min_silence_ms: int = 300,
    vad_speech_pad_ms: int = 200,
    rough_cut_strategy: str = "priority_silence_v3",
    *,
    validate_input: bool = True,
) -> list[str]:
    command = _base_slice_command(
        "slide_rule",
        input_path,
        output_path,
        model_path,
        aligner_path,
        backend,
        device,
        dtype,
        batch_size,
        min_seg_sec,
        max_seg_sec,
        language,
        vad_backend,
        asr_backend_kwargs,
        forced_aligner_kwargs,
        aligner_max_batch_size,
        aligner_concurrency,
        target_sample_rate,
        max_source_seconds,
        allowed_extensions,
        preprocess_chunk_mode,
        preprocess_chunk_sec,
        preprocess_min_chunk_sec,
        preprocess_max_chunk_sec,
        rms_silence_frame_ms,
        rms_silence_hop_ms,
        rms_silence_percentile,
        rms_silence_threshold_multiplier,
        rms_min_silence_ms,
        vad_threshold,
        vad_min_speech_ms,
        vad_min_silence_ms,
        vad_speech_pad_ms,
        validate_input=validate_input,
    )
    if rough_cut_strategy.strip():
        command.extend(["--rough-cut-strategy", rough_cut_strategy.strip()])
    if punctuation:
        command.append("--enable-punctuation-correction")
    if overwrite:
        command.append("--overwrite")
    if dry_run:
        command.append("--dry-run")
    return command


def llm_slice_command(
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
    llm_model: str,
    punct_llm_model: str,
    rough_llm_model: str,
    llm_provider: str,
    env_path: str,
    llm_concurrency: int,
    llm_max_rounds: int,
    asr_backend_kwargs: str = "{}",
    forced_aligner_kwargs: str = "{}",
    aligner_max_batch_size: int = 1,
    aligner_concurrency: int = 1,
    target_sample_rate: int = 16000,
    max_source_seconds: float = 1800.0,
    allowed_extensions: str = "",
    preprocess_chunk_mode: str = "rms_silence",
    preprocess_chunk_sec: float = 30.0,
    preprocess_min_chunk_sec: float = 5.0,
    preprocess_max_chunk_sec: float = 15.0,
    rms_silence_frame_ms: float = 25.0,
    rms_silence_hop_ms: float = 5.0,
    rms_silence_percentile: float = 25.0,
    rms_silence_threshold_multiplier: float = 1.8,
    rms_min_silence_ms: float = 80.0,
    vad_threshold: float = 0.5,
    vad_min_speech_ms: int = 250,
    vad_min_silence_ms: int = 300,
    vad_speech_pad_ms: int = 200,
    rough_cut_strategy: str = "llm_slice_v1",
    *,
    validate_input: bool = True,
) -> list[str]:
    if not llm_model.strip():
        raise gr.Error("LLM-assisted slicing requires an LLM model.")
    command = _base_slice_command(
        "slide_LLM",
        input_path,
        output_path,
        model_path,
        aligner_path,
        backend,
        device,
        dtype,
        batch_size,
        min_seg_sec,
        max_seg_sec,
        language,
        vad_backend,
        asr_backend_kwargs,
        forced_aligner_kwargs,
        aligner_max_batch_size,
        aligner_concurrency,
        target_sample_rate,
        max_source_seconds,
        allowed_extensions,
        preprocess_chunk_mode,
        preprocess_chunk_sec,
        preprocess_min_chunk_sec,
        preprocess_max_chunk_sec,
        rms_silence_frame_ms,
        rms_silence_hop_ms,
        rms_silence_percentile,
        rms_silence_threshold_multiplier,
        rms_min_silence_ms,
        vad_threshold,
        vad_min_speech_ms,
        vad_min_silence_ms,
        vad_speech_pad_ms,
        validate_input=validate_input,
    )
    command.extend(
        [
            "--llm-model",
            llm_model.strip(),
            "--llm-concurrency",
            str(int(llm_concurrency)),
            "--llm-max-rounds",
            str(int(llm_max_rounds)),
        ]
    )
    if rough_cut_strategy.strip():
        command.extend(["--rough-cut-strategy", rough_cut_strategy.strip()])
    if punct_llm_model.strip():
        command.extend(["--punct-llm-model", punct_llm_model.strip()])
    if rough_llm_model.strip():
        command.extend(["--rough-llm-model", rough_llm_model.strip()])
    if punctuation:
        command.append("--enable-punctuation-correction")
    if llm_provider.strip():
        command.extend(["--llm-provider", llm_provider.strip()])
    if env_path.strip():
        command.extend(["--env-path", str(resolve_path(env_path))])
    if overwrite:
        command.append("--overwrite")
    if dry_run:
        command.append("--dry-run")
    return command


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
    allow_empty_output: bool = False,
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
    if allow_empty_output:
        command.append("--allow-empty-output")
    return command


def model_download_command(
    models: list[str],
    provider: str,
    download_path: str,
    asr_dir: str,
    aligner_dir: str,
    mossformer_path: str,
    force: bool,
) -> list[str]:
    selected = models or ["all"]
    command = [
        sys.executable,
        script_path("utils", "download_models.py"),
        "--models",
        *selected,
        "--provider",
        provider,
        "--download-path",
        str(resolve_path(download_path)),
        "--asr-dir",
        asr_dir,
        "--aligner-dir",
        aligner_dir,
        "--mossformer-path",
        str(resolve_path(mossformer_path)),
    ]
    if force:
        command.append("--force")
    return command
