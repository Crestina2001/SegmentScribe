"""Gradio layout builders for the SegmentScribe WebUI."""

from __future__ import annotations

import gradio as gr

from .actions import (
    preview_segment,
    refresh_results,
    run_convert,
    run_denoise,
    run_full_pipeline,
    run_long_split,
    run_model_download,
    run_music,
    run_slice,
    run_speaker_filter,
    run_volume_normalize,
)
from .commands import LLM_PROVIDERS, RULE_SLICE_MODE, SLICE_MODES
from .dialogs import browse_folder, browse_jsonl_file, browse_jsonl_save
from .paths import DEFAULT_WORK_DIR, PROJECT_ROOT


def build_tabs() -> None:
    gr.Markdown("# SegmentScribe WebUI")
    gr.Markdown("Convert audio, remove music, denoise speech, pre-split long files, slice segments, and review results.")
    _build_full_pipeline_tab()
    _build_single_stages_tab()
    _build_results_tab()
    _build_speaker_filter_tab()
    _build_volume_tab()
    _build_models_tab()


def _shared_slice_controls(prefix: str, *, include_device: bool = True, include_run_flags: bool = True):
    mode = gr.Dropdown(list(SLICE_MODES), value=RULE_SLICE_MODE, label="Slicing mode")
    model = gr.Textbox(label="ASR model path", value=str(PROJECT_ROOT / "checkpoints" / "Qwen3-ASR-1.7B"))
    aligner = gr.Textbox(label="Forced aligner path", value=str(PROJECT_ROOT / "checkpoints" / "Qwen3-ForcedAligner-0.6B"))
    with gr.Row():
        browse_model = gr.Button("Browse ASR model...")
        browse_aligner = gr.Button("Browse aligner...")
    with gr.Row():
        backend = gr.Dropdown(["transformers", "vllm"], value="transformers", label="ASR backend")
        device = gr.Textbox(label="Device", value="cuda:0") if include_device else None
        dtype = gr.Dropdown(["bfloat16", "float16", "float32"], value="bfloat16", label="dtype")
        batch = gr.Number(label="ASR batch size", value=1, precision=0)
    with gr.Row():
        min_sec = gr.Number(label="Min segment seconds", value=3.0)
        max_sec = gr.Number(label="Max segment seconds", value=10.0)
        vad = gr.Dropdown(["auto", "silero", "librosa"], value="auto", label="VAD backend")
    with gr.Row():
        language = gr.Textbox(label="Language hint", value="")
        punctuation = gr.Checkbox(label="Rule punctuation correction", value=False)
        overwrite = gr.Checkbox(label="Overwrite", value=True) if include_run_flags else None
        dry = gr.Checkbox(label="Dry run", value=False) if include_run_flags else None
    with gr.Accordion("LLM slicing settings", open=False):
        llm_model = gr.Textbox(label="LLM model", value="")
        with gr.Row():
            punct_llm_model = gr.Textbox(label="Punctuation LLM override", value="")
            rough_llm_model = gr.Textbox(label="Rough-cut LLM override", value="")
        with gr.Row():
            llm_provider = gr.Dropdown(list(LLM_PROVIDERS), value="", label="LLM provider")
            env_path = gr.Textbox(label="Env path", value=".env")
        with gr.Row():
            llm_concurrency = gr.Number(label="LLM concurrency", value=8, precision=0)
            llm_max_rounds = gr.Number(label="LLM max rounds", value=5, precision=0)
    browse_model.click(browse_folder, inputs=model, outputs=model)
    browse_aligner.click(browse_folder, inputs=aligner, outputs=aligner)
    controls = {
        f"{prefix}_mode": mode,
        f"{prefix}_model": model,
        f"{prefix}_aligner": aligner,
        f"{prefix}_backend": backend,
        f"{prefix}_dtype": dtype,
        f"{prefix}_batch": batch,
        f"{prefix}_min": min_sec,
        f"{prefix}_max": max_sec,
        f"{prefix}_vad": vad,
        f"{prefix}_language": language,
        f"{prefix}_punctuation": punctuation,
        f"{prefix}_llm_model": llm_model,
        f"{prefix}_punct_llm_model": punct_llm_model,
        f"{prefix}_rough_llm_model": rough_llm_model,
        f"{prefix}_llm_provider": llm_provider,
        f"{prefix}_env_path": env_path,
        f"{prefix}_llm_concurrency": llm_concurrency,
        f"{prefix}_llm_max_rounds": llm_max_rounds,
    }
    if device is not None:
        controls[f"{prefix}_device"] = device
    if overwrite is not None:
        controls[f"{prefix}_overwrite"] = overwrite
    if dry is not None:
        controls[f"{prefix}_dry"] = dry
    return controls


def _build_full_pipeline_tab() -> None:
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
            do_presplit = gr.Checkbox(label="4. Pre-split long audio", value=False)
            do_slice = gr.Checkbox(label="5. Slice audios", value=True)
            do_speaker_filter = gr.Checkbox(label="7. Filter speaker outliers", value=False)
            do_volume_normalize = gr.Checkbox(label="8. Normalize volume", value=False)

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
            zip_model = gr.Textbox(label="ZipEnhancer model", value=str(PROJECT_ROOT / "checkpoints" / "ZipEnhancer"))
            with gr.Row():
                zip_normalize = gr.Dropdown(["match_original", "none", "library_median"], value="match_original", label="Zip normalize")
                zip_alignment_metric = gr.Dropdown(["rms", "peak"], value="rms", label="Zip alignment metric")

        with gr.Accordion("Pre-split long audio settings", open=False):
            with gr.Row():
                presplit_target = gr.Number(label="Target piece seconds", value=300.0)
                presplit_window = gr.Number(label="Search window seconds", value=60.0)
                presplit_frame = gr.Number(label="Frame ms", value=20.0)
            with gr.Row():
                presplit_hop = gr.Number(label="Hop ms", value=5.0)
                presplit_silence = gr.Number(label="Min silence ms", value=30.0)

        with gr.Accordion("Slicing settings", open=False):
            slice_controls = _shared_slice_controls("full", include_device=False, include_run_flags=False)

        with gr.Accordion("Speaker filter settings", open=False):
            with gr.Row():
                speaker_device = gr.Textbox(label="Speaker filter device", value="cuda:0")
                speaker_mad = gr.Number(label="MAD multiplier", value=3.0)
                speaker_max_prune = gr.Number(label="Max prune ratio", value=0.10)
                speaker_min_rows = gr.Number(label="Min rows", value=8, precision=0)

        with gr.Accordion("Volume normalization settings", open=False):
            with gr.Row():
                volume_target = gr.Number(label="Target LUFS", value=-20.0)
                volume_max_gain = gr.Number(label="Max upward gain dB", value=12.0)
                volume_dynamic = gr.Number(label="Max dynamic range dB", value=24.0)
            with gr.Row():
                volume_peak_margin = gr.Number(label="Peak margin dB", value=1.0)
                volume_active_ratio = gr.Number(label="Min active ratio", value=0.03)
                volume_sample_rate = gr.Number(label="Output sample rate (0 keeps source)", value=0, precision=0)

        browse_full_source.click(browse_folder, inputs=source_path, outputs=source_path)
        browse_full_workspace.click(browse_folder, inputs=workspace, outputs=workspace)
        browse_full_mossformer.click(browse_folder, inputs=mossformer_model_path, outputs=mossformer_model_path)

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
                do_presplit,
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
                presplit_target,
                presplit_window,
                presplit_frame,
                presplit_hop,
                presplit_silence,
                slice_controls["full_mode"],
                slice_controls["full_model"],
                slice_controls["full_aligner"],
                slice_controls["full_backend"],
                slice_controls["full_dtype"],
                slice_controls["full_batch"],
                slice_controls["full_min"],
                slice_controls["full_max"],
                slice_controls["full_language"],
                slice_controls["full_vad"],
                slice_controls["full_punctuation"],
                slice_controls["full_llm_model"],
                slice_controls["full_punct_llm_model"],
                slice_controls["full_rough_llm_model"],
                slice_controls["full_llm_provider"],
                slice_controls["full_env_path"],
                slice_controls["full_llm_concurrency"],
                slice_controls["full_llm_max_rounds"],
                do_speaker_filter,
                speaker_device,
                speaker_mad,
                speaker_max_prune,
                speaker_min_rows,
                do_volume_normalize,
                volume_target,
                volume_max_gain,
                volume_dynamic,
                volume_peak_margin,
                volume_active_ratio,
                volume_sample_rate,
            ],
            outputs=pipeline_log,
        )


def _build_single_stages_tab() -> None:
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
            c_run.click(run_convert, inputs=[c_input, c_output, c_recursive, c_overwrite, c_sample_rate, c_start, c_digits], outputs=c_log)

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
            m_run.click(run_music, inputs=[m_input, m_output, m_device, m_model, m_shifts, m_jobs, m_segment, m_recursive, m_overwrite], outputs=m_log)

        with gr.Accordion("3. Eliminate noise", open=False):
            d_method = gr.Dropdown(["MossFormer2", "ZipEnhancer"], value="MossFormer2", label="Method")
            d_input = gr.Textbox(label="Input folder")
            d_output = gr.Textbox(label="Output folder", value=str(DEFAULT_WORK_DIR / "03_denoised"))
            with gr.Row():
                d_browse_input = gr.Button("Browse input...")
                d_browse_output = gr.Button("Browse output...")
                d_browse_checkpoint = gr.Button("Browse checkpoint...")
            d_device = gr.Textbox(label="Device", value="cuda:0")
            d_mossformer_path = gr.Textbox(label="MossFormer checkpoint path", value=str(PROJECT_ROOT / "checkpoints" / "MossFormer2_SE_48K"))
            d_cpu = gr.Checkbox(label="Force MossFormer CPU", value=False)
            with gr.Row():
                d_workers = gr.Number(label="Preprocess workers", value=1, precision=0)
                d_batch = gr.Number(label="Inference batch size", value=1, precision=0)
            d_zip_model = gr.Textbox(label="ZipEnhancer model", value=str(PROJECT_ROOT / "checkpoints" / "ZipEnhancer"))
            with gr.Row():
                d_zip_normalize = gr.Dropdown(["match_original", "none", "library_median"], value="match_original", label="Normalize")
                d_zip_metric = gr.Dropdown(["rms", "peak"], value="rms", label="Alignment metric")
                d_overwrite = gr.Checkbox(label="Overwrite", value=True)
            d_run = gr.Button("Run denoise")
            d_log = gr.Textbox(label="Denoise log", lines=10)
            d_run.click(
                run_denoise,
                inputs=[d_method, d_input, d_output, d_device, d_mossformer_path, d_cpu, d_workers, d_batch, d_zip_model, d_zip_normalize, d_zip_metric, d_overwrite],
                outputs=d_log,
            )

        with gr.Accordion("4. Pre-split long audio", open=False):
            p_input = gr.Textbox(label="Input file or folder")
            p_output = gr.Textbox(label="Output folder", value=str(DEFAULT_WORK_DIR / "03_presliced"))
            with gr.Row():
                p_browse_input = gr.Button("Browse input...")
                p_browse_output = gr.Button("Browse output...")
            with gr.Row():
                p_recursive = gr.Checkbox(label="Recursive", value=True)
                p_overwrite = gr.Checkbox(label="Overwrite", value=True)
                p_target = gr.Number(label="Target piece seconds", value=300.0)
                p_window = gr.Number(label="Search window seconds", value=60.0)
            with gr.Row():
                p_frame = gr.Number(label="Frame ms", value=20.0)
                p_hop = gr.Number(label="Hop ms", value=5.0)
                p_silence = gr.Number(label="Min silence ms", value=30.0)
            p_run = gr.Button("Run pre-split")
            p_log = gr.Textbox(label="Pre-split log", lines=10)
            p_run.click(run_long_split, inputs=[p_input, p_output, p_recursive, p_overwrite, p_target, p_window, p_frame, p_hop, p_silence], outputs=p_log)

        with gr.Accordion("5. Slice audios", open=False):
            s_input = gr.Textbox(label="Input file or folder")
            s_output = gr.Textbox(label="Output sliced folder", value=str(DEFAULT_WORK_DIR / "04_sliced"))
            with gr.Row():
                s_browse_input = gr.Button("Browse input...")
                s_browse_output = gr.Button("Browse output...")
            slice_controls = _shared_slice_controls("single")
            s_run = gr.Button("Run slicing")
            s_log = gr.Textbox(label="Slicing log", lines=10)
            s_run.click(
                run_slice,
                inputs=[
                    slice_controls["single_mode"],
                    s_input,
                    s_output,
                    slice_controls["single_model"],
                    slice_controls["single_aligner"],
                    slice_controls["single_backend"],
                    slice_controls["single_device"],
                    slice_controls["single_dtype"],
                    slice_controls["single_batch"],
                    slice_controls["single_min"],
                    slice_controls["single_max"],
                    slice_controls["single_language"],
                    slice_controls["single_vad"],
                    slice_controls["single_punctuation"],
                    slice_controls["single_overwrite"],
                    slice_controls["single_dry"],
                    slice_controls["single_llm_model"],
                    slice_controls["single_punct_llm_model"],
                    slice_controls["single_rough_llm_model"],
                    slice_controls["single_llm_provider"],
                    slice_controls["single_env_path"],
                    slice_controls["single_llm_concurrency"],
                    slice_controls["single_llm_max_rounds"],
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
        p_browse_input.click(browse_folder, inputs=p_input, outputs=p_input)
        p_browse_output.click(browse_folder, inputs=p_output, outputs=p_output)
        s_browse_input.click(browse_folder, inputs=s_input, outputs=s_input)
        s_browse_output.click(browse_folder, inputs=s_output, outputs=s_output)


def _build_results_tab() -> None:
    with gr.Tab("6. Check result"):
        result_dir = gr.Textbox(label="Sliced output folder", value=str(DEFAULT_WORK_DIR / "04_sliced"))
        result_browse_folder = gr.Button("Browse result folder...")
        refresh = gr.Button("Refresh results", variant="primary")
        summary = gr.Textbox(label="summary.json", lines=12)
        manifest = gr.Dataframe(headers=["audio_relpath", "duration_sec", "sample_rate", "transcript"], label="manifest.tsv", interactive=False, wrap=True)
        segment = gr.Dropdown(label="Segment")
        with gr.Row():
            audio = gr.Audio(label="Audio", type="filepath")
            transcript = gr.Textbox(label="Transcript", lines=4)
        refresh.click(refresh_results, inputs=result_dir, outputs=[summary, manifest, segment])
        result_browse_folder.click(browse_folder, inputs=result_dir, outputs=result_dir)
        segment.change(preview_segment, inputs=[result_dir, segment], outputs=[audio, transcript])


def _build_speaker_filter_tab() -> None:
    with gr.Tab("7. Filter speaker outliers"):
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
            inputs=[speaker_input, speaker_output, speaker_root, speaker_device, speaker_mad, speaker_max_prune, speaker_min_rows, speaker_dry_run, speaker_overwrite],
            outputs=speaker_log,
        )


def _build_volume_tab() -> None:
    with gr.Tab("8. Normalize volume"):
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
            inputs=[volume_input, volume_output, volume_jsonl, volume_target, volume_max_gain, volume_dynamic, volume_peak_margin, volume_active_ratio, volume_sample_rate, volume_dry_run, volume_overwrite],
            outputs=volume_log,
        )


def _build_models_tab() -> None:
    with gr.Tab("Models"):
        gr.Markdown("Download checkpoints into the default folders used by the WebUI.")
        download_models = gr.CheckboxGroup(["all", "qwen3", "asr", "aligner", "mossformer", "zipenhancer", "speaker"], value=["all"], label="Models")
        download_provider = gr.Dropdown(["modelscope", "huggingface"], value="modelscope", label="Download source")
        download_path = gr.Textbox(label="Checkpoint root path", value=str(PROJECT_ROOT / "checkpoints"))
        download_asr_dir = gr.Textbox(label="ASR directory name", value="Qwen3-ASR-1.7B")
        download_aligner_dir = gr.Textbox(label="Aligner directory name", value="Qwen3-ForcedAligner-0.6B")
        download_mossformer_path = gr.Textbox(label="MossFormer checkpoint path", value=str(PROJECT_ROOT / "checkpoints" / "MossFormer2_SE_48K"))
        with gr.Row():
            browse_download_root = gr.Button("Browse checkpoint root...")
            browse_download_mossformer = gr.Button("Browse MossFormer path...")
        download_force = gr.Checkbox(label="Force re-download", value=False)
        download_button = gr.Button("Download models", variant="primary")
        download_log = gr.Textbox(label="Download log", lines=18)
        browse_download_root.click(browse_folder, inputs=download_path, outputs=download_path)
        browse_download_mossformer.click(browse_folder, inputs=download_mossformer_path, outputs=download_mossformer_path)
        download_button.click(
            run_model_download,
            inputs=[download_models, download_provider, download_path, download_asr_dir, download_aligner_dir, download_mossformer_path, download_force],
            outputs=download_log,
        )
