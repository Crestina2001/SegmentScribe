#!/usr/bin/env python
"""SegmentScribe Chinese Gradio WebUI."""

from __future__ import annotations

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "缺少依赖 gradio，请先安装：\n\n"
        "  python -m pip install gradio\n\n"
        "然后重新运行：python webui_cn.py"
    ) from exc

from ui_utils.actions import (
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
from ui_utils.commands import LLM_PROVIDERS, LLM_SLICE_MODE, RULE_SLICE_MODE
from ui_utils.dialogs import browse_folder, browse_jsonl_file, browse_jsonl_save
from ui_utils.paths import DEFAULT_WORK_DIR, PROJECT_ROOT


SLICE_MODE_CHOICES = [("规则切分", RULE_SLICE_MODE), ("LLM 辅助切分", LLM_SLICE_MODE)]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SegmentScribe 中文 WebUI") as app:
        build_tabs()
    return app


def build_tabs() -> None:
    gr.Markdown("# SegmentScribe 中文 WebUI")
    gr.Markdown("转换音频、去除背景音乐、语音降噪、预切长音频、切分片段，并检查处理结果。")
    _build_full_pipeline_tab()
    _build_single_stages_tab()
    _build_results_tab()
    _build_speaker_filter_tab()
    _build_volume_tab()
    _build_models_tab()


def _shared_slice_controls(prefix: str, *, include_device: bool = True, include_run_flags: bool = True):
    mode = gr.Dropdown(SLICE_MODE_CHOICES, value=RULE_SLICE_MODE, label="切分模式")
    model = gr.Textbox(label="ASR 模型路径", value=str(PROJECT_ROOT / "checkpoints" / "Qwen3-ASR-1.7B"))
    aligner = gr.Textbox(label="强制对齐模型路径", value=str(PROJECT_ROOT / "checkpoints" / "Qwen3-ForcedAligner-0.6B"))
    with gr.Row():
        browse_model = gr.Button("选择 ASR 模型...")
        browse_aligner = gr.Button("选择对齐模型...")
    with gr.Row():
        backend = gr.Dropdown(["transformers", "vllm"], value="transformers", label="ASR 后端")
        device = gr.Textbox(label="设备", value="cuda:0") if include_device else None
        dtype = gr.Dropdown(["bfloat16", "float16", "float32"], value="bfloat16", label="数据类型")
        batch = gr.Number(label="ASR 批大小", value=1, precision=0)
    with gr.Row():
        min_sec = gr.Number(label="最短片段秒数", value=3.0)
        max_sec = gr.Number(label="最长片段秒数", value=10.0)
        vad = gr.Dropdown(["auto", "silero", "librosa"], value="auto", label="VAD 后端")
    with gr.Row():
        language = gr.Textbox(label="语言提示", value="")
        punctuation = gr.Checkbox(label="启用规则标点修正", value=False)
        overwrite = gr.Checkbox(label="覆盖输出", value=True) if include_run_flags else None
        dry = gr.Checkbox(label="试运行", value=False) if include_run_flags else None
    with gr.Accordion("LLM 切分设置", open=False):
        llm_model = gr.Textbox(label="LLM 模型", value="")
        with gr.Row():
            punct_llm_model = gr.Textbox(label="标点 LLM 覆盖模型", value="")
            rough_llm_model = gr.Textbox(label="粗切 LLM 覆盖模型", value="")
        with gr.Row():
            llm_provider = gr.Dropdown(list(LLM_PROVIDERS), value="", label="LLM 服务商")
            env_path = gr.Textbox(label="环境变量文件路径", value=".env")
        with gr.Row():
            llm_concurrency = gr.Number(label="LLM 并发数", value=8, precision=0)
            llm_max_rounds = gr.Number(label="LLM 最大轮数", value=5, precision=0)
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
    with gr.Tab("完整流水线"):
        source_path = gr.Textbox(label="源音频文件夹或文件", value=str(PROJECT_ROOT / "audios"))
        workspace = gr.Textbox(label="工作目录", value=str(DEFAULT_WORK_DIR))
        with gr.Row():
            browse_full_source = gr.Button("选择源路径...")
            browse_full_workspace = gr.Button("选择工作目录...")
        with gr.Row():
            overwrite = gr.Checkbox(label="覆盖输出", value=True)
            recursive = gr.Checkbox(label="递归扫描文件夹", value=True)
        with gr.Row():
            do_convert = gr.Checkbox(label="1. 格式转换", value=True)
            do_music = gr.Checkbox(label="2. 去除背景音乐", value=False)
            do_denoise = gr.Checkbox(label="3. 语音降噪", value=True)
            do_presplit = gr.Checkbox(label="4. 预切长音频", value=False)
            do_slice = gr.Checkbox(label="5. 音频切分", value=True)
            do_speaker_filter = gr.Checkbox(label="7. 过滤说话人异常", value=False)
            do_volume_normalize = gr.Checkbox(label="8. 音量归一化", value=False)

        with gr.Accordion("格式转换设置", open=False):
            sample_rate = gr.Number(label="采样率", value=16000, precision=0)

        with gr.Accordion("背景音乐去除设置", open=False):
            demucs_model = gr.Dropdown(["htdemucs", "htdemucs_ft"], value="htdemucs", label="Demucs 模型")
            with gr.Row():
                device = gr.Textbox(label="设备", value="cuda:0")
                demucs_shifts = gr.Number(label="Demucs shifts", value=1, precision=0)
                demucs_jobs = gr.Number(label="Demucs 并行任务数", value=0, precision=0)
                demucs_segment = gr.Number(label="Demucs 分段秒数", value=0, precision=1)

        with gr.Accordion("语音降噪设置", open=False):
            denoise_method = gr.Dropdown(["MossFormer2", "ZipEnhancer"], value="MossFormer2", label="方法")
            mossformer_model_path = gr.Textbox(
                label="MossFormer checkpoint 路径",
                value=str(PROJECT_ROOT / "checkpoints" / "MossFormer2_SE_48K"),
            )
            browse_full_mossformer = gr.Button("选择 MossFormer checkpoint...")
            mossformer_cpu = gr.Checkbox(label="强制 MossFormer 使用 CPU", value=False)
            with gr.Row():
                preprocess_workers = gr.Number(label="预处理线程数", value=1, precision=0)
                inference_batch_size = gr.Number(label="推理批大小", value=1, precision=0)
            zip_model = gr.Textbox(label="ZipEnhancer 模型", value="iic/speech_zipenhancer_ans_multiloss_16k_base")
            with gr.Row():
                zip_normalize = gr.Dropdown(["match_original", "none", "library_median"], value="match_original", label="Zip 音量对齐")
                zip_alignment_metric = gr.Dropdown(["rms", "peak"], value="rms", label="Zip 对齐指标")

        with gr.Accordion("长音频预切设置", open=False):
            with gr.Row():
                presplit_target = gr.Number(label="目标片段秒数", value=300.0)
                presplit_window = gr.Number(label="搜索窗口秒数", value=60.0)
                presplit_frame = gr.Number(label="帧长 ms", value=20.0)
            with gr.Row():
                presplit_hop = gr.Number(label="步长 ms", value=5.0)
                presplit_silence = gr.Number(label="最短静音 ms", value=30.0)

        with gr.Accordion("切分设置", open=False):
            slice_controls = _shared_slice_controls("full", include_device=False, include_run_flags=False)

        with gr.Accordion("说话人过滤设置", open=False):
            with gr.Row():
                speaker_device = gr.Textbox(label="说话人过滤设备", value="cuda:0")
                speaker_mad = gr.Number(label="MAD 倍数", value=3.0)
                speaker_max_prune = gr.Number(label="最大剔除比例", value=0.10)
                speaker_min_rows = gr.Number(label="最少行数", value=8, precision=0)

        with gr.Accordion("音量归一化设置", open=False):
            with gr.Row():
                volume_target = gr.Number(label="目标 LUFS", value=-20.0)
                volume_max_gain = gr.Number(label="最大增益 dB", value=12.0)
                volume_dynamic = gr.Number(label="最大动态范围 dB", value=24.0)
            with gr.Row():
                volume_peak_margin = gr.Number(label="峰值余量 dB", value=1.0)
                volume_active_ratio = gr.Number(label="最小有效音频比例", value=0.03)
                volume_sample_rate = gr.Number(label="输出采样率（0 保持原始）", value=0, precision=0)

        browse_full_source.click(browse_folder, inputs=source_path, outputs=source_path)
        browse_full_workspace.click(browse_folder, inputs=workspace, outputs=workspace)
        browse_full_mossformer.click(browse_folder, inputs=mossformer_model_path, outputs=mossformer_model_path)

        run_pipeline = gr.Button("运行选中的流水线", variant="primary")
        pipeline_log = gr.Textbox(label="日志", lines=24)
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
    with gr.Tab("单步流程"):
        with gr.Accordion("1. 格式转换", open=True):
            c_input = gr.Textbox(label="输入文件夹")
            c_output = gr.Textbox(label="输出 WAV 文件夹", value=str(DEFAULT_WORK_DIR / "01_numbered_wavs"))
            with gr.Row():
                c_browse_input = gr.Button("选择输入...")
                c_browse_output = gr.Button("选择输出...")
            with gr.Row():
                c_recursive = gr.Checkbox(label="递归", value=True)
                c_overwrite = gr.Checkbox(label="覆盖", value=True)
                c_sample_rate = gr.Number(label="采样率", value=16000, precision=0)
                c_start = gr.Number(label="起始编号", value=1, precision=0)
                c_digits = gr.Number(label="编号位数", value=3, precision=0)
            c_run = gr.Button("运行格式转换")
            c_log = gr.Textbox(label="格式转换日志", lines=10)
            c_run.click(run_convert, inputs=[c_input, c_output, c_recursive, c_overwrite, c_sample_rate, c_start, c_digits], outputs=c_log)

        with gr.Accordion("2. 去除背景音乐", open=False):
            m_input = gr.Textbox(label="输入文件或文件夹")
            m_output = gr.Textbox(label="输出人声 WAV/文件夹", value=str(DEFAULT_WORK_DIR / "02_vocals"))
            with gr.Row():
                m_browse_input = gr.Button("选择输入...")
                m_browse_output = gr.Button("选择输出...")
            with gr.Row():
                m_device = gr.Textbox(label="设备", value="cuda:0")
                m_model = gr.Dropdown(["htdemucs", "htdemucs_ft"], value="htdemucs", label="模型")
                m_shifts = gr.Number(label="Shifts", value=1, precision=0)
                m_jobs = gr.Number(label="并行任务数", value=0, precision=0)
                m_segment = gr.Number(label="分段秒数", value=0, precision=1)
            with gr.Row():
                m_recursive = gr.Checkbox(label="递归", value=True)
                m_overwrite = gr.Checkbox(label="覆盖", value=True)
            m_run = gr.Button("运行背景音乐去除")
            m_log = gr.Textbox(label="背景音乐去除日志", lines=10)
            m_run.click(run_music, inputs=[m_input, m_output, m_device, m_model, m_shifts, m_jobs, m_segment, m_recursive, m_overwrite], outputs=m_log)

        with gr.Accordion("3. 语音降噪", open=False):
            d_method = gr.Dropdown(["MossFormer2", "ZipEnhancer"], value="MossFormer2", label="方法")
            d_input = gr.Textbox(label="输入文件夹")
            d_output = gr.Textbox(label="输出文件夹", value=str(DEFAULT_WORK_DIR / "03_denoised"))
            with gr.Row():
                d_browse_input = gr.Button("选择输入...")
                d_browse_output = gr.Button("选择输出...")
                d_browse_checkpoint = gr.Button("选择 checkpoint...")
            d_device = gr.Textbox(label="设备", value="cuda:0")
            d_mossformer_path = gr.Textbox(label="MossFormer checkpoint 路径", value=str(PROJECT_ROOT / "checkpoints" / "MossFormer2_SE_48K"))
            d_cpu = gr.Checkbox(label="强制 MossFormer 使用 CPU", value=False)
            with gr.Row():
                d_workers = gr.Number(label="预处理线程数", value=1, precision=0)
                d_batch = gr.Number(label="推理批大小", value=1, precision=0)
            d_zip_model = gr.Textbox(label="ZipEnhancer 模型", value="iic/speech_zipenhancer_ans_multiloss_16k_base")
            with gr.Row():
                d_zip_normalize = gr.Dropdown(["match_original", "none", "library_median"], value="match_original", label="音量对齐")
                d_zip_metric = gr.Dropdown(["rms", "peak"], value="rms", label="对齐指标")
                d_overwrite = gr.Checkbox(label="覆盖", value=True)
            d_run = gr.Button("运行语音降噪")
            d_log = gr.Textbox(label="语音降噪日志", lines=10)
            d_run.click(
                run_denoise,
                inputs=[d_method, d_input, d_output, d_device, d_mossformer_path, d_cpu, d_workers, d_batch, d_zip_model, d_zip_normalize, d_zip_metric, d_overwrite],
                outputs=d_log,
            )

        with gr.Accordion("4. 预切长音频", open=False):
            p_input = gr.Textbox(label="输入文件或文件夹")
            p_output = gr.Textbox(label="输出文件夹", value=str(DEFAULT_WORK_DIR / "03_presliced"))
            with gr.Row():
                p_browse_input = gr.Button("选择输入...")
                p_browse_output = gr.Button("选择输出...")
            with gr.Row():
                p_recursive = gr.Checkbox(label="递归", value=True)
                p_overwrite = gr.Checkbox(label="覆盖", value=True)
                p_target = gr.Number(label="目标片段秒数", value=300.0)
                p_window = gr.Number(label="搜索窗口秒数", value=60.0)
            with gr.Row():
                p_frame = gr.Number(label="帧长 ms", value=20.0)
                p_hop = gr.Number(label="步长 ms", value=5.0)
                p_silence = gr.Number(label="最短静音 ms", value=30.0)
            p_run = gr.Button("运行预切")
            p_log = gr.Textbox(label="预切日志", lines=10)
            p_run.click(run_long_split, inputs=[p_input, p_output, p_recursive, p_overwrite, p_target, p_window, p_frame, p_hop, p_silence], outputs=p_log)

        with gr.Accordion("5. 音频切分", open=False):
            s_input = gr.Textbox(label="输入文件或文件夹")
            s_output = gr.Textbox(label="输出切分文件夹", value=str(DEFAULT_WORK_DIR / "04_sliced"))
            with gr.Row():
                s_browse_input = gr.Button("选择输入...")
                s_browse_output = gr.Button("选择输出...")
            slice_controls = _shared_slice_controls("single")
            s_run = gr.Button("运行切分")
            s_log = gr.Textbox(label="切分日志", lines=10)
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
    with gr.Tab("6. 检查结果"):
        result_dir = gr.Textbox(label="切分输出文件夹", value=str(DEFAULT_WORK_DIR / "04_sliced"))
        result_browse_folder = gr.Button("选择结果文件夹...")
        refresh = gr.Button("刷新结果", variant="primary")
        summary = gr.Textbox(label="summary.json", lines=12)
        manifest = gr.Dataframe(headers=["audio_relpath", "duration_sec", "sample_rate", "transcript"], label="manifest.tsv", interactive=False, wrap=True)
        segment = gr.Dropdown(label="片段")
        with gr.Row():
            audio = gr.Audio(label="音频", type="filepath")
            transcript = gr.Textbox(label="转写文本", lines=4)
        refresh.click(refresh_results, inputs=result_dir, outputs=[summary, manifest, segment])
        result_browse_folder.click(browse_folder, inputs=result_dir, outputs=result_dir)
        segment.change(preview_segment, inputs=[result_dir, segment], outputs=[audio, transcript])


def _build_speaker_filter_tab() -> None:
    with gr.Tab("7. 过滤说话人异常"):
        speaker_input = gr.Textbox(label="输入 VoxCPM JSONL")
        speaker_output = gr.Textbox(label="过滤后输出 JSONL")
        speaker_root = gr.Textbox(label="数据集根目录（可选）")
        with gr.Row():
            speaker_browse_input = gr.Button("选择输入 JSONL...")
            speaker_browse_output = gr.Button("选择输出 JSONL...")
            speaker_browse_root = gr.Button("选择数据集根目录...")
        with gr.Row():
            speaker_device = gr.Textbox(label="设备", value="cuda:0")
            speaker_mad = gr.Number(label="MAD 倍数", value=3.0)
            speaker_max_prune = gr.Number(label="最大剔除比例", value=0.10)
            speaker_min_rows = gr.Number(label="最少行数", value=8, precision=0)
        with gr.Row():
            speaker_dry_run = gr.Checkbox(label="试运行", value=False)
            speaker_overwrite = gr.Checkbox(label="覆盖", value=True)
        speaker_run = gr.Button("运行说话人过滤", variant="primary")
        speaker_log = gr.Textbox(label="说话人过滤日志", lines=16)
        speaker_browse_input.click(browse_jsonl_file, inputs=speaker_input, outputs=speaker_input)
        speaker_browse_output.click(browse_jsonl_save, inputs=speaker_output, outputs=speaker_output)
        speaker_browse_root.click(browse_folder, inputs=speaker_root, outputs=speaker_root)
        speaker_run.click(
            run_speaker_filter,
            inputs=[speaker_input, speaker_output, speaker_root, speaker_device, speaker_mad, speaker_max_prune, speaker_min_rows, speaker_dry_run, speaker_overwrite],
            outputs=speaker_log,
        )


def _build_volume_tab() -> None:
    with gr.Tab("8. 音量归一化"):
        volume_input = gr.Textbox(label="切分输入文件夹", value=str(DEFAULT_WORK_DIR / "04_sliced"))
        volume_output = gr.Textbox(label="归一化输出文件夹", value=str(DEFAULT_WORK_DIR / "05_normalized"))
        volume_jsonl = gr.Textbox(label="JSONL 文件名/路径（可选）", value="")
        with gr.Row():
            volume_browse_input = gr.Button("选择输入...")
            volume_browse_output = gr.Button("选择输出...")
        with gr.Row():
            volume_target = gr.Number(label="目标 LUFS", value=-20.0)
            volume_max_gain = gr.Number(label="最大增益 dB", value=12.0)
            volume_dynamic = gr.Number(label="最大动态范围 dB", value=24.0)
        with gr.Row():
            volume_peak_margin = gr.Number(label="峰值余量 dB", value=1.0)
            volume_active_ratio = gr.Number(label="最小有效音频比例", value=0.03)
            volume_sample_rate = gr.Number(label="输出采样率（0 保持原始）", value=0, precision=0)
        with gr.Row():
            volume_dry_run = gr.Checkbox(label="试运行", value=False)
            volume_overwrite = gr.Checkbox(label="覆盖", value=True)
        volume_run = gr.Button("运行音量归一化", variant="primary")
        volume_log = gr.Textbox(label="音量归一化日志", lines=16)
        volume_browse_input.click(browse_folder, inputs=volume_input, outputs=volume_input)
        volume_browse_output.click(browse_folder, inputs=volume_output, outputs=volume_output)
        volume_run.click(
            run_volume_normalize,
            inputs=[volume_input, volume_output, volume_jsonl, volume_target, volume_max_gain, volume_dynamic, volume_peak_margin, volume_active_ratio, volume_sample_rate, volume_dry_run, volume_overwrite],
            outputs=volume_log,
        )


def _build_models_tab() -> None:
    with gr.Tab("模型下载"):
        gr.Markdown("下载 WebUI 默认使用的模型 checkpoint。")
        download_models = gr.CheckboxGroup(["all", "qwen3", "asr", "aligner", "mossformer"], value=["all"], label="模型")
        download_provider = gr.Dropdown(["modelscope", "huggingface"], value="modelscope", label="下载来源")
        download_path = gr.Textbox(label="检查点根目录", value=str(PROJECT_ROOT / "checkpoints"))
        download_asr_dir = gr.Textbox(label="ASR 目录名", value="Qwen3-ASR-1.7B")
        download_aligner_dir = gr.Textbox(label="对齐模型目录名", value="Qwen3-ForcedAligner-0.6B")
        download_mossformer_path = gr.Textbox(label="MossFormer checkpoint 路径", value=str(PROJECT_ROOT / "checkpoints" / "MossFormer2_SE_48K"))
        with gr.Row():
            browse_download_root = gr.Button("选择下载根目录...")
            browse_download_mossformer = gr.Button("选择 MossFormer 路径...")
        download_force = gr.Checkbox(label="强制重新下载", value=False)
        download_button = gr.Button("下载模型", variant="primary")
        download_log = gr.Textbox(label="下载日志", lines=18)
        browse_download_root.click(browse_folder, inputs=download_path, outputs=download_path)
        browse_download_mossformer.click(browse_folder, inputs=download_mossformer_path, outputs=download_mossformer_path)
        download_button.click(
            run_model_download,
            inputs=[download_models, download_provider, download_path, download_asr_dir, download_aligner_dir, download_mossformer_path, download_force],
            outputs=download_log,
        )


if __name__ == "__main__":
    build_app().queue().launch()
