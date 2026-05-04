"""Orchestrates the rule-based slide slicing pipeline."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from slicing_utils.asr import AsrBackend
from slicing_utils.filter_write import (
    Phase5Result,
    SegmentRecord,
    prepare_source_dirs,
    run_filter_write_phase,
    sort_records,
    voxcpm_jsonl_name,
    write_manifest,
    write_summary,
    write_text_atomic,
    write_voxcpm_jsonl,
)
from slicing_utils.prepass import (
    AsyncPrepassScheduler,
    FullPrepass,
    RmsSilenceConfig,
    VadConfig,
    assemble_full_prepass,
    prepare_full_prepass_plan,
)
from slicing_utils.rough_cut import Phase3Result, run_rough_cut_phase

from .config import AUDIO_EXTENSIONS, RuleWorkflowConfig
from .punctuation import (
    RulePunctuationResult,
    run_identity_punctuation_phase,
    run_rule_punctuation_phase,
)
from .thin_cut import run_thin_cut_phase


logger = logging.getLogger(__name__)


@dataclass
class SourceResult:
    source_path: str
    status: str
    kept_records: list[SegmentRecord] = field(default_factory=list)
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    phase_counts: dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None


def iter_audio_files(input_path: Path, allowed_extensions: Sequence[str]) -> list[Path]:
    exts = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in allowed_extensions
    }
    if input_path.is_file():
        return [input_path]
    return [
        candidate
        for candidate in sorted(input_path.rglob("*"))
        if candidate.is_file() and candidate.suffix.lower() in exts
    ]


def load_audio_mono(path: Path, target_sample_rate: int) -> tuple[np.ndarray, int, float]:
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    duration_sec = float(len(audio) / sample_rate) if sample_rate > 0 else 0.0
    if sample_rate != target_sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate
    return audio, sample_rate, duration_sec


class SlideRulePipeline:
    def __init__(self, config: RuleWorkflowConfig) -> None:
        self.config = config

        self.asr_backend = AsrBackend(
            model_path=config.model_path,
            aligner_path=config.aligner_path,
            backend=config.asr_backend,
            device=config.device,
            dtype=config.dtype,
            max_inference_batch_size=config.asr_max_batch_size,
            max_aligner_batch_size=config.aligner_max_batch_size,
            language=config.language,
            backend_kwargs=config.asr_backend_kwargs,
            forced_aligner_kwargs=config.forced_aligner_kwargs,
        )
        self.prepass_scheduler = AsyncPrepassScheduler(
            self.asr_backend,
            asr_max_batch_size=config.asr_max_batch_size,
            aligner_num_workers=config.aligner_num_workers,
            aligner_max_batch_size=config.aligner_max_batch_size,
        )

    async def run(self) -> dict[str, Any]:
        cfg = self.config
        workflow_started = time.perf_counter()
        logger.info(
            "Starting slide_rule: input=%s output_dir=%s backend=%s device=%s dtype=%s "
            "asr_batch=%s aligner_workers=%s aligner_batch=%s source_concurrency=%s "
            "chunk_mode=%s segment_bounds=%.2f-%.2fs dry_run=%s overwrite=%s",
            cfg.input_path,
            cfg.output_dir,
            cfg.asr_backend,
            cfg.device,
            cfg.dtype,
            cfg.asr_max_batch_size,
            cfg.aligner_num_workers,
            cfg.aligner_max_batch_size,
            cfg.source_concurrency,
            cfg.preprocess_chunk_mode,
            cfg.min_seg_sec,
            cfg.max_seg_sec,
            cfg.dry_run,
            cfg.overwrite,
        )
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        source_files = iter_audio_files(cfg.input_path, cfg.allowed_extensions)
        if not source_files:
            raise SystemExit(f"No audio files found under {cfg.input_path}.")

        try:
            _raise_for_duplicate_stems(source_files)
            logger.info("Discovered %d source audio file(s)", len(source_files))
            self.prepass_scheduler.start()
            source_gate = asyncio.Semaphore(max(1, cfg.source_concurrency))

            async def run_indexed(path: Path, *, index: int) -> tuple[int, SourceResult]:
                async with source_gate:
                    return index, await self._process_source(path, index=index, total=len(source_files))

            tasks = [
                asyncio.create_task(run_indexed(path, index=index))
                for index, path in enumerate(source_files, start=1)
            ]
            indexed_results = await asyncio.gather(*tasks)
            results = [result for _, result in sorted(indexed_results, key=lambda item: item[0])]
            kept_records = sort_records(record for result in results for record in result.kept_records)
            write_manifest(cfg.output_dir, kept_records)
            jsonl_path = write_voxcpm_jsonl(
                cfg.output_dir,
                voxcpm_jsonl_name(cfg.input_path),
                kept_records,
            )
            summary = self._build_summary(source_files, kept_records, results)
            summary["voxcpm_jsonl"] = str(jsonl_path)
            write_summary(cfg.output_dir, summary)
            logger.info(
                "Finished slide_rule in %.2fs: sources=%d kept_segments=%d kept_audio_hours=%.4f "
                "output_dir=%s jsonl=%s",
                time.perf_counter() - workflow_started,
                len(source_files),
                len(kept_records),
                summary["kept_audio_hours"],
                cfg.output_dir,
                jsonl_path,
            )
            return summary
        finally:
            await self.prepass_scheduler.close()

    async def _process_source(self, source_path: Path, *, index: int, total: int) -> SourceResult:
        cfg = self.config
        result = SourceResult(source_path=str(source_path), status="ok")
        source_started = time.perf_counter()
        logger.info("Source %d/%d started: %s", index, total, source_path)

        try:
            phase_started = time.perf_counter()
            audio, sr, total_duration = await asyncio.to_thread(
                load_audio_mono,
                source_path,
                cfg.target_sample_rate,
            )
        except Exception as exc:
            result.status = "read_error"
            result.error = str(exc)
            logger.error("Source %d/%d read_error: %s", index, total, exc)
            return self._finish_source(result, index=index, total=total, started_at=source_started)
        loaded_duration = float(len(audio) / sr) if sr > 0 else 0.0
        logger.info(
            "Source %d/%d audio loaded in %.2fs: sr=%d original_duration=%.2fs "
            "loaded_duration=%.2fs samples=%d",
            index,
            total,
            time.perf_counter() - phase_started,
            sr,
            total_duration,
            loaded_duration,
            len(audio),
        )

        if total_duration > cfg.max_source_seconds:
            result.status = "source_too_long"
            result.error = f"duration {total_duration:.1f}s > max {cfg.max_source_seconds}s"
            logger.error("Source %d/%d source_too_long: %s", index, total, result.error)
            return self._finish_source(result, index=index, total=total, started_at=source_started)

        src_stem = source_path.stem
        src_dirs = prepare_source_dirs(cfg.output_dir, src_stem)
        vad_cfg = VadConfig(
            backend=cfg.vad_backend,
            threshold=cfg.vad_threshold,
            min_speech_ms=cfg.vad_min_speech_ms,
            min_silence_ms=cfg.vad_min_silence_ms,
            speech_pad_ms=cfg.vad_speech_pad_ms,
        )
        rms_silence_cfg = RmsSilenceConfig(
            min_chunk_sec=cfg.preprocess_min_chunk_sec,
            max_chunk_sec=cfg.preprocess_max_chunk_sec,
            frame_ms=cfg.rms_silence_frame_ms,
            hop_ms=cfg.rms_silence_hop_ms,
            silence_percentile=cfg.rms_silence_percentile,
            threshold_multiplier=cfg.rms_silence_threshold_multiplier,
            min_silence_ms=cfg.rms_min_silence_ms,
        )

        try:
            phase_started = time.perf_counter()
            plan = await asyncio.to_thread(
                prepare_full_prepass_plan,
                audio,
                sr,
                chunk_sec=cfg.preprocess_chunk_sec,
                chunk_mode=cfg.preprocess_chunk_mode,
                rms_silence_cfg=rms_silence_cfg,
                vad_cfg=vad_cfg,
            )
            transcribed = await self.prepass_scheduler.run_source_prepass(
                audio=audio,
                sample_rate=sr,
                chunks=plan.chunks,
                source_label=str(source_path),
            )
            prepass: FullPrepass = await asyncio.to_thread(
                assemble_full_prepass,
                audio,
                sr,
                transcribed,
            )
        except Exception as exc:
            result.status = "prepass_error"
            result.error = f"{exc}\n{traceback.format_exc()}"
            logger.error("Source %d/%d prepass_error: %s", index, total, exc)
            return self._finish_source(result, index=index, total=total, started_at=source_started)

        _write_json(src_dirs.prepass_dir / "stream.json", prepass.to_stream_dump())
        logger.info(
            "Source %d/%d prepass completed in %.2fs: chunks=%d chars=%d warnings=%d "
            "speech=%.2fs",
            index,
            total,
            time.perf_counter() - phase_started,
            len(prepass.chunk_spans),
            len(prepass.global_chars),
            len(prepass.warnings),
            prepass.total_speech_sec,
        )
        if not prepass.global_chars:
            result.status = "asr_empty"
            result.error = "prepass produced no aligned characters"
            logger.error("Source %d/%d asr_empty: %s", index, total, result.error)
            return self._finish_source(result, index=index, total=total, started_at=source_started)

        phase_started = time.perf_counter()
        phase2 = (
            run_rule_punctuation_phase(prepass)
            if cfg.enable_punctuation_correction
            else run_identity_punctuation_phase(prepass)
        )
        _write_json(
            src_dirs.traces_dir / "punctuation.json",
            {
                "mode": "rule" if cfg.enable_punctuation_correction else "disabled",
                "enabled": cfg.enable_punctuation_correction,
                "edit_count": len(phase2.applied_windows),
                "corrected_full_text": phase2.corrected_full_text,
                "edits": [dataclasses.asdict(edit) for edit in phase2.applied_windows],
            },
        )
        _write_json(src_dirs.intermediate_dir / "phase2_punctuation.json", dataclasses.asdict(phase2))
        logger.info(
            "Source %d/%d punctuation completed in %.2fs: mode=%s edits=%d",
            index,
            total,
            time.perf_counter() - phase_started,
            "rule" if cfg.enable_punctuation_correction else "disabled",
            len(phase2.applied_windows),
        )

        try:
            phase_started = time.perf_counter()
            phase3: Phase3Result = await run_rough_cut_phase(
                audio=audio,
                sample_rate=sr,
                prepass=prepass,
                corrected_full_text=phase2.corrected_full_text,
                corrected_token_to_positions=phase2.corrected_token_to_positions,
                min_seg_sec=cfg.min_seg_sec,
                max_seg_sec=cfg.max_seg_sec,
                strategy=cfg.rough_cut_strategy,
            )
        except Exception as exc:
            result.status = "rough_cut_error"
            result.error = f"{exc}\n{traceback.format_exc()}"
            logger.error("Source %d/%d rough_cut_error: %s", index, total, exc)
            return self._finish_source(result, index=index, total=total, started_at=source_started)

        planner_error = next((trace.error for trace in phase3.block_traces if trace.error), None)
        _write_json(
            src_dirs.traces_dir / "rough_cut.json",
            {
                "segment_count": len(phase3.segments),
                "segments": [dataclasses.asdict(segment) for segment in phase3.segments],
                "blocks": [dataclasses.asdict(trace) for trace in phase3.block_traces],
            },
        )
        _write_json(src_dirs.intermediate_dir / "phase3_rough_cut.json", dataclasses.asdict(phase3))
        logger.info(
            "Source %d/%d rough cut completed in %.2fs: segments=%d blocks=%d planner_error=%s",
            index,
            total,
            time.perf_counter() - phase_started,
            len(phase3.segments),
            len(phase3.block_traces),
            planner_error or "none",
        )

        phase_started = time.perf_counter()
        phase4 = run_thin_cut_phase(audio=audio, sample_rate=sr, segments=phase3.segments)
        _write_json(
            src_dirs.traces_dir / "thin_cut.json",
            {
                "mode": "rule",
                "segment_count": len(phase4.refined),
                "refined": [
                    {
                        "rough": dataclasses.asdict(refined.rough),
                        "cut_start_sec": refined.cut_start_sec,
                        "cut_end_sec": refined.cut_end_sec,
                        "duration_sec": refined.duration_sec,
                        "zoom_reason": refined.zoom_reason,
                        "zoom_error": refined.zoom_error,
                        "zoom_id": refined.zoom_id,
                    }
                    for refined in phase4.refined
                ],
                "traces": [dataclasses.asdict(trace) for trace in phase4.traces],
            },
        )
        _write_json(src_dirs.intermediate_dir / "phase4_zoom_cut.json", dataclasses.asdict(phase4))
        trim_fallbacks = sum(1 for refined in phase4.refined if refined.zoom_error)
        logger.info(
            "Source %d/%d thin cut completed in %.2fs: refined=%d trim_fallbacks=%d",
            index,
            total,
            time.perf_counter() - phase_started,
            len(phase4.refined),
            trim_fallbacks,
        )

        try:
            phase_started = time.perf_counter()
            phase5: Phase5Result = await asyncio.to_thread(
                run_filter_write_phase,
                audio=audio,
                sample_rate=sr,
                source_path=source_path,
                src_stem=src_stem,
                src_dirs=src_dirs,
                corrected_full_text=phase2.corrected_full_text,
                corrected_token_to_positions=phase2.corrected_token_to_positions,
                global_chars=prepass.global_chars,
                refined_segments=phase4.refined,
                min_seg_sec=cfg.min_seg_sec,
                max_seg_sec=cfg.max_seg_sec,
                overwrite=cfg.overwrite,
                dry_run=cfg.dry_run,
                output_dir=cfg.output_dir,
            )
        except Exception as exc:
            result.status = "filter_error"
            result.error = f"{exc}\n{traceback.format_exc()}"
            logger.error("Source %d/%d filter_error: %s", index, total, exc)
            return self._finish_source(result, index=index, total=total, started_at=source_started)
        status_counts: dict[str, int] = {}
        for outcome in phase5.outcomes:
            status_counts[outcome.status] = status_counts.get(outcome.status, 0) + 1
        logger.info(
            "Source %d/%d filter/write completed in %.2fs: kept=%d dropped=%d statuses=%s",
            index,
            total,
            time.perf_counter() - phase_started,
            len(phase5.kept_records),
            len(phase5.outcomes) - len(phase5.kept_records),
            _format_counts(status_counts),
        )

        for outcome in phase5.outcomes:
            if outcome.status == "kept" and outcome.record is not None:
                _write_json(
                    src_dirs.traces_dir / f"{outcome.record.stem}.json",
                    {
                        "segment_index": outcome.segment_index,
                        "source_path": str(source_path),
                        "source_start_sec": outcome.record.source_start_sec,
                        "source_end_sec": outcome.record.source_end_sec,
                        "duration_sec": outcome.record.duration_sec,
                        "rough": dataclasses.asdict(outcome.rough),
                        "refined": {
                            "cut_start_sec": outcome.refined.cut_start_sec,
                            "cut_end_sec": outcome.refined.cut_end_sec,
                            "duration_sec": outcome.refined.duration_sec,
                            "zoom_reason": outcome.refined.zoom_reason,
                            "zoom_error": outcome.refined.zoom_error,
                            "zoom_id": outcome.refined.zoom_id,
                        },
                        "transcript": outcome.transcript,
                        "audio_relpath": outcome.record.audio_relpath,
                        "transcript_relpath": outcome.record.transcript_relpath,
                    },
                )

        for outcome in phase5.outcomes:
            result.outcomes.append(
                {
                    "segment_index": outcome.segment_index,
                    "status": outcome.status,
                    "reason": outcome.reason,
                    "source_start_sec": outcome.rough.start_sec,
                    "segment_end_sec": outcome.refined.cut_start_sec,
                    "duration_sec": max(0.0, outcome.refined.cut_start_sec - outcome.rough.start_sec),
                    "audio_relpath": outcome.record.audio_relpath if outcome.record else None,
                    "transcript_relpath": outcome.record.transcript_relpath if outcome.record else None,
                }
            )
        result.kept_records.extend(phase5.kept_records)

        result.phase_counts = {
            "warnings": len(prepass.warnings),
            "rule_punctuation_edits": len(phase2.applied_windows),
            "rough_blocks": len(phase3.block_traces),
            "rough_segments": len(phase3.segments),
            "refined_segments": len(phase4.refined),
            **{f"status_{key}": value for key, value in status_counts.items()},
        }
        return self._finish_source(result, index=index, total=total, started_at=source_started)

    def _finish_source(
        self,
        result: SourceResult,
        *,
        index: int,
        total: int,
        started_at: float,
    ) -> SourceResult:
        logger.info(
            "Source %d/%d finished: status=%s kept=%d elapsed=%.2fs",
            index,
            total,
            result.status,
            len(result.kept_records),
            time.perf_counter() - started_at,
        )
        return result

    def _build_summary(
        self,
        source_files: Sequence[Path],
        kept_records: Sequence[SegmentRecord],
        results: Sequence[SourceResult],
    ) -> dict[str, Any]:
        cfg = self.config
        total_seconds = sum(record.duration_sec for record in kept_records)
        status_counts: dict[str, int] = {}
        for source_result in results:
            for outcome in source_result.outcomes:
                status = str(outcome.get("status", "unknown"))
                status_counts[status] = status_counts.get(status, 0) + 1
        return {
            "pipeline_mode": "rule",
            "input_path": str(cfg.input_path),
            "output_dir": str(cfg.output_dir),
            "source_audio_count": len(source_files),
            "kept_segment_count": len(kept_records),
            "kept_audio_hours": round(total_seconds / 3600.0, 4),
            "status_counts": status_counts,
            "per_source": [
                {
                    "source_path": source_result.source_path,
                    "status": source_result.status,
                    "error": source_result.error,
                    "phase_counts": source_result.phase_counts,
                    "kept_segment_count": len(source_result.kept_records),
                }
                for source_result in results
            ],
            "config": {
                "pipeline_mode": "rule",
                "enable_punctuation_correction": cfg.enable_punctuation_correction,
                "model_path": cfg.model_path,
                "aligner_path": cfg.aligner_path,
                "asr_backend": cfg.asr_backend,
                "device": cfg.device,
                "dtype": cfg.dtype,
                "source_concurrency": cfg.source_concurrency,
                "asr_max_batch_size": cfg.asr_max_batch_size,
                "aligner_num_workers": cfg.aligner_num_workers,
                "aligner_max_batch_size": cfg.aligner_max_batch_size,
                "min_seg_sec": cfg.min_seg_sec,
                "max_seg_sec": cfg.max_seg_sec,
                "preprocess_chunk_sec": cfg.preprocess_chunk_sec,
                "preprocess_chunk_mode": cfg.preprocess_chunk_mode,
                "preprocess_min_chunk_sec": cfg.preprocess_min_chunk_sec,
                "preprocess_max_chunk_sec": cfg.preprocess_max_chunk_sec,
                "vad_backend": cfg.vad_backend,
                "target_sample_rate": cfg.target_sample_rate,
                "max_source_seconds": cfg.max_source_seconds,
                "overwrite": cfg.overwrite,
                "dry_run": cfg.dry_run,
                "rough_cut_strategy": cfg.rough_cut_strategy,
            },
            "generated_at_unix": int(time.time()),
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _raise_for_duplicate_stems(source_files: Sequence[Path]) -> None:
    paths_by_stem: dict[str, list[Path]] = {}
    for path in source_files:
        paths_by_stem.setdefault(path.stem, []).append(path)
    duplicates = {stem: paths for stem, paths in paths_by_stem.items() if len(paths) > 1}
    if not duplicates:
        return
    details = "; ".join(
        f"{stem}: {', '.join(str(path) for path in paths)}"
        for stem, paths in sorted(duplicates.items())
    )
    raise SystemExit(f"Duplicate source filename stems would collide in output directories: {details}")


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


async def run_rule_workflow(config: RuleWorkflowConfig) -> dict[str, Any]:
    return await SlideRulePipeline(config).run()


def run_rule_workflow_sync(config: RuleWorkflowConfig) -> dict[str, Any]:
    return asyncio.run(run_rule_workflow(config))
