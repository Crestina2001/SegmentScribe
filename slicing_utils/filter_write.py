"""Phase 5 - length filter + output writer.

Deterministic step that consumes :class:`RefinedSegment` records, drops any
segment flagged ``drop=true`` in Phase 3 or whose duration falls outside
``[min_seg_sec, max_seg_sec]``, and writes audio/transcript/manifest/jsonl
files in the same layout as ``slide_LLM`` so downstream tooling can reuse
them unchanged.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .asr import CharToken

from .rough_cut import RoughSegment
from .shared import is_punct_or_space, token_to_text_positions

RefinedSegment = Any


logger = logging.getLogger(__name__)


@dataclass
class SegmentRecord:
    stem: str
    source_path: str
    segment_index: int
    audio_relpath: str
    transcript_relpath: str
    trace_relpath: str
    source_start_sec: float
    source_end_sec: float
    duration_sec: float
    sample_rate: int
    transcript: str


@dataclass
class FilterOutcome:
    segment_index: int
    status: str  # "kept" | "dropped_llm" | "dropped_length"
    reason: str
    rough: RoughSegment
    refined: RefinedSegment
    transcript: str
    record: Optional[SegmentRecord] = None
    audio_path: Optional[Path] = None
    transcript_path: Optional[Path] = None


@dataclass
class SourceDirs:
    audios_dir: Path
    transcripts_dir: Path
    traces_dir: Path
    discarded_dir: Path
    prepass_dir: Path
    intermediate_dir: Path


@dataclass
class Phase5Result:
    outcomes: list[FilterOutcome] = field(default_factory=list)
    kept_records: list[SegmentRecord] = field(default_factory=list)


def prepare_source_dirs(output_dir: Path, src_stem: str) -> SourceDirs:
    dirs = SourceDirs(
        audios_dir=output_dir / "audios" / src_stem,
        transcripts_dir=output_dir / "transcripts" / src_stem,
        traces_dir=output_dir / "traces" / src_stem,
        discarded_dir=output_dir / "discarded_audios" / src_stem,
        prepass_dir=output_dir / "prepass" / src_stem,
        intermediate_dir=output_dir / "intermediate" / src_stem,
    )
    for d in (
        dirs.audios_dir,
        dirs.transcripts_dir,
        dirs.traces_dir,
        dirs.discarded_dir,
        dirs.prepass_dir,
        dirs.intermediate_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def transcript_for_segment(
    corrected_full_text: str,
    token_to_positions: dict[int, list[int]],
    global_chars: Sequence[CharToken],
    char_start_idx: int,
    char_end_idx: int,
) -> str:
    """Project the corrected full text onto ``[char_start_idx, char_end_idx]``."""
    if not corrected_full_text or char_end_idx < char_start_idx:
        return ""
    positions: list[int] = []
    for token_idx in range(char_start_idx, char_end_idx + 1):
        positions.extend(token_to_positions.get(token_idx, []))
    if not positions:
        return "".join(str(c.char) for c in global_chars[char_start_idx : char_end_idx + 1]).strip()
    start_pos = min(positions)
    end_pos = max(positions) + 1
    while end_pos < len(corrected_full_text) and is_punct_or_space(
        corrected_full_text[end_pos]
    ):
        end_pos += 1
    return corrected_full_text[start_pos:end_pos].strip()


def run_filter_write_phase(
    *,
    audio: np.ndarray,
    sample_rate: int,
    source_path: Path,
    src_stem: str,
    src_dirs: SourceDirs,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    global_chars: Sequence[CharToken],
    refined_segments: Sequence[RefinedSegment],
    min_seg_sec: float,
    max_seg_sec: float,
    overwrite: bool,
    dry_run: bool,
    output_dir: Path,
) -> Phase5Result:
    result = Phase5Result()

    for seg_idx, refined in enumerate(refined_segments, start=1):
        rough = refined.rough
        transcript_text = transcript_for_segment(
            corrected_full_text,
            corrected_token_to_positions,
            global_chars,
            rough.char_start_idx,
            rough.char_end_idx,
        )

        seg_duration = max(0.0, refined.cut_start_sec - rough.start_sec)

        if rough.drop:
            outcome = FilterOutcome(
                segment_index=seg_idx,
                status="dropped_llm",
                reason=f"rough_cut drop=true ({rough.reason})",
                rough=rough,
                refined=refined,
                transcript=transcript_text,
            )
            _write_discarded(
                src_dirs=src_dirs,
                src_stem=src_stem,
                source_path=source_path,
                seg_idx=seg_idx,
                refined=refined,
                seg_audio=_slice_audio(audio, sample_rate, rough.start_sec, refined.cut_start_sec),
                sample_rate=sample_rate,
                transcript_text=transcript_text,
                outcome_status="dropped_llm",
                dry_run=dry_run,
                overwrite=overwrite,
            )
            result.outcomes.append(outcome)
            continue

        if seg_duration < min_seg_sec - 1e-6 or seg_duration > max_seg_sec + 1e-6:
            reason = (
                f"duration {seg_duration:.3f}s outside [{min_seg_sec}, {max_seg_sec}]"
            )
            outcome = FilterOutcome(
                segment_index=seg_idx,
                status="dropped_length",
                reason=reason,
                rough=rough,
                refined=refined,
                transcript=transcript_text,
            )
            _write_discarded(
                src_dirs=src_dirs,
                src_stem=src_stem,
                source_path=source_path,
                seg_idx=seg_idx,
                refined=refined,
                seg_audio=_slice_audio(audio, sample_rate, rough.start_sec, refined.cut_start_sec),
                sample_rate=sample_rate,
                transcript_text=transcript_text,
                outcome_status="dropped_length",
                dry_run=dry_run,
                overwrite=overwrite,
            )
            result.outcomes.append(outcome)
            continue

        stem = f"{seg_idx:06d}"
        audio_rel = f"audios/{src_stem}/{stem}.wav"
        transcript_rel = f"transcripts/{src_stem}/{stem}.txt"
        trace_rel = f"traces/{src_stem}/{stem}.json"
        audio_path = src_dirs.audios_dir / f"{stem}.wav"
        transcript_path = src_dirs.transcripts_dir / f"{stem}.txt"
        seg_audio = _slice_audio(audio, sample_rate, rough.start_sec, refined.cut_start_sec)

        if not dry_run:
            _write_audio(audio_path, seg_audio, sample_rate, overwrite=overwrite)
            _write_text(transcript_path, transcript_text, overwrite=overwrite)

        record = SegmentRecord(
            stem=stem,
            source_path=str(source_path),
            segment_index=seg_idx,
            audio_relpath=audio_rel,
            transcript_relpath=transcript_rel,
            trace_relpath=trace_rel,
            source_start_sec=rough.start_sec,
            source_end_sec=refined.cut_start_sec,
            duration_sec=seg_duration,
            sample_rate=sample_rate,
            transcript=transcript_text,
        )
        outcome = FilterOutcome(
            segment_index=seg_idx,
            status="kept",
            reason=rough.reason or "natural pause",
            rough=rough,
            refined=refined,
            transcript=transcript_text,
            record=record,
            audio_path=audio_path,
            transcript_path=transcript_path,
        )
        result.outcomes.append(outcome)
        result.kept_records.append(record)

    return result


def _slice_audio(
    audio: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    start_sample = max(0, int(round(start_sec * sample_rate)))
    end_sample = max(start_sample, int(round(end_sec * sample_rate)))
    end_sample = min(end_sample, len(audio))
    return np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int, *, overwrite: bool) -> None:
    import soundfile as sf

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _write_discarded(
    *,
    src_dirs: SourceDirs,
    src_stem: str,
    source_path: Path,
    seg_idx: int,
    refined: RefinedSegment,
    seg_audio: np.ndarray,
    sample_rate: int,
    transcript_text: str,
    outcome_status: str,
    dry_run: bool,
    overwrite: bool,
) -> Optional[str]:
    if dry_run:
        return None
    import soundfile as sf

    stem = f"{seg_idx:06d}"
    try:
        src_dirs.discarded_dir.mkdir(parents=True, exist_ok=True)
        audio_path = src_dirs.discarded_dir / f"{stem}.wav"
        txt_path = src_dirs.discarded_dir / f"{stem}.txt"
        json_path = src_dirs.discarded_dir / f"{stem}.json"
        if seg_audio.size > 0 and (overwrite or not audio_path.exists()):
            sf.write(str(audio_path), seg_audio, sample_rate, subtype="PCM_16")
        if overwrite or not txt_path.exists():
            txt_path.write_text((transcript_text or "") + "\n", encoding="utf-8")
        meta = {
            "status": outcome_status,
            "source_path": str(source_path),
            "segment_index": seg_idx,
            "rough": {
                "char_start_idx": refined.rough.char_start_idx,
                "char_end_idx": refined.rough.char_end_idx,
                "start_sec": refined.rough.start_sec,
                "end_sec": refined.rough.end_sec,
                "drop": refined.rough.drop,
                "reason": refined.rough.reason,
            },
            "refined": {
                "cut_start_sec": refined.cut_start_sec,
                "cut_end_sec": refined.cut_end_sec,
                "duration_sec": refined.duration_sec,
                "zoom_reason": refined.zoom_reason,
                "zoom_error": refined.zoom_error,
                "zoom_id": refined.zoom_id,
            },
            "transcript": transcript_text,
        }
        if overwrite or not json_path.exists():
            write_text_atomic(json_path, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        return f"discarded_audios/{src_stem}/{stem}"
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to write discarded segment %s: %s", stem, exc)
        return None


def write_manifest(output_dir: Path, kept_records: Sequence[SegmentRecord]) -> None:
    manifest_path = output_dir / "manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "audio_relpath",
                "transcript",
                "duration_sec",
                "sample_rate",
                "source_path",
                "source_start_sec",
                "source_end_sec",
                "segment_index",
                "trace_relpath",
            ]
        )
        for record in kept_records:
            writer.writerow(
                [
                    record.audio_relpath,
                    record.transcript,
                    f"{record.duration_sec:.3f}",
                    record.sample_rate,
                    record.source_path,
                    f"{record.source_start_sec:.3f}",
                    f"{record.source_end_sec:.3f}",
                    record.segment_index,
                    record.trace_relpath,
                ]
            )


def voxcpm_jsonl_name(input_path: Path) -> str:
    return f"{input_path.stem}_voxcpm.jsonl"


def write_voxcpm_jsonl(
    output_dir: Path,
    jsonl_name: str,
    kept_records: Sequence[SegmentRecord],
) -> Path:
    jsonl_path = output_dir / jsonl_name
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in kept_records:
            payload = {
                "audio": record.audio_relpath,
                "text": record.transcript,
                "duration": round(record.duration_sec, 3),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return jsonl_path


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    write_text_atomic(output_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def sort_records(records: Iterable[SegmentRecord]) -> list[SegmentRecord]:
    return sorted(records, key=lambda r: (r.source_path, r.segment_index))
