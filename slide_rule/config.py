"""Lightweight configuration for ``slide_rule``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}


@dataclass
class RuleWorkflowConfig:
    input_path: Path
    output_dir: Path
    model_path: str
    aligner_path: str
    asr_backend: str = "transformers"
    asr_backend_kwargs: dict[str, Any] = field(default_factory=dict)
    forced_aligner_kwargs: dict[str, Any] = field(default_factory=dict)
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    asr_max_batch_size: int = 1
    min_seg_sec: float = 3.0
    max_seg_sec: float = 10.0
    language: Optional[str] = None
    target_sample_rate: int = 16000
    max_source_seconds: float = 30.0 * 60.0
    allowed_extensions: tuple[str, ...] = tuple(sorted(AUDIO_EXTENSIONS))
    overwrite: bool = False
    dry_run: bool = False
    preprocess_chunk_sec: float = 30.0
    vad_backend: str = "auto"
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 200
    enable_punctuation_correction: bool = False
