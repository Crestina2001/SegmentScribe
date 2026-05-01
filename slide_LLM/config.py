"""Configuration for the LLM-driven slide pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from llm_gateway.models import ProviderName
from slide_rule.config import AUDIO_EXTENSIONS


@dataclass
class LLMWorkflowConfig:
    input_path: Path
    output_dir: Path
    model_path: str
    aligner_path: str
    llm_model: str
    punct_llm_model: Optional[str] = None
    rough_llm_model: Optional[str] = None
    llm_provider: Optional[ProviderName] = None
    env_path: Optional[str] = None
    llm_max_rounds: int = 5
    asr_backend: str = "transformers"
    asr_backend_kwargs: dict[str, Any] = field(default_factory=dict)
    forced_aligner_kwargs: dict[str, Any] = field(default_factory=dict)
    llm_concurrency: int = 8
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    asr_max_batch_size: int = 1
    aligner_max_batch_size: int = 1
    min_seg_sec: float = 3.0
    max_seg_sec: float = 10.0
    language: Optional[str] = None
    target_sample_rate: int = 16000
    max_source_seconds: float = 30.0 * 60.0
    allowed_extensions: tuple[str, ...] = tuple(sorted(AUDIO_EXTENSIONS))
    overwrite: bool = False
    dry_run: bool = False
    preprocess_chunk_sec: float = 30.0
    preprocess_chunk_mode: str = "rms_silence"
    preprocess_min_chunk_sec: float = 5.0
    preprocess_max_chunk_sec: float = 15.0
    rms_silence_frame_ms: float = 25.0
    rms_silence_hop_ms: float = 5.0
    rms_silence_percentile: float = 25.0
    rms_silence_threshold_multiplier: float = 1.8
    rms_min_silence_ms: float = 80.0
    vad_backend: str = "auto"
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 200

    @property
    def punctuation_model(self) -> str:
        return self.punct_llm_model or self.llm_model

    @property
    def rough_model(self) -> str:
        return self.rough_llm_model or self.llm_model
