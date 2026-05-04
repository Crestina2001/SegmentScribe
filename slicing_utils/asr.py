"""Thin wrapper around Qwen3-ASR + Qwen3-ForcedAligner.

The wrapper exposes a :class:`WindowTranscript` with per-character timestamps
in seconds, relative to the window start. Callers translate to absolute time
by adding the window cursor.

Model paths may be either a local directory or a ModelScope / Hugging Face
repo id. Non-directory inputs are resolved through ModelScope first, then
Hugging Face, so users can pass e.g. ``Qwen/Qwen3-ASR-1.7B`` directly.
"""

from __future__ import annotations

import os
import sys
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

ASR_SAMPLE_RATE = 16000
DTYPE_NAMES = {"float16", "bfloat16", "float32"}
logger = logging.getLogger(__name__)


@dataclass
class _AsrOnlyResult:
    text: str
    language: str


def resolve_model_path(path: str, *, label: str = "model") -> str:
    """Return a local directory for ``path``.

    Resolution order:
      1. If ``path`` expands to an existing directory, return its absolute path.
      2. Otherwise, try ModelScope, then Hugging Face snapshot downloads and
         return the local cache directory. This lets users pass repo ids
         (e.g. ``Qwen/Qwen3-ASR-1.7B``) directly.
      3. If both fail, raise :class:`SystemExit` with a helpful error.
    """
    if not path or not str(path).strip():
        raise SystemExit(f"{label} path is empty.")

    local = Path(os.path.expanduser(str(path)))
    if local.is_dir():
        return str(local.resolve())

    errors: list[str] = []
    try:
        from modelscope import snapshot_download as modelscope_snapshot_download

        try:
            resolved = modelscope_snapshot_download(str(path))
            print(
                f"[slide_LLM] Resolved {label} '{path}' via modelscope -> {resolved}",
                file=sys.stderr,
            )
            return str(resolved)
        except Exception as exc:
            errors.append(f"ModelScope rejected it: {exc}")
    except ImportError as exc:
        errors.append(f"ModelScope package is not installed: {exc}")

    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        try:
            resolved = hf_snapshot_download(repo_id=str(path))
            print(
                f"[slide_LLM] Resolved {label} '{path}' via Hugging Face -> {resolved}",
                file=sys.stderr,
            )
            return str(resolved)
        except Exception as exc:
            errors.append(f"Hugging Face rejected it: {exc}")
    except ImportError as exc:
        errors.append(f"huggingface_hub package is not installed: {exc}")

    detail = "\n  - ".join(errors)
    raise SystemExit(
        f"Failed to resolve {label} path '{path}'. It is not an existing local "
        f"directory, and no model hub accepted it.\n  - {detail}\n"
        f"Either download the model first and pass the local directory, or pass a "
        f"valid ModelScope/Hugging Face repo id."
    )


@dataclass(frozen=True)
class CharToken:
    """One aligned character/word from the forced aligner."""

    idx: int
    char: str
    start_sec: float
    end_sec: float


@dataclass
class WindowTranscript:
    """ASR result for one sliding window.

    All times are seconds relative to the window start (not the source file).
    ``chars`` is an index-stable list: ``chars[i].idx == i``.
    """

    text: str
    language: str
    chars: list[CharToken]
    duration_sec: float


class AsrBackend:
    """Loads Qwen3-ASR with forced alignment and transcribes short windows."""

    def __init__(
        self,
        model_path: str,
        aligner_path: str,
        *,
        backend: str = "transformers",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_inference_batch_size: int = 1,
        max_aligner_batch_size: Optional[int] = None,
        aligner_concurrency: int = 1,
        max_new_tokens: int = 512,
        language: Optional[str] = None,
        backend_kwargs: Optional[dict[str, Any]] = None,
        forced_aligner_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if aligner_path is None or not str(aligner_path).strip():
            raise ValueError(
                "AsrBackend requires a forced-aligner path for per-character timestamps."
            )
        if backend not in {"transformers", "vllm"}:
            raise ValueError(f"Unsupported ASR backend: {backend}")

        if dtype not in DTYPE_NAMES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        Qwen3ASRModel, Qwen3ForcedAligner, torch = _load_qwen_asr_runtime()
        torch_dtype = getattr(torch, dtype)
        resolved_model_path = resolve_model_path(model_path, label="ASR model")
        resolved_aligner_path = resolve_model_path(aligner_path, label="forced aligner")
        backend_kwargs = dict(backend_kwargs or {})
        forced_aligner_kwargs = _coerce_torch_dtype_kwargs(dict(forced_aligner_kwargs or {}))
        self.backend = backend
        self.max_inference_batch_size = max_inference_batch_size
        self.max_aligner_batch_size = (
            max(1, int(max_aligner_batch_size))
            if max_aligner_batch_size is not None
            else max(1, int(max_inference_batch_size or 1))
        )
        self.aligner_concurrency = max(1, int(aligner_concurrency or 1))
        self._forced_aligner_cls = Qwen3ForcedAligner
        self._forced_aligner_path = resolved_aligner_path
        self._forced_aligner_kwargs = forced_aligner_kwargs
        self._forced_aligners: list[Any | None] = [None] * self.aligner_concurrency
        self._forced_aligner_slots: queue.Queue[int] = queue.Queue()
        for slot in range(self.aligner_concurrency):
            self._forced_aligner_slots.put(slot)
        self._forced_aligner_lock = threading.RLock()

        if backend == "transformers":
            model_kwargs = {
                "dtype": torch_dtype,
                "device_map": device,
                "max_inference_batch_size": max_inference_batch_size,
                "max_new_tokens": max_new_tokens,
                **_coerce_torch_dtype_kwargs(backend_kwargs),
            }
            self.model = Qwen3ASRModel.from_pretrained(
                resolved_model_path,
                forced_aligner=None,
                **model_kwargs,
            )
        else:
            _prepare_vllm_cuda_visible_devices(device)
            self.model = Qwen3ASRModel.LLM(
                model=resolved_model_path,
                forced_aligner=None,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
                **backend_kwargs,
            )
        self.language = language
        self.sample_rate = ASR_SAMPLE_RATE

    def transcribe_window(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> WindowTranscript:
        """Transcribe a single window and return a :class:`WindowTranscript`.

        The Qwen3-ASR model expects 16 kHz mono float32 samples. The caller is
        responsible for resampling the source audio; this wrapper only checks
        the incoming sample rate.
        """
        return self.transcribe_windows([(audio, sample_rate)])[0]

    def transcribe_windows(
        self,
        windows: Sequence[tuple[np.ndarray, int]],
    ) -> list[WindowTranscript]:
        """Transcribe a batch of windows and return results in input order."""
        prepared, durations = self._prepare_windows(windows)
        results = self._run_asr(prepared)
        alignments = self._align_transcripts(prepared, results)
        return self._build_transcripts(results, alignments, durations)

    def transcribe_windows_asr_only(
        self,
        windows: Sequence[tuple[np.ndarray, int]],
    ) -> list[WindowTranscript]:
        """Run only Qwen3-ASR for a batch, without forced alignment."""
        prepared, durations = self._prepare_windows(windows)
        results = self._run_asr(prepared)
        return self._build_transcripts(results, [None] * len(results), durations)

    def align_window_transcripts(
        self,
        windows: Sequence[tuple[np.ndarray, int]],
        transcripts: Sequence[WindowTranscript],
    ) -> list[WindowTranscript]:
        """Run forced alignment for existing ASR text over matching windows."""
        prepared, durations = self._prepare_windows(windows)
        results = [_AsrOnlyResult(text=tr.text, language=tr.language) for tr in transcripts]
        alignments = self._align_transcripts(prepared, results)
        return self._build_transcripts(results, alignments, durations)

    def _prepare_windows(
        self,
        windows: Sequence[tuple[np.ndarray, int]],
    ) -> tuple[list[tuple[np.ndarray, int]], list[float]]:
        prepared: list[tuple[np.ndarray, int]] = []
        durations: list[float] = []
        for audio, sample_rate in windows:
            if audio.ndim != 1:
                raise ValueError(f"Expected mono audio, got shape {audio.shape}.")
            if sample_rate != self.sample_rate:
                raise ValueError(
                    f"AsrBackend expects {self.sample_rate} Hz audio, got {sample_rate} Hz."
                )
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32, copy=False)
            prepared.append((audio, sample_rate))
            durations.append(float(len(audio) / sample_rate) if sample_rate > 0 else 0.0)
        return prepared, durations

    def _run_asr(self, prepared: Sequence[tuple[np.ndarray, int]]) -> list[Any]:
        return list(self.model.transcribe(
            audio=prepared,
            language=self.language,
            return_time_stamps=False,
        ))

    def _build_transcripts(
        self,
        results: Sequence[Any],
        alignments: Sequence[Any | None],
        durations: Sequence[float],
    ) -> list[WindowTranscript]:
        transcripts: list[WindowTranscript] = []
        for result, alignment, duration_sec in zip(results, alignments, durations):
            chars: list[CharToken] = []
            time_stamps = alignment
            items = list(time_stamps) if time_stamps is not None else []
            for idx, item in enumerate(items):
                chars.append(
                    CharToken(
                        idx=idx,
                        char=str(item.text),
                        start_sec=float(item.start_time),
                        end_sec=float(item.end_time),
                    )
                )
            transcripts.append(
                WindowTranscript(
                    text=str(result.text or ""),
                    language=str(result.language or ""),
                    chars=chars,
                    duration_sec=duration_sec,
                )
            )
        return transcripts

    def _align_transcripts(
        self,
        prepared: Sequence[tuple[np.ndarray, int]],
        results: Sequence[Any],
    ) -> list[Any | None]:
        pending_audio: list[tuple[np.ndarray, int]] = []
        pending_text: list[str] = []
        pending_language: list[str] = []
        pending_indices: list[int] = []
        alignments: list[Any | None] = [None] * len(results)

        for index, ((audio, sample_rate), result) in enumerate(zip(prepared, results)):
            text = str(getattr(result, "text", "") or "")
            if not text.strip():
                continue
            language = str(getattr(result, "language", "") or self.language or "")
            pending_audio.append((audio, sample_rate))
            pending_text.append(text)
            pending_language.append(language)
            pending_indices.append(index)

        if not pending_audio:
            return alignments

        batch_size = max(1, int(self.max_aligner_batch_size or 1))
        jobs = sorted(
            zip(pending_indices, pending_audio, pending_text, pending_language),
            key=lambda item: len(item[1][0]),
        )
        progress = tqdm(
            total=len(jobs),
            desc="Qwen3-ForcedAligner",
            unit="win",
            dynamic_ncols=True,
            leave=False,
        )
        try:
            slot = self._forced_aligner_slots.get()
            try:
                with self._forced_aligner_lock:
                    aligner = self._get_forced_aligner_unlocked(slot)
                for offset in range(0, len(jobs), batch_size):
                    job_batch = jobs[offset : offset + batch_size]
                    index_batch = [item[0] for item in job_batch]
                    audio_batch = [item[1] for item in job_batch]
                    text_batch = [item[2] for item in job_batch]
                    language_batch = [item[3] for item in job_batch]
                    aligned_batch = aligner.align(
                        audio=audio_batch,
                        text=text_batch,
                        language=language_batch,
                    )
                    for index, alignment in zip(index_batch, aligned_batch):
                        alignments[index] = alignment
                    progress.update(len(audio_batch))
            finally:
                self._forced_aligner_slots.put(slot)
        finally:
            progress.close()
        return alignments

    def _get_forced_aligner(self) -> Any:
        with self._forced_aligner_lock:
            return self._get_forced_aligner_unlocked(0)

    def _get_forced_aligner_unlocked(self, slot: int) -> Any:
        if self._forced_aligners[slot] is None:
            logger.info(
                "Loading Qwen3 forced aligner slot %d/%d: %s",
                slot + 1,
                self.aligner_concurrency,
                self._forced_aligner_path,
            )
            self._forced_aligners[slot] = self._forced_aligner_cls.from_pretrained(
                self._forced_aligner_path,
                **self._forced_aligner_kwargs,
            )
        return self._forced_aligners[slot]


def _should_log_progress(offset: int, batch_len: int, total: int, batch_size: int) -> bool:
    if total <= 10 or batch_size > 1:
        return True
    current = offset + batch_len
    return offset == 0 or current >= total or current % 10 == 0


def _coerce_torch_dtype_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if not kwargs:
        return kwargs
    _, _, torch = _load_qwen_asr_runtime()
    for key in ("dtype", "torch_dtype"):
        value = kwargs.get(key)
        if isinstance(value, str) and value in DTYPE_NAMES:
            kwargs[key] = getattr(torch, value)
    return kwargs


def _load_qwen_asr_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
    except ImportError as exc:
        raise SystemExit(
            "slicing_utils.asr requires the Qwen-ASR runtime dependencies. "
            "Install this repository/package with its runtime requirements, "
            "including torch and qwen_asr, before running ASR."
        ) from exc
    return Qwen3ASRModel, Qwen3ForcedAligner, torch


def _prepare_vllm_cuda_visible_devices(device: str) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    if not isinstance(device, str) or not device.startswith("cuda:"):
        return
    index = device.split(":", 1)[1].strip()
    if index.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = index
