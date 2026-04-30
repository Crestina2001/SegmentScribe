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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ASR_SAMPLE_RATE = 16000
DTYPE_NAMES = {"float16", "bfloat16", "float32"}


def resolve_model_path(path: str, *, label: str = "model") -> str:
    """Return a local directory for ``path``.

    Resolution order:
      1. If ``path`` expands to an existing directory, return that directory.
      2. Otherwise, try ModelScope, then Hugging Face snapshot downloads and
         return the local cache directory. This lets users pass repo ids
         (e.g. ``Qwen/Qwen3-ASR-1.7B``) directly.
      3. If both fail, raise :class:`SystemExit` with a helpful error.
    """
    if not path or not str(path).strip():
        raise SystemExit(f"{label} path is empty.")

    local = Path(os.path.expanduser(str(path)))
    if local.is_dir():
        return str(local)

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
        Qwen3ASRModel, torch = _load_qwen_asr_runtime()
        torch_dtype = getattr(torch, dtype)
        resolved_model_path = resolve_model_path(model_path, label="ASR model")
        resolved_aligner_path = resolve_model_path(aligner_path, label="forced aligner")
        backend_kwargs = dict(backend_kwargs or {})
        forced_aligner_kwargs = _coerce_torch_dtype_kwargs(dict(forced_aligner_kwargs or {}))
        self.backend = backend
        self.max_inference_batch_size = max_inference_batch_size

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
                forced_aligner=resolved_aligner_path,
                forced_aligner_kwargs=forced_aligner_kwargs or None,
                **model_kwargs,
            )
        else:
            _prepare_vllm_cuda_visible_devices(device)
            self.model = Qwen3ASRModel.LLM(
                model=resolved_model_path,
                forced_aligner=resolved_aligner_path,
                forced_aligner_kwargs=forced_aligner_kwargs or None,
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

        results = self.model.transcribe(
            audio=prepared,
            language=self.language,
            return_time_stamps=True,
        )
        transcripts: list[WindowTranscript] = []
        for result, duration_sec in zip(results, durations):
            chars: list[CharToken] = []
            time_stamps = getattr(result, "time_stamps", None)
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


def _coerce_torch_dtype_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if not kwargs:
        return kwargs
    _, torch = _load_qwen_asr_runtime()
    for key in ("dtype", "torch_dtype"):
        value = kwargs.get(key)
        if isinstance(value, str) and value in DTYPE_NAMES:
            kwargs[key] = getattr(torch, value)
    return kwargs


def _load_qwen_asr_runtime() -> tuple[Any, Any]:
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise SystemExit(
            "slicing_utils.asr requires the Qwen-ASR runtime dependencies. "
            "Install this repository/package with its runtime requirements, "
            "including torch and qwen_asr, before running ASR."
        ) from exc
    return Qwen3ASRModel, torch


def _prepare_vllm_cuda_visible_devices(device: str) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    if not isinstance(device, str) or not device.startswith("cuda:"):
        return
    index = device.split(":", 1)[1].strip()
    if index.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = index
