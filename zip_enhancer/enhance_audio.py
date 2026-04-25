#!/usr/bin/env python
"""Enhance long-form speech audio with ModelScope ZipEnhancer.

The tool slices a long input into speech regions, enhances only those speech
regions, and concatenates them into a speech-only 16 kHz mono WAV.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional for Silero VAD
    torch = None

try:
    from silero_vad import get_speech_timestamps, load_silero_vad
except ImportError:  # pragma: no cover - optional dependency
    get_speech_timestamps = None
    load_silero_vad = None


DEFAULT_MODEL = "iic/speech_zipenhancer_ans_multiloss_16k_base"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}
PEAK_CEILING_DBFS = -1.0
NORMALIZE_MODES = {"none", "match_original", "library_median"}


@dataclass(frozen=True)
class SpeechSlice:
    start: int
    end: int

    @property
    def duration_samples(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class PreparedAudio:
    input_path: Path
    output_path: Path
    audio: np.ndarray
    sample_rate: int
    source_duration: float
    speech_regions: list[SpeechSlice]
    speech_slices: list[SpeechSlice]
    hard_cut_count: int


def require_librosa():
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "Missing dependency 'librosa'. Install it with: pip install librosa"
        ) from exc
    return librosa


def require_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "Missing dependency 'soundfile'. Install it with: pip install soundfile"
        ) from exc
    return sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slice long audio into speech regions, enhance each region with "
            "ModelScope ZipEnhancer, and write a speech-only WAV."
        )
    )
    parser.add_argument("--input", required=True, help="Input audio path.")
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output enhanced WAV path for a file input, or output directory "
            "for a directory input."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "ModelScope model id or local model path. If your ModelScope version "
            "does not accept the default iic id, try "
            "damo/speech_zipenhancer_ans_multiloss_16k_base."
        ),
    )
    parser.add_argument(
        "--vad-backend",
        default="auto",
        choices=("auto", "silero", "librosa"),
        help="VAD backend. 'auto' prefers Silero and falls back to librosa.",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=16000,
        help="Sample rate used for enhancement and output.",
    )
    parser.add_argument(
        "--min-speech-seconds",
        type=float,
        default=0.25,
        help="Discard detected speech regions shorter than this duration.",
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=120,
        help="Padding added to both sides of detected speech regions.",
    )
    parser.add_argument(
        "--max-slice-seconds",
        type=float,
        default=30.0,
        help=(
            "Legacy hard split length used only when --no-silence-aware-splits "
            "is passed. Default: 30."
        ),
    )
    parser.add_argument(
        "--silence-aware-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Split long speech by searching VAD non-speech gaps between "
            "--silence-search-min-seconds and --silence-search-max-seconds. "
            "Default: true."
        ),
    )
    parser.add_argument(
        "--silence-search-min-seconds",
        type=float,
        default=5.0,
        help="Earliest cut search point for silence-aware splitting. Default: 5.",
    )
    parser.add_argument(
        "--silence-search-max-seconds",
        type=float,
        default=15.0,
        help="Latest cut search point for silence-aware splitting. Default: 15.",
    )
    parser.add_argument(
        "--fallback-slice-seconds",
        type=float,
        default=10.0,
        help="Hard cut length when no VAD non-speech gap is found. Default: 10.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=2,
        help=(
            "Directory mode only: number of background workers for decode, "
            "resample, and VAD while the GPU enhances prepared files. Default: 2."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device passed to the ModelScope pipeline when supported.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD threshold, or librosa energy sensitivity mapping.",
    )
    parser.add_argument(
        "--min-silence-duration-ms",
        type=int,
        default=300,
        help="Silence needed to separate nearby speech regions.",
    )
    parser.add_argument(
        "--normalize",
        default="none",
        help=(
            "Normalization mode: none, match_original, library_median, or a numeric "
            "dBFS target such as -20. library_median estimates one corpus-level RMS "
            "target using a duration-weighted speech median."
        ),
    )
    parser.add_argument(
        "--alignment-metric",
        default="rms",
        choices=("rms", "peak"),
        help="Metric used by --normalize: rms or peak. Default: rms.",
    )
    parser.add_argument(
        "--use-modelscope-preprocess",
        action="store_true",
        help=(
            "Use ModelScope's file-based preprocess path. This keeps upstream "
            "behavior but applies ModelScope audio_norm(), which changes gain."
        ),
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --input is a directory, scan audio files recursively. Default: true.",
    )
    parser.add_argument(
        "--allowed-extensions",
        nargs="*",
        default=sorted(AUDIO_EXTENSIONS),
        help="Audio extensions to process when --input is a directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --output if it already exists.",
    )
    return parser.parse_args()


def load_audio_mono(path: Path, target_sample_rate: int) -> tuple[np.ndarray, int, float]:
    try:
        sf = require_soundfile()
        audio, sample_rate = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = np.asarray(audio, dtype=np.float32)

        source_duration = float(len(audio) / sample_rate) if sample_rate > 0 else 0.0
        if sample_rate != target_sample_rate:
            librosa = require_librosa()
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate
    except Exception as exc:
        librosa = require_librosa()
        try:
            audio, sample_rate = librosa.load(
                str(path),
                sr=target_sample_rate,
                mono=True,
                dtype=np.float32,
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Could not decode audio with soundfile or librosa: {path}"
            ) from fallback_exc
        source_duration = float(len(audio) / target_sample_rate) if target_sample_rate > 0 else 0.0
        sample_rate = target_sample_rate
        print(f"Decoded with librosa fallback: {path} ({exc})")

    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate, source_duration


def normalize_extensions(extensions: Sequence[str]) -> set[str]:
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}


def iter_audio_files(input_dir: Path, allowed_extensions: Sequence[str], recursive: bool) -> list[Path]:
    exts = normalize_extensions(allowed_extensions)
    paths = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in exts)


def choose_vad_backend(requested: str) -> str:
    if requested == "silero":
        if load_silero_vad is None or get_speech_timestamps is None or torch is None:
            raise SystemExit("silero-vad is not installed. Install it with: pip install silero-vad")
        return "silero"
    if requested == "librosa":
        return "librosa"
    return "silero" if load_silero_vad is not None and get_speech_timestamps is not None and torch is not None else "librosa"


def merge_regions(regions: Sequence[SpeechSlice], max_gap_samples: int) -> list[SpeechSlice]:
    merged: list[SpeechSlice] = []
    for region in sorted(regions, key=lambda item: item.start):
        if region.end <= region.start:
            continue
        if merged and region.start - merged[-1].end <= max_gap_samples:
            merged[-1] = SpeechSlice(merged[-1].start, max(merged[-1].end, region.end))
        else:
            merged.append(region)
    return merged


def detect_speech_slices(
    audio: np.ndarray,
    sample_rate: int,
    backend: str,
    threshold: float,
    min_speech_seconds: float,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
) -> list[SpeechSlice]:
    min_speech_samples = max(1, int(round(min_speech_seconds * sample_rate)))
    pad_samples = max(0, int(round(speech_pad_ms / 1000.0 * sample_rate)))
    max_gap_samples = max(0, int(round(min_silence_duration_ms / 1000.0 * sample_rate)))

    if backend == "silero":
        vad_model = load_silero_vad()
        speech = torch.from_numpy(audio)
        raw_regions = get_speech_timestamps(
            speech,
            vad_model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=int(round(min_speech_seconds * 1000.0)),
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=False,
        )
        regions = [
            SpeechSlice(max(0, int(item["start"])), min(len(audio), int(item["end"])))
            for item in raw_regions
        ]
    else:
        librosa = require_librosa()
        top_db = max(10, int(round((1.0 - threshold) * 60)))
        intervals = librosa.effects.split(
            audio,
            top_db=top_db,
            frame_length=2048,
            hop_length=512,
        )
        regions = [
            SpeechSlice(max(0, int(start) - pad_samples), min(len(audio), int(end) + pad_samples))
            for start, end in intervals.tolist()
        ]

    filtered = [region for region in regions if region.duration_samples >= min_speech_samples]
    return merge_regions(filtered, max_gap_samples=max_gap_samples)


def split_long_slices(slices: Sequence[SpeechSlice], max_slice_seconds: float, sample_rate: int) -> list[SpeechSlice]:
    max_samples = max(1, int(round(max_slice_seconds * sample_rate)))
    split_slices: list[SpeechSlice] = []
    for region in slices:
        start = region.start
        while start < region.end:
            end = min(region.end, start + max_samples)
            split_slices.append(SpeechSlice(start=start, end=end))
            start = end
    return split_slices


def find_longest_vad_gap_cut(
    speech_regions: Sequence[SpeechSlice],
    start: int,
    end: int,
    sample_rate: int,
    search_min_seconds: float,
    search_max_seconds: float,
) -> int | None:
    search_start = start + int(round(search_min_seconds * sample_rate))
    search_end = min(end, start + int(round(search_max_seconds * sample_rate)))
    if search_end <= search_start:
        return None

    best_gap: SpeechSlice | None = None
    for previous, current in zip(speech_regions, speech_regions[1:]):
        gap_start = max(previous.end, start)
        gap_end = min(current.start, end)
        if gap_end <= gap_start:
            continue
        overlap_start = max(gap_start, search_start)
        overlap_end = min(gap_end, search_end)
        if overlap_end <= overlap_start:
            continue

        candidate = SpeechSlice(start=overlap_start, end=overlap_end)
        if best_gap is None or candidate.duration_samples > best_gap.duration_samples:
            best_gap = candidate

    if best_gap is None:
        return None
    return best_gap.start + best_gap.duration_samples // 2


def split_slices_on_vad_gaps(
    speech_regions: Sequence[SpeechSlice],
    sample_rate: int,
    args: argparse.Namespace,
) -> tuple[list[SpeechSlice], int]:
    if not args.silence_aware_splits:
        return split_long_slices(speech_regions, args.max_slice_seconds, sample_rate), 0
    if not speech_regions:
        return [], 0

    max_samples = max(1, int(round(args.silence_search_max_seconds * sample_rate)))
    fallback_samples = max(1, int(round(args.fallback_slice_seconds * sample_rate)))
    split_slices: list[SpeechSlice] = []
    hard_cut_count = 0

    start = speech_regions[0].start
    final_end = speech_regions[-1].end
    while start < final_end:
        if final_end - start <= max_samples:
            split_slices.append(SpeechSlice(start=start, end=final_end))
            break

        cut = find_longest_vad_gap_cut(
            speech_regions=speech_regions,
            start=start,
            end=final_end,
            sample_rate=sample_rate,
            search_min_seconds=args.silence_search_min_seconds,
            search_max_seconds=args.silence_search_max_seconds,
        )
        if cut is None:
            cut = min(final_end, start + fallback_samples)
            hard_cut_count += 1
        if cut <= start:
            cut = min(final_end, start + fallback_samples)
            hard_cut_count += 1

        split_slices.append(SpeechSlice(start=start, end=cut))
        start = cut

    return split_slices, hard_cut_count


def build_enhancer(model: str, device: str | None):
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
    except ImportError as exc:  # pragma: no cover - import guard
        missing = getattr(exc, "name", None)
        if missing and missing != "modelscope":
            raise SystemExit(
                f"Missing dependency required by ModelScope pipelines: {missing}. "
                f"Install it with: pip install {missing}"
            ) from exc
        raise SystemExit(
            "Missing dependency 'modelscope'. Install it with: pip install -U modelscope"
        ) from exc

    kwargs = {"model": model}
    if device:
        kwargs["device"] = device

    try:
        enhancer = pipeline(Tasks.acoustic_noise_suppression, **kwargs)
    except TypeError:
        if not device:
            raise
        print(
            "ModelScope pipeline rejected --device; retrying without device.",
            file=sys.stderr,
        )
        enhancer = pipeline(Tasks.acoustic_noise_suppression, model=model)

    effective_device = resolve_enhancer_device(enhancer, device)
    prepare_enhancer_model(enhancer)
    if effective_device:
        move_enhancer_to_device(enhancer, effective_device)

    return enhancer


def resolve_enhancer_device(enhancer, requested_device: str | None) -> str | None:
    if requested_device:
        return requested_device
    device = getattr(enhancer, "device", None)
    if device is None:
        device = getattr(enhancer, "device_name", None)
    return str(device) if device is not None else None


def prepare_enhancer_model(enhancer) -> None:
    prepare_model = getattr(enhancer, "prepare_model", None)
    if prepare_model is None:
        return
    try:
        prepare_model()
    except Exception as exc:
        print(
            f"Warning: ModelScope prepare_model() failed; forcing device sync instead: {exc}",
            file=sys.stderr,
        )


def move_enhancer_to_device(enhancer, device: str) -> None:
    if hasattr(enhancer, "device"):
        enhancer.device = device

    seen: set[int] = set()

    def move_obj(obj) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if hasattr(obj, "device"):
            try:
                obj.device = device
            except Exception:
                pass

        if torch is not None and isinstance(obj, torch.nn.Module):
            obj.to(device)
            obj.eval()
            return

        if hasattr(obj, "to"):
            try:
                obj.to(device)
            except Exception:
                pass

        if isinstance(obj, dict):
            for value in obj.values():
                move_obj(value)
            return

        if isinstance(obj, (list, tuple, set)):
            for value in obj:
                move_obj(value)
            return

        if hasattr(obj, "__dict__"):
            for key, value in vars(obj).items():
                if key.startswith("__"):
                    continue
                if isinstance(value, (str, bytes, int, float, bool, Path, type(None))):
                    continue
                move_obj(value)

    move_obj(enhancer)


def write_temp_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf = require_soundfile()
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def read_output_audio(path: Path, sample_rate: int) -> np.ndarray:
    sf = require_soundfile()
    audio, output_sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if output_sample_rate != sample_rate:
        librosa = require_librosa()
        audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=output_sample_rate, target_sr=sample_rate)
    return np.ascontiguousarray(audio, dtype=np.float32)


def rms_level(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    data = audio.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(data * data)))


def peak_level(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def linear_to_dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-12)))


def dbfs_to_linear(dbfs: float) -> float:
    return float(10.0 ** (dbfs / 20.0))


def parse_normalize(value: str | float) -> str | float:
    if isinstance(value, str) and value.lower() in NORMALIZE_MODES:
        return value.lower()
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "--normalize must be one of: none, match_original, library_median, "
            "or a numeric dBFS target such as -20."
        ) from exc


def normalize_output_audio(
    audio: np.ndarray,
    mode: str,
    target_dbfs: float,
    peak_ceiling_dbfs: float = PEAK_CEILING_DBFS,
) -> tuple[np.ndarray, float, str]:
    if mode == "none":
        return audio, 1.0, "disabled"

    eps = 1e-8
    current_rms = rms_level(audio)
    current_peak = peak_level(audio)
    peak_ceiling = min(dbfs_to_linear(peak_ceiling_dbfs), 1.0)

    if current_peak <= eps:
        return audio, 1.0, "silent output"

    if peak_ceiling <= eps:
        raise ValueError("Internal peak ceiling is too low; choose a value closer to 0 dBFS.")

    if mode == "rms":
        if current_rms <= eps:
            return audio, 1.0, "near-silent output"
        target_rms = dbfs_to_linear(target_dbfs)
        rms_scale = target_rms / current_rms
        peak_scale = peak_ceiling / current_peak
        scale = min(rms_scale, peak_scale)
        reason = (
            f"target RMS {target_dbfs:.1f} dBFS"
            if scale == rms_scale
            else f"limited by peak ceiling {peak_ceiling_dbfs:.1f} dBFS"
        )
    elif mode == "peak":
        requested_peak = dbfs_to_linear(target_dbfs)
        target_peak = min(requested_peak, peak_ceiling)
        scale = target_peak / current_peak
        reason = (
            f"target peak {target_dbfs:.1f} dBFS"
            if target_peak == requested_peak
            else f"limited by peak ceiling {peak_ceiling_dbfs:.1f} dBFS"
        )
    else:
        raise ValueError(f"Unsupported output normalization mode: {mode}")

    normalized = audio * float(scale)
    normalized = np.clip(normalized, -peak_ceiling, peak_ceiling)
    return np.ascontiguousarray(normalized, dtype=np.float32), float(scale), reason


def speech_stats_for_file(
    input_path: Path,
    args: argparse.Namespace,
    vad_backend: str,
) -> tuple[float, float, float] | None:
    audio, sample_rate, source_duration = load_audio_mono(input_path, args.target_sample_rate)
    speech_regions = detect_speech_slices(
        audio=audio,
        sample_rate=sample_rate,
        backend=vad_backend,
        threshold=args.vad_threshold,
        min_speech_seconds=args.min_speech_seconds,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
    )
    speech_slices, _hard_cut_count = split_slices_on_vad_gaps(speech_regions, sample_rate, args)
    if not speech_slices:
        print(f"Auto RMS skipped: no speech detected in {input_path}", file=sys.stderr)
        return None

    sum_squares = 0.0
    sample_count = 0
    peak = 0.0
    for speech_slice in speech_slices:
        chunk = audio[speech_slice.start : speech_slice.end].astype(np.float64, copy=False)
        sum_squares += float(np.sum(chunk * chunk))
        sample_count += int(chunk.size)
        if chunk.size:
            peak = max(peak, float(np.max(np.abs(chunk))))

    if sample_count <= 0 or peak <= 1e-8:
        print(f"Auto RMS skipped: near-silent speech in {input_path}", file=sys.stderr)
        return None

    rms = float(np.sqrt(sum_squares / sample_count))
    speech_duration = sample_count / sample_rate
    print(
        f"Auto RMS stats: {input_path} "
        f"(source {source_duration:.2f}s, speech {speech_duration:.2f}s, "
        f"RMS {linear_to_dbfs(rms):.2f} dBFS, peak {linear_to_dbfs(peak):.2f} dBFS)"
    )
    return rms, peak, speech_duration


def estimate_target_rms_dbfs(
    source_files: Sequence[Path],
    args: argparse.Namespace,
    vad_backend: str,
) -> float:
    weighted_rms_dbfs_values: list[tuple[float, float]] = []
    max_safe_targets: list[float] = []

    print("\nEstimating duration-weighted median RMS target from input speech levels...")
    for index, source_file in enumerate(source_files, start=1):
        print(f"[auto {index}/{len(source_files)}] {source_file}")
        stats = speech_stats_for_file(source_file, args, vad_backend)
        if stats is None:
            continue
        rms, peak, speech_duration = stats
        rms_dbfs = linear_to_dbfs(rms)
        peak_dbfs = linear_to_dbfs(peak)
        if speech_duration > 0.0:
            weighted_rms_dbfs_values.append((rms_dbfs, speech_duration))
        max_safe_targets.append(rms_dbfs + (PEAK_CEILING_DBFS - peak_dbfs))

    if not weighted_rms_dbfs_values:
        raise SystemExit("Could not estimate --normalize library_median: no usable speech was detected.")

    unconstrained_target = weighted_median(
        [value for value, _weight in weighted_rms_dbfs_values],
        [weight for _value, weight in weighted_rms_dbfs_values],
    )
    no_clip_target = min(max_safe_targets)
    target = min(unconstrained_target, no_clip_target)
    limited_note = (
        ""
        if target == unconstrained_target
        else f"; limited from {unconstrained_target:.2f} dBFS by peak ceiling"
    )
    total_weight = sum(weight for _value, weight in weighted_rms_dbfs_values)
    weighted_absolute_loss = (
        sum(abs(value - target) * weight for value, weight in weighted_rms_dbfs_values) / total_weight
        if total_weight > 0.0
        else 0.0
    )
    print(
        f"Auto RMS target: {target:.2f} dBFS "
        f"(files {len(weighted_rms_dbfs_values)}, duration-weighted median, "
        f"avg absolute gain shift {weighted_absolute_loss:.3f} dB"
        f"{limited_note})\n"
    )
    return target


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("weighted_median requires equal numbers of values and weights.")
    if not values:
        raise ValueError("weighted_median requires at least one value.")

    pairs = sorted(
        (float(value), float(weight))
        for value, weight in zip(values, weights)
        if float(weight) > 0.0
    )
    if not pairs:
        raise ValueError("weighted_median requires at least one positive weight.")

    total_weight = sum(weight for _value, weight in pairs)
    midpoint = total_weight / 2.0
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= midpoint:
            return value
    return pairs[-1][0]


def restore_gain(enhanced: np.ndarray, reference: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    if mode == "none":
        return enhanced, 1.0

    eps = 1e-8
    ref_rms = rms_level(reference)
    ref_peak = peak_level(reference)
    out_rms = rms_level(enhanced)
    out_peak = peak_level(enhanced)

    if ref_peak <= eps or out_peak <= eps:
        return enhanced, 1.0

    if mode == "rms":
        scale = ref_rms / max(out_rms, eps)
    elif mode == "peak":
        scale = ref_peak / out_peak
    elif mode == "rms-peak":
        rms_scale = ref_rms / max(out_rms, eps)
        peak_scale = ref_peak / out_peak
        scale = min(rms_scale, peak_scale)
    else:
        raise ValueError(f"Unsupported gain mode: {mode}")

    scaled = enhanced * float(scale)
    scaled_peak = peak_level(scaled)
    if scaled_peak > 1.0:
        clip_guard = 1.0 / max(scaled_peak, eps)
        scaled = scaled * clip_guard
        scale *= clip_guard

    return np.ascontiguousarray(scaled, dtype=np.float32), float(scale)


def gain_mode_for_normalize(normalize: str | float, alignment_metric: str) -> str:
    if normalize != "match_original":
        return "none"
    return "rms-peak" if alignment_metric == "rms" else "peak"


def output_normalization_for_normalize(
    normalize: str | float,
    alignment_metric: str,
) -> tuple[str, float | None]:
    if isinstance(normalize, float):
        return alignment_metric, normalize
    if normalize == "library_median":
        return "rms", None
    return "none", None


def enhance_slice(
    enhancer,
    audio: np.ndarray,
    sample_rate: int,
    temp_dir: Path,
    index: int,
    use_modelscope_preprocess: bool,
) -> np.ndarray:
    if not use_modelscope_preprocess:
        expected_sample_rate = int(getattr(enhancer, "SAMPLE_RATE", sample_rate))
        if sample_rate != expected_sample_rate:
            raise RuntimeError(
                f"Direct ZipEnhancer mode expects {expected_sample_rate} Hz audio, got {sample_rate} Hz."
            )
        inputs = {
            "ndarray": np.reshape(np.ascontiguousarray(audio, dtype=np.float32), [1, audio.shape[0]]),
            "nsamples": int(audio.shape[0]),
        }
        result = enhancer.forward(inputs)
        output_pcm = result.get("output_pcm") if isinstance(result, dict) else None
        if output_pcm is None:
            raise RuntimeError(f"ModelScope enhancer did not produce output for slice {index}.")
        pcm = np.frombuffer(output_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return np.ascontiguousarray(pcm, dtype=np.float32)

    input_path = temp_dir / f"slice_{index:06d}_input.wav"
    output_path = temp_dir / f"slice_{index:06d}_enhanced.wav"
    write_temp_wav(input_path, audio, sample_rate)

    result = enhancer(str(input_path), output_path=str(output_path))
    if output_path.exists():
        return read_output_audio(output_path, sample_rate)

    output_pcm = result.get("output_pcm") if isinstance(result, dict) else None
    if output_pcm is None:
        raise RuntimeError(f"ModelScope enhancer did not produce output for slice {index}.")

    pcm = np.frombuffer(output_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return np.ascontiguousarray(pcm, dtype=np.float32)


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def output_path_for_input(input_file: Path, input_root: Path, output_root: Path) -> Path:
    relpath = input_file.relative_to(input_root)
    return output_root / relpath.with_suffix(".wav")


def prepare_file(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    vad_backend: str,
) -> PreparedAudio | None:
    ensure_output_path(output_path, overwrite=args.overwrite)

    audio, sample_rate, source_duration = load_audio_mono(input_path, args.target_sample_rate)
    print(f"Loaded: {input_path} ({source_duration:.2f}s -> {len(audio) / sample_rate:.2f}s at {sample_rate} Hz)")

    speech_regions = detect_speech_slices(
        audio=audio,
        sample_rate=sample_rate,
        backend=vad_backend,
        threshold=args.vad_threshold,
        min_speech_seconds=args.min_speech_seconds,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
    )
    speech_slices, hard_cut_count = split_slices_on_vad_gaps(speech_regions, sample_rate, args)
    total_speech_sec = sum(item.duration_samples for item in speech_slices) / sample_rate
    print(
        f"Speech regions: {len(speech_regions)}; enhancement slices: "
        f"{len(speech_slices)}; speech: {total_speech_sec:.2f}s"
    )
    if hard_cut_count:
        print(
            f"Warning: no VAD non-speech gap found for {hard_cut_count} slice split(s); "
            f"used {args.fallback_slice_seconds:.2f}s hard fallback.",
            file=sys.stderr,
        )

    if not speech_slices:
        print(f"Skipped: no speech detected in {input_path}", file=sys.stderr)
        return None

    return PreparedAudio(
        input_path=input_path,
        output_path=output_path,
        audio=audio,
        sample_rate=sample_rate,
        source_duration=source_duration,
        speech_regions=speech_regions,
        speech_slices=speech_slices,
        hard_cut_count=hard_cut_count,
    )


def enhance_prepared_file(
    prepared: PreparedAudio,
    args: argparse.Namespace,
    enhancer,
) -> bool:
    try:
        gain_mode = gain_mode_for_normalize(args.normalize, args.alignment_metric)
        output_mode, output_target_dbfs = output_normalization_for_normalize(
            args.normalize,
            args.alignment_metric,
        )
        enhanced_chunks: list[np.ndarray] = []
        with tempfile.TemporaryDirectory(prefix="zipenhancer_") as temp_root:
            temp_dir = Path(temp_root)
            for index, speech_slice in enumerate(prepared.speech_slices, start=1):
                chunk = np.ascontiguousarray(
                    prepared.audio[speech_slice.start : speech_slice.end],
                    dtype=np.float32,
                )
                enhanced = enhance_slice(
                    enhancer,
                    chunk,
                    prepared.sample_rate,
                    temp_dir,
                    index,
                    use_modelscope_preprocess=args.use_modelscope_preprocess,
                )
                enhanced, gain_scale = restore_gain(enhanced, chunk, gain_mode)
                for speech_region in prepared.speech_regions:
                    overlap_start = max(speech_region.start, speech_slice.start)
                    overlap_end = min(speech_region.end, speech_slice.end)
                    if overlap_end <= overlap_start:
                        continue
                    relative_start = max(0, overlap_start - speech_slice.start)
                    relative_end = min(len(enhanced), overlap_end - speech_slice.start)
                    if relative_end > relative_start:
                        enhanced_chunks.append(enhanced[relative_start:relative_end])
                gain_note = f", gain scale {gain_scale:.3f}" if gain_mode != "none" else ""
                print(
                    f"Enhanced slice {index}/{len(prepared.speech_slices)} "
                    f"({len(chunk) / prepared.sample_rate:.2f}s{gain_note})"
                )

        output_audio = np.concatenate(enhanced_chunks).astype(np.float32, copy=False)
        if output_mode != "none":
            if output_target_dbfs is None:
                raise RuntimeError("Output normalization target was not initialized.")
            output_audio, output_gain, normalize_note = normalize_output_audio(
                output_audio,
                mode=output_mode,
                target_dbfs=output_target_dbfs,
            )
            print(
                f"Output normalization: {output_mode}, "
                f"gain scale {output_gain:.3f} ({normalize_note})"
            )
        output_audio = np.clip(output_audio, -1.0, 1.0)
        sf = require_soundfile()
        sf.write(str(prepared.output_path), output_audio, prepared.sample_rate, subtype="PCM_16")
        print(
            f"Wrote: {prepared.output_path} "
            f"({len(output_audio) / prepared.sample_rate:.2f}s, {prepared.sample_rate} Hz mono)"
        )
        return True
    except Exception as exc:
        print(f"Failed: {prepared.input_path}: {exc}", file=sys.stderr)
        return False


def enhance_file(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    enhancer,
    vad_backend: str,
) -> bool:
    try:
        prepared = prepare_file(input_path, output_path, args, vad_backend)
    except Exception as exc:
        print(f"Failed: {input_path}: {exc}", file=sys.stderr)
        return False
    if prepared is None:
        return False
    return enhance_prepared_file(prepared, args, enhancer)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    if not input_path.exists():
        raise SystemExit(f"Input audio does not exist: {input_path}")
    if args.target_sample_rate <= 0:
        raise SystemExit("--target-sample-rate must be greater than 0.")
    if args.max_slice_seconds <= 0:
        raise SystemExit("--max-slice-seconds must be greater than 0.")
    if args.silence_search_min_seconds <= 0:
        raise SystemExit("--silence-search-min-seconds must be greater than 0.")
    if args.silence_search_max_seconds <= args.silence_search_min_seconds:
        raise SystemExit("--silence-search-max-seconds must be greater than --silence-search-min-seconds.")
    if args.fallback_slice_seconds <= 0:
        raise SystemExit("--fallback-slice-seconds must be greater than 0.")
    if args.preprocess_workers <= 0:
        raise SystemExit("--preprocess-workers must be greater than 0.")
    if args.min_speech_seconds <= 0:
        raise SystemExit("--min-speech-seconds must be greater than 0.")
    args.normalize = parse_normalize(args.normalize)
    if args.normalize == "library_median" and args.alignment_metric != "rms":
        raise SystemExit("--normalize library_median requires --alignment-metric rms.")

    vad_backend = choose_vad_backend(args.vad_backend)
    print(f"Using VAD backend: {vad_backend}")

    if input_path.is_dir():
        source_files = iter_audio_files(input_path, args.allowed_extensions, args.recursive)
        if not source_files:
            raise SystemExit(f"No audio files found under: {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Found {len(source_files)} audio files under {input_path}")

        if args.normalize == "library_median":
            args.normalize = estimate_target_rms_dbfs(source_files, args, vad_backend)

        enhancer = build_enhancer(args.model, args.device)
        succeeded = 0
        failed = 0
        if args.preprocess_workers == 1:
            for index, source_file in enumerate(source_files, start=1):
                destination = output_path_for_input(source_file, input_path, output_path)
                print(f"\n[{index}/{len(source_files)}] {source_file} -> {destination}")
                if enhance_file(source_file, destination, args, enhancer, vad_backend):
                    succeeded += 1
                else:
                    failed += 1
        else:
            print(f"Preparing files with {args.preprocess_workers} background worker(s)")
            with ThreadPoolExecutor(max_workers=args.preprocess_workers) as executor:
                future_to_job = {}
                next_index = 1
                max_pending = max(1, args.preprocess_workers * 2)

                def submit_next() -> bool:
                    nonlocal next_index
                    if next_index > len(source_files):
                        return False
                    source_file = source_files[next_index - 1]
                    destination = output_path_for_input(source_file, input_path, output_path)
                    future = executor.submit(prepare_file, source_file, destination, args, vad_backend)
                    future_to_job[future] = (next_index, source_file, destination)
                    next_index += 1
                    return True

                while len(future_to_job) < max_pending and submit_next():
                    pass

                while future_to_job:
                    done, _pending = wait(future_to_job, return_when=FIRST_COMPLETED)
                    for future in done:
                        index, source_file, destination = future_to_job.pop(future)
                        while len(future_to_job) < max_pending and submit_next():
                            pass

                        print(f"\n[{index}/{len(source_files)}] {source_file} -> {destination}")
                        try:
                            prepared = future.result()
                        except Exception as exc:
                            print(f"Failed: {source_file}: {exc}", file=sys.stderr)
                            failed += 1
                            continue
                        if prepared is None:
                            failed += 1
                            continue
                        if enhance_prepared_file(prepared, args, enhancer):
                            succeeded += 1
                        else:
                            failed += 1

        print(f"\nDone. Succeeded: {succeeded}; failed/skipped: {failed}; output: {output_path}")
        if failed:
            raise SystemExit(1)
        return

    if args.normalize == "library_median":
        args.normalize = estimate_target_rms_dbfs([input_path], args, vad_backend)

    enhancer = build_enhancer(args.model, args.device)
    if output_path.exists() and output_path.is_dir():
        output_path = output_path / input_path.with_suffix(".wav").name
    if not enhance_file(input_path, output_path, args, enhancer, vad_backend):
        raise SystemExit(1)


if __name__ == "__main__":
    # Avoid oversubscribing CPU threads when ModelScope uses torch internally.
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
