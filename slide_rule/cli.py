"""Command-line entry point for the rule-based slide pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import AUDIO_EXTENSIONS, RuleWorkflowConfig


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slide_rule.cli",
        description="Rule-based slide slicing: pre-pass, rule punctuation, rough cut, thin cut, write.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--aligner-path", required=True)
    parser.add_argument("--asr-backend", default="transformers", choices=("transformers", "vllm"))
    parser.add_argument("--asr-backend-kwargs", default="{}")
    parser.add_argument("--forced-aligner-kwargs", default="{}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--asr-max-batch-size", type=int, default=8)
    parser.add_argument("--aligner-max-batch-size", type=int, default=1)
    parser.add_argument(
        "--aligner-concurrency",
        type=int,
        default=1,
        help="Number of forced-aligner instances allowed to run concurrently.",
    )
    parser.add_argument("--min-seg-sec", type=float, default=3.0)
    parser.add_argument("--max-seg-sec", type=float, default=10.0)
    parser.add_argument("--language", default=None)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--max-source-seconds", type=float, default=30.0 * 60.0)
    parser.add_argument("--allowed-extensions", nargs="*", default=sorted(AUDIO_EXTENSIONS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preprocess-chunk-sec", type=float, default=30.0)
    parser.add_argument("--preprocess-chunk-mode", default="rms_silence", choices=("vad", "rms_silence"))
    parser.add_argument("--preprocess-min-chunk-sec", type=float, default=5.0)
    parser.add_argument("--preprocess-max-chunk-sec", type=float, default=15.0)
    parser.add_argument("--rms-silence-frame-ms", type=float, default=25.0)
    parser.add_argument("--rms-silence-hop-ms", type=float, default=5.0)
    parser.add_argument("--rms-silence-percentile", type=float, default=25.0)
    parser.add_argument("--rms-silence-threshold-multiplier", type=float, default=1.8)
    parser.add_argument(
        "--rms-min-silence-ms",
        type=float,
        default=80.0,
        help="Deprecated compatibility option; RMS prepass now cuts at the best silent position without a minimum run length.",
    )
    parser.add_argument("--vad-backend", default="auto", choices=("auto", "silero", "librosa"))
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech-ms", type=int, default=250)
    parser.add_argument("--vad-min-silence-ms", type=int, default=300)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=200)
    parser.add_argument(
        "--rough-cut-strategy",
        default="priority_silence_v3",
        choices=("legacy_dp", "priority_silence_v1", "priority_silence_v2", "priority_silence_v3", "dp_strategy_2"),
        help="Rough-cut planner strategy. Default: priority_silence_v3.",
    )
    parser.add_argument(
        "--enable-punctuation-correction",
        action="store_true",
        help="Enable rule-based punctuation edits. Disabled by default.",
    )
    # Compatibility no-ops for existing command snippets.
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--punct-llm-model", default=None)
    parser.add_argument("--zoom-llm-model", default=None)
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--env-path", default=None)
    parser.add_argument("--punct-max-window-sec", type=float, default=None)
    return parser


def args_to_config(args: argparse.Namespace):
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"--input path does not exist: {input_path}")
    if args.min_seg_sec <= 0 or args.max_seg_sec <= 0 or args.min_seg_sec > args.max_seg_sec:
        raise SystemExit("Invalid segment bounds: require 0 < min_seg_sec <= max_seg_sec.")
    if args.asr_max_batch_size <= 0:
        raise SystemExit("--asr-max-batch-size must be > 0.")
    if args.aligner_max_batch_size <= 0:
        raise SystemExit("--aligner-max-batch-size must be > 0.")
    if args.aligner_concurrency <= 0:
        raise SystemExit("--aligner-concurrency must be > 0.")
    if (
        args.preprocess_min_chunk_sec <= 0
        or args.preprocess_max_chunk_sec <= 0
        or args.preprocess_min_chunk_sec > args.preprocess_max_chunk_sec
    ):
        raise SystemExit(
            "Invalid prepass chunk bounds: require 0 < preprocess_min_chunk_sec <= preprocess_max_chunk_sec."
        )

    return RuleWorkflowConfig(
        input_path=input_path,
        output_dir=output_dir,
        model_path=args.model_path,
        aligner_path=args.aligner_path,
        asr_backend=args.asr_backend,
        asr_backend_kwargs=_parse_json_dict(args.asr_backend_kwargs, name="--asr-backend-kwargs"),
        forced_aligner_kwargs=_parse_json_dict(args.forced_aligner_kwargs, name="--forced-aligner-kwargs"),
        device=args.device,
        dtype=args.dtype,
        asr_max_batch_size=args.asr_max_batch_size,
        aligner_max_batch_size=args.aligner_max_batch_size,
        aligner_concurrency=args.aligner_concurrency,
        min_seg_sec=args.min_seg_sec,
        max_seg_sec=args.max_seg_sec,
        language=args.language,
        target_sample_rate=args.target_sample_rate,
        max_source_seconds=args.max_source_seconds,
        allowed_extensions=tuple(args.allowed_extensions),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        preprocess_chunk_sec=args.preprocess_chunk_sec,
        preprocess_chunk_mode=args.preprocess_chunk_mode,
        preprocess_min_chunk_sec=args.preprocess_min_chunk_sec,
        preprocess_max_chunk_sec=args.preprocess_max_chunk_sec,
        rms_silence_frame_ms=args.rms_silence_frame_ms,
        rms_silence_hop_ms=args.rms_silence_hop_ms,
        rms_silence_percentile=args.rms_silence_percentile,
        rms_silence_threshold_multiplier=args.rms_silence_threshold_multiplier,
        rms_min_silence_ms=args.rms_min_silence_ms,
        vad_backend=args.vad_backend,
        vad_threshold=args.vad_threshold,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        enable_punctuation_correction=args.enable_punctuation_correction,
        rough_cut_strategy=args.rough_cut_strategy,
    )


def _parse_json_dict(raw_value: str, *, name: str) -> dict:
    try:
        parsed = json.loads(raw_value or "{}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be a valid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{name} must be a JSON object.")
    return parsed


def configure_logging() -> None:
    """Configure stderr-only runtime logs for slide_rule."""
    _configure_package_logger("slide_rule")
    _configure_package_logger("slicing_utils")
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)


def _configure_package_logger(name: str) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = args_to_config(args)
    configure_logging()
    from .pipeline import run_rule_workflow_sync

    summary = run_rule_workflow_sync(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
