"""Command-line entry point for the LLM-driven slide pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from llm_gateway.models import ProviderName
from slide_rule.config import AUDIO_EXTENSIONS

from .config import LLMWorkflowConfig


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
PROVIDERS: tuple[ProviderName, ...] = ("openai", "minimax", "anthropic", "gemini", "deepseek")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slide_LLM",
        description="LLM-driven slide slicing: pre-pass, LLM punctuation, LLM rough cut, thin cut, write.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--aligner-path", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--punct-llm-model", default=None)
    parser.add_argument("--rough-llm-model", default=None)
    parser.add_argument("--llm-provider", choices=PROVIDERS, default=None)
    parser.add_argument("--env-path", default=None)
    parser.add_argument("--llm-max-rounds", type=int, default=5)
    parser.add_argument("--asr-backend", default="transformers", choices=("transformers", "vllm"))
    parser.add_argument("--asr-backend-kwargs", default="{}")
    parser.add_argument("--forced-aligner-kwargs", default="{}")
    parser.add_argument("--llm-concurrency", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--asr-max-batch-size", type=int, default=1)
    parser.add_argument("--min-seg-sec", type=float, default=3.0)
    parser.add_argument("--max-seg-sec", type=float, default=10.0)
    parser.add_argument("--language", default=None)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--max-source-seconds", type=float, default=30.0 * 60.0)
    parser.add_argument("--allowed-extensions", nargs="*", default=sorted(AUDIO_EXTENSIONS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preprocess-chunk-sec", type=float, default=30.0)
    parser.add_argument("--vad-backend", default="auto", choices=("auto", "silero", "librosa"))
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech-ms", type=int, default=250)
    parser.add_argument("--vad-min-silence-ms", type=int, default=300)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=200)
    return parser


def args_to_config(args: argparse.Namespace) -> LLMWorkflowConfig:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"--input path does not exist: {input_path}")
    if args.min_seg_sec <= 0 or args.max_seg_sec <= 0 or args.min_seg_sec > args.max_seg_sec:
        raise SystemExit("Invalid segment bounds: require 0 < min_seg_sec <= max_seg_sec.")
    if args.asr_max_batch_size <= 0:
        raise SystemExit("--asr-max-batch-size must be > 0.")
    if args.llm_concurrency <= 0:
        raise SystemExit("--llm-concurrency must be > 0.")
    if args.llm_max_rounds <= 0:
        raise SystemExit("--llm-max-rounds must be > 0.")

    return LLMWorkflowConfig(
        input_path=input_path,
        output_dir=output_dir,
        model_path=args.model_path,
        aligner_path=args.aligner_path,
        llm_model=args.llm_model,
        punct_llm_model=args.punct_llm_model,
        rough_llm_model=args.rough_llm_model,
        llm_provider=args.llm_provider,
        env_path=args.env_path,
        llm_max_rounds=args.llm_max_rounds,
        asr_backend=args.asr_backend,
        asr_backend_kwargs=_parse_json_dict(args.asr_backend_kwargs, name="--asr-backend-kwargs"),
        forced_aligner_kwargs=_parse_json_dict(args.forced_aligner_kwargs, name="--forced-aligner-kwargs"),
        llm_concurrency=args.llm_concurrency,
        device=args.device,
        dtype=args.dtype,
        asr_max_batch_size=args.asr_max_batch_size,
        min_seg_sec=args.min_seg_sec,
        max_seg_sec=args.max_seg_sec,
        language=args.language,
        target_sample_rate=args.target_sample_rate,
        max_source_seconds=args.max_source_seconds,
        allowed_extensions=tuple(args.allowed_extensions),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        preprocess_chunk_sec=args.preprocess_chunk_sec,
        vad_backend=args.vad_backend,
        vad_threshold=args.vad_threshold,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
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
    for name in ("slide_LLM", "slicing_utils"):
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
    from .pipeline import run_llm_workflow_sync

    summary = run_llm_workflow_sync(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
