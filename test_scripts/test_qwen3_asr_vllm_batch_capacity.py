#!/usr/bin/env python
"""Probe the largest Qwen3-ASR vLLM batch size this machine can run.

By default the model is loaded once and all candidate batch sizes are tested in
the same process, so reported timings exclude repeated vLLM startup cost. A
subprocess mode is also available when crash/OOM isolation matters more.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ASR_SAMPLE_RATE = 16000
RESULT_PREFIX = "RESULT_JSON:"
DEFAULT_MAX_MODEL_LEN = 8192

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the maximum Qwen3-ASR vLLM inference batch size that succeeds."
    )
    parser.add_argument("--model-path", default="checkpoints/Qwen3-ASR-1.7B")
    parser.add_argument("--aligner-path", default="checkpoints/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default=None)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=("single-process", "subprocess"),
        default="single-process",
        help=(
            "single-process loads vLLM once and removes startup from per-batch "
            "timings; subprocess isolates each candidate but repeats startup."
        ),
    )
    parser.add_argument("--search", choices=("linear", "binary"), default="linear")
    parser.add_argument("--audio-seconds", type=float, default=30.0)
    parser.add_argument("--audio-file", default=None, help="Optional probe audio file.")
    parser.add_argument(
        "--probe-audio",
        choices=("tone", "silence", "noise"),
        default="tone",
        help="Synthetic probe audio used when --audio-file is not provided.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--warmup-batch-size",
        type=int,
        default=1,
        help="Batch size used for the warmup call in single-process mode. Use 0 to skip.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=(
            "vLLM max_model_len to use unless --backend-kwargs already includes "
            f"max_model_len. Use 0 to leave vLLM's model default unchanged."
        ),
    )
    parser.add_argument("--backend-kwargs", default="{}")
    parser.add_argument("--forced-aligner-kwargs", default="{}")
    parser.add_argument("--timeout-sec", type=float, default=900.0)
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Defaults to test_scripts/logs/qwen3_asr_vllm_batch_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--keep-going-after-fail",
        action="store_true",
        help="With linear search, continue testing larger sizes after a failure.",
    )
    parser.add_argument("--_child-batch-size", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args._child_batch_size is not None:
        return run_child_probe(args)
    return run_parent_probe(args)


def run_parent_probe(args: argparse.Namespace) -> int:
    if args.min_batch_size <= 0 or args.max_batch_size <= 0:
        raise SystemExit("Batch sizes must be positive.")
    if args.min_batch_size > args.max_batch_size:
        raise SystemExit("--min-batch-size must be <= --max-batch-size.")

    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else REPO_ROOT
        / "test_scripts"
        / "logs"
        / f"qwen3_asr_vllm_batch_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_dir / "results.jsonl"
    backend_kwargs = effective_backend_kwargs(args)

    print(f"Writing probe logs to: {log_dir}")
    print(f"Testing Qwen3-ASR vLLM batch capacity up to {args.max_batch_size}")
    print(f"Mode: {args.mode}")
    print(f"Effective vLLM backend kwargs: {json.dumps(backend_kwargs, sort_keys=True)}")

    if args.mode == "single-process":
        results, startup = run_single_process_probe(args, jsonl_path)
    else:
        results = run_subprocess_probe(args, log_dir, jsonl_path)
        startup = None

    successful = [item["batch_size"] for item in results if item["ok"]]
    failed = [item["batch_size"] for item in results if not item["ok"]]
    max_success = max(successful) if successful else None
    summary = {
        "max_successful_batch_size": max_success,
        "successful_batch_sizes": sorted(successful),
        "failed_batch_sizes": sorted(failed),
        "audio_seconds": args.audio_seconds,
        "max_new_tokens": args.max_new_tokens,
        "backend_kwargs": backend_kwargs,
        "mode": args.mode,
        "startup": startup,
        "log_dir": str(log_dir),
        "results_jsonl": str(jsonl_path),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("Summary:")
    print(json.dumps(summary, indent=2))
    return 0 if max_success is not None else 1


def run_subprocess_probe(
    args: argparse.Namespace,
    log_dir: Path,
    jsonl_path: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if args.search == "linear":
        for batch_size in range(args.min_batch_size, args.max_batch_size + 1):
            result = run_candidate(args, batch_size, log_dir)
            results.append(result)
            append_jsonl(jsonl_path, result)
            print(format_result(result))
            if not result["ok"] and not args.keep_going_after_fail:
                break
        return results

    low = args.min_batch_size
    high = args.max_batch_size
    best = None
    while low <= high:
        batch_size = (low + high) // 2
        result = run_candidate(args, batch_size, log_dir)
        results.append(result)
        append_jsonl(jsonl_path, result)
        print(format_result(result))
        if result["ok"]:
            best = batch_size
            low = batch_size + 1
        else:
            high = batch_size - 1
    if best is not None and all(item["batch_size"] != args.min_batch_size for item in results):
        result = run_candidate(args, args.min_batch_size, log_dir)
        results.append(result)
        append_jsonl(jsonl_path, result)
        print(format_result(result))
    return results


def run_single_process_probe(
    args: argparse.Namespace,
    jsonl_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from slicing_utils.asr import AsrBackend

    backend_kwargs = effective_backend_kwargs(args)
    forced_aligner_kwargs = parse_json_dict(
        args.forced_aligner_kwargs,
        "--forced-aligner-kwargs",
    )
    audio = load_probe_audio(args)
    max_engine_batch_size = args.max_batch_size

    print("Loading vLLM engine once...")
    load_started = time.perf_counter()
    asr = AsrBackend(
        model_path=args.model_path,
        aligner_path=args.aligner_path,
        backend="vllm",
        device=args.device,
        max_inference_batch_size=max_engine_batch_size,
        max_new_tokens=args.max_new_tokens,
        language=args.language,
        backend_kwargs=backend_kwargs,
        forced_aligner_kwargs=forced_aligner_kwargs,
    )
    load_elapsed = time.perf_counter() - load_started
    print(f"Loaded vLLM engine in {load_elapsed:.3f}s")

    warmup: dict[str, Any] | None = None
    if args.warmup_batch_size > 0:
        warmup_size = min(args.warmup_batch_size, max_engine_batch_size)
        print(f"Warmup batch={warmup_size} ...")
        warmup = run_loaded_candidate(asr, audio, warmup_size, is_warmup=True)
        print(format_result(warmup).replace("OK ", "WARMUP OK ", 1))
        if not warmup["ok"]:
            return [warmup], {"load_elapsed_sec": round(load_elapsed, 3), "warmup": warmup}

    results: list[dict[str, Any]] = []
    if args.search == "linear":
        candidates = range(args.min_batch_size, args.max_batch_size + 1)
        for batch_size in candidates:
            result = run_loaded_candidate(asr, audio, batch_size)
            results.append(result)
            append_jsonl(jsonl_path, result)
            print(format_result(result))
            if not result["ok"] and not args.keep_going_after_fail:
                break
    else:
        low = args.min_batch_size
        high = args.max_batch_size
        while low <= high:
            batch_size = (low + high) // 2
            result = run_loaded_candidate(asr, audio, batch_size)
            results.append(result)
            append_jsonl(jsonl_path, result)
            print(format_result(result))
            if result["ok"]:
                low = batch_size + 1
            else:
                high = batch_size - 1

    startup = {
        "load_elapsed_sec": round(load_elapsed, 3),
        "warmup": warmup,
    }
    return results, startup


def run_loaded_candidate(
    asr: Any,
    audio: np.ndarray,
    batch_size: int,
    *,
    is_warmup: bool = False,
) -> dict[str, Any]:
    windows = [(audio.copy(), ASR_SAMPLE_RATE) for _ in range(batch_size)]
    started = time.perf_counter()
    try:
        transcripts = asr.transcribe_windows(windows)
        elapsed = time.perf_counter() - started
        ok = len(transcripts) == batch_size
        return {
            "ok": ok,
            "batch_size": batch_size,
            "transcript_count": len(transcripts),
            "elapsed_sec": round(elapsed, 3),
            "per_item_sec": round(elapsed / batch_size, 3),
            "is_warmup": is_warmup,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "batch_size": batch_size,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(elapsed, 3),
            "per_item_sec": None,
            "is_warmup": is_warmup,
        }


def run_candidate(args: argparse.Namespace, batch_size: int, log_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-path",
        args.model_path,
        "--aligner-path",
        args.aligner_path,
        "--device",
        args.device,
        "--max-batch-size",
        str(args.max_batch_size),
        "--min-batch-size",
        str(args.min_batch_size),
        "--audio-seconds",
        str(args.audio_seconds),
        "--probe-audio",
        args.probe_audio,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--backend-kwargs",
        args.backend_kwargs,
        "--forced-aligner-kwargs",
        args.forced_aligner_kwargs,
        "--_child-batch-size",
        str(batch_size),
    ]
    if args.language:
        command.extend(["--language", args.language])
    if args.audio_file:
        command.extend(["--audio-file", args.audio_file])

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else str(REPO_ROOT) + os.pathsep + existing_pythonpath
    )

    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_sec,
        )
        returncode = completed.returncode
        output = completed.stdout or ""
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        output += f"\nTimed out after {args.timeout_sec} seconds.\n"
    elapsed = time.time() - started
    log_path = log_dir / f"batch_{batch_size:04d}.log"
    log_path.write_text(output, encoding="utf-8", errors="replace")

    child_result = parse_child_result(output)
    if child_result is None:
        child_result = {
            "ok": False,
            "error_type": "subprocess_error",
            "error": f"exit code {returncode}; no {RESULT_PREFIX} line found",
        }
    root_cause = extract_root_cause(output)
    if root_cause:
        child_result["root_cause"] = root_cause

    child_result.update(
        {
            "batch_size": batch_size,
            "returncode": returncode,
            "elapsed_sec": round(elapsed, 3),
            "log_path": str(log_path),
        }
    )
    if returncode != 0:
        child_result["ok"] = False
    return child_result


def run_child_probe(args: argparse.Namespace) -> int:
    batch_size = int(args._child_batch_size)
    started = time.time()
    try:
        from slicing_utils.asr import AsrBackend

        backend_kwargs = effective_backend_kwargs(args)
        forced_aligner_kwargs = parse_json_dict(
            args.forced_aligner_kwargs,
            "--forced-aligner-kwargs",
        )
        audio = load_probe_audio(args)
        windows = [(audio.copy(), ASR_SAMPLE_RATE) for _ in range(batch_size)]
        asr = AsrBackend(
            model_path=args.model_path,
            aligner_path=args.aligner_path,
            backend="vllm",
            device=args.device,
            max_inference_batch_size=batch_size,
            max_new_tokens=args.max_new_tokens,
            language=args.language,
            backend_kwargs=backend_kwargs,
            forced_aligner_kwargs=forced_aligner_kwargs,
        )
        transcripts = asr.transcribe_windows(windows)
        result = {
            "ok": len(transcripts) == batch_size,
            "transcript_count": len(transcripts),
            "audio_seconds": round(len(audio) / ASR_SAMPLE_RATE, 3),
            "backend_kwargs": backend_kwargs,
            "elapsed_sec": round(time.time() - started, 3),
        }
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 0 if result["ok"] else 2
    except Exception as exc:
        result = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(time.time() - started, 3),
        }
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 2


def load_probe_audio(args: argparse.Namespace) -> np.ndarray:
    if args.audio_file:
        import librosa

        audio, _sample_rate = librosa.load(
            args.audio_file,
            sr=ASR_SAMPLE_RATE,
            mono=True,
            duration=args.audio_seconds,
        )
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise ValueError(f"Probe audio file is empty: {args.audio_file}")
        return audio

    sample_count = max(1, int(args.audio_seconds * ASR_SAMPLE_RATE))
    if args.probe_audio == "silence":
        return np.zeros(sample_count, dtype=np.float32)
    if args.probe_audio == "noise":
        rng = np.random.default_rng(seed=1234)
        return rng.normal(0.0, 0.01, sample_count).astype(np.float32)

    t = np.arange(sample_count, dtype=np.float32) / ASR_SAMPLE_RATE
    return (0.03 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


def parse_json_dict(raw_value: str, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return parsed


def effective_backend_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    backend_kwargs = parse_json_dict(args.backend_kwargs, "--backend-kwargs")
    if args.max_model_len > 0 and "max_model_len" not in backend_kwargs:
        backend_kwargs["max_model_len"] = int(args.max_model_len)
    return backend_kwargs


def parse_child_result(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    return None


def extract_root_cause(output: str) -> str | None:
    lines = [strip_ansi(line).strip() for line in output.splitlines()]
    needles = (
        "ValueError:",
        "torch.OutOfMemoryError:",
        "CUDA out of memory",
        "RuntimeError: CUDA",
    )
    matches = [line for line in lines if any(needle in line for needle in needles)]
    if matches:
        return matches[-1]
    for line in reversed(lines):
        if "ERROR" in line and line:
            return line
    return None


def strip_ansi(value: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def format_result(result: dict[str, Any]) -> str:
    status = "OK" if result["ok"] else "FAIL"
    detail = f"{status} batch={result['batch_size']} elapsed={result['elapsed_sec']}s"
    if result.get("per_item_sec") is not None:
        detail += f" per_item={result['per_item_sec']}s"
    if not result["ok"]:
        detail += f" error={result.get('error_type')}: {result.get('error')}"
        if result.get("root_cause"):
            detail += f" root_cause={result['root_cause']}"
    return detail


if __name__ == "__main__":
    raise SystemExit(main())
