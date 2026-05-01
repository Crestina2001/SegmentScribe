#!/usr/bin/env python
"""Benchmark Qwen3-ASR plus Qwen3-ForcedAligner throughput.

The parent process sweeps candidate configurations in subprocesses so vLLM
startup, CUDA graphs, and failed/OOM candidates do not poison the next run.
Each child reports ASR-only time, forced-aligner-only time, and combined
inference throughput on fixed-duration chunks cut from real audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ASR_SAMPLE_RATE = 16000
RESULT_PREFIX = "RESULT_JSON:"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class Candidate:
    name: str
    backend: str
    batch_size: int
    chunk_sec: float
    backend_kwargs: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep Qwen3-ASR + forced-aligner throughput configurations."
    )
    parser.add_argument("--input", default="audios/tian_raw_denoised")
    parser.add_argument("--model-path", default="checkpoints/Qwen3-ASR-1.7B")
    parser.add_argument("--aligner-path", default="checkpoints/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--language", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--chunk-seconds", default="30,60")
    parser.add_argument("--backends", default="vllm,transformers")
    parser.add_argument("--max-chunks", type=int, default=16)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    parser.add_argument("--keep-going-after-fail", action="store_true")
    parser.add_argument(
        "--vllm-profiles",
        default="default,fast,wide",
        help="Comma-separated built-in vLLM profiles: default, fast, wide, eager.",
    )
    parser.add_argument(
        "--candidate-jsonl",
        default=None,
        help=(
            "Optional JSONL file with explicit candidates. Each row may contain "
            "name, backend, batch_size, chunk_sec, backend_kwargs."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Defaults to test_scripts/logs/qwen3_asr_aligner_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--_child-candidate", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args._child_candidate:
        return run_child(args)
    return run_parent(args)


def run_parent(args: argparse.Namespace) -> int:
    candidates = load_candidates(args)
    if not candidates:
        raise SystemExit("No candidates to benchmark.")

    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else REPO_ROOT
        / "test_scripts"
        / "logs"
        / f"qwen3_asr_aligner_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = log_dir / "results.jsonl"
    results_csv = log_dir / "results.csv"

    print(f"Writing benchmark logs to: {log_dir}")
    print(f"Candidates: {len(candidates)}")

    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"\n[{index}/{len(candidates)}] {candidate.name}")
        result = run_candidate(args, candidate, log_dir)
        results.append(result)
        append_jsonl(results_jsonl, result)
        print(format_result(result))
        if not result.get("ok") and not args.keep_going_after_fail:
            print("Stopping after first failed candidate. Pass --keep-going-after-fail to continue.")
            break

    write_csv(results_csv, results)
    successful = [result for result in results if result.get("ok")]
    best = max(successful, key=lambda item: item["inference_audio_hours_per_hour"], default=None)
    summary = {
        "best": best,
        "result_count": len(results),
        "successful_count": len(successful),
        "results_jsonl": str(results_jsonl),
        "results_csv": str(results_csv),
        "log_dir": str(log_dir),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if best else 1


def run_candidate(args: argparse.Namespace, candidate: Candidate, log_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input",
        args.input,
        "--model-path",
        args.model_path,
        "--aligner-path",
        args.aligner_path,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-chunks",
        str(args.max_chunks),
        "--warmup-batches",
        str(args.warmup_batches),
        "--_child-candidate",
        candidate_to_json(candidate),
    ]
    if args.language:
        command.extend(["--language", args.language])

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else str(REPO_ROOT) + os.pathsep + existing_pythonpath
    )
    started = time.perf_counter()
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
        output = completed.stdout or ""
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        output += f"\nTimed out after {args.timeout_sec} seconds.\n"
        returncode = 124

    elapsed = time.perf_counter() - started
    log_path = log_dir / f"{safe_filename(candidate.name)}.log"
    log_path.write_text(output, encoding="utf-8", errors="replace")
    child = parse_child_result(output) or {
        "ok": False,
        "error_type": "subprocess_error",
        "error": f"exit code {returncode}; no {RESULT_PREFIX} line found",
    }
    child.update(
        {
            "returncode": returncode,
            "subprocess_elapsed_sec": round(elapsed, 3),
            "log_path": str(log_path),
        }
    )
    if returncode != 0:
        child["ok"] = False
    return child


def run_child(args: argparse.Namespace) -> int:
    candidate = candidate_from_json(args._child_candidate or "{}")
    started = time.perf_counter()
    try:
        audio_chunks = load_audio_chunks(Path(args.input), candidate.chunk_sec, args.max_chunks)
        if not audio_chunks:
            raise ValueError(f"No audio chunks found under {args.input}")

        load_started = time.perf_counter()
        model, aligner, torch = load_models(args, candidate)
        load_sec = time.perf_counter() - load_started

        warmup_sec = 0.0
        if args.warmup_batches > 0:
            warmup_windows = audio_chunks[: min(candidate.batch_size, len(audio_chunks))]
            warmup_started = time.perf_counter()
            run_asr(model, warmup_windows, args.language)
            warmup_sec = time.perf_counter() - warmup_started
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        asr_started = time.perf_counter()
        asr_results = []
        for batch in batched(audio_chunks, candidate.batch_size):
            asr_results.extend(run_asr(model, batch, args.language))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        asr_sec = time.perf_counter() - asr_started

        align_started = time.perf_counter()
        aligned_count = 0
        for chunk_batch, result_batch in zip(
            batched(audio_chunks, candidate.batch_size),
            batched(asr_results, candidate.batch_size),
        ):
            texts = [str(getattr(result, "text", "") or "") for result in result_batch]
            languages = [
                str(getattr(result, "language", "") or args.language or "")
                for result in result_batch
            ]
            non_empty = [
                (chunk, text, language)
                for chunk, text, language in zip(chunk_batch, texts, languages)
                if text.strip()
            ]
            if not non_empty:
                continue
            aligner.align(
                audio=[item[0] for item in non_empty],
                text=[item[1] for item in non_empty],
                language=[item[2] for item in non_empty],
            )
            aligned_count += len(non_empty)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        align_sec = time.perf_counter() - align_started

        audio_sec = sum(len(audio) / ASR_SAMPLE_RATE for audio, _sr in audio_chunks)
        inference_sec = asr_sec + align_sec
        result = {
            "ok": True,
            "candidate": candidate_to_dict(candidate),
            "chunk_count": len(audio_chunks),
            "aligned_count": aligned_count,
            "audio_sec": round(audio_sec, 3),
            "load_sec": round(load_sec, 3),
            "warmup_sec": round(warmup_sec, 3),
            "asr_sec": round(asr_sec, 3),
            "align_sec": round(align_sec, 3),
            "inference_sec": round(inference_sec, 3),
            "end_to_end_sec": round(time.perf_counter() - started, 3),
            "asr_audio_hours_per_hour": round(audio_sec / max(asr_sec, 1e-9), 3),
            "align_audio_hours_per_hour": round(audio_sec / max(align_sec, 1e-9), 3),
            "inference_audio_hours_per_hour": round(audio_sec / max(inference_sec, 1e-9), 3),
        }
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        result = {
            "ok": False,
            "candidate": candidate_to_dict(candidate),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "end_to_end_sec": round(time.perf_counter() - started, 3),
        }
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 2


def load_models(args: argparse.Namespace, candidate: Candidate) -> tuple[Any, Any, Any]:
    import torch
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

    from slicing_utils.asr import resolve_model_path

    model_path = resolve_model_path(args.model_path, label="ASR model")
    aligner_path = resolve_model_path(args.aligner_path, label="forced aligner")
    torch_dtype = getattr(torch, args.dtype)
    backend_kwargs = dict(candidate.backend_kwargs)
    if candidate.backend == "vllm":
        model = Qwen3ASRModel.LLM(
            model=model_path,
            forced_aligner=None,
            max_inference_batch_size=candidate.batch_size,
            max_new_tokens=args.max_new_tokens,
            **backend_kwargs,
        )
    else:
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            forced_aligner=None,
            dtype=torch_dtype,
            device_map=args.device,
            max_inference_batch_size=candidate.batch_size,
            max_new_tokens=args.max_new_tokens,
            **backend_kwargs,
        )
    aligner = Qwen3ForcedAligner.from_pretrained(aligner_path)
    return model, aligner, torch


def run_asr(model: Any, windows: Sequence[tuple[np.ndarray, int]], language: str | None) -> list[Any]:
    return list(model.transcribe(audio=list(windows), language=language, return_time_stamps=False))


def load_audio_chunks(input_path: Path, chunk_sec: float, max_chunks: int) -> list[tuple[np.ndarray, int]]:
    import librosa

    files = list(iter_audio_files(input_path))
    chunks: list[tuple[np.ndarray, int]] = []
    chunk_samples = max(1, int(chunk_sec * ASR_SAMPLE_RATE))
    for path in files:
        audio, _ = librosa.load(str(path), sr=ASR_SAMPLE_RATE, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            if len(chunk) < ASR_SAMPLE_RATE:
                continue
            chunks.append((np.ascontiguousarray(chunk, dtype=np.float32), ASR_SAMPLE_RATE))
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def iter_audio_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def load_candidates(args: argparse.Namespace) -> list[Candidate]:
    if args.candidate_jsonl:
        rows = [
            json.loads(line)
            for line in Path(args.candidate_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [candidate_from_mapping(row) for row in rows]

    batch_sizes = parse_int_list(args.batch_sizes)
    chunk_secs = parse_float_list(args.chunk_seconds)
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    vllm_profiles = [item.strip() for item in args.vllm_profiles.split(",") if item.strip()]
    candidates: list[Candidate] = []
    for backend in backends:
        if backend == "vllm":
            for profile in vllm_profiles:
                kwargs = vllm_profile_kwargs(profile)
                for batch_size in batch_sizes:
                    for chunk_sec in chunk_secs:
                        candidates.append(
                            Candidate(
                                name=f"vllm_{profile}_b{batch_size}_c{int(chunk_sec)}",
                                backend=backend,
                                batch_size=batch_size,
                                chunk_sec=chunk_sec,
                                backend_kwargs=kwargs,
                            )
                        )
        elif backend == "transformers":
            for batch_size in batch_sizes:
                for chunk_sec in chunk_secs:
                    candidates.append(
                        Candidate(
                            name=f"transformers_b{batch_size}_c{int(chunk_sec)}",
                            backend=backend,
                            batch_size=batch_size,
                            chunk_sec=chunk_sec,
                            backend_kwargs={},
                        )
                    )
        else:
            raise SystemExit(f"Unsupported backend in --backends: {backend}")
    return candidates


def vllm_profile_kwargs(profile: str) -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {
        "default": {"max_model_len": 8192},
        "fast": {
            "gpu_memory_utilization": 0.92,
            "max_model_len": 8192,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 32,
        },
        "wide": {
            "gpu_memory_utilization": 0.95,
            "max_model_len": 8192,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 64,
        },
        "eager": {
            "gpu_memory_utilization": 0.92,
            "max_model_len": 8192,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 32,
            "enforce_eager": True,
        },
    }
    if profile not in profiles:
        raise SystemExit(f"Unknown vLLM profile: {profile}")
    return dict(profiles[profile])


def candidate_from_mapping(row: dict[str, Any]) -> Candidate:
    return Candidate(
        name=str(row.get("name") or f"{row['backend']}_b{row['batch_size']}_c{row['chunk_sec']}"),
        backend=str(row["backend"]),
        batch_size=int(row["batch_size"]),
        chunk_sec=float(row["chunk_sec"]),
        backend_kwargs=dict(row.get("backend_kwargs") or {}),
    )


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "backend": candidate.backend,
        "batch_size": candidate.batch_size,
        "chunk_sec": candidate.chunk_sec,
        "backend_kwargs": candidate.backend_kwargs,
    }


def candidate_to_json(candidate: Candidate) -> str:
    return json.dumps(candidate_to_dict(candidate), separators=(",", ":"))


def candidate_from_json(raw: str) -> Candidate:
    return candidate_from_mapping(json.loads(raw))


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def batched(items: Sequence[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def parse_child_result(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    return None


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    fields = [
        "ok",
        "name",
        "backend",
        "batch_size",
        "chunk_sec",
        "chunk_count",
        "audio_sec",
        "load_sec",
        "warmup_sec",
        "asr_sec",
        "align_sec",
        "inference_sec",
        "inference_audio_hours_per_hour",
        "error_type",
        "error",
        "log_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            candidate = result.get("candidate") or {}
            row = {
                "ok": result.get("ok"),
                "name": candidate.get("name"),
                "backend": candidate.get("backend"),
                "batch_size": candidate.get("batch_size"),
                "chunk_sec": candidate.get("chunk_sec"),
                "chunk_count": result.get("chunk_count"),
                "audio_sec": result.get("audio_sec"),
                "load_sec": result.get("load_sec"),
                "warmup_sec": result.get("warmup_sec"),
                "asr_sec": result.get("asr_sec"),
                "align_sec": result.get("align_sec"),
                "inference_sec": result.get("inference_sec"),
                "inference_audio_hours_per_hour": result.get("inference_audio_hours_per_hour"),
                "error_type": result.get("error_type"),
                "error": result.get("error"),
                "log_path": result.get("log_path"),
            }
            writer.writerow(row)


def format_result(result: dict[str, Any]) -> str:
    candidate = result.get("candidate") or {}
    if not result.get("ok"):
        return (
            f"FAIL {candidate.get('name')} "
            f"{result.get('error_type')}: {result.get('error')} "
            f"log={result.get('log_path')}"
        )
    return (
        f"OK {candidate.get('name')} "
        f"throughput={result['inference_audio_hours_per_hour']}x realtime "
        f"asr={result['asr_sec']}s align={result['align_sec']}s "
        f"chunks={result['chunk_count']}"
    )


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


if __name__ == "__main__":
    raise SystemExit(main())
