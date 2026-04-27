#!/usr/bin/env python
"""Filter likely speaker outliers from a VoxCPM-style JSONL manifest.

The tool extracts SpeechBrain ECAPA speaker embeddings for each listed audio
segment, scores each segment against the dataset centroid, and writes a
filtered JSONL plus audit reports. It never copies or modifies audio files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = "checkpoints/spkrec-ecapa-voxceleb"
TARGET_SAMPLE_RATE = 16000
MAD_NORMAL_SCALE = 1.4826
EPSILON = 1e-12


@dataclass(frozen=True)
class JsonlRecord:
    line_number: int
    payload: dict[str, Any]
    audio_relpath: str
    audio_path: Path


@dataclass(frozen=True)
class ScoreRow:
    record: JsonlRecord
    similarity: float
    rank: int
    pruned: bool
    reason: str


@dataclass(frozen=True)
class ThresholdStats:
    median: float
    mad: float
    threshold: float
    threshold_prune_count: int
    capped_prune_count: int
    max_allowed_prune_count: int
    pruning_disabled_reason: str


EmbeddingExtractor = Callable[[Path], np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use SpeechBrain ECAPA embeddings to remove likely speaker outliers "
            "from a VoxCPM-style JSONL manifest."
        )
    )
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL manifest.")
    parser.add_argument("--output-jsonl", required=True, help="Filtered output JSONL.")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Dataset root for relative audio paths. Default: input JSONL parent.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device for SpeechBrain.")
    parser.add_argument("--model-source", default=MODEL_SOURCE, help="SpeechBrain model source.")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Local pretrained model directory.")
    parser.add_argument(
        "--mad-multiplier",
        type=float,
        default=3.0,
        help="Robust threshold multiplier. Default: 3.0.",
    )
    parser.add_argument(
        "--max-prune-ratio",
        type=float,
        default=0.10,
        help="Maximum fraction of rows to prune. Default: 0.10.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=8,
        help="Below this row count, write reports but do not prune. Default: 8.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and write reports, but do not write filtered/pruned JSONL files.",
    )
    return parser


def normalize_manifest_path(path_value: str) -> str:
    return str(Path(path_value)).replace("\\", "/")


def read_jsonl_records(jsonl_path: Path, dataset_root: Path) -> list[JsonlRecord]:
    records: list[JsonlRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {jsonl_path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise SystemExit(f"JSONL row {line_number} must be an object.")

            audio_value = payload.get("audio")
            if not isinstance(audio_value, str) or not audio_value:
                raise SystemExit(f"JSONL row {line_number} is missing a string 'audio' field.")

            audio_rel = Path(audio_value)
            if audio_rel.is_absolute():
                raise SystemExit(f"JSONL row {line_number} uses an absolute audio path: {audio_value}")
            if any(part == ".." for part in audio_rel.parts):
                raise SystemExit(f"JSONL row {line_number} uses an unsafe audio path: {audio_value}")

            audio_path = dataset_root / audio_rel
            if not audio_path.exists() or not audio_path.is_file():
                raise SystemExit(f"JSONL row {line_number} references missing audio: {audio_path}")

            records.append(
                JsonlRecord(
                    line_number=line_number,
                    payload=payload,
                    audio_relpath=normalize_manifest_path(audio_value),
                    audio_path=audio_path,
                )
            )

    if not records:
        raise SystemExit(f"No usable rows found in JSONL: {jsonl_path}")
    return records


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if not math.isfinite(norm) or norm <= EPSILON:
        raise ValueError("embedding has zero or non-finite norm")
    return flat / norm


def centroid_for_embeddings(embeddings: np.ndarray) -> np.ndarray:
    centroid = np.mean(embeddings, axis=0)
    return l2_normalize(centroid)


def score_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    centroid = centroid_for_embeddings(embeddings)
    return np.asarray(embeddings @ centroid, dtype=np.float32)


def decide_pruned_indices(
    similarities: Sequence[float],
    *,
    mad_multiplier: float,
    max_prune_ratio: float,
    min_rows: int,
) -> tuple[set[int], ThresholdStats]:
    scores = np.asarray(similarities, dtype=np.float32)
    if scores.size == 0:
        raise ValueError("similarities must be non-empty")

    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    threshold = float(median - mad_multiplier * MAD_NORMAL_SCALE * mad)
    below_threshold = [idx for idx, score in enumerate(scores) if float(score) < threshold]

    max_allowed = int(math.floor(scores.size * max_prune_ratio))
    if max_prune_ratio > 0 and scores.size > 0:
        max_allowed = max(1, max_allowed)
    max_allowed = min(max_allowed, scores.size)

    disabled_reason = ""
    if scores.size < min_rows:
        disabled_reason = f"row count {scores.size} is below min_rows {min_rows}"
        pruned: set[int] = set()
    elif max_allowed <= 0:
        disabled_reason = "max_prune_ratio allows zero rows"
        pruned = set()
    else:
        sorted_below = sorted(below_threshold, key=lambda idx: (float(scores[idx]), idx))
        pruned = set(sorted_below[:max_allowed])

    return pruned, ThresholdStats(
        median=median,
        mad=mad,
        threshold=threshold,
        threshold_prune_count=len(below_threshold),
        capped_prune_count=len(pruned),
        max_allowed_prune_count=max_allowed,
        pruning_disabled_reason=disabled_reason,
    )


def build_score_rows(
    records: Sequence[JsonlRecord],
    similarities: Sequence[float],
    pruned_indices: set[int],
) -> list[ScoreRow]:
    ranked_indices = sorted(range(len(similarities)), key=lambda idx: (float(similarities[idx]), idx))
    rank_by_index = {idx: rank for rank, idx in enumerate(ranked_indices, start=1)}
    rows: list[ScoreRow] = []
    for index, (record, similarity) in enumerate(zip(records, similarities, strict=True)):
        pruned = index in pruned_indices
        rows.append(
            ScoreRow(
                record=record,
                similarity=float(similarity),
                rank=rank_by_index[index],
                pruned=pruned,
                reason="below speaker similarity threshold" if pruned else "",
            )
        )
    return rows


def extract_embeddings(records: Sequence[JsonlRecord], extractor: EmbeddingExtractor) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    for offset, record in enumerate(records, start=1):
        embedding = l2_normalize(extractor(record.audio_path))
        embeddings.append(embedding)
        print(f"[{offset}/{len(records)}] embedded row {record.line_number}: {record.audio_relpath}")
    return np.stack(embeddings, axis=0).astype(np.float32, copy=False)


def make_speechbrain_extractor(
    *,
    model_source: str,
    model_dir: Path,
    device: str,
) -> EmbeddingExtractor:
    try:
        import librosa
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for speaker filtering. Install dependencies with:\n"
            "  pip install -r requirements.txt\n"
            "and make sure PyTorch is installed for your CUDA or CPU setup."
        ) from exc

    classifier = EncoderClassifier.from_hparams(
        source=model_source,
        savedir=str(model_dir),
        run_opts={"device": device},
    )

    def extract(path: Path) -> np.ndarray:
        audio, _ = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
        if audio.size == 0:
            raise ValueError("empty audio")
        signal = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = classifier.encode_batch(signal)
        return embedding.detach().cpu().numpy()

    return extract


def ensure_outputs_available(
    *,
    output_jsonl: Path,
    report_csv: Path,
    pruned_jsonl: Path,
    summary_json: Path,
    dry_run: bool,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    paths = [report_csv, summary_json]
    if not dry_run:
        paths.extend([output_jsonl, pruned_jsonl])
    existing = [path for path in paths if path.exists()]
    if existing:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise SystemExit(f"Output files already exist. Pass --overwrite to replace them.\n{rendered}")


def write_jsonl(path: Path, payloads: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_report_csv(path: Path, rows: Sequence[ScoreRow], threshold_stats: ThresholdStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = (
            "jsonl_row",
            "audio",
            "audio_path",
            "similarity",
            "rank_lowest_first",
            "threshold",
            "median",
            "mad",
            "pruned",
            "reason",
        )
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "jsonl_row": row.record.line_number,
                    "audio": row.record.audio_relpath,
                    "audio_path": str(row.record.audio_path),
                    "similarity": f"{row.similarity:.8f}",
                    "rank_lowest_first": row.rank,
                    "threshold": f"{threshold_stats.threshold:.8f}",
                    "median": f"{threshold_stats.median:.8f}",
                    "mad": f"{threshold_stats.mad:.8f}",
                    "pruned": row.pruned,
                    "reason": row.reason,
                }
            )


def pruned_payload(row: ScoreRow) -> dict[str, Any]:
    payload = dict(row.record.payload)
    payload["speaker_outlier_pruned"] = True
    payload["speaker_similarity"] = row.similarity
    payload["speaker_outlier_reason"] = row.reason
    return payload


def write_summary_json(
    path: Path,
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    dataset_root: Path,
    rows: Sequence[ScoreRow],
    threshold_stats: ThresholdStats,
    device: str,
    model_source: str,
    model_dir: Path,
    dry_run: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pruned_count = sum(1 for row in rows if row.pruned)
    summary = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "dataset_root": str(dataset_root),
        "row_count": len(rows),
        "kept_count": len(rows) - pruned_count,
        "pruned_count": pruned_count,
        "threshold": threshold_stats.threshold,
        "median": threshold_stats.median,
        "mad": threshold_stats.mad,
        "threshold_prune_count": threshold_stats.threshold_prune_count,
        "capped_prune_count": threshold_stats.capped_prune_count,
        "max_allowed_prune_count": threshold_stats.max_allowed_prune_count,
        "pruning_disabled_reason": threshold_stats.pruning_disabled_reason,
        "device": device,
        "model_source": model_source,
        "model_dir": str(model_dir),
        "dry_run": dry_run,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_filter(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    dataset_root: Path,
    device: str,
    model_source: str,
    model_dir: Path,
    mad_multiplier: float,
    max_prune_ratio: float,
    min_rows: int,
    dry_run: bool,
    overwrite: bool,
    extractor: EmbeddingExtractor | None = None,
) -> dict[str, Any]:
    report_csv = output_jsonl.with_name("speaker_similarity_report.csv")
    pruned_jsonl = output_jsonl.with_name("pruned_speaker_outliers.jsonl")
    summary_json = output_jsonl.with_name("speaker_filter_summary.json")

    ensure_outputs_available(
        output_jsonl=output_jsonl,
        report_csv=report_csv,
        pruned_jsonl=pruned_jsonl,
        summary_json=summary_json,
        dry_run=dry_run,
        overwrite=overwrite,
    )

    records = read_jsonl_records(input_jsonl, dataset_root)
    if extractor is None:
        extractor = make_speechbrain_extractor(
            model_source=model_source,
            model_dir=model_dir,
            device=device,
        )

    print(f"Embedding {len(records)} JSONL-listed audio file(s)...")
    embeddings = extract_embeddings(records, extractor)
    similarities = score_embeddings(embeddings)
    pruned_indices, threshold_stats = decide_pruned_indices(
        similarities,
        mad_multiplier=mad_multiplier,
        max_prune_ratio=max_prune_ratio,
        min_rows=min_rows,
    )
    rows = build_score_rows(records, similarities, pruned_indices)
    kept_rows = [row for row in rows if not row.pruned]
    pruned_rows = [row for row in rows if row.pruned]

    write_report_csv(report_csv, rows, threshold_stats)
    write_summary_json(
        summary_json,
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        dataset_root=dataset_root,
        rows=rows,
        threshold_stats=threshold_stats,
        device=device,
        model_source=model_source,
        model_dir=model_dir,
        dry_run=dry_run,
    )

    if not dry_run:
        write_jsonl(output_jsonl, [row.record.payload for row in kept_rows])
        if pruned_rows:
            write_jsonl(pruned_jsonl, [pruned_payload(row) for row in pruned_rows])

    return {
        "input_rows": len(rows),
        "kept_rows": len(kept_rows),
        "pruned_rows": len(pruned_rows),
        "threshold": threshold_stats.threshold,
        "median": threshold_stats.median,
        "mad": threshold_stats.mad,
        "report_csv": str(report_csv),
        "summary_json": str(summary_json),
        "output_jsonl": str(output_jsonl),
        "pruned_jsonl": str(pruned_jsonl) if pruned_rows and not dry_run else None,
        "dry_run": dry_run,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.mad_multiplier < 0:
        raise SystemExit("--mad-multiplier must be non-negative.")
    if args.max_prune_ratio < 0 or args.max_prune_ratio > 1:
        raise SystemExit("--max-prune-ratio must be between 0 and 1.")
    if args.min_rows < 1:
        raise SystemExit("--min-rows must be at least 1.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    if not input_jsonl.exists() or not input_jsonl.is_file():
        raise SystemExit(f"Input JSONL file does not exist: {input_jsonl}")

    output_jsonl = Path(args.output_jsonl).expanduser()
    if not output_jsonl.is_absolute():
        output_jsonl = (Path.cwd() / output_jsonl).resolve()
    if input_jsonl == output_jsonl:
        raise SystemExit("--output-jsonl must be different from --input-jsonl.")

    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root
        else input_jsonl.parent
    )
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    summary = run_filter(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        dataset_root=dataset_root,
        device=args.device,
        model_source=args.model_source,
        model_dir=Path(args.model_dir).expanduser(),
        mad_multiplier=args.mad_multiplier,
        max_prune_ratio=args.max_prune_ratio,
        min_rows=args.min_rows,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
