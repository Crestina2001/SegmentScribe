#!/usr/bin/env python
"""Compare two JSONL strategy outputs.

The default comparison is designed for VoxCPM-style rows such as:

    {"audio": "audios/001.wav", "text": "...", "duration": 3.42}

Rows are matched by ``audio`` by default. Use ``--key _line`` when two
strategies do not share stable IDs and you want a line-by-line comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LINE_KEY = "_line"
DEFAULT_IGNORE_FIELDS = {"audio"}
STRONG_ENDINGS = "。！？!?…"
WEAK_ENDINGS = "，、；：,"


@dataclass(frozen=True)
class JsonlRow:
    line_number: int
    key: str
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two JSONL files from different slicing/processing strategies."
    )
    parser.add_argument("left", help="Baseline JSONL file.")
    parser.add_argument("right", help="Candidate JSONL file.")
    parser.add_argument(
        "--left-name",
        default="left",
        help="Label for the baseline file in the report. Default: left.",
    )
    parser.add_argument(
        "--right-name",
        default="right",
        help="Label for the candidate file in the report. Default: right.",
    )
    parser.add_argument(
        "--key",
        default="audio",
        help="Field used to match rows. Use '_line' for line-by-line comparison. Default: audio.",
    )
    parser.add_argument(
        "--ignore-field",
        action="append",
        default=[],
        help=(
            "Field to ignore when detecting changed rows. Can be passed multiple times. "
            "The key field is ignored automatically."
        ),
    )
    parser.add_argument(
        "--float-tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for comparing numeric values. Default: 1e-6.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=8,
        help="Number of changed/added/removed examples to show. Default: 8.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Report format. Default: text.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the report. The report is still printed to stdout.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, key_field: str) -> tuple[list[JsonlRow], list[str]]:
    rows: list[JsonlRow] = []
    errors: list[str] = []
    seen = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc})")
                continue
            if not isinstance(payload, dict):
                errors.append(f"line {line_number}: expected object, got {type(payload).__name__}")
                continue

            if key_field == LINE_KEY:
                key = str(line_number)
            else:
                value = payload.get(key_field)
                if value is None:
                    errors.append(f"line {line_number}: missing key field '{key_field}'")
                    key = f"{LINE_KEY}:{line_number}"
                else:
                    key = str(value)

            seen[key] += 1
            if seen[key] > 1:
                key = f"{key}#{seen[key]}"
                errors.append(
                    f"line {line_number}: duplicate key, comparing this row as '{key}'"
                )
            rows.append(JsonlRow(line_number=line_number, key=key, payload=payload))

    return rows, errors


def almost_equal(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), abs_tol=tolerance)
    return left == right


def changed_fields(
    left: dict[str, Any],
    right: dict[str, Any],
    ignored: set[str],
    tolerance: float,
) -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}
    for field in sorted((set(left) | set(right)) - ignored):
        left_value = left.get(field)
        right_value = right.get(field)
        if not almost_equal(left_value, right_value, tolerance):
            changes[field] = {"left": left_value, "right": right_value}
    return changes


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = max(0, min(len(values) - 1, math.ceil((pct / 100.0) * len(values)) - 1))
    return values[index]


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "mean": None, "p95": None, "max": None}
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "mean": sum(values) / len(values),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def text_stats(rows: list[JsonlRow]) -> dict[str, Any]:
    texts = [str(row.payload.get("text", "")) for row in rows]
    lengths = [len(text) for text in texts]
    endings = Counter(text.strip()[-1] if text.strip() else "" for text in texts)
    strong_end = sum(1 for text in texts if text.strip().endswith(tuple(STRONG_ENDINGS)))
    weak_end = sum(1 for text in texts if text.strip().endswith(tuple(WEAK_ENDINGS)))
    return {
        "text_len": numeric_summary([float(length) for length in lengths]),
        "strong_end_count": strong_end,
        "weak_end_count": weak_end,
        "top_endings": endings.most_common(8),
    }


def dataset_stats(rows: list[JsonlRow]) -> dict[str, Any]:
    durations = [
        float(row.payload["duration"])
        for row in rows
        if isinstance(row.payload.get("duration"), (int, float))
    ]
    return {
        "rows": len(rows),
        "duration": numeric_summary(durations),
        **text_stats(rows),
    }


def pct(count: int, total: int) -> float:
    return round((100.0 * count / total), 2) if total else 0.0


def compact_value(value: Any, max_len: int = 120) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) > max_len:
        return f"{text[: max_len - 3]}..."
    return text


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    left_path = Path(args.left)
    right_path = Path(args.right)
    left_rows, left_errors = read_jsonl(left_path, args.key)
    right_rows, right_errors = read_jsonl(right_path, args.key)

    ignored = set(args.ignore_field) | {args.key}
    if args.key == "audio":
        ignored |= DEFAULT_IGNORE_FIELDS

    left_by_key = {row.key: row for row in left_rows}
    right_by_key = {row.key: row for row in right_rows}
    left_keys = set(left_by_key)
    right_keys = set(right_by_key)
    matched_keys = sorted(left_keys & right_keys)
    removed_keys = sorted(left_keys - right_keys)
    added_keys = sorted(right_keys - left_keys)

    changed: list[dict[str, Any]] = []
    field_counts = Counter()
    duration_deltas: list[float] = []
    text_len_deltas: list[float] = []

    for key in matched_keys:
        left_row = left_by_key[key]
        right_row = right_by_key[key]

        left_duration = left_row.payload.get("duration")
        right_duration = right_row.payload.get("duration")
        if isinstance(left_duration, (int, float)) and isinstance(right_duration, (int, float)):
            duration_deltas.append(float(right_duration) - float(left_duration))

        left_text = str(left_row.payload.get("text", ""))
        right_text = str(right_row.payload.get("text", ""))
        text_len_deltas.append(float(len(right_text) - len(left_text)))

        fields = changed_fields(
            left_row.payload,
            right_row.payload,
            ignored,
            args.float_tolerance,
        )
        if not fields:
            continue
        changed.append(
            {
                "key": key,
                "left_line": left_row.line_number,
                "right_line": right_row.line_number,
                "fields": fields,
            }
        )
        field_counts.update(fields.keys())

    return {
        "left": {"name": args.left_name, "path": str(left_path), "errors": left_errors},
        "right": {"name": args.right_name, "path": str(right_path), "errors": right_errors},
        "key": args.key,
        "ignored_fields": sorted(ignored),
        "summary": {
            "left_rows": len(left_rows),
            "right_rows": len(right_rows),
            "matched": len(matched_keys),
            "unchanged": len(matched_keys) - len(changed),
            "changed": len(changed),
            "removed": len(removed_keys),
            "added": len(added_keys),
            "changed_pct_of_matched": pct(len(changed), len(matched_keys)),
        },
        "stats": {
            args.left_name: dataset_stats(left_rows),
            args.right_name: dataset_stats(right_rows),
            "candidate_minus_baseline": {
                "duration_delta": numeric_summary(duration_deltas),
                "text_len_delta": numeric_summary(text_len_deltas),
            },
        },
        "field_change_counts": dict(field_counts.most_common()),
        "examples": {
            "changed": changed[: args.show],
            "removed": [
                {"key": key, "line": left_by_key[key].line_number, "row": left_by_key[key].payload}
                for key in removed_keys[: args.show]
            ],
            "added": [
                {"key": key, "line": right_by_key[key].line_number, "row": right_by_key[key].payload}
                for key in added_keys[: args.show]
            ],
        },
    }


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_numeric_summary(summary: dict[str, Any]) -> str:
    return (
        f"min={fmt_num(summary['min'])}, p50={fmt_num(summary['p50'])}, "
        f"mean={fmt_num(summary['mean'])}, p95={fmt_num(summary['p95'])}, "
        f"max={fmt_num(summary['max'])}"
    )


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    left = report["left"]
    right = report["right"]
    summary = report["summary"]

    lines.append("JSONL Strategy Comparison")
    lines.append("=" * 24)
    lines.append(f"{left['name']}: {left['path']}")
    lines.append(f"{right['name']}: {right['path']}")
    lines.append(f"match key: {report['key']}")
    lines.append(f"ignored fields: {', '.join(report['ignored_fields']) or '(none)'}")
    lines.append("")

    lines.append("Summary")
    lines.append("-" * 7)
    lines.append(f"{left['name']} rows: {summary['left_rows']}")
    lines.append(f"{right['name']} rows: {summary['right_rows']}")
    lines.append(f"matched: {summary['matched']}")
    lines.append(
        f"unchanged: {summary['unchanged']} | changed: {summary['changed']} "
        f"({summary['changed_pct_of_matched']}% of matched)"
    )
    lines.append(f"removed from {left['name']}: {summary['removed']}")
    lines.append(f"added in {right['name']}: {summary['added']}")
    lines.append("")

    if left["errors"] or right["errors"]:
        lines.append("Read Warnings")
        lines.append("-" * 13)
        for label, errors in ((left["name"], left["errors"]), (right["name"], right["errors"])):
            for error in errors[:20]:
                lines.append(f"{label}: {error}")
            if len(errors) > 20:
                lines.append(f"{label}: ... {len(errors) - 20} more warnings")
        lines.append("")

    lines.append("Dataset Stats")
    lines.append("-" * 13)
    for label in (left["name"], right["name"]):
        stats = report["stats"][label]
        lines.append(f"{label} duration: {format_numeric_summary(stats['duration'])}")
        lines.append(f"{label} text_len: {format_numeric_summary(stats['text_len'])}")
        lines.append(
            f"{label} endings: strong={stats['strong_end_count']}, "
            f"weak={stats['weak_end_count']}, top={stats['top_endings']}"
        )
    delta = report["stats"]["candidate_minus_baseline"]
    lines.append(f"duration delta ({right['name']} - {left['name']}): {format_numeric_summary(delta['duration_delta'])}")
    lines.append(f"text length delta ({right['name']} - {left['name']}): {format_numeric_summary(delta['text_len_delta'])}")
    lines.append("")

    lines.append("Changed Fields")
    lines.append("-" * 14)
    if report["field_change_counts"]:
        for field, count in report["field_change_counts"].items():
            lines.append(f"{field}: {count}")
    else:
        lines.append("(none)")
    lines.append("")

    examples = report["examples"]
    lines.append("Changed Examples")
    lines.append("-" * 16)
    if examples["changed"]:
        for item in examples["changed"]:
            lines.append(
                f"{item['key']} ({left['name']} line {item['left_line']}, "
                f"{right['name']} line {item['right_line']})"
            )
            for field, change in item["fields"].items():
                lines.append(
                    f"  {field}: {compact_value(change['left'])} -> {compact_value(change['right'])}"
                )
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(f"Removed Examples ({left['name']} only)")
    lines.append("-" * (19 + len(left["name"])))
    if examples["removed"]:
        for item in examples["removed"]:
            lines.append(f"{item['key']} line {item['line']}: {compact_value(item['row'])}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(f"Added Examples ({right['name']} only)")
    lines.append("-" * (17 + len(right["name"])))
    if examples["added"]:
        for item in examples["added"]:
            lines.append(f"{item['key']} line {item['line']}: {compact_value(item['row'])}")
    else:
        lines.append("(none)")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.format == "json":
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
    else:
        rendered = render_text(report)

    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
