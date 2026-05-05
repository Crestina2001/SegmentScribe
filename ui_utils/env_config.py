"""Read WebUI defaults from the full-pipeline shell env file."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from .commands import LLM_SLICE_MODE, RULE_SLICE_MODE
from .paths import PROJECT_ROOT


FULL_PIPELINE_ENV = PROJECT_ROOT / "configs" / "full_pipeline.env"


def load_full_pipeline_env(path: Path = FULL_PIPELINE_ENV) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        try:
            parts = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        if not parts or "=" not in parts[0]:
            continue
        key, value = parts[0].split("=", 1)
        values[key.strip()] = value
    return values


FULL_PIPELINE_DEFAULTS = load_full_pipeline_env()


def env_str(name: str, default: str = "") -> str:
    return FULL_PIPELINE_DEFAULTS.get(name, default)


def env_int(name: str, default: int) -> int:
    try:
        return int(float(env_str(name, str(default))))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = env_str(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def env_path(name: str, default: str) -> str:
    value = env_str(name, default)
    if not value:
        return default
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def env_choice(name: str, choices: tuple[str, ...] | list[str], default: str) -> str:
    value = env_str(name, default)
    return value if value in choices else default


def env_slice_mode(default: str = RULE_SLICE_MODE) -> str:
    raw = env_str("SLICE_MODE", "").strip().lower()
    if raw == "llm":
        return LLM_SLICE_MODE
    if raw == "rule":
        return RULE_SLICE_MODE
    return default


def env_optional_json(name: str, default: str = "{}") -> str:
    value = env_str(name, default).strip()
    return value or default


def env_optional_text(name: str, default: str = "") -> str:
    return env_str(name, default)
