"""Lightweight waveform metadata types needed by rule-based slicing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WarningSpan:
    left_idx: int
    right_idx: int
    gap_ms: float
    category: str
    reason: str
    left_char: str
    right_char: str
