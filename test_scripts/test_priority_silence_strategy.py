from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import slicing_utils.rough_cut as rough_cut
from slicing_utils.asr import CharToken
from slicing_utils.preprocess import PausePercentiles
from slicing_utils.rough_cut import RoughBoundary
from slide_rule.cli import build_parser


def _stats(
    *,
    strong_p60: float = 60.0,
    strong_p80: float = 200.0,
    weak_p20: float = 20.0,
    weak_p60: float = 100.0,
):
    strong = PausePercentiles(10, 10.0, 20.0, 40.0, 50.0, strong_p60, strong_p80, 95.0)
    weak = PausePercentiles(10, 10.0, weak_p20, 40.0, 50.0, weak_p60, 80.0, 95.0)
    return SimpleNamespace(strong=strong, weak=weak)


def _chars(count: int) -> list[CharToken]:
    return [
        CharToken(
            idx=idx,
            char=chr(ord("a") + idx),
            start_sec=float(idx) * 0.5,
            end_sec=float(idx) * 0.5 + 0.4,
        )
        for idx in range(count)
    ]


def _boundary(
    token_idx: int,
    *,
    kind: str,
    pause_ms: float,
    candidate_id: int | None = None,
) -> RoughBoundary:
    return RoughBoundary(
        candidate_id=token_idx if candidate_id is None else candidate_id,
        token_idx=token_idx,
        text_pos=token_idx,
        boundary_text="<final>" if kind == "final" else ".",
        boundary_kind=kind,
        quality="perfect",
        cut_policy="normal",
        pause_ms=pause_ms,
        cut_sec=float(token_idx) + 0.8,
        next_token_start_sec=float(token_idx + 1),
        reason="test",
    )


def _patch_raw_duration_estimator(monkeypatch):
    def raw_duration(_audio, _sample_rate, start_sec, end_sec, _cache):
        return max(0.0, float(end_sec) - float(start_sec))

    monkeypatch.setattr(rough_cut, "_estimate_trimmed_duration_sec", raw_duration)


def _priority_plan(boundaries: list[RoughBoundary], monkeypatch, *, max_seg_sec: float = 10.0):
    _patch_raw_duration_estimator(monkeypatch)
    return rough_cut._plan_segments_priority_silence_v1(
        audio=np.zeros(100, dtype=np.float32),
        sample_rate=10,
        prepass=_stats(),
        chars=_chars(4),
        boundaries=boundaries,
        min_seg_sec=0.5,
        max_seg_sec=max_seg_sec,
    )


def _priority_plan_v2(
    boundaries: list[RoughBoundary],
    monkeypatch,
    *,
    weak_p60: float = 100.0,
    max_seg_sec: float = 10.0,
    char_count: int = 4,
):
    _patch_raw_duration_estimator(monkeypatch)
    return rough_cut._plan_segments_priority_silence_v2(
        audio=np.zeros(100, dtype=np.float32),
        sample_rate=10,
        prepass=_stats(weak_p60=weak_p60),
        chars=_chars(char_count),
        boundaries=boundaries,
        min_seg_sec=0.5,
        max_seg_sec=max_seg_sec,
    )


def _dp2_plan(
    boundaries: list[RoughBoundary],
    monkeypatch,
    *,
    strong_p60: float = 120.0,
    strong_p80: float = 300.0,
    weak_p20: float = 30.0,
    weak_p60: float = 100.0,
    max_seg_sec: float = 10.0,
    char_count: int = 4,
):
    _patch_raw_duration_estimator(monkeypatch)
    return rough_cut._plan_segments_dp_strategy_2(
        audio=np.zeros(100, dtype=np.float32),
        sample_rate=10,
        prepass=_stats(
            strong_p60=strong_p60,
            strong_p80=strong_p80,
            weak_p20=weak_p20,
            weak_p60=weak_p60,
        ),
        chars=_chars(char_count),
        boundaries=boundaries,
        min_seg_sec=0.5,
        max_seg_sec=max_seg_sec,
    )


def test_priority_must_cut_requires_strong_pause_threshold_or_final():
    prepass = _stats(strong_p80=200.0)

    assert rough_cut._priority_must_cut(_boundary(0, kind="strong", pause_ms=219.0), prepass) is False
    assert rough_cut._priority_must_cut(_boundary(0, kind="strong", pause_ms=220.0), prepass) is True
    assert rough_cut._priority_must_cut(_boundary(0, kind="weak", pause_ms=999.0), prepass) is False
    assert rough_cut._priority_must_cut(_boundary(3, kind="final", pause_ms=0.0), prepass) is True
    assert rough_cut._priority_valid_cut(_boundary(3, kind="final", pause_ms=0.0), prepass) is False


def test_priority_good_cut_splits_valid_parent_when_total_duration_ties(monkeypatch):
    segments, chosen_cuts, error = _priority_plan(
        [
            _boundary(1, kind="strong", pause_ms=130.0),
            _boundary(3, kind="final", pause_ms=0.0),
        ],
        monkeypatch,
        max_seg_sec=1.1,
    )

    assert error is None
    assert [(seg.char_start_idx, seg.char_end_idx) for seg in segments] == [(0, 1), (2, 3)]
    assert [cut["strategy_phase"] for cut in chosen_cuts] == ["good_cut", "good_cut"]


def test_priority_valid_cut_uses_weak_boundary_after_good_phase(monkeypatch):
    segments, chosen_cuts, error = _priority_plan(
        [
            _boundary(1, kind="weak", pause_ms=150.0),
            _boundary(3, kind="final", pause_ms=0.0),
        ],
        monkeypatch,
    )

    assert error is None
    assert [(seg.char_start_idx, seg.char_end_idx) for seg in segments] == [(0, 1), (2, 3)]
    assert [cut["strategy_phase"] for cut in chosen_cuts] == ["valid_cut", "valid_cut"]


def test_priority_later_phase_does_not_split_locked_good_segments(monkeypatch):
    segments, chosen_cuts, error = _priority_plan(
        [
            _boundary(1, kind="strong", pause_ms=130.0),
            _boundary(2, kind="weak", pause_ms=500.0),
            _boundary(3, kind="final", pause_ms=0.0),
        ],
        monkeypatch,
    )

    assert error is None
    assert [(seg.char_start_idx, seg.char_end_idx) for seg in segments] == [(0, 1), (2, 3)]
    assert [cut["strategy_phase"] for cut in chosen_cuts] == ["good_cut", "good_cut"]


def test_slide_rule_cli_accepts_rough_cut_strategy():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--input",
            ".",
            "--output-dir",
            "out",
            "--model-path",
            "asr",
            "--aligner-path",
            "aligner",
            "--rough-cut-strategy",
            "priority_silence_v1",
        ]
    )

    assert args.rough_cut_strategy == "priority_silence_v1"


def test_slide_rule_cli_accepts_priority_silence_v2():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--input",
            ".",
            "--output-dir",
            "out",
            "--model-path",
            "asr",
            "--aligner-path",
            "aligner",
            "--rough-cut-strategy",
            "priority_silence_v2",
        ]
    )

    assert args.rough_cut_strategy == "priority_silence_v2"


def test_slide_rule_cli_accepts_dp_strategy_2():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--input",
            ".",
            "--output-dir",
            "out",
            "--model-path",
            "asr",
            "--aligner-path",
            "aligner",
            "--rough-cut-strategy",
            "dp_strategy_2",
        ]
    )

    assert args.rough_cut_strategy == "dp_strategy_2"


def test_dp_strategy_2_cut_scores_use_punctuation_and_pause_bands():
    prepass = _stats(strong_p60=120.0, strong_p80=300.0, weak_p20=30.0, weak_p60=100.0)

    assert rough_cut._dp_strategy_2_legal_cut(_boundary(1, kind="weak", pause_ms=30.0), prepass) is False
    assert rough_cut._dp_strategy_2_cut_score(_boundary(1, kind="weak", pause_ms=50.0), prepass) == -2
    assert rough_cut._dp_strategy_2_cut_score(_boundary(1, kind="weak", pause_ms=110.0), prepass) == 0
    assert rough_cut._dp_strategy_2_cut_score(_boundary(1, kind="strong", pause_ms=130.0), prepass) == 4
    assert rough_cut._dp_strategy_2_must_cut(_boundary(1, kind="strong", pause_ms=300.0), prepass) is True


def test_dp_strategy_2_uses_punctuation_cuts_to_make_short_valid_segments(monkeypatch):
    segments, chosen_cuts, error, meta = _dp2_plan(
        [
            _boundary(1, kind="weak", pause_ms=500.0),
            _boundary(3, kind="strong", pause_ms=130.0),
            _boundary(5, kind="final", pause_ms=0.0),
        ],
        monkeypatch,
        max_seg_sec=2.0,
        char_count=6,
    )

    assert error is None
    assert meta["dp_strategy_2_legal_candidate_count"] == 2
    assert [(seg.char_start_idx, seg.char_end_idx, seg.drop) for seg in segments] == [
        (0, 1, False),
        (2, 3, False),
        (4, 5, False),
    ]
    assert [cut["strategy_phase"] for cut in chosen_cuts] == [
        "dp_strategy_2",
        "dp_strategy_2",
        "dp_strategy_2",
    ]


def test_dp_strategy_2_invalid_unsolved_region_is_leftover_drop(monkeypatch):
    segments, chosen_cuts, error, _meta = _dp2_plan(
        [_boundary(3, kind="final", pause_ms=0.0)],
        monkeypatch,
        max_seg_sec=1.1,
    )

    assert error is None
    assert [(seg.char_start_idx, seg.char_end_idx, seg.drop) for seg in segments] == [(0, 3, True)]
    assert [cut["strategy_phase"] for cut in chosen_cuts] == ["leftover"]


def test_priority_v2_legal_cut_uses_weak_p60_without_120_floor():
    prepass = _stats(weak_p60=80.0)

    assert rough_cut._priority_v2_legal_cut(_boundary(1, kind="weak", pause_ms=90.0), prepass) is True
    assert rough_cut._priority_valid_cut(_boundary(1, kind="weak", pause_ms=90.0), prepass) is False
    assert rough_cut._priority_v2_good_cut(_boundary(1, kind="strong", pause_ms=90.0), prepass) is True
    assert rough_cut._priority_good_cut(_boundary(1, kind="strong", pause_ms=90.0), prepass) is False


def test_priority_v2_top50_uses_top_strong_when_strong_can_fill():
    candidates = [
        _boundary(0, kind="strong", pause_ms=300.0),
        _boundary(1, kind="strong", pause_ms=250.0),
        _boundary(2, kind="weak", pause_ms=500.0),
        _boundary(3, kind="weak", pause_ms=450.0),
    ]

    top, remaining = rough_cut._priority_v2_partition_candidates(candidates)

    assert [candidate.boundary_kind for candidate in top] == ["strong", "strong"]
    assert {candidate.candidate_id for candidate in remaining} == {2, 3}


def test_priority_v2_top50_uses_all_strong_then_top_weak_when_needed():
    candidates = [
        _boundary(0, kind="strong", pause_ms=300.0),
        _boundary(1, kind="weak", pause_ms=250.0),
        _boundary(2, kind="weak", pause_ms=500.0),
        _boundary(3, kind="weak", pause_ms=450.0),
    ]

    top, remaining = rough_cut._priority_v2_partition_candidates(candidates)

    assert [(candidate.boundary_kind, candidate.pause_ms) for candidate in top] == [
        ("strong", 300.0),
        ("weak", 500.0),
    ]
    assert {candidate.candidate_id for candidate in remaining} == {1, 3}


def test_priority_v2_step2_locked_span_is_not_resliced(monkeypatch):
    segments, chosen_cuts, error, meta = _priority_plan_v2(
        [
            _boundary(1, kind="strong", pause_ms=130.0),
            _boundary(2, kind="weak", pause_ms=500.0),
            _boundary(3, kind="final", pause_ms=0.0),
        ],
        monkeypatch,
        max_seg_sec=1.1,
    )

    assert error is None
    assert meta["priority_v2_total_legal_candidate_count"] == 0
    assert [(seg.char_start_idx, seg.char_end_idx, seg.drop) for seg in segments] == [
        (0, 1, False),
        (2, 3, False),
    ]
    assert [cut["strategy_phase"] for cut in chosen_cuts] == ["good_cut", "good_cut"]


def test_priority_v2_step4_can_reuse_failed_top50_position(monkeypatch):
    segments, chosen_cuts, error, meta = _priority_plan_v2(
        [
            _boundary(1, kind="weak", pause_ms=110.0),
            _boundary(3, kind="weak", pause_ms=300.0),
            _boundary(5, kind="final", pause_ms=0.0),
        ],
        monkeypatch,
        weak_p60=100.0,
        max_seg_sec=1.1,
        char_count=6,
    )

    assert error is None
    assert meta["priority_v2_top50_candidate_count"] == 1
    assert meta["priority_v2_partition_remaining50_candidate_count"] == 1
    assert meta["priority_v2_remaining50_candidate_count"] == 2
    assert [cut["strategy_phase"] for cut in chosen_cuts] == [
        "remaining50_cut",
        "remaining50_cut",
        "remaining50_cut",
    ]
    assert [(seg.char_start_idx, seg.char_end_idx, seg.drop) for seg in segments] == [
        (0, 1, False),
        (2, 3, False),
        (4, 5, False),
    ]


def test_priority_v2_unresolved_leftover_is_dropped(monkeypatch):
    segments, chosen_cuts, error, _meta = _priority_plan_v2(
        [
            _boundary(3, kind="final", pause_ms=0.0),
        ],
        monkeypatch,
        max_seg_sec=1.1,
    )

    assert error is None
    assert [(seg.char_start_idx, seg.char_end_idx, seg.drop) for seg in segments] == [(0, 3, True)]
    assert [cut["strategy_phase"] for cut in chosen_cuts] == ["leftover"]
