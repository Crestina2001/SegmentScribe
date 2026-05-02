from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import slicing_utils.rough_cut as shared_rough_cut
from slicing_utils.asr import CharToken
from slicing_utils.prepass import FullPrepass
from slicing_utils.preprocess import PausePercentiles
from slicing_utils.rough_cut import RoughBoundary
from slide_LLM import rough_cut
from slide_LLM.cli import args_to_config, build_parser
from slide_LLM.pipeline import _identity_punctuation_result


def _stats(
    *,
    strong_p80: float = 300.0,
    weak_p20: float = 40.0,
    weak_p50: float = 100.0,
):
    strong = PausePercentiles(10, 10.0, 20.0, 40.0, 60.0, 90.0, strong_p80, 500.0)
    weak = PausePercentiles(10, 10.0, weak_p20, 70.0, weak_p50, 130.0, 200.0, 400.0)
    no_punc = PausePercentiles(10, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    return SimpleNamespace(strong=strong, weak=weak, no_punc=no_punc)


def _chars(text: str) -> list[CharToken]:
    return [
        CharToken(idx=idx, char=ch, start_sec=idx * 0.5, end_sec=idx * 0.5 + 0.4)
        for idx, ch in enumerate(text)
        if ch not in "，。"
    ]


def _prepass(text: str, *, pauses: dict[int, float] | None = None) -> FullPrepass:
    chars = _chars(text)
    stats = _stats()
    return FullPrepass(
        strong=stats.strong,
        weak=stats.weak,
        no_punc=stats.no_punc,
        global_chars=chars,
        full_text=text,
        warnings=[],
        pause_ms_by_token=pauses or {},
    )


def _boundary(token_idx: int, *, kind: str, pause_ms: float, label: str = "ok") -> RoughBoundary:
    return RoughBoundary(
        candidate_id=token_idx,
        token_idx=token_idx,
        text_pos=token_idx,
        boundary_text="<final>" if kind == "final" else "，",
        boundary_kind=kind,
        quality=label,
        cut_policy="normal",
        pause_ms=pause_ms,
        cut_sec=float(token_idx) + 0.4,
        next_token_start_sec=float(token_idx) + 0.5,
        reason="test",
    )


def _patch_raw_duration_estimator(monkeypatch):
    def raw_duration(_audio, _sample_rate, start_sec, end_sec, _cache):
        return max(0.0, float(end_sec) - float(start_sec))

    monkeypatch.setattr(shared_rough_cut, "_estimate_trimmed_duration_sec", raw_duration)


def test_slide_llm_cli_defaults_to_classifier_strategy_and_disabled_punctuation(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--input",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--model-path",
            "asr",
            "--aligner-path",
            "aligner",
            "--llm-model",
            "model",
        ]
    )
    cfg = args_to_config(args)

    assert cfg.enable_punctuation_correction is False
    assert cfg.rough_cut_strategy == "llm_pause_priority_silence_v2"


def test_slide_llm_cli_accepts_punctuation_and_legacy_tool_strategy(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--input",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--model-path",
            "asr",
            "--aligner-path",
            "aligner",
            "--llm-model",
            "model",
            "--enable-punctuation-correction",
            "--rough-cut-strategy",
            "llm_tool",
        ]
    )
    cfg = args_to_config(args)

    assert cfg.enable_punctuation_correction is True
    assert cfg.rough_cut_strategy == "llm_tool"


def test_identity_punctuation_result_preserves_text_and_mapping():
    prepass = _prepass("甲，乙。")

    result = _identity_punctuation_result(prepass)

    assert result.corrected_full_text == "甲，乙。"
    assert result.applied_windows == []
    assert result.corrected_token_to_positions == {0: [0], 1: [2]}


def test_grouped_classifier_prompt_indexes_targets_but_not_padding():
    text = "甲，乙，丙，丁，戊，己，庚。"
    prepass = _prepass(text)
    mapping = {idx: [idx * 2] for idx in range(7)}
    boundaries = shared_rough_cut._extract_punctuation_boundaries(text, mapping, prepass.global_chars, prepass)
    target_boundaries = [b for b in boundaries if b.boundary_kind != "final"][1:6]

    prompt, marker_map = rough_cut._build_pause_classification_prompt(
        corrected_full_text=text,
        corrected_token_to_positions=mapping,
        prepass=prepass,
        all_boundaries=boundaries,
        target_boundaries=target_boundaries,
        punctuation_start_index=1,
    )

    assert prompt["target_markers"] == [1, 2, 3, 4, 5]
    assert "[1]乙，" in prompt["text"]
    assert "[5]己，" in prompt["text"]
    assert prompt["text"].startswith("甲，")
    assert "庚" not in prompt["text"]
    assert marker_map[1].token_idx == 1
    assert marker_map[5].token_idx == 5


def test_pause_label_validation_accepts_exact_good_ok_bad_set():
    parsed, error = rough_cut._validate_pause_labels(
        [{"idx": 1, "label": "good"}, {"idx": 2, "label": "ok"}, {"idx": 3, "label": "bad"}],
        {1, 2, 3},
    )

    assert error == ""
    assert parsed == {1: "good", 2: "ok", 3: "bad"}


def test_pause_label_validation_rejects_missing_extra_duplicate_and_bad_label():
    assert rough_cut._validate_pause_labels([{"idx": 1, "label": "good"}], {1, 2})[0] is None
    assert rough_cut._validate_pause_labels([{"idx": 3, "label": "good"}], {1, 2})[0] is None
    assert rough_cut._validate_pause_labels(
        [{"idx": 1, "label": "good"}, {"idx": 1, "label": "ok"}],
        {1},
    )[0] is None
    assert rough_cut._validate_pause_labels([{"idx": 1, "label": "perfect"}], {1})[0] is None
    assert rough_cut._validate_pause_labels([{"idx": "1", "label": "good"}], {1})[0] is None


def test_llm_pause_step_rules_use_requested_thresholds():
    prepass = _stats(strong_p80=300.0, weak_p20=40.0, weak_p50=100.0)

    assert rough_cut._llm_pause_step(_boundary(1, kind="strong", pause_ms=300.0), prepass, "good") == "must_cut"
    assert rough_cut._llm_pause_step(_boundary(1, kind="weak", pause_ms=100.0), prepass, "good") == "step2_good_median"
    assert rough_cut._llm_pause_step(_boundary(1, kind="weak", pause_ms=41.0), prepass, "good") == "step3_good_or_ok"
    assert rough_cut._llm_pause_step(_boundary(1, kind="weak", pause_ms=100.0), prepass, "ok") == "step3_good_or_ok"
    assert rough_cut._llm_pause_step(_boundary(1, kind="weak", pause_ms=41.0), prepass, "ok") == "step4_not_bad"
    assert rough_cut._llm_pause_step(_boundary(1, kind="weak", pause_ms=999.0), prepass, "bad") == "not_candidate"
    assert rough_cut._llm_pause_step(_boundary(3, kind="final", pause_ms=0.0), prepass, "bad") == "must_cut"


def test_llm_pause_priority_planner_uses_labeled_steps(monkeypatch):
    _patch_raw_duration_estimator(monkeypatch)
    prepass = _stats(strong_p80=300.0, weak_p20=40.0, weak_p50=100.0)
    boundaries = [
        _boundary(1, kind="weak", pause_ms=100.0),
        _boundary(3, kind="weak", pause_ms=100.0),
        _boundary(5, kind="final", pause_ms=0.0),
    ]

    segments, cuts, error, meta = rough_cut._plan_segments_llm_pause_priority_silence_v2(
        audio=np.zeros(100, dtype=np.float32),
        sample_rate=10,
        prepass=prepass,
        chars=[
            CharToken(idx=idx, char="x", start_sec=idx * 0.5, end_sec=idx * 0.5 + 0.4)
            for idx in range(6)
        ],
        boundaries=boundaries,
        labels_by_candidate_id={1: "good", 3: "ok"},
        min_seg_sec=0.5,
        max_seg_sec=1.1,
    )

    assert error is None
    assert [(segment.char_start_idx, segment.char_end_idx, segment.drop) for segment in segments] == [
        (0, 1, False),
        (2, 3, False),
        (4, 5, False),
    ]
    assert [cut["llm_pause_label"] for cut in cuts] == ["good", "ok", "final"]
    assert meta["llm_pause_step2_candidate_count"] == 1
