"""Phase 3 - LLM rough cut planning for ``slide_LLM``."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from llm_gateway import MemoryManager, Tool, ToolDefinition, UnifiedClient
from llm_gateway.models import ProviderName
from slicing_utils.prepass import FullPrepass, compact_pause_stats, pause_ms_after_token
from slicing_utils.rough_cut import (
    BlockCallTrace,
    Phase3Result,
    RoughBoundary,
    RoughSegment,
    _extract_punctuation_boundaries,
    _priority_leftover_spans,
    _priority_span_to_cut,
    _priority_span_to_segment,
    _solve_priority_regions,
    _subtract_priority_spans,
    _PriorityRegion,
    _segment_end_sec,
    _segment_start_sec,
)
from slicing_utils.shared import STRONG_STOPS, WEAK_STOPS

from .local_prompts import (
    LLM_SLICE_SYSTEM_PROMPT,
    PAUSE_CLASSIFICATION_SYSTEM_PROMPT,
    ROUGH_CUT_SYSTEM_PROMPT,
    build_llm_slice_base_prompt,
    build_llm_slice_submit_prompt,
    build_pause_classification_prompt as build_pause_classification_prompt_payload,
    build_rough_cut_base_prompt,
    build_retry_tool_choice_prompt,
    build_submit_cut_prompt,
    build_submit_segments_prompt,
)
from .text_rules import is_punct_or_space


TRIM_TOP_DB = 60
TRIM_PADDING_SEC = 0.1
MAX_EXTENSION_SENTENCES = 2
PAUSE_CLASSIFICATION_SENTENCE_GROUP_SIZE = 5
PAUSE_CLASSIFICATION_PADDING = 1
PAUSE_CLASSIFICATION_RETRY_LIMIT = 2
PAUSE_LABELS = frozenset({"good", "ok", "bad"})
LLM_SLICE_SENTENCE_GROUP_SIZE = 5
LLM_SLICE_PADDING = 1


@dataclass(frozen=True)
class CandidateBoundary:
    token_idx: int
    text_pos: int
    boundary_text: str
    boundary_kind: str
    pause_ms: float
    cut_sec: float
    middle_sec: float


@dataclass(frozen=True)
class PausePromptRank:
    rank: int
    total: int


@dataclass(frozen=True)
class LengthCheck:
    start_idx: int
    end_idx: int
    start_sec: float
    end_sec: float
    raw_duration_sec: float
    trimmed_duration_sec: float
    min_seg_sec: float
    max_seg_sec: float

    @property
    def valid(self) -> bool:
        return self.min_seg_sec <= self.trimmed_duration_sec <= self.max_seg_sec

    @property
    def status(self) -> str:
        if self.trimmed_duration_sec < self.min_seg_sec:
            return "too_short"
        if self.trimmed_duration_sec > self.max_seg_sec:
            return "too_long"
        return "valid"


@dataclass(frozen=True)
class SegmentProposal:
    end_idx: int
    keep: bool
    marker_id: int | None = None


@dataclass(frozen=True)
class CutChoice:
    marker_id: int
    end_idx: int


@dataclass(frozen=True)
class PauseClassificationResult:
    labels_by_candidate_id: dict[int, str]
    group_traces: list[dict[str, Any]]


@dataclass(frozen=True)
class PauseSentence:
    sentence_index: int
    token_start_idx: int
    token_end_idx: int
    boundaries: tuple[RoughBoundary, ...]


async def run_llm_rough_cut_phase(
    *,
    audio: np.ndarray,
    sample_rate: int,
    prepass: FullPrepass,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    min_seg_sec: float,
    max_seg_sec: float,
    llm_client: UnifiedClient,
    model: str,
    provider: ProviderName | None = None,
    max_rounds: int = 5,
    strategy: str = "llm_slice_v1",
) -> Phase3Result:
    if strategy == "llm_pause_priority_silence_v2":
        return await run_llm_pause_priority_silence_v2_phase(
            audio=audio,
            sample_rate=sample_rate,
            prepass=prepass,
            corrected_full_text=corrected_full_text,
            corrected_token_to_positions=corrected_token_to_positions,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            llm_client=llm_client,
            model=model,
            provider=provider,
        )
    if strategy == "llm_slice_v1":
        return await run_llm_slice_v1_phase(
            audio=audio,
            sample_rate=sample_rate,
            prepass=prepass,
            corrected_full_text=corrected_full_text,
            corrected_token_to_positions=corrected_token_to_positions,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            llm_client=llm_client,
            model=model,
            provider=provider,
            max_rounds=max_rounds,
        )
    if strategy != "llm_tool":
        raise ValueError(f"unknown LLM rough cut strategy: {strategy}")

    result = Phase3Result()
    chars = prepass.global_chars
    if not chars:
        return result

    boundaries = _extract_candidate_boundaries(corrected_full_text, corrected_token_to_positions, prepass)
    if not boundaries:
        boundaries = [
            CandidateBoundary(
                token_idx=len(chars) - 1,
                text_pos=len(corrected_full_text) - 1,
                boundary_text="<final>",
                boundary_kind="final",
                pause_ms=0.0,
                cut_sec=float(chars[-1].end_sec),
                middle_sec=float(chars[-1].end_sec),
            )
        ]

    pause_rankings = _pause_prompt_rankings(boundaries)
    trim_cache: dict[tuple[int, int], LengthCheck] = {}
    start_idx = 0
    block_index = 1
    while start_idx < len(chars):
        plan_window = _select_window(start_idx=start_idx, boundaries=boundaries, chars=chars)
        if not plan_window:
            plan_window = [boundaries[-1]]
        applied, trace = await _plan_one_window(
            audio=audio,
            sample_rate=sample_rate,
            prepass=prepass,
            corrected_full_text=corrected_full_text,
            corrected_token_to_positions=corrected_token_to_positions,
            boundaries=plan_window,
            chars=chars,
            start_idx=start_idx,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            llm_client=llm_client,
            model=model,
            provider=provider,
            max_rounds=max_rounds,
            block_index=block_index,
            trim_cache=trim_cache,
            pause_rankings=pause_rankings,
        )
        if not applied:
            break
        result.segments.extend(applied)
        result.block_traces.append(trace)
        next_start = max(seg.char_end_idx for seg in applied) + 1
        if next_start <= start_idx:
            break
        start_idx = next_start
        block_index += 1
    return result


async def run_llm_pause_priority_silence_v2_phase(
    *,
    audio: np.ndarray,
    sample_rate: int,
    prepass: FullPrepass,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    min_seg_sec: float,
    max_seg_sec: float,
    llm_client: UnifiedClient,
    model: str,
    provider: ProviderName | None = None,
) -> Phase3Result:
    result = Phase3Result()
    chars = prepass.global_chars
    if not chars:
        return result

    boundaries = _extract_punctuation_boundaries(
        corrected_full_text,
        corrected_token_to_positions,
        chars,
        prepass,
    )
    classification = await _classify_pause_boundaries(
        llm_client=llm_client,
        model=model,
        provider=provider,
        prepass=prepass,
        corrected_full_text=corrected_full_text,
        corrected_token_to_positions=corrected_token_to_positions,
        boundaries=boundaries,
    )
    segments, chosen_cuts, error, planner_meta = _plan_segments_llm_pause_priority_silence_v2(
        audio=audio,
        sample_rate=sample_rate,
        prepass=prepass,
        chars=chars,
        boundaries=boundaries,
        labels_by_candidate_id=classification.labels_by_candidate_id,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
    )
    result.segments.extend(segments)
    response_cuts = [
        _boundary_trace(boundary, prepass, classification.labels_by_candidate_id.get(boundary.candidate_id, "final"))
        for boundary in boundaries
    ]
    result.block_traces.append(
        BlockCallTrace(
            block_index=1,
            char_start_idx=0,
            char_end_idx=len(chars) - 1,
            block_start_sec=float(chars[0].start_sec),
            block_end_sec=float(chars[-1].end_sec),
            prompt_summary={
                "planner": "llm_pause_priority_silence_v2",
                "strategy": "llm_pause_priority_silence_v2",
                "classification_mode": "llm_pause_labels",
                "classifier_sentence_group_size": PAUSE_CLASSIFICATION_SENTENCE_GROUP_SIZE,
                "classifier_sentence_padding": PAUSE_CLASSIFICATION_PADDING,
                "candidate_boundary_count": len(boundaries),
                "classified_boundary_count": len(classification.labels_by_candidate_id),
                "good_boundary_count": sum(1 for label in classification.labels_by_candidate_id.values() if label == "good"),
                "ok_boundary_count": sum(1 for label in classification.labels_by_candidate_id.values() if label == "ok"),
                "bad_boundary_count": sum(1 for label in classification.labels_by_candidate_id.values() if label == "bad"),
                "thresholds_ms": _llm_pause_thresholds(prepass),
                "text_preview": _extract_span_text(
                    corrected_full_text,
                    corrected_token_to_positions,
                    0,
                    len(chars) - 1,
                )[:160],
                "trim_top_db": TRIM_TOP_DB,
                **planner_meta,
            },
            response_cuts=response_cuts,
            applied_cuts=chosen_cuts,
            carry_over_from_idx=None,
            error=error,
        )
    )
    if classification.group_traces:
        result.block_traces[0].prompt_summary["classifier_traces"] = classification.group_traces
    return result


@dataclass(frozen=True)
class _LlmSliceMarker:
    marker_id: int
    boundary: RoughBoundary
    relative_sec: float


@dataclass(frozen=True)
class _LlmSliceGroup:
    group_index: int
    target_sentences: tuple[PauseSentence, ...]
    padding_sentence: PauseSentence | None
    markers: tuple[_LlmSliceMarker, ...]
    target_first_token_idx: int
    target_last_token_idx: int

    @property
    def final_marker_id(self) -> int:
        return self.markers[-1].marker_id


async def run_llm_slice_v1_phase(
    *,
    audio: np.ndarray,
    sample_rate: int,
    prepass: FullPrepass,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    min_seg_sec: float,
    max_seg_sec: float,
    llm_client: UnifiedClient,
    model: str,
    provider: ProviderName | None = None,
    max_rounds: int = 5,
) -> Phase3Result:
    result = Phase3Result()
    chars = prepass.global_chars
    if not chars:
        return result

    boundaries = _extract_punctuation_boundaries(
        corrected_full_text,
        corrected_token_to_positions,
        chars,
        prepass,
    )
    sentences = _pause_classification_sentences(boundaries)
    groups = _build_llm_slice_groups(sentences=sentences, all_boundaries=boundaries, chars=chars)
    if not groups:
        return result

    pause_rankings = _pause_prompt_rankings(boundaries)
    trim_cache: dict[tuple[int, int], LengthCheck] = {}

    for group in groups:
        block_trace, segments = await _run_llm_slice_group(
            audio=audio,
            sample_rate=sample_rate,
            prepass=prepass,
            corrected_full_text=corrected_full_text,
            corrected_token_to_positions=corrected_token_to_positions,
            chars=chars,
            group=group,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            llm_client=llm_client,
            model=model,
            provider=provider,
            max_rounds=max_rounds,
            trim_cache=trim_cache,
            pause_rankings=pause_rankings,
        )
        result.segments.extend(segments)
        result.block_traces.append(block_trace)

    return result


def _build_llm_slice_groups(
    *,
    sentences: Sequence[PauseSentence],
    all_boundaries: Sequence[RoughBoundary],
    chars: Sequence[Any],
) -> list[_LlmSliceGroup]:
    groups: list[_LlmSliceGroup] = []
    if not sentences:
        return groups
    for group_index, target_start in enumerate(
        range(0, len(sentences), LLM_SLICE_SENTENCE_GROUP_SIZE),
        start=1,
    ):
        target_sentences = tuple(
            sentences[target_start : target_start + LLM_SLICE_SENTENCE_GROUP_SIZE]
        )
        if not target_sentences:
            continue
        padding_index = target_start + LLM_SLICE_SENTENCE_GROUP_SIZE
        padding_sentence = (
            sentences[padding_index]
            if LLM_SLICE_PADDING > 0 and padding_index < len(sentences)
            else None
        )
        target_first_token_idx = target_sentences[0].token_start_idx
        target_last_token_idx = target_sentences[-1].token_end_idx
        target_boundaries = [
            boundary
            for boundary in all_boundaries
            if target_first_token_idx <= boundary.token_idx <= target_last_token_idx
        ]
        if not target_boundaries:
            continue
        window_start_sec = _segment_start_sec(chars, target_first_token_idx)
        markers = tuple(
            _LlmSliceMarker(
                marker_id=marker_id,
                boundary=boundary,
                relative_sec=max(0.0, _llm_slice_boundary_sec(boundary) - window_start_sec),
            )
            for marker_id, boundary in enumerate(target_boundaries, start=1)
        )
        groups.append(
            _LlmSliceGroup(
                group_index=group_index,
                target_sentences=target_sentences,
                padding_sentence=padding_sentence,
                markers=markers,
                target_first_token_idx=target_first_token_idx,
                target_last_token_idx=target_last_token_idx,
            )
        )
    return groups


def _llm_slice_boundary_sec(boundary: RoughBoundary) -> float:
    if boundary.boundary_kind == "final":
        return float(boundary.cut_sec)
    return float(boundary.cut_sec) + (float(boundary.next_token_start_sec) - float(boundary.cut_sec)) * 0.5


async def _run_llm_slice_group(
    *,
    audio: np.ndarray,
    sample_rate: int,
    prepass: FullPrepass,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    chars: Sequence[Any],
    group: _LlmSliceGroup,
    min_seg_sec: float,
    max_seg_sec: float,
    llm_client: UnifiedClient,
    model: str,
    provider: ProviderName | None,
    max_rounds: int,
    trim_cache: dict[tuple[int, int], LengthCheck],
    pause_rankings: dict[int, PausePromptRank],
) -> tuple[BlockCallTrace, list[RoughSegment]]:
    expected_marker_ids = {marker.marker_id for marker in group.markers}
    target_marker_ids = sorted(expected_marker_ids)
    target_sentence_indices = [sentence.sentence_index for sentence in group.target_sentences]
    padding_sentence_index = (
        group.padding_sentence.sentence_index if group.padding_sentence is not None else None
    )
    marked_text = _marked_llm_slice_text(
        corrected_full_text=corrected_full_text,
        corrected_token_to_positions=corrected_token_to_positions,
        group=group,
        prepass=prepass,
        pause_rankings=pause_rankings,
    )
    text_preview = _extract_span_text(
        corrected_full_text,
        corrected_token_to_positions,
        group.target_first_token_idx,
        group.target_last_token_idx,
    )[:160]
    base_prompt = build_llm_slice_base_prompt(
        marked_text=marked_text,
        padding_sentence_index=padding_sentence_index,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
    )

    memory_manager = MemoryManager(system_prompt=LLM_SLICE_SYSTEM_PROMPT, stateful=True)
    rounds_log: list[dict[str, Any]] = []
    last_valid_indices: list[int] | None = None
    last_segments_feedback: list[dict[str, Any]] | None = None
    error: str | None = None

    def submit_slices(cut_indices: list[int]) -> list[dict[str, Any]]:
        round_log: dict[str, Any] = {"round": len(rounds_log) + 1, "tool": "submit_slices"}
        parse_error, indices = _parse_cut_indices_from_raw(cut_indices)
        if parse_error is not None:
            round_log["error"] = parse_error
            if indices is not None:
                round_log["raw_indices"] = indices
            rounds_log.append(round_log)
            raise ValueError(parse_error)

        validation_error = _validate_cut_indices(
            indices=indices,
            expected_marker_ids=expected_marker_ids,
            final_marker_id=group.final_marker_id,
        )
        if validation_error is not None:
            round_log["error"] = validation_error
            round_log["indices"] = list(indices)
            rounds_log.append(round_log)
            raise ValueError(validation_error)

        segment_results, segments_feedback = _llm_slice_length_feedback(
            audio=audio,
            sample_rate=sample_rate,
            chars=chars,
            group=group,
            indices=indices,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            cache=trim_cache,
        )
        nonlocal last_valid_indices, last_segments_feedback
        last_valid_indices = list(indices)
        last_segments_feedback = segments_feedback
        round_log["indices"] = list(indices)
        round_log["feedback"] = segment_results
        rounds_log.append(round_log)
        return segment_results

    def confirm_slices() -> dict[str, Any]:
        round_log: dict[str, Any] = {"round": len(rounds_log) + 1, "tool": "confirm_slices"}
        if last_valid_indices is None:
            err = "confirm_slices is only legal after at least one valid submit_slices call"
            round_log["error"] = err
            rounds_log.append(round_log)
            raise ValueError(err)
        round_log["committed_cut_indices"] = list(last_valid_indices)
        rounds_log.append(round_log)
        return {"committed_cut_indices": list(last_valid_indices)}

    final_response_text = ""
    try:
        submit_response = await llm_client.send_prompt_autoTC(
            memory_manager,
            json.dumps(build_llm_slice_submit_prompt(base_prompt), ensure_ascii=False, indent=2),
            provider=provider,
            model=model,
            temperature=0,
            tools=[_submit_slices_tool(func=submit_slices), _confirm_slices_tool(func=confirm_slices)],
            tool_choice="required",
            max_tool_rounds=max_rounds,
            trace_name="slide_LLM.rough_cut.llm_slice_v1.submit",
            metadata={"phase": "rough_cut_llm_slice_v1", "block_index": group.group_index},
        )
        final_response_text = submit_response.text
    except RuntimeError as exc:
        if "Maximum tool-call rounds exceeded" not in str(exc):
            raise
        error = f"fallback: {exc}; auto-committed latest valid submission if available"

    committed_from_model = last_valid_indices is not None
    if last_valid_indices is None:
        last_valid_indices = _llm_slice_fallback_indices(group)
        _, last_segments_feedback = _llm_slice_length_feedback(
            audio=audio,
            sample_rate=sample_rate,
            chars=chars,
            group=group,
            indices=last_valid_indices,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            cache=trim_cache,
        )
        error = "fallback: no valid LLM submission; used per-sentence cuts"

    segments, applied_cuts = _emit_llm_slice_segments(
        audio=audio,
        sample_rate=sample_rate,
        chars=chars,
        group=group,
        indices=last_valid_indices,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
        cache=trim_cache,
    )

    block_trace = BlockCallTrace(
        block_index=group.group_index,
        char_start_idx=int(group.target_first_token_idx),
        char_end_idx=int(group.target_last_token_idx),
        block_start_sec=float(chars[group.target_first_token_idx].start_sec),
        block_end_sec=float(chars[group.target_last_token_idx].end_sec),
        prompt_summary={
            "planner": "llm_slice_v1",
            "strategy": "llm_slice_v1",
            "target_sentence_indices": target_sentence_indices,
            "padding_sentence_index": padding_sentence_index,
            "available_marker_ids": target_marker_ids,
            "final_marker_id": group.final_marker_id,
            "target_marker_count": len(target_marker_ids),
            "expected_range_sec": [min_seg_sec, max_seg_sec],
            "max_rounds": max_rounds,
            "sentence_group_size": LLM_SLICE_SENTENCE_GROUP_SIZE,
            "padding_sentence_count": LLM_SLICE_PADDING,
            "text_preview": text_preview,
            "trim_top_db": TRIM_TOP_DB,
            "committed_cut_indices": list(last_valid_indices),
            "committed_via_tool": committed_from_model,
            "final_response_text": final_response_text,
            "committed_segments": last_segments_feedback,
        },
        response_cuts=rounds_log,
        applied_cuts=applied_cuts,
        carry_over_from_idx=None,
        error=error,
    )
    return block_trace, segments


def _marked_llm_slice_text(
    *,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    group: _LlmSliceGroup,
    prepass: FullPrepass,
    pause_rankings: dict[int, PausePromptRank],
) -> str:
    target_positions: list[int] = []
    for token_idx in range(group.target_first_token_idx, group.target_last_token_idx + 1):
        target_positions.extend(corrected_token_to_positions.get(token_idx, []))
    if not target_positions:
        return ""
    start_pos = min(target_positions)
    end_pos = max(target_positions) + 1
    while end_pos < len(corrected_full_text) and is_punct_or_space(corrected_full_text[end_pos]):
        end_pos += 1

    insertions: list[tuple[int, str]] = []
    for marker in group.markers:
        boundary = marker.boundary
        if boundary.boundary_kind == "final":
            insert_pos = end_pos
        else:
            insert_pos = boundary.text_pos + 1 + len(boundary.boundary_text)
            insert_pos = min(max(insert_pos, start_pos), end_pos)
        insertions.append(
            (
                insert_pos - start_pos,
                _inline_marker(
                    marker.marker_id,
                    marker.relative_sec,
                    _pause_prompt_label(boundary, prepass, pause_rankings),
                ),
            )
        )

    text = corrected_full_text[start_pos:end_pos]
    for rel_pos, marker_str in sorted(insertions, reverse=True):
        text = text[:rel_pos] + marker_str + text[rel_pos:]

    if group.padding_sentence is not None:
        padding_text = _extract_span_text(
            corrected_full_text,
            corrected_token_to_positions,
            group.padding_sentence.token_start_idx,
            group.padding_sentence.token_end_idx,
        )
        text = text + padding_text

    return text


def _submit_slices_tool(*, func: Any | None = None) -> Tool | ToolDefinition:
    tool_kwargs = dict(
        name="submit_slices",
        description=(
            "Submit a strictly increasing list of visible marker IDs that define cut positions "
            "inside the 5 target sentences. The last value MUST be the last visible marker. "
            "Each consecutive pair (previous_end+1, end) defines one segment. "
            "The tool replies with a list of segments containing start_idx, end_idx, length, and keep."
        ),
        parameters={
            "type": "object",
            "properties": {
                "cut_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Strictly increasing visible marker IDs shown in text_with_cut_markers. "
                        "The last value must be the last visible marker."
                    ),
                }
            },
            "required": ["cut_indices"],
            "additionalProperties": False,
        },
    )
    if func is None:
        return ToolDefinition(**tool_kwargs)
    return Tool(**tool_kwargs, func=func)


def _confirm_slices_tool(*, func: Any | None = None) -> Tool | ToolDefinition:
    tool_kwargs = dict(
        name="confirm_slices",
        description=(
            "Confirm the latest valid submit_slices result as the final slicing decision. "
            "This tool takes no input parameters."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )
    if func is None:
        return ToolDefinition(**tool_kwargs)
    return Tool(**tool_kwargs, func=func)


def _parse_cut_indices(response: Any) -> tuple[Optional[str], Optional[list[int]]]:
    for call in getattr(response, "tool_calls", []) or []:
        if call.name == "submit_slices":
            return _parse_cut_indices_from_raw(call.arguments.get("cut_indices"))
    return "missing submit_slices tool call", None


def _parse_cut_indices_from_raw(raw: Any) -> tuple[Optional[str], Optional[list[int]]]:
    if not isinstance(raw, list):
        return "submit_slices.cut_indices must be a list of integers", None
    indices: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            return "submit_slices.cut_indices must contain only integers", list(raw)
        indices.append(int(item))
    return None, indices


def _validate_cut_indices(
    *,
    indices: Sequence[int],
    expected_marker_ids: set[int],
    final_marker_id: int,
) -> Optional[str]:
    if not indices:
        return "cut_indices must be a non-empty list"
    last = -1
    for value in indices:
        if value not in expected_marker_ids:
            return (
                f"cut_indices contains unknown marker id {value}; "
                f"valid range is {sorted(expected_marker_ids)}"
            )
        if value <= last:
            return "cut_indices must be strictly increasing"
        last = value
    if indices[-1] != final_marker_id:
        return (
            f"last value of cut_indices must be the last visible marker ({final_marker_id}); "
            f"got {indices[-1]}"
        )
    return None


def _llm_slice_length_feedback(
    *,
    audio: np.ndarray,
    sample_rate: int,
    chars: Sequence[Any],
    group: _LlmSliceGroup,
    indices: Sequence[int],
    min_seg_sec: float,
    max_seg_sec: float,
    cache: dict[tuple[int, int], LengthCheck],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    marker_by_id = {marker.marker_id: marker for marker in group.markers}
    segments_payload: list[dict[str, Any]] = []
    prev_end_token_idx = group.target_first_token_idx - 1
    for marker_id in indices:
        marker = marker_by_id[marker_id]
        start_token_idx = prev_end_token_idx + 1
        end_token_idx = marker.boundary.token_idx
        check = _check_length(
            audio=audio,
            sample_rate=sample_rate,
            chars=chars,
            start_idx=start_token_idx,
            end_idx=end_token_idx,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            cache=cache,
        )
        segments_payload.append(
            {
                "start_idx": int(start_token_idx),
                "end_idx": int(end_token_idx),
                "length": round(float(check.trimmed_duration_sec), 3),
                "keep": bool(check.valid),
            }
        )
        prev_end_token_idx = end_token_idx
    return segments_payload, segments_payload


def _emit_llm_slice_segments(
    *,
    audio: np.ndarray,
    sample_rate: int,
    chars: Sequence[Any],
    group: _LlmSliceGroup,
    indices: Sequence[int],
    min_seg_sec: float,
    max_seg_sec: float,
    cache: dict[tuple[int, int], LengthCheck],
) -> tuple[list[RoughSegment], list[dict[str, Any]]]:
    marker_by_id = {marker.marker_id: marker for marker in group.markers}
    segments: list[RoughSegment] = []
    applied_cuts: list[dict[str, Any]] = []
    prev_end_token_idx = group.target_first_token_idx - 1
    for marker_id in indices:
        marker = marker_by_id[marker_id]
        start_token_idx = prev_end_token_idx + 1
        end_token_idx = marker.boundary.token_idx
        check = _check_length(
            audio=audio,
            sample_rate=sample_rate,
            chars=chars,
            start_idx=start_token_idx,
            end_idx=end_token_idx,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            cache=cache,
        )
        drop = not check.valid
        reason = (
            f"llm_slice_v1 accepted marker {marker_id}"
            if not drop
            else f"llm_slice_v1 not accepted (length {check.status}) at marker {marker_id}"
        )
        segment = _make_segment(start_token_idx, end_token_idx, check, drop=drop, reason=reason)
        segments.append(segment)
        applied_cuts.append(
            {
                "tool": "submit_slices",
                "marker_id": int(marker_id),
                "cut_char_idx": int(end_token_idx),
                "boundary_kind": marker.boundary.boundary_kind,
                "boundary_text": marker.boundary.boundary_text,
                "pause_ms": marker.boundary.pause_ms,
                "trimmed_duration_sec": round(float(check.trimmed_duration_sec), 3),
                "raw_duration_sec": round(float(check.raw_duration_sec), 3),
                "length_status": check.status,
                "accepted": check.valid,
                "drop": drop,
                "strategy_cut_policy": "llm_slice_v1",
            }
        )
        prev_end_token_idx = end_token_idx
    return segments, applied_cuts


def _llm_slice_fallback_indices(group: _LlmSliceGroup) -> list[int]:
    sentence_end_token_idxs = {sentence.token_end_idx for sentence in group.target_sentences}
    fallback: list[int] = []
    for marker in group.markers:
        if marker.boundary.token_idx in sentence_end_token_idxs:
            fallback.append(marker.marker_id)
    if not fallback or fallback[-1] != group.final_marker_id:
        fallback.append(group.final_marker_id)
    deduped: list[int] = []
    for value in fallback:
        if not deduped or value > deduped[-1]:
            deduped.append(value)
    return deduped


async def _classify_pause_boundaries(
    *,
    llm_client: UnifiedClient,
    model: str,
    provider: ProviderName | None,
    prepass: FullPrepass,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    boundaries: Sequence[RoughBoundary],
) -> PauseClassificationResult:
    sentences = _pause_classification_sentences(boundaries)
    targetable_sentences = [sentence for sentence in sentences if sentence.boundaries]
    if not targetable_sentences:
        return PauseClassificationResult(labels_by_candidate_id={}, group_traces=[])

    labels_by_candidate_id: dict[int, str] = {}
    traces: list[dict[str, Any]] = []
    for group_index, target_start in enumerate(
        range(0, len(targetable_sentences), PAUSE_CLASSIFICATION_SENTENCE_GROUP_SIZE),
        start=1,
    ):
        target_sentences = targetable_sentences[
            target_start : target_start + PAUSE_CLASSIFICATION_SENTENCE_GROUP_SIZE
        ]
        prompt, marker_to_boundary = _build_pause_classification_prompt(
            corrected_full_text=corrected_full_text,
            corrected_token_to_positions=corrected_token_to_positions,
            prepass=prepass,
            sentences=sentences,
            target_sentences=target_sentences,
        )
        expected_markers = set(marker_to_boundary)
        memory_manager = MemoryManager(system_prompt=PAUSE_CLASSIFICATION_SYSTEM_PROMPT, stateful=True)
        attempts: list[dict[str, Any]] = []
        validated: dict[int, str] | None = None
        validation_error = ""
        for attempt in range(1, PAUSE_CLASSIFICATION_RETRY_LIMIT + 1):
            payload = dict(prompt)
            if validation_error:
                payload["previous_validation_error"] = validation_error
                payload["retry_instruction"] = "Resubmit labels for all and only the visible marker IDs."
            response = await llm_client.send_prompt(
                memory_manager,
                json.dumps(payload, ensure_ascii=False, indent=2),
                provider=provider,
                model=model,
                temperature=0,
                tools=[_submit_pause_labels_tool()],
                tool_choice="required",
                trace_name="slide_LLM.rough_cut.pause_classification",
                metadata={"phase": "rough_cut_pause_classification", "group_index": group_index, "attempt": attempt},
            )
            raw_labels = _parse_pause_labels(response)
            validated, validation_error = _validate_pause_labels(raw_labels, expected_markers)
            _append_tool_result_messages(
                memory_manager,
                response,
                {
                    "tool": "submit_pause_labels",
                    "valid": validated is not None,
                    "labels": validated or raw_labels,
                    "validation_error": validation_error or None,
                },
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_labels": raw_labels,
                    "validation_error": validation_error or None,
                }
            )
            if validated is not None:
                break
        if validated is None:
            validated = {marker_id: "ok" for marker_id in sorted(expected_markers)}
            attempts.append(
                {
                    "attempt": "fallback",
                    "raw_labels": validated,
                    "validation_error": validation_error or "classifier did not return a valid label set",
                }
            )
        for marker_id, label in validated.items():
            labels_by_candidate_id[marker_to_boundary[marker_id].candidate_id] = label
        traces.append(
            {
                "group_index": group_index,
                "target_sentence_indices": [sentence.sentence_index for sentence in target_sentences],
                "prompt": prompt,
                "target_marker_map": {
                    str(marker_id): {
                        "candidate_id": boundary.candidate_id,
                        "token_idx": boundary.token_idx,
                        "boundary_text": boundary.boundary_text,
                        "pause_ms": boundary.pause_ms,
                    }
                    for marker_id, boundary in marker_to_boundary.items()
                },
                "validated_labels": validated,
                "attempts": attempts,
            }
        )
    return PauseClassificationResult(labels_by_candidate_id=labels_by_candidate_id, group_traces=traces)


def _build_pause_classification_prompt(
    *,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    prepass: FullPrepass,
    sentences: Sequence[PauseSentence],
    target_sentences: Sequence[PauseSentence],
) -> tuple[dict[str, Any], dict[int, RoughBoundary]]:
    if not target_sentences:
        raise ValueError("target_sentences must be non-empty")
    target_sentence_ids = {sentence.sentence_index for sentence in target_sentences}
    target_start = min(target_sentence_ids)
    target_end = max(target_sentence_ids)
    context_start = max(0, target_start - PAUSE_CLASSIFICATION_PADDING)
    context_end = min(len(sentences) - 1, target_end + PAUSE_CLASSIFICATION_PADDING)
    context_sentences = [sentence for sentence in sentences if context_start <= sentence.sentence_index <= context_end]
    target_boundaries = [
        boundary
        for sentence in target_sentences
        for boundary in sentence.boundaries
    ]
    marker_to_boundary = {marker_id: boundary for marker_id, boundary in enumerate(target_boundaries, start=1)}
    boundary_to_marker = {boundary.candidate_id: marker_id for marker_id, boundary in marker_to_boundary.items()}
    marked_text = _marked_pause_classification_text(
        corrected_full_text=corrected_full_text,
        corrected_token_to_positions=corrected_token_to_positions,
        context_sentences=context_sentences,
        boundary_to_marker=boundary_to_marker,
    )
    return (
        build_pause_classification_prompt_payload(
            marked_text=marked_text,
            target_sentence_indices=sorted(target_sentence_ids),
            target_markers=sorted(marker_to_boundary),
            marker_pause_ms={
                str(marker_id): marker_to_boundary[marker_id].pause_ms
                for marker_id in sorted(marker_to_boundary)
            },
            pause_stats=_rough_cut_pause_stats(prepass),
        ),
        marker_to_boundary,
    )


def _marked_pause_classification_text(
    *,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    context_sentences: Sequence[PauseSentence],
    boundary_to_marker: dict[int, int],
) -> str:
    pieces: list[str] = []
    for sentence in context_sentences:
        cursor = sentence.token_start_idx
        for boundary in sentence.boundaries:
            if cursor <= boundary.token_idx:
                piece = _extract_span_text(
                    corrected_full_text,
                    corrected_token_to_positions,
                    cursor,
                    boundary.token_idx,
                )
                marker = boundary_to_marker.get(boundary.candidate_id)
                if marker is not None:
                    piece = f"[{marker}]" + piece
                pieces.append(piece)
            cursor = boundary.token_idx + 1
        if cursor <= sentence.token_end_idx:
            pieces.append(
                _extract_span_text(
                    corrected_full_text,
                    corrected_token_to_positions,
                    cursor,
                    sentence.token_end_idx,
                )
            )
    return "".join(pieces)


def _pause_classification_sentences(boundaries: Sequence[RoughBoundary]) -> list[PauseSentence]:
    sentences: list[PauseSentence] = []
    token_start_idx = 0
    sentence_boundaries: list[RoughBoundary] = []
    for boundary in boundaries:
        if boundary.boundary_kind != "final":
            sentence_boundaries.append(boundary)
        if boundary.boundary_kind in {"strong", "final"}:
            sentences.append(
                PauseSentence(
                    sentence_index=len(sentences),
                    token_start_idx=token_start_idx,
                    token_end_idx=boundary.token_idx,
                    boundaries=tuple(sentence_boundaries),
                )
            )
            token_start_idx = boundary.token_idx + 1
            sentence_boundaries = []
    if sentence_boundaries:
        sentences.append(
            PauseSentence(
                sentence_index=len(sentences),
                token_start_idx=token_start_idx,
                token_end_idx=sentence_boundaries[-1].token_idx,
                boundaries=tuple(sentence_boundaries),
            )
        )
    return sentences


def _submit_pause_labels_tool() -> ToolDefinition:
    return ToolDefinition(
        name="submit_pause_labels",
        description="Submit semantic/audio cut quality labels for visible punctuation markers.",
        parameters={
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "idx": {"type": "integer", "description": "Visible marker ID, e.g. 1 for [1]."},
                            "label": {"type": "string", "enum": ["good", "ok", "bad"]},
                        },
                        "required": ["idx", "label"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["labels"],
            "additionalProperties": False,
        },
    )


def _parse_pause_labels(response: Any) -> list[dict[str, Any]]:
    for call in getattr(response, "tool_calls", []) or []:
        if call.name == "submit_pause_labels":
            labels = call.arguments.get("labels")
            return labels if isinstance(labels, list) else []
    return []


def _validate_pause_labels(
    raw_labels: Sequence[Any],
    expected_markers: set[int],
) -> tuple[dict[int, str] | None, str]:
    parsed: dict[int, str] = {}
    for item in raw_labels:
        if not isinstance(item, dict):
            return None, "each label item must be an object"
        marker_id = item.get("idx")
        label = item.get("label")
        if not isinstance(marker_id, int) or isinstance(marker_id, bool):
            return None, "each label item must contain integer idx"
        if marker_id in parsed:
            return None, f"duplicate marker idx: {marker_id}"
        if marker_id not in expected_markers:
            return None, f"unexpected marker idx: {marker_id}; expected {sorted(expected_markers)}"
        if not isinstance(label, str) or label not in PAUSE_LABELS:
            return None, f"invalid label for marker {marker_id}: {label!r}"
        parsed[marker_id] = label
    missing = expected_markers - set(parsed)
    if missing:
        return None, f"missing marker labels: {sorted(missing)}"
    return parsed, ""


def _llm_pause_thresholds(prepass: FullPrepass) -> dict[str, float]:
    return {
        "strong_p80_ms": float(prepass.strong.p80_ms),
        "weak_p20_ms": float(prepass.weak.p20_ms),
        "weak_p50_ms": float(prepass.weak.p50_ms),
    }


def _llm_pause_step(boundary: RoughBoundary, prepass: FullPrepass, label: str) -> str:
    if boundary.boundary_kind == "final":
        return "must_cut"
    pause_ms = float(boundary.pause_ms)
    if pause_ms >= float(prepass.strong.p80_ms) and label == "good":
        return "must_cut"
    if pause_ms >= float(prepass.weak.p50_ms) and label == "good":
        return "step2_good_median"
    if (pause_ms > float(prepass.weak.p20_ms) and label == "good") or (
        pause_ms >= float(prepass.weak.p50_ms) and label == "ok"
    ):
        return "step3_good_or_ok"
    if pause_ms > float(prepass.weak.p20_ms) and label != "bad":
        return "step4_not_bad"
    return "not_candidate"


def _plan_segments_llm_pause_priority_silence_v2(
    *,
    audio: np.ndarray,
    sample_rate: int,
    prepass: FullPrepass,
    chars: Sequence[Any],
    boundaries: Sequence[RoughBoundary],
    labels_by_candidate_id: dict[int, str],
    min_seg_sec: float,
    max_seg_sec: float,
) -> tuple[list[RoughSegment], list[dict[str, Any]], Optional[str], dict[str, Any]]:
    if not boundaries:
        return [], [], "no candidate punctuation boundaries found", {}

    trim_cache: dict[tuple[int, int], float] = {}
    must_boundaries = [
        boundary
        for boundary in boundaries
        if _llm_pause_step(boundary, prepass, labels_by_candidate_id.get(boundary.candidate_id, "bad")) == "must_cut"
    ]
    if not must_boundaries or must_boundaries[-1].boundary_kind != "final":
        return [], [], "llm_pause_priority_silence_v2 found no final must_cut boundary", {}

    regions: list[_PriorityRegion] = []
    region_start = 0
    for region_end in must_boundaries:
        if region_start <= region_end.token_idx:
            regions.append(_PriorityRegion(start_token_idx=region_start, end_boundary=region_end))
        region_start = region_end.token_idx + 1

    all_solved: list[Any] = []
    current_regions = regions
    phase_specs = [
        ("step2_good_median", "segments_before_silence"),
        ("step3_good_or_ok", "silence_before_segments"),
        ("step4_not_bad", "silence_before_segments"),
    ]
    for phase, split_priority in phase_specs:
        candidates = [
            boundary
            for boundary in boundaries
            if boundary.boundary_kind != "final"
            and _llm_pause_step(boundary, prepass, labels_by_candidate_id.get(boundary.candidate_id, "bad")) == phase
        ]
        solved = _solve_priority_regions(
            audio=audio,
            sample_rate=sample_rate,
            chars=chars,
            regions=current_regions,
            candidate_boundaries=candidates,
            phase=phase,
            split_priority=split_priority,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            trim_cache=trim_cache,
            require_internal_candidate=(phase == "step2_good_median"),
        )
        all_solved.extend(solved)
        current_regions = _subtract_priority_spans(current_regions, solved, chars)

    all_spans: list[Any] = []
    solved_spans = sorted(all_solved, key=lambda span: (span.start_token_idx, span.end_boundary.token_idx))
    for region in regions:
        region_spans = [
            span
            for span in solved_spans
            if region.start_token_idx <= span.start_token_idx <= span.end_boundary.token_idx <= region.end_boundary.token_idx
        ]
        all_spans.extend(
            _priority_leftover_spans(
                audio=audio,
                sample_rate=sample_rate,
                chars=chars,
                region_start_token_idx=region.start_token_idx,
                region_end_boundary=region.end_boundary,
                chosen_spans=region_spans,
                min_seg_sec=min_seg_sec,
                max_seg_sec=max_seg_sec,
                trim_cache=trim_cache,
            )
        )

    strategy = "llm_pause_priority_silence_v2"
    segments = [_priority_span_to_segment(span, chars, strategy=strategy) for span in all_spans]
    chosen_cuts = [
        _priority_span_to_cut(span, strategy=strategy)
        | {
            "llm_pause_label": labels_by_candidate_id.get(span.end_boundary.candidate_id, "final"),
            "llm_pause_step": _llm_pause_step(
                span.end_boundary,
                prepass,
                labels_by_candidate_id.get(span.end_boundary.candidate_id, "bad"),
            ),
            "thresholds_ms": _llm_pause_thresholds(prepass),
        }
        for span in all_spans
    ]
    meta = {
        "llm_pause_must_cut_boundary_count": len(must_boundaries),
        "llm_pause_step2_candidate_count": sum(
            1 for b in boundaries if _llm_pause_step(b, prepass, labels_by_candidate_id.get(b.candidate_id, "bad")) == "step2_good_median"
        ),
        "llm_pause_step3_candidate_count": sum(
            1 for b in boundaries if _llm_pause_step(b, prepass, labels_by_candidate_id.get(b.candidate_id, "bad")) == "step3_good_or_ok"
        ),
        "llm_pause_step4_candidate_count": sum(
            1 for b in boundaries if _llm_pause_step(b, prepass, labels_by_candidate_id.get(b.candidate_id, "bad")) == "step4_not_bad"
        ),
    }
    return segments, chosen_cuts, None, meta


def _boundary_trace(boundary: RoughBoundary, prepass: FullPrepass, label: str) -> dict[str, Any]:
    return {
        "candidate_id": int(boundary.candidate_id),
        "cut_char_idx": int(boundary.token_idx),
        "boundary_kind": boundary.boundary_kind,
        "boundary_quality": boundary.quality,
        "cut_policy": boundary.cut_policy,
        "boundary_text": boundary.boundary_text,
        "pause_ms": boundary.pause_ms,
        "llm_pause_label": label,
        "strategy_cut_policy": _llm_pause_step(boundary, prepass, label if label in PAUSE_LABELS else "bad"),
        "thresholds_ms": _llm_pause_thresholds(prepass),
        "reason": boundary.reason,
    }


async def _plan_one_window(
    *,
    audio: np.ndarray,
    sample_rate: int,
    prepass: FullPrepass,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    boundaries: Sequence[CandidateBoundary],
    chars: Sequence[Any],
    start_idx: int,
    min_seg_sec: float,
    max_seg_sec: float,
    llm_client: UnifiedClient,
    model: str,
    provider: ProviderName | None,
    max_rounds: int,
    block_index: int,
    trim_cache: dict[tuple[int, int], LengthCheck],
    pause_rankings: dict[int, PausePromptRank],
) -> tuple[list[RoughSegment], BlockCallTrace]:
    prompt_summary = _prompt_summary(
        prepass=prepass,
        corrected_full_text=corrected_full_text,
        corrected_token_to_positions=corrected_token_to_positions,
        boundaries=boundaries,
        start_idx=start_idx,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
        pause_rankings=pause_rankings,
    )
    marker_to_boundary = {marker_id: boundary for marker_id, boundary in enumerate(boundaries, start=1)}
    memory_manager = MemoryManager(system_prompt=ROUGH_CUT_SYSTEM_PROMPT, stateful=True)
    response_cuts: list[dict[str, Any]] = []
    first_cut_idx: Optional[int] = None
    first_cut_marker: Optional[int] = None
    first_legal_cut_idx: Optional[int] = None
    first_legal_cut_marker: Optional[int] = None
    first_legal_cut_check: Optional[LengthCheck] = None
    latest_cut_idx: Optional[int] = None
    last_cut_retry_allowed = True
    legal_submit_cut_count = 0
    latest_format_ok_segments: Optional[list[SegmentProposal]] = None
    error: Optional[str] = None

    first_response = await _call_submit_cut(
        llm_client=llm_client,
        memory_manager=memory_manager,
        model=model,
        provider=provider,
        prompt=prompt_summary,
        block_index=block_index,
    )
    first_cut_marker = _parse_first_cut(first_response)
    window_end_idx = boundaries[-1].token_idx if boundaries else len(chars) - 1
    if first_cut_marker is not None:
        first_boundary = marker_to_boundary.get(first_cut_marker)
        if first_boundary is None:
            response_cuts.append(
                {
                    "tool": "submit_cut",
                    "cut_idx": first_cut_marker,
                    "error": f"cut_idx must be one of visible marker ids: {sorted(marker_to_boundary)}",
                }
            )
            first_cut_marker = None
        else:
            first_cut_idx = first_boundary.token_idx
            latest_cut_idx = first_cut_idx
            first_legal_cut_idx = first_cut_idx
            first_legal_cut_marker = first_cut_marker
            legal_submit_cut_count = 1
    if first_cut_idx is not None and not (start_idx <= first_cut_idx <= window_end_idx):
        response_cuts.append(
            {
                "tool": "submit_cut",
                "marker_id": first_cut_marker,
                "cut_idx": first_cut_idx,
                "error": f"cut_idx must be within [{start_idx}, {window_end_idx}]",
            }
        )
        first_cut_idx = None
    if first_cut_idx is not None:
        check = _check_length(
            audio=audio,
            sample_rate=sample_rate,
            chars=chars,
            start_idx=start_idx,
            end_idx=first_cut_idx,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            cache=trim_cache,
        )
        response_cuts.append(
            {
                "tool": "submit_cut",
                "marker_id": first_cut_marker,
                "cut_idx": first_cut_idx,
                "length": dataclasses.asdict(check),
                "length_status": check.status,
                "valid": check.valid,
            }
        )
        _append_tool_result_messages(memory_manager, first_response, response_cuts[-1])
        if check.valid:
            segment = _make_segment(start_idx, first_cut_idx, check, drop=False, reason="LLM first cut length-valid")
            return [segment], _make_trace(
                block_index=block_index,
                start_idx=start_idx,
                end_idx=first_cut_idx,
                chars=chars,
                prompt_summary=prompt_summary,
                response_cuts=response_cuts,
                applied=[_segment_trace(segment, check)],
                error=None,
            )
        first_legal_cut_check = check
        error = _length_error_message(check)
        last_cut_retry_allowed = check.status == "too_short"
    else:
        error = "first tool call missing or invalid submit_cut.cut_idx"
        response_cuts.append({"tool": "submit_cut", "error": error})
        _append_tool_result_messages(memory_manager, first_response, response_cuts[-1])
        last_cut_retry_allowed = True

    for round_index in range(1, max_rounds + 1):
        retry_response = await _call_retry_tool_choice(
            llm_client=llm_client,
            memory_manager=memory_manager,
            model=model,
            provider=provider,
            prompt=prompt_summary,
            feedback=error or "",
            first_given_idx=first_cut_marker,
            block_index=block_index,
            round_index=round_index,
        )
        chosen_tool = _chosen_tool_name(retry_response)
        if chosen_tool == "submit_cut":
            if not last_cut_retry_allowed:
                error = "submit_cut retry is only legal after an invalid cut or a too_short length error"
                response_cuts.append({"tool": "submit_cut", "round": round_index, "error": error})
                _append_tool_result_messages(memory_manager, retry_response, response_cuts[-1])
                continue
            cut_choice, cut_error = _parse_and_resolve_cut(retry_response, marker_to_boundary)
            if cut_error:
                error = cut_error
                response_cuts.append({"tool": "submit_cut", "round": round_index, "error": error})
                _append_tool_result_messages(memory_manager, retry_response, response_cuts[-1])
                last_cut_retry_allowed = True
                continue
            assert cut_choice is not None
            latest_cut_idx = cut_choice.end_idx
            legal_submit_cut_count += 1
            check = _check_length(
                audio=audio,
                sample_rate=sample_rate,
                chars=chars,
                start_idx=start_idx,
                end_idx=cut_choice.end_idx,
                min_seg_sec=min_seg_sec,
                max_seg_sec=max_seg_sec,
                cache=trim_cache,
            )
            response_cuts.append(
                {
                    "tool": "submit_cut",
                    "round": round_index,
                    "marker_id": cut_choice.marker_id,
                    "cut_idx": cut_choice.end_idx,
                    "length": dataclasses.asdict(check),
                    "length_status": check.status,
                    "valid": check.valid,
                }
            )
            _append_tool_result_messages(memory_manager, retry_response, response_cuts[-1])
            if check.valid:
                segment = _make_segment(
                    start_idx,
                    cut_choice.end_idx,
                    check,
                    drop=False,
                    reason="LLM extension cut length-valid",
                )
                return [segment], _make_trace(
                    block_index=block_index,
                    start_idx=start_idx,
                    end_idx=cut_choice.end_idx,
                    chars=chars,
                    prompt_summary=prompt_summary,
                    response_cuts=response_cuts,
                    applied=[_segment_trace(segment, check)],
                    error=None,
                )
            if legal_submit_cut_count >= 2 and check.status == "too_long":
                assert first_legal_cut_idx is not None
                discard_check = first_legal_cut_check or _check_length(
                    audio=audio,
                    sample_rate=sample_rate,
                    chars=chars,
                    start_idx=start_idx,
                    end_idx=first_legal_cut_idx,
                    min_seg_sec=min_seg_sec,
                    max_seg_sec=max_seg_sec,
                    cache=trim_cache,
                )
                segment = _make_segment(
                    start_idx,
                    first_legal_cut_idx,
                    discard_check,
                    drop=True,
                    reason="second legal submit_cut was too_long; discarded first submitted cut",
                )
                response_cuts.append(
                    {
                        "tool": "submit_cut",
                        "round": round_index,
                        "action": "discard_first_submitted_cut",
                        "first_marker_id": first_legal_cut_marker,
                        "first_cut_idx": first_legal_cut_idx,
                    }
                )
                return [segment], _make_trace(
                    block_index=block_index,
                    start_idx=start_idx,
                    end_idx=first_legal_cut_idx,
                    chars=chars,
                    prompt_summary=prompt_summary,
                    response_cuts=response_cuts,
                    applied=[_segment_trace(segment, discard_check)],
                    error=None,
                )
            error = _length_error_message(check)
            last_cut_retry_allowed = check.status == "too_short"
            continue

        if chosen_tool != "submit_segments":
            error = "retry must call submit_cut or submit_segments"
            response_cuts.append({"tool": "retry_tool_choice", "round": round_index, "error": error})
            _append_tool_result_messages(memory_manager, retry_response, response_cuts[-1])
            continue

        second_response = retry_response
        proposals, format_error = _parse_segment_proposals(second_response)
        if format_error:
            error = format_error
            response_cuts.append({"tool": "submit_segments", "round": round_index, "error": error})
            _append_tool_result_messages(memory_manager, second_response, response_cuts[-1])
            continue
        proposals, marker_error = _resolve_segment_markers(proposals or [], marker_to_boundary)
        if marker_error:
            error = marker_error
            response_cuts.append({"tool": "submit_segments", "round": round_index, "error": error})
            _append_tool_result_messages(memory_manager, second_response, response_cuts[-1])
            continue
        if first_cut_marker is not None and proposals and proposals[-1].marker_id != first_cut_marker:
            error = (
                "last segments[].end_idx must equal first_given_idx "
                f"({first_cut_marker}); got {proposals[-1].marker_id}"
            )
            response_cuts.append({"tool": "submit_segments", "round": round_index, "error": error})
            _append_tool_result_messages(memory_manager, second_response, response_cuts[-1])
            continue
        latest_format_ok_segments = proposals
        assert proposals is not None
        checks = [
            _check_length(
                audio=audio,
                sample_rate=sample_rate,
                chars=chars,
                start_idx=start_idx if i == 0 else proposals[i - 1].end_idx + 1,
                end_idx=proposal.end_idx,
                min_seg_sec=min_seg_sec,
                max_seg_sec=max_seg_sec,
                cache=trim_cache,
            )
            for i, proposal in enumerate(proposals)
        ]
        mismatches = [
            {
                "end_idx": proposal.marker_id,
                "resolved_token_idx": proposal.end_idx,
                "keep": proposal.keep,
                "actual_valid": check.valid,
            }
            for proposal, check in zip(proposals, checks)
            if proposal.keep != check.valid
        ]
        response_cuts.append(
            {
                "tool": "submit_segments",
                "round": round_index,
                "segments": [dataclasses.asdict(p) for p in proposals],
                "lengths": [dataclasses.asdict(c) | {"valid": c.valid} for c in checks],
                "mismatches": mismatches,
            }
        )
        _append_tool_result_messages(memory_manager, second_response, response_cuts[-1])
        if not mismatches:
            segments = [
                _make_segment(
                    check.start_idx,
                    check.end_idx,
                    check,
                    drop=not proposal.keep,
                    reason="LLM keep/discard length-consistent",
                )
                for proposal, check in zip(proposals, checks)
            ]
            return segments, _make_trace(
                block_index=block_index,
                start_idx=start_idx,
                end_idx=segments[-1].char_end_idx,
                chars=chars,
                prompt_summary=prompt_summary,
                response_cuts=response_cuts,
                applied=[_segment_trace(s, c) for s, c in zip(segments, checks)],
                error=None,
            )
        error = "keep/discard mismatch: " + json.dumps(mismatches, ensure_ascii=False)

    fallback_segments, fallback_checks, fallback_error = _fallback_segments(
        audio=audio,
        sample_rate=sample_rate,
        chars=chars,
        boundaries=boundaries,
        start_idx=start_idx,
        first_cut_idx=first_cut_idx,
        latest_cut_idx=latest_cut_idx,
        latest_format_ok_segments=latest_format_ok_segments,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
        cache=trim_cache,
    )
    return fallback_segments, _make_trace(
        block_index=block_index,
        start_idx=start_idx,
        end_idx=fallback_segments[-1].char_end_idx if fallback_segments else start_idx,
        chars=chars,
        prompt_summary=prompt_summary,
        response_cuts=response_cuts,
        applied=[_segment_trace(s, c) for s, c in zip(fallback_segments, fallback_checks)],
        error=fallback_error or error,
    )


def _append_tool_result_messages(
    memory_manager: MemoryManager,
    response: Any,
    result: dict[str, Any],
) -> None:
    for call in getattr(response, "tool_calls", []) or []:
        message: dict[str, Any] = {
            "role": "tool",
            "name": call.name,
            "content": json.dumps(result, ensure_ascii=False),
        }
        if call.id is not None:
            message["tool_call_id"] = call.id
        memory_manager.append_message(message)


def _fallback_segments(
    *,
    audio: np.ndarray,
    sample_rate: int,
    chars: Sequence[Any],
    boundaries: Sequence[CandidateBoundary],
    start_idx: int,
    first_cut_idx: Optional[int],
    latest_cut_idx: Optional[int],
    latest_format_ok_segments: Optional[list[SegmentProposal]],
    min_seg_sec: float,
    max_seg_sec: float,
    cache: dict[tuple[int, int], LengthCheck],
) -> tuple[list[RoughSegment], list[LengthCheck], str]:
    if latest_format_ok_segments:
        segments: list[RoughSegment] = []
        checks: list[LengthCheck] = []
        piece_start = start_idx
        for proposal in latest_format_ok_segments:
            check = _check_length(
                audio=audio,
                sample_rate=sample_rate,
                chars=chars,
                start_idx=piece_start,
                end_idx=proposal.end_idx,
                min_seg_sec=min_seg_sec,
                max_seg_sec=max_seg_sec,
                cache=cache,
            )
            checks.append(check)
            segments.append(
                _make_segment(
                    piece_start,
                    check.end_idx,
                    check,
                    drop=not check.valid,
                    reason="fallback auto-flipped keep by measured length",
                )
            )
            piece_start = proposal.end_idx + 1
        return segments, checks, "fallback: auto-flipped format-valid submit_segments payload"

    fallback_cut_idx = latest_cut_idx if latest_cut_idx is not None else first_cut_idx
    if fallback_cut_idx is not None:
        check = _check_length(
            audio=audio,
            sample_rate=sample_rate,
            chars=chars,
            start_idx=start_idx,
            end_idx=fallback_cut_idx,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            cache=cache,
        )
        return (
            [_make_segment(start_idx, fallback_cut_idx, check, drop=True, reason="fallback discarded latest cut")],
            [check],
            "fallback: used latest submit_cut as discard",
        )

    first_sentence_end = boundaries[0].token_idx if boundaries else len(chars) - 1
    check = _check_length(
        audio=audio,
        sample_rate=sample_rate,
        chars=chars,
        start_idx=start_idx,
        end_idx=first_sentence_end,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
        cache=cache,
    )
    return (
        [
            _make_segment(
                start_idx,
                first_sentence_end,
                check,
                drop=not check.valid,
                reason="fallback first sentence, keep only if length-valid",
            )
        ],
        [check],
        "fallback: sliced first sentence",
    )


async def _call_submit_cut(
    *,
    llm_client: UnifiedClient,
    memory_manager: MemoryManager,
    model: str,
    provider: ProviderName | None,
    prompt: dict[str, Any],
    block_index: int,
):
    return await llm_client.send_prompt(
        memory_manager,
        json.dumps(build_submit_cut_prompt(prompt), ensure_ascii=False, indent=2),
        provider=provider,
        model=model,
        temperature=0,
        tools=[_submit_cut_tool()],
        tool_choice="required",
        trace_name="slide_LLM.rough_cut.submit_cut",
        metadata={"phase": "rough_cut", "block_index": block_index},
    )


async def _call_submit_segments(
    *,
    llm_client: UnifiedClient,
    memory_manager: MemoryManager,
    model: str,
    provider: ProviderName | None,
    prompt: dict[str, Any],
    feedback: str,
    first_given_idx: int | None,
    block_index: int,
    round_index: int,
):
    payload = build_submit_segments_prompt(
        base_prompt=prompt,
        feedback=feedback,
        first_given_idx=first_given_idx,
    )
    return await llm_client.send_prompt(
        memory_manager,
        json.dumps(payload, ensure_ascii=False, indent=2),
        provider=provider,
        model=model,
        temperature=0,
        tools=[_submit_segments_tool()],
        tool_choice="required",
        trace_name="slide_LLM.rough_cut.submit_segments",
        metadata={"phase": "rough_cut", "block_index": block_index, "round": round_index},
    )


async def _call_retry_tool_choice(
    *,
    llm_client: UnifiedClient,
    memory_manager: MemoryManager,
    model: str,
    provider: ProviderName | None,
    prompt: dict[str, Any],
    feedback: str,
    first_given_idx: int | None,
    block_index: int,
    round_index: int,
):
    payload = build_retry_tool_choice_prompt(
        base_prompt=prompt,
        feedback=feedback,
        first_given_idx=first_given_idx,
    )
    return await llm_client.send_prompt(
        memory_manager,
        json.dumps(payload, ensure_ascii=False, indent=2),
        provider=provider,
        model=model,
        temperature=0,
        tools=[_submit_cut_tool(), _submit_segments_tool()],
        tool_choice="required",
        trace_name="slide_LLM.rough_cut.retry_tool_choice",
        metadata={"phase": "rough_cut", "block_index": block_index, "round": round_index},
    )


def _submit_cut_tool() -> ToolDefinition:
    return ToolDefinition(
        name="submit_cut",
        description=(
            "First tool call. Submit one rough cut candidate using a visible bracket marker number. "
            "This can be used again only when the first reports 'too short' or 'illegal' error."
        ),
        parameters={
            "type": "object",
            "properties": {
                "cut_idx": {
                    "type": "integer",
                    "description": "Visible bracket marker number, for example 2 for [2: 3.5s: 820ms].",
                }
            },
            "required": ["cut_idx"],
            "additionalProperties": False,
        },
    )


def _submit_segments_tool() -> ToolDefinition:
    return ToolDefinition(
        name="submit_segments",
        description=(
            "Submit rough segment end marker numbers and keep/discard states. "
            "For submit_segments split submissions, the last segment end_idx must equal first_given_idx "
            "from the prompt, meaning the list partitions the first attempted slice."
        ),
        parameters={
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "end_idx": {
                                "type": "integer",
                                "description": (
                                    "Visible bracket marker number where this piece ends, for example 3 for [3: 4.2s: long pause]. "
                                    "The final segment object's end_idx must equal first_given_idx from the prompt."
                                ),
                            },
                            "keep": {"type": "boolean"},
                        },
                        "required": ["end_idx", "keep"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["segments"],
            "additionalProperties": False,
        },
    )


def _parse_first_cut(response: Any) -> Optional[int]:
    for call in getattr(response, "tool_calls", []) or []:
        if call.name == "submit_cut":
            value = call.arguments.get("cut_idx")
            if isinstance(value, int):
                return value
    return None


def _chosen_tool_name(response: Any) -> Optional[str]:
    for call in getattr(response, "tool_calls", []) or []:
        if call.name in {"submit_cut", "submit_segments"}:
            return call.name
    return None


def _parse_and_resolve_cut(
    response: Any,
    marker_to_boundary: dict[int, CandidateBoundary],
) -> tuple[Optional[CutChoice], Optional[str]]:
    marker_id = _parse_first_cut(response)
    if marker_id is None:
        return None, "missing or invalid submit_cut.cut_idx"
    boundary = marker_to_boundary.get(marker_id)
    if boundary is None:
        return None, f"cut_idx must be one of visible marker ids: {sorted(marker_to_boundary)}"
    return CutChoice(marker_id=marker_id, end_idx=boundary.token_idx), None


def _parse_segment_proposals(response: Any) -> tuple[Optional[list[SegmentProposal]], Optional[str]]:
    arguments: Optional[dict[str, Any]] = None
    for call in getattr(response, "tool_calls", []) or []:
        if call.name == "submit_segments":
            arguments = call.arguments
            break
    if arguments is None:
        return None, "missing submit_segments tool call"
    raw_segments = arguments.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        return None, "segments must be a non-empty list"
    proposals: list[SegmentProposal] = []
    last_end = -1
    for item in raw_segments:
        if not isinstance(item, dict) or not isinstance(item.get("end_idx"), int) or not isinstance(item.get("keep"), bool):
            return None, "each segment must contain integer end_idx and boolean keep"
        end_idx = int(item["end_idx"])
        if end_idx <= last_end:
            return None, "end_idx values must be strictly increasing"
        proposals.append(SegmentProposal(end_idx=end_idx, keep=bool(item["keep"])))
        last_end = end_idx
    return proposals, None


def _extract_candidate_boundaries(
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    prepass: FullPrepass,
) -> list[CandidateBoundary]:
    boundaries: list[CandidateBoundary] = []
    chars = prepass.global_chars
    for token_idx, left in enumerate(chars[:-1]):
        positions = corrected_token_to_positions.get(token_idx)
        if not positions:
            continue
        cursor = max(positions) + 1
        punct_run: list[str] = []
        while cursor < len(corrected_full_text) and is_punct_or_space(corrected_full_text[cursor]):
            punct_run.append(corrected_full_text[cursor])
            cursor += 1
        visible = [ch for ch in punct_run if not ch.isspace()]
        if not visible:
            continue
        kind = "strong" if any(ch in STRONG_STOPS or ch in ".!?。？！" for ch in visible) else "weak"
        pause_ms = pause_ms_after_token(prepass, token_idx)
        boundaries.append(
            CandidateBoundary(
                token_idx=token_idx,
                text_pos=max(positions),
                boundary_text="".join(punct_run),
                boundary_kind=kind,
                pause_ms=round(pause_ms, 1),
                cut_sec=float(left.end_sec),
                middle_sec=float(left.end_sec) + (float(chars[token_idx + 1].start_sec) - float(left.end_sec)) * 0.5,
            )
        )
    if chars:
        boundaries.append(
            CandidateBoundary(
                token_idx=len(chars) - 1,
                text_pos=max(corrected_token_to_positions.get(len(chars) - 1, [len(corrected_full_text) - 1])),
                boundary_text="<final>",
                boundary_kind="final",
                pause_ms=0.0,
                cut_sec=float(chars[-1].end_sec),
                middle_sec=float(chars[-1].end_sec),
            )
        )
    return boundaries


def _select_window(
    *,
    start_idx: int,
    boundaries: Sequence[CandidateBoundary],
    chars: Sequence[Any],
) -> list[CandidateBoundary]:
    start_sec = _segment_start_sec(chars, start_idx)
    selected: list[CandidateBoundary] = []
    sentence_count = 0
    base_done = False
    for boundary in boundaries:
        if boundary.token_idx < start_idx:
            continue
        if not base_done:
            selected.append(boundary)
            if boundary.boundary_kind in {"strong", "final"}:
                sentence_count += 1
            duration = _segment_end_sec(chars, boundary.token_idx) - start_sec
            if duration >= 15.0 or sentence_count >= 3 or boundary.boundary_kind == "final":
                base_done = True
            continue
        if len([b for b in selected if b.boundary_kind in {"strong", "final"}]) >= sentence_count + MAX_EXTENSION_SENTENCES:
            break
        selected.append(boundary)
        if boundary.boundary_kind == "final":
            break
    return selected


def _prompt_summary(
    *,
    prepass: FullPrepass,
    corrected_full_text: str,
    corrected_token_to_positions: dict[int, list[int]],
    boundaries: Sequence[CandidateBoundary],
    start_idx: int,
    min_seg_sec: float,
    max_seg_sec: float,
    pause_rankings: dict[int, PausePromptRank],
) -> dict[str, Any]:
    end_idx = boundaries[-1].token_idx if boundaries else start_idx
    marked_text = _extract_marked_span_text(
        corrected_full_text=corrected_full_text,
        token_to_positions_map=corrected_token_to_positions,
        first_token_idx=start_idx,
        last_token_idx=end_idx,
        boundaries=boundaries,
        marker_origin_sec=_segment_start_sec(prepass.global_chars, start_idx),
        prepass=prepass,
        pause_rankings=pause_rankings,
    )
    return build_rough_cut_base_prompt(
        start_idx=start_idx,
        marked_text=marked_text,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
    )


def _rough_cut_pause_stats(prepass: FullPrepass) -> dict[str, Any]:
    raw = compact_pause_stats(prepass)

    def _pick(name: str) -> dict[str, Any]:
        bucket = raw[name]
        return {
            "p20_ms": bucket["p20_ms"],
            "p50_ms": bucket["p50_ms"],
            "p80_ms": bucket["p80_ms"],
        }

    return {
        "explanation": (
            "Pause-ms distribution for this timbre/source. strong_stops are pauses after sentence-ending punctuation; "
            "weak_stops are pauses after comma-like punctuation; no_pause means pauses after boundaries without recognized strong/weak punctuation, "
            "including plain character-to-character flow and punctuation/symbols outside the strong/weak sets. "
            "Only p20, p50, and p80 are shown."
        ),
        "strong_stops": _pick("strong_stops"),
        "weak_stops": _pick("weak_stops"),
        "no_pause": _pick("no_punc"),
    }


def _pause_prompt_rankings(boundaries: Sequence[Any]) -> dict[int, PausePromptRank]:
    rankings: dict[int, PausePromptRank] = {}
    ranked = sorted(
        (
            (int(boundary.token_idx), float(boundary.pause_ms))
            for boundary in boundaries
            if getattr(boundary, "boundary_kind", None) in {"weak", "strong"}
        ),
        key=lambda item: (-item[1], item[0]),
    )
    total = len(ranked)
    previous_pause: float | None = None
    current_rank = 0
    for index, (token_idx, pause_ms) in enumerate(ranked, start=1):
        if previous_pause is None or pause_ms != previous_pause:
            current_rank = index
            previous_pause = pause_ms
        rankings[token_idx] = PausePromptRank(rank=current_rank, total=total)
    return rankings


def _pause_prompt_label(boundary: Any, prepass: FullPrepass, rankings: dict[int, PausePromptRank]) -> str:
    if boundary.boundary_kind == "final":
        return "final boundary"
    pause_class = _pause_class_label(float(boundary.pause_ms), prepass, rankings.get(int(boundary.token_idx)))
    rank = rankings.get(int(boundary.token_idx))
    if rank is None:
        return pause_class
    return pause_class


def _pause_class_label(pause_ms: float, prepass: FullPrepass, rank: PausePromptRank | None = None) -> str:
    if rank is None or rank.total <= 0:
        weak = prepass.weak
        strong = prepass.strong
        if pause_ms < min(float(weak.p20_ms), float(strong.p20_ms)):
            return "very short pause"
        if pause_ms < min(float(weak.p40_ms), float(strong.p40_ms)):
            return "short pause"
        if pause_ms < max(float(weak.p60_ms), float(strong.p60_ms)):
            return "median pause"
        if pause_ms <= max(float(weak.p80_ms), float(strong.p80_ms)):
            return "long pause"
        return "very long pause"

    percentile = rank.rank / float(rank.total)
    if percentile <= 0.2:
        return "very long pause"
    if percentile <= 0.4:
        return "long pause"
    if percentile <= 0.6:
        return "median pause"
    if percentile <= 0.8:
        return "short pause"
    return "very short pause"


def _resolve_segment_markers(
    proposals: list[SegmentProposal],
    marker_to_boundary: dict[int, CandidateBoundary],
) -> tuple[Optional[list[SegmentProposal]], Optional[str]]:
    resolved: list[SegmentProposal] = []
    for proposal in proposals:
        boundary = marker_to_boundary.get(proposal.end_idx)
        if boundary is None:
            return None, f"end_idx must be one of visible marker ids: {sorted(marker_to_boundary)}"
        resolved.append(SegmentProposal(end_idx=boundary.token_idx, keep=proposal.keep, marker_id=proposal.end_idx))
    last_end = -1
    for proposal in resolved:
        if proposal.end_idx <= last_end:
            return None, "resolved end_idx values must be strictly increasing"
        last_end = proposal.end_idx
    return resolved, None


def _extract_marked_span_text(
    *,
    corrected_full_text: str,
    token_to_positions_map: dict[int, list[int]],
    first_token_idx: int,
    last_token_idx: int,
    boundaries: Sequence[CandidateBoundary],
    prepass: FullPrepass,
    pause_rankings: dict[int, PausePromptRank],
    marker_origin_sec: float = 0.0,
) -> str:
    positions: list[int] = []
    for token_idx in range(first_token_idx, last_token_idx + 1):
        positions.extend(token_to_positions_map.get(token_idx, []))
    if not positions:
        return ""

    start_pos = min(positions)
    end_pos = max(positions) + 1
    while end_pos < len(corrected_full_text) and is_punct_or_space(corrected_full_text[end_pos]):
        end_pos += 1

    insertions: list[tuple[int, str]] = []
    for marker_id, boundary in enumerate(boundaries, start=1):
        if boundary.token_idx < first_token_idx or boundary.token_idx > last_token_idx:
            continue
        if boundary.boundary_kind == "final":
            insert_pos = end_pos
        else:
            insert_pos = boundary.text_pos + 1 + len(boundary.boundary_text)
            insert_pos = min(max(insert_pos, start_pos), end_pos)
        relative_sec = max(0.0, float(boundary.middle_sec) - float(marker_origin_sec))
        insertions.append(
            (
                insert_pos - start_pos,
                _inline_marker(
                    marker_id,
                    relative_sec,
                    _pause_prompt_label(boundary, prepass, pause_rankings),
                ),
            )
        )

    text = corrected_full_text[start_pos:end_pos]
    for rel_pos, marker in sorted(insertions, reverse=True):
        text = text[:rel_pos] + marker + text[rel_pos:]

    return text


def _inline_marker(marker_id: int, relative_sec: float, pause_label: str) -> str:
    return f"[{marker_id}: {float(relative_sec):.1f}s: {pause_label}]"


def _extract_span_text(
    corrected_full_text: str,
    token_to_positions_map: dict[int, list[int]],
    first_token_idx: int,
    last_token_idx: int,
) -> str:
    positions: list[int] = []
    for token_idx in range(first_token_idx, last_token_idx + 1):
        positions.extend(token_to_positions_map.get(token_idx, []))
    if not positions:
        return ""
    start_pos = min(positions)
    end_pos = max(positions) + 1
    while end_pos < len(corrected_full_text) and is_punct_or_space(corrected_full_text[end_pos]):
        end_pos += 1
    return corrected_full_text[start_pos:end_pos]


def _check_length(
    *,
    audio: np.ndarray,
    sample_rate: int,
    chars: Sequence[Any],
    start_idx: int,
    end_idx: int,
    min_seg_sec: float,
    max_seg_sec: float,
    cache: dict[tuple[int, int], LengthCheck],
) -> LengthCheck:
    start_idx = max(0, min(start_idx, len(chars) - 1))
    end_idx = max(start_idx, min(end_idx, len(chars) - 1))
    start_sec = _segment_start_sec(chars, start_idx)
    end_sec = _segment_end_sec(chars, end_idx)
    start_sample = max(0, int(round(start_sec * sample_rate)))
    end_sample = min(len(audio), max(start_sample, int(round(end_sec * sample_rate))))
    key = (start_sample, end_sample)
    cached = cache.get(key)
    if cached is not None:
        return cached
    seg_audio = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
    if seg_audio.size == 0:
        trimmed_sec = 0.0
    else:
        import librosa

        trimmed, index = librosa.effects.trim(seg_audio, top_db=TRIM_TOP_DB)
        if trimmed.size == 0 or len(index) != 2 or int(index[1]) <= int(index[0]):
            trimmed_sec = 0.0
        else:
            pad = max(0, int(round(TRIM_PADDING_SEC * sample_rate)))
            trim_start = max(0, int(index[0]) - pad)
            trim_end = min(len(seg_audio), int(index[1]) + pad)
            trimmed_sec = max(0.0, (trim_end - trim_start) / float(sample_rate))
    check = LengthCheck(
        start_idx=start_idx,
        end_idx=end_idx,
        start_sec=start_sec,
        end_sec=end_sec,
        raw_duration_sec=max(0.0, end_sec - start_sec),
        trimmed_duration_sec=trimmed_sec,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
    )
    cache[key] = check
    return check


def _make_segment(start_idx: int, end_idx: int, check: LengthCheck, *, drop: bool, reason: str) -> RoughSegment:
    return RoughSegment(
        char_start_idx=int(start_idx),
        char_end_idx=int(end_idx),
        start_sec=float(check.start_sec),
        end_sec=float(check.end_sec),
        drop=bool(drop),
        reason=(
            f"{reason}; trimmed {check.trimmed_duration_sec:.3f}s "
            f"against [{check.min_seg_sec:.3f}, {check.max_seg_sec:.3f}]"
        ),
    )


def _make_trace(
    *,
    block_index: int,
    start_idx: int,
    end_idx: int,
    chars: Sequence[Any],
    prompt_summary: dict[str, Any],
    response_cuts: list[dict[str, Any]],
    applied: list[dict[str, Any]],
    error: Optional[str],
) -> BlockCallTrace:
    return BlockCallTrace(
        block_index=block_index,
        char_start_idx=int(start_idx),
        char_end_idx=int(end_idx),
        block_start_sec=float(chars[start_idx].start_sec) if chars else 0.0,
        block_end_sec=float(chars[end_idx].end_sec) if chars else 0.0,
        prompt_summary=prompt_summary,
        response_cuts=response_cuts,
        applied_cuts=applied,
        carry_over_from_idx=None,
        error=error,
    )


def _segment_trace(segment: RoughSegment, check: LengthCheck) -> dict[str, Any]:
    return {
        "cut_char_idx": segment.char_end_idx,
        "drop": segment.drop,
        "reason": segment.reason,
        "length": dataclasses.asdict(check) | {"valid": check.valid},
    }


def _length_error_message(check: LengthCheck) -> str:
    return (
        f"Length error: current slice is {check.trimmed_duration_sec:.3f}s after trim; "
        f"expected [{check.min_seg_sec:.3f}, {check.max_seg_sec:.3f}] seconds."
    )
