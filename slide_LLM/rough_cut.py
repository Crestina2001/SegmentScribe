"""Phase 3 - LLM rough cut planning for ``slide_LLM``."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from llm_gateway import MemoryManager, UnifiedClient
from llm_gateway.models import ProviderName, ToolDefinition
from slicing_utils.prepass import FullPrepass, compact_pause_stats, pause_ms_after_token
from slicing_utils.rough_cut import (
    BlockCallTrace,
    Phase3Result,
    RoughSegment,
    _segment_end_sec,
    _segment_start_sec,
)
from slicing_utils.shared import STRONG_STOPS, WEAK_STOPS

from .local_prompts import (
    ROUGH_CUT_SYSTEM_PROMPT,
    build_rough_cut_base_prompt,
    build_retry_tool_choice_prompt,
    build_submit_cut_prompt,
    build_submit_segments_prompt,
)
from .text_rules import is_punct_or_space


TRIM_TOP_DB = 60
TRIM_PADDING_SEC = 0.1
MAX_EXTENSION_SENTENCES = 2


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
) -> Phase3Result:
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
) -> tuple[list[RoughSegment], BlockCallTrace]:
    prompt_summary = _prompt_summary(
        prepass=prepass,
        corrected_full_text=corrected_full_text,
        corrected_token_to_positions=corrected_token_to_positions,
        boundaries=boundaries,
        start_idx=start_idx,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
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
                                    "Visible bracket marker number where this piece ends, for example 3 for [3: 4.2s: 560ms]. "
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
) -> dict[str, Any]:
    end_idx = boundaries[-1].token_idx if boundaries else start_idx
    marked_text = _extract_marked_span_text(
        corrected_full_text=corrected_full_text,
        token_to_positions_map=corrected_token_to_positions,
        first_token_idx=start_idx,
        last_token_idx=end_idx,
        boundaries=boundaries,
        marker_origin_sec=_segment_start_sec(prepass.global_chars, start_idx),
    )
    return build_rough_cut_base_prompt(
        start_idx=start_idx,
        marked_text=marked_text,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
        pause_stats=_rough_cut_pause_stats(prepass),
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
        insertions.append((insert_pos - start_pos, _inline_marker(marker_id, relative_sec, boundary.pause_ms)))

    text = corrected_full_text[start_pos:end_pos]
    for rel_pos, marker in sorted(insertions, reverse=True):
        text = text[:rel_pos] + marker + text[rel_pos:]

    return text


def _inline_marker(marker_id: int, relative_sec: float, pause_ms: float) -> str:
    rounded_pause_ms = int(round(float(pause_ms)))
    return f"[{marker_id}: {float(relative_sec):.1f}s: {rounded_pause_ms}ms]"


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
