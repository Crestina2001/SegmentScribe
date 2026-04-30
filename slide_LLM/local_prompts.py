"""Local prompt text and prompt payload builders for ``slide_LLM``."""

from __future__ import annotations

from typing import Any


PUNCTUATION_SYSTEM_PROMPT = (
    "You are a punctuation correction expert. "
    "Return only the fully revised text. Do not revise the text itself, only the punctuation. "
    "The input sentence may contain inline warning markers like {too long: 500 ms} or {too short: 0 ms}. "
    "Use those markers only as pause-timing hints, and never include them in your output. "
    "Validation rule: the text shall not be revised; if text is revised, an error is raised and you must resubmit.\n"
    "## principles\n"
    "1, If a punctuation is semantically plausible but the pause is short, keep it.\n"
    "2, If adding a punctuation to a position where pause is long is semantically weird, don't add it.\n"
    "3, Add or remove a punctuation only when the revised version makes sense semantically."

)

ROUGH_CUT_SYSTEM_PROMPT = (
    "You are a rough cut planning expert. "
    "## Workflow:\n"
    "1. decide the closest position from the beginning where semantic pause agrees with pause_ms;\n"
    "2. Call submit_cut: choose one visible marker as cut_idx. "
    "Use submit_cut for the initial submission and only use submit_cut again when submitted length is too short;\n"
    "3. if the tool call raises a length error, decide on one of these strategies:\n"
    "submitted length is too short: strategy A: call submit_cut again to extend the cut_idx; Strategy B: discard with submit_segments."
    "submitted length is too long: Strategy A: split the segment into several pieces and each piece fits the requirements;\n"
    "Strategy B: When strategy A is not feasible, you may split the segment into several pieces and set some as 'keep=False'(discard) to keep the core segments:\n"
    "    Identify if any sub-part of the sentence can stand alone as a semantically complete unit and meet the length requirement.\n"
    "    If yes, keep that part and discard the rest.\n"
    "    Try to retain as much information as possible.\n"
    "Strategy C: if no semantically meaningful split can satisfy the length requirement, discard the whole segment.\n"
    "4. Use submit_segments to split the already attempted first slice and submit each segment with the state 'keep'/'discard'. "
    "submit_segments succeeds if keep/discard decisions align with real length constraints.\n"
    "5. For submit_segments, the last segments[].end_idx must equal first_given_idx, "
    "the marker submitted in the first submit_cut call. The split list partitions only that first attempted slice."
    "## Notes"
    "1, If [start, first good pause] is too short,  [start, second good pause] is too long, use submit_segments to discard [start, first good pause], "
    "NEVER cut in this way: [start, a bad pause position] to satisfy the length requirement.\n"
    "2, For the first submit of submit_cut, only consider where it is a good FIRST pause. Don't consider length requirement."
)

ROUGH_CUT_STRATEGIES = [
    "A: split the segment into several pieces and each piece fits the requirements;",
    (
        "B: When strategy A is not feasible, you may split the segment into several pieces and set some as 'keep=False'(discard) to keep the core segments:\n"
        "    Identify if any sub-part of the sentence can stand alone as a semantically complete unit and meet the length requirement.\n"
        "    If yes, keep that part and discard the rest.\n"
        "    Try to retain as much information as possible."
    ),
    "C: extend the segment to meet the requirements;",
    "D: if no semantically meaningful split can satisfy the length requirement, discard the whole segment.",
]


def build_punctuation_prompt(
    *,
    window_index: int,
    marked_text: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sentence": marked_text,
        "warning_markers": {
            "{too long: N ms}": "The pause at this boundary is unusually long for continuous speech; consider whether punctuation should be stronger or moved here.",
            "{too short: N ms}": "The pause at this boundary is unusually short for the current punctuation; consider whether punctuation should be weaker or removed.",
            "output_rule": "Do not include any warning marker in the revised text.",
        },
        "statistics": stats,
    }


def build_punctuation_retry_prompt(*, validation_error: str) -> dict[str, Any]:
    return {
        "validation_error": validation_error,
        "instruction": (
            "Your previous revision changed spoken text. Please resubmit only the fully revised text "
            "for the same sentence, changing punctuation only."
        ),
    }


def build_rough_cut_base_prompt(
    *,
    start_idx: int,
    marked_text: str,
    min_seg_sec: float,
    max_seg_sec: float,
    pause_stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "text_with_cut_markers": marked_text,
        "index_rule": (
            "Bracket markers like [1: 2.3s: 200ms] in text_with_cut_markers are the only valid indices. "
            "The middle field is seconds from the beginning of the current segment window to that marker position, "
            "and the last field is the pause length. "
            "submit_cut.cut_idx and submit_segments[].end_idx must be one of these marker numbers, "
            "not a hidden/global character or token index. For example, if text contains 'industry, [2: 3.5s: 120ms] then', "
            "submitting 2 means the segment ends at the punctuation immediately before [2]."
        ),
        "statistics": pause_stats,
        "expected_range_sec": [min_seg_sec, max_seg_sec],
        #"strategies": ROUGH_CUT_STRATEGIES,
    }


def build_submit_cut_prompt(base_prompt: dict[str, Any]) -> dict[str, Any]:
    return dict(base_prompt) | {"required_tool": "submit_cut"}


def build_submit_segments_prompt(
    *,
    base_prompt: dict[str, Any],
    feedback: str,
    first_given_idx: int | None,
) -> dict[str, Any]:
    return dict(base_prompt) | {
        "required_tool": "submit_segments",
        "accurate_current_slice_length_feedback": feedback,
        "first_given_idx": first_given_idx,
        "split_constraint": (
            "For this submit_segments split submission, the last segments[].end_idx must equal first_given_idx. "
            "Do not extend beyond or stop before the first submitted cut marker in submit_segments."
        ),
        #"strategies": ROUGH_CUT_STRATEGIES,
    }


def build_retry_tool_choice_prompt(
    *,
    base_prompt: dict[str, Any],
    feedback: str,
    first_given_idx: int | None,
) -> dict[str, Any]:
    return dict(base_prompt) | {
        "required_tool": "choose either submit_cut or submit_segments",
        "accurate_current_slice_length_feedback": feedback,
        "first_given_idx": first_given_idx,
        "tool_choice_rule": (
            "Call submit_cut only when the valid first submit reports 'too short' error."
            "Call submit_segments for split-or-discard decisions inside the submit_cut attempted slice."
        ),
        "split_constraint": (
            "If you call submit_segments, the last segments[].end_idx must equal first_given_idx. "
            "Do not extend beyond or stop before the first submitted cut marker in submit_segments."
        ),
        #"strategies": ROUGH_CUT_STRATEGIES,
    }
