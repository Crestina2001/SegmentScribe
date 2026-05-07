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
    "1. decide the closest position from the beginning where semantic pause agrees with the pause class and rank;\n"
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


LLM_SLICE_SYSTEM_PROMPT = (
    "You are a TTS slicing expert. You choose cut positions inside a window of 5 target sentences "
    "plus 1 padding sentence shown only as context.\n"
    "Each visible marker [idx: T sec: pause class] sits at a punctuation boundary inside the 5 target sentences: "
    "idx is the marker ID, T is seconds from the start of the window to that boundary, "
    "and pause class is the bucketed silence length using all weak and strong punctuation boundaries together. "
    "The padding sentence after the 5 target sentences has no markers and must not be sliced.\n"
    "## Workflow\n"
    "1. Call submit_slices with cut_indices: a strictly increasing list of marker IDs. "
    "The last index MUST be the last visible marker in the text. "
    "Each consecutive pair (previous_end+1, end) defines one segment.\n"
    "2. submit_slices returns, for every segment, its trimmed audio length and whether it is accepted "
    "(trimmed length within expected_range_sec).\n"
    "3. You may call submit_slices again with revised cut_indices to improve the result. "
    "Each round you see fresh length feedback. Up to a few rounds are allowed. "
    "When the latest submission is good enough, call the confirm_slices tool. "
    "The latest valid submit_slices call before confirm_slices becomes the final slicing decision. "
    "The confirm_slices tool takes no input parameters.\n"
    "Accepted segments are kept; rejected (length-invalid) segments are dropped from the dataset.\n"
    "## Priority: semantic integrity over keeping segments\n"
    "- Prefer cutting at boundaries where the preceding span forms a coherent mini-thought "
    "(clause, list item, contrast, cause/effect, discourse beat).\n"
    "- Do NOT cut where it would break a coherent unit (subject from verb, fixed phrase, "
    "preposition from object, conjunction from following clause), even if length would be valid.\n"
    "- Avoid cutting around very short pause/short pause just to make segment length valid. "
    "- It is acceptable for some segments to be rejected when no semantically clean cut fits the length range; "
    "those rejected segments are simply dropped, while the others are kept. "
    "Do NOT introduce a semantically bad cut just to keep more audio.\n"
    "- Within those constraints, try to maximize the number of accepted segments."
)


PAUSE_CLASSIFICATION_SYSTEM_PROMPT = (
    "Task: Semantic Transcript Segmentation\n"
    "You are a linguistic expert specializing in discourse analysis. "
    "Your objective is to evaluate potential segmentation points (slicing boundaries) without being overly strict. "
    "The main goal is to find comma-like boundaries that are good or acceptable cut points; "
    "periods, question marks, exclamation marks, and other terminal sentence boundaries should be treated as good by default "
    "unless cutting there would clearly break meaning.\n"
    "The bracket marker before a text span, like [1], refers to the punctuation at the end of that marked span. "
    "Return labels using the submit_pause_labels tool only.\n"
    "## Classification Criteria\n"
    "Good (Preferred Boundary): A natural cut point. "
    "Label terminal sentence punctuation as good by default. "
    "For comma-like punctuation, label good when the preceding span forms a coherent mini-thought, list item, contrast, "
    "cause/effect unit, speaker turn, or discourse beat that can stand alone naturally.\n"
    "OK (Acceptable Boundary): A safe but non-ideal break point. "
    "Use ok generously for comma-like punctuation when cutting there would preserve grammar and be understandable, "
    "even if the surrounding sentences belong to the same paragraph or thought process. "
    "OK means usable, not perfect.\n"
    "Bad (Invalid Boundary): Reserve bad for clear semantic or grammatical fractures. "
    "Examples: separating a subject from its verb, splitting a fixed phrase, separating a preposition from its object, "
    "cutting immediately after a conjunction that needs the following clause, or leaving the listener unable to parse the meaning. "
    "Do not mark a boundary bad merely because another nearby boundary is better."
)


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


def build_pause_classification_prompt(
    *,
    marked_text: str,
    target_sentence_indices: list[int],
    target_markers: list[int],
    marker_pause_ms: dict[str, float],
    pause_stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": "Classify the semantic/audio cut quality around each visible punctuation marker.",
        "text": marked_text,
        "marker_rule": (
            "A marker such as [1] appears before the text span whose ending punctuation is being classified. "
            "The prompt contains five target sentences when available; padding sentences have no marker and must not receive a label."
        ),
        "labels": {
            "good": "Natural, semantically complete, and suitable as a cut if pause thresholds allow it.",
            "ok": "Usable but weaker than good; only cut when pause timing is strong enough.",
            "bad": "Abrupt, semantically awkward, or should be avoided as a cut.",
        },
        "target_sentence_indices": target_sentence_indices,
        "target_markers": target_markers,
        "pause_stats_ms": pause_stats,
        "boundary_pauses_ms": marker_pause_ms,
        "output_rule": "Call submit_pause_labels with one label for every target marker and no padding markers.",
    }


def build_rough_cut_base_prompt(
    *,
    start_idx: int,
    marked_text: str,
    min_seg_sec: float,
    max_seg_sec: float,
) -> dict[str, Any]:
    return {
        "text_with_cut_markers": marked_text,
        "index_rule": (
            "Bracket markers like [1: 2.3s: long pause] in text_with_cut_markers are the only valid indices. "
            "The middle field is seconds from the beginning of the current segment window to that marker position, "
            "and the last field is the pause class computed from all weak and strong punctuation boundaries together. "
            "submit_cut.cut_idx and submit_segments[].end_idx must be one of these marker numbers, "
            "not a hidden/global character or token index. For example, if text contains 'industry, [2: 3.5s: median pause] then', "
            "submitting 2 means the segment ends at the punctuation immediately before [2]."
        ),
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


def build_llm_slice_base_prompt(
    *,
    marked_text: str,
    padding_sentence_index: int | None,
    min_seg_sec: float,
    max_seg_sec: float,
) -> dict[str, Any]:
    return {
        "task": "Slice the 5 target sentences into segments by choosing cut marker IDs.",
        "text_with_cut_markers": marked_text,
        "marker_rule": (
            "Bracket markers like [3: 4.2s: very long pause] sit at punctuation boundaries inside the 5 target sentences. "
            "idx is the marker ID, the middle field is seconds from the start of this window to the cut position, "
            "and the last field is the pause class computed from all weak and strong punctuation boundaries together. "
            "Only marker IDs shown in text_with_cut_markers are valid; the padding sentence has no markers and cannot be sliced."
        ),
        "padding_sentence_index": padding_sentence_index,
        "expected_range_sec": [min_seg_sec, max_seg_sec],
        "submit_rule": (
            "First call submit_slices with cut_indices: a strictly increasing list of visible marker IDs. "
            "The last value of cut_indices MUST be the last visible marker in text_with_cut_markers. "
            "Each consecutive pair (previous_end+1, end) defines one segment. "
            "After submit_slices returns segment lengths and keep states, either call submit_slices again "
            "with improved cut_indices or call confirm_slices with no parameters. "
            "The latest valid submit_slices call before confirm_slices is the final decision."
        ),
        "cut_quality_rule": (
            "Avoid cutting around weak boundaries just to make segment length valid. "
            "Weak comma-like boundaries with short pauses should be skipped or allowed to create rejected spans "
            "unless the cut is semantically natural and audio-pause-safe."
        ),
    }


def build_llm_slice_submit_prompt(base_prompt: dict[str, Any]) -> dict[str, Any]:
    return dict(base_prompt) | {"required_tool": "submit_slices"}


def build_llm_slice_retry_prompt(
    *,
    base_prompt: dict[str, Any],
    feedback: dict[str, Any] | str,
    round_index: int,
    max_rounds: int,
) -> dict[str, Any]:
    return dict(base_prompt) | {
        "required_tool": "submit_slices",
        "round_index": round_index,
        "max_rounds": max_rounds,
        "tool_result_summary": feedback,
        "tool_choice_rule": (
            "Call submit_slices to revise cut_indices based on the latest tool result. "
            "If the latest result is good enough, stop calling tools and answer with a short final confirmation."
        ),
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
