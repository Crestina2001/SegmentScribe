# Technical Report: `slide_rule` Slicing Pipeline

## Purpose

`slide_rule` converts prepared speech audio into transcript-aligned clips for TTS data preparation. It takes one audio file or a directory of audio files, runs ASR and forced alignment, chooses punctuation-aware cut points, trims silence, filters clips by duration, and writes audio, transcripts, traces, manifests, and VoxCPM-style JSONL.

The key user-facing duration controls are:

| Option | Role |
| --- | --- |
| `--min-seg-sec` | Minimum accepted final segment duration. |
| `--max-seg-sec` | Maximum accepted final segment duration. |
| `--preprocess-chunk-sec` | Maximum packed ASR chunk size during the prepass. |
| `--target-sample-rate` | Sample rate used internally after loading/resampling. |

The implementation is split across:

- `slide_rule.pipeline`: orchestration, source iteration, summary writing.
- `slicing_utils.prepass`: VAD packing, ASR/alignment stitching, pause statistics.
- `slide_rule.punctuation`: optional rule-based punctuation edits.
- `slicing_utils.rough_cut`: punctuation-boundary rough cut planning.
- `slide_rule.thin_cut`: silence-aware boundary refinement.
- `slicing_utils.filter_write`: final filtering and output writing.

## End-to-End Flow

### 1. CLI Configuration And Source Discovery

The CLI builds a `RuleWorkflowConfig` from command-line arguments, then runs `SlideRulePipeline`. The input can be either a single audio file or a directory. Directory inputs are recursively scanned for allowed audio extensions.

Before processing, `slide_rule` creates the output directory and constructs one `AsrBackend` with the configured model path, forced-aligner path, backend, device, dtype, batch size, language hint, and backend-specific kwargs.

### 2. Audio Loading

Each source file is loaded with `soundfile`, converted to mono by averaging channels when needed, converted to `float32`, and resampled with `librosa` if the source sample rate differs from `--target-sample-rate`.

The pipeline rejects a source early when its original loaded duration exceeds `--max-source-seconds`.

### 3. Full-Source Prepass

The prepass in `slicing_utils.prepass` creates the global transcript and alignment structure used by every later phase.

The prepass does four important things:

1. Runs VAD using the configured backend (`auto`, `silero`, or `librosa`).
2. Packs speech regions into chunks, usually bounded by `--preprocess-chunk-sec`.
3. Runs ASR and forced alignment on those chunks through `AsrBackend`.
4. Stitches all chunk-level character timestamps into one source-wide character stream.

The stitched output is represented by `FullPrepass`:

| Field | Meaning |
| --- | --- |
| `full_text` | Concatenated ASR text for the whole source. |
| `global_chars` | Character tokens with source-absolute timestamps. |
| `chunk_spans` | Per-chunk timing, text, and character index ranges. |
| `strong`, `weak`, `no_punc` | Pause percentile statistics by punctuation category. |
| `warnings` | Suspicious pause spans based on source-wide statistics. |

The prepass writes `prepass/<source>/stream.json`, which is the best first stop for checking whether ASR text and timestamps are plausible.

### 4. Pause Statistics And Warnings

After stitching, the pipeline computes pause distributions for three categories:

| Category | Meaning |
| --- | --- |
| `strong` | Pause after strong sentence-ending punctuation. |
| `weak` | Pause after weaker punctuation such as commas or semicolons. |
| `no_punc` | Pause after a non-punctuation character. |

These distributions produce percentiles such as `p5_ms`, `p20_ms`, `p50_ms`, `p80_ms`, and `p95_ms`. Later phases use those percentiles to decide whether a boundary has a natural pause, an unusually short pause, or a missing punctuation mark.

Warnings identify suspicious timing/punctuation combinations, for example a long pause where there is no punctuation or a very short pause after strong punctuation.

### 5. Optional Rule-Based Punctuation Correction

Punctuation correction is disabled by default. When `--enable-punctuation-correction` is set, `slide_rule.punctuation` applies conservative punctuation-only edits based on pause statistics.

The correction phase can:

- Remove strong punctuation after unusually short silence.
- Remove weak punctuation after unusually short silence.
- Add a comma-like punctuation mark after unusually long silence without punctuation.

The correction phase has a defensive guard: if the spoken content changes after ignoring punctuation and whitespace, all edits are rejected. This keeps the phase punctuation-only.

The output is written to:

- `traces/<source>/punctuation.json`
- `intermediate/<source>/phase2_punctuation.json`

When correction is disabled, the phase passes the original ASR text through unchanged while still producing the mappings needed by later phases.

## Rough Cut Planning

The rough cut phase in `slicing_utils.rough_cut` chooses candidate segment boundaries from punctuation and character-level timestamps. It is deterministic and does not call an LLM.

### Candidate Boundaries

Boundaries are extracted from punctuation runs after aligned characters. The phase also appends one synthetic final boundary so the end of the source can be planned.

Each boundary records:

| Field | Meaning |
| --- | --- |
| `boundary_kind` | `strong`, `weak`, or `final`. |
| `quality` | `perfect`, `ok`, or `bad`. |
| `cut_policy` | `must_cut`, `normal`, or `no_cut`. |
| `pause_ms` | Gap between the current character and the next character. |
| `cut_sec` | Timestamp used as the candidate cut position. |
| `reason` | Human-readable explanation of the classification. |

Strong punctuation includes sentence-ending punctuation. Weak punctuation includes comma-like punctuation. Any final boundary is always treated as `must_cut`.

### Boundary Quality

Boundary quality is assigned by comparing the observed pause against source-specific pause percentiles.

- Strong boundaries become `perfect` when the pause is at least the stronger of the strong median, weak p80, or 180 ms.
- Strong boundaries become `ok` when the pause is at least the stronger of strong p20, weak median, or 90 ms.
- Weak boundaries become `perfect` when the pause is at least the stronger of weak p80, no-punctuation p95, or 220 ms.
- Weak boundaries become `ok` when the pause is at least the stronger of weak median, no-punctuation p80, or 120 ms.
- Otherwise the boundary is `bad`.

### Cut Policy

Cut policy controls what the dynamic planner is allowed to do:

| Policy | Meaning |
| --- | --- |
| `must_cut` | The planner may not skip this boundary. |
| `normal` | The planner may cut here if it improves the plan. |
| `no_cut` | The planner should avoid cutting here. |

Long pauses after strong or weak punctuation tend to become `must_cut`. Very short weak pauses can become `no_cut`.

### Dynamic Planning

The planner works across anchor windows. It prefers an anchor boundary in the 15-30 second range when possible, then plans one or more segments inside that window.

For each possible segment, it estimates the duration after silence trimming using `librosa.effects.trim(top_db=60)`. It then scores candidate paths in this order:

1. Valid final duration inside `[--min-seg-sec, --max-seg-sec]`.
2. Maybe-valid duration within a small slack window or valid raw duration.
3. Boundary quality, preferring fewer `bad` and then fewer `ok` boundaries.
4. Deviation from the midpoint of the configured duration range.

The rough planner may still emit segments that are too short or too long. That is intentional: phase 5 is the final authority that keeps or discards segments.

Rough cut details are written to `intermediate/<source>/phase3_rough_cut.json` and summarized in `traces/<source>/rough_cut.json`.

## Thin Cut Refinement

The thin cut phase in `slide_rule.thin_cut` refines each rough segment using `librosa.effects.trim(top_db=60)`.

For each rough segment:

1. Slice the rough audio range.
2. Trim leading and trailing silence.
3. Move the segment start to the trimmed start.
4. Move the final cut point to the trimmed end.
5. Record the trim samples and any fallback error.

If trimming fails or produces an empty segment, the phase falls back to the original rough boundaries and records `zoom_error="trim_empty"`.

Thin cut details are written to:

- `intermediate/<source>/phase4_zoom_cut.json`
- `traces/<source>/thin_cut.json`

The word `zoom` appears in some field names for compatibility with earlier workflow traces; in `slide_rule`, this phase is rule-based trimming rather than a model call.

## Final Filtering And Output Writing

The final phase in `slicing_utils.filter_write` converts refined segments into files.

For each refined segment:

1. Project the corrected full transcript onto the segment character range.
2. Compute final duration as `refined.cut_start_sec - rough.start_sec`.
3. Drop segments marked `drop=true`.
4. Drop segments outside `[--min-seg-sec, --max-seg-sec]`.
5. Write kept audio and transcript files.
6. Write discarded audio, transcript, and JSON metadata for dropped segments when not in dry-run mode.

Kept segments become `SegmentRecord` entries. Those records drive `manifest.tsv`, the VoxCPM JSONL file, and `summary.json`.

## Output Layout

For an output directory such as `audios/zhouquanquan_final_4`, the writer creates this layout:

| Path | Contents |
| --- | --- |
| `audios/<source>/<segment>.wav` | Final kept audio clips. |
| `transcripts/<source>/<segment>.txt` | Transcript text for kept clips. |
| `traces/<source>/<segment>.json` | Per-kept-segment rough/refined timing and transcript trace. |
| `discarded_audios/<source>/<segment>.*` | Dropped clips, transcript text, and metadata. |
| `prepass/<source>/stream.json` | Full-source ASR text, aligned chars, chunks, pause stats, warnings. |
| `traces/<source>/punctuation.json` | Punctuation correction mode and edits. |
| `traces/<source>/rough_cut.json` | Rough-cut trace summary. |
| `traces/<source>/thin_cut.json` | Thin-cut trace summary. |
| `intermediate/<source>/phase2_punctuation.json` | Full punctuation phase object. |
| `intermediate/<source>/phase3_rough_cut.json` | Full rough-cut phase object. |
| `intermediate/<source>/phase4_zoom_cut.json` | Full thin-cut phase object. |
| `manifest.tsv` | Tab-separated manifest for kept clips. |
| `summary.json` | Run-level status, counts, config, and per-source summaries. |
| `<input_stem>_voxcpm.jsonl` | VoxCPM-style JSONL with `audio`, `text`, and `duration`. |

## How To Debug A Segment

Start with the per-segment trace, for example:

```text
traces/001/000008.json
```

Important fields:

| Field | What to inspect |
| --- | --- |
| `source_start_sec`, `source_end_sec`, `duration_sec` | Final kept segment timing. |
| `rough.reason` | Why the rough planner chose this boundary. |
| `rough.char_start_idx`, `rough.char_end_idx` | Character range used for transcript projection. |
| `refined.zoom_reason` | Whether thin cut used `librosa.trim`. |
| `refined.zoom_error` | Whether trimming fell back. |
| `transcript` | Final transcript written for the clip. |
| `audio_relpath`, `transcript_relpath` | Output files connected to this trace. |

For a trace like segment `000008`, the `rough.reason` may look like:

```text
normal / bad strong pause at '...' (0ms); must-valid segment (raw 4.16s, trimmed 4.16s)
```

That means the boundary was a strong punctuation boundary, but the measured pause was short enough to classify as `bad`. The segment was still accepted because the planner found it duration-valid, and phase 5 kept it because the final duration was inside the configured range.

To investigate further:

1. Open `intermediate/<source>/phase3_rough_cut.json` and find the matching `char_start_idx` / `char_end_idx`.
2. Check `applied_cuts` to see the planner's selected cut path and boundary scores.
3. Open `intermediate/<source>/phase4_zoom_cut.json` and find the matching `segment_index`.
4. Compare original rough timing with trimmed timing.
5. If the segment was dropped, inspect `discarded_audios/<source>/<segment>.json` for the final drop reason.

## Notes And Limitations

- Segment quality depends heavily on ASR and forced-aligner timestamp quality. Bad timestamps can produce bad pause measurements and boundary choices.
- Punctuation correction is intentionally conservative. It only edits punctuation and rejects all edits if spoken content would change.
- Rough-cut planning can emit invalid-length segments when required by boundary policy or source structure. Final filtering drops those segments.
- Thin cut uses amplitude-based silence trimming, so background noise or music can reduce trimming accuracy.
- Some trace text or punctuation may appear mojibake-like depending on upstream model output or encoding interpretation. The slicer records that data as produced; the report does not attempt to normalize or correct it.
- Field names such as `zoom_*` are compatibility names from earlier workflows. In `slide_rule`, thin cutting is implemented by local `librosa` trimming.

