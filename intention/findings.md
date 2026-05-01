# Experimental Findings

## Qwen3-ASR + Forced Aligner Throughput

Date: 2026-05-01

Environment from `loc_dev.md`:

```bash
cd /mnt/d/Conceal/python_repo/tts_data_prep/SegmentScribe
conda activate qwen3-asr
```

Benchmark script:

```bash
python test_scripts/benchmark_qwen3_asr_aligner_throughput.py
```

The benchmark measures three timings separately:

- ASR-only inference
- Qwen3 forced-aligner-only inference
- combined ASR + forced-aligner throughput

### Smoke Test

Command:

```bash
python test_scripts/benchmark_qwen3_asr_aligner_throughput.py \
  --input audios/enhanced_wav_gpu1_batch1 \
  --backends transformers \
  --batch-sizes 1 \
  --chunk-seconds 15 \
  --max-chunks 2 \
  --warmup-batches 0 \
  --timeout-sec 900 \
  --keep-going-after-fail
```

Result:

- candidate: `transformers_b1_c15`
- chunks: `2`
- audio: `30.0s`
- ASR time: `2.625s`
- forced-aligner time: `3.764s`
- combined inference time: `6.389s`
- throughput: `4.696x` realtime

### Tiny vLLM Comparison

Command:

```bash
python test_scripts/benchmark_qwen3_asr_aligner_throughput.py \
  --input audios/enhanced_wav_gpu1_batch1 \
  --backends vllm \
  --vllm-profiles default,eager \
  --batch-sizes 1,2 \
  --chunk-seconds 15 \
  --max-chunks 2 \
  --warmup-batches 0 \
  --timeout-sec 300 \
  --keep-going-after-fail
```

Best result:

- candidate: `vllm_default_b2_c15`
- chunks: `2`
- audio: `30.0s`
- ASR time: `2.577s`
- forced-aligner time: `3.787s`
- combined inference time: `6.364s`
- throughput: `4.714x` realtime

Notes:

- `vllm_eager_b1_c15` and `vllm_eager_b2_c15` failed engine initialization in this environment.

### 30-Second Chunk Sweep

Command:

```bash
python test_scripts/benchmark_qwen3_asr_aligner_throughput.py \
  --input audios/enhanced_wav_gpu1_batch1 \
  --backends transformers,vllm \
  --vllm-profiles default \
  --batch-sizes 1,4,8 \
  --chunk-seconds 30 \
  --max-chunks 8 \
  --warmup-batches 1 \
  --timeout-sec 600 \
  --keep-going-after-fail
```

Results:

| Candidate | ASR sec | Align sec | Combined sec | Throughput |
| --- | ---: | ---: | ---: | ---: |
| `transformers_b1_c30` | `13.647` | `27.101` | `40.748` | `5.579x` |
| `transformers_b4_c30` | `4.638` | `30.696` | `35.334` | `6.434x` |
| `transformers_b8_c30` | `2.850` | `32.419` | `35.269` | `6.446x` |
| `vllm_default_b1_c30` | `4.700` | `27.500` | `32.200` | `7.060x` |
| `vllm_default_b4_c30` | `1.530` | `30.914` | `32.444` | `7.007x` |
| `vllm_default_b8_c30` | `0.791` | `32.724` | `33.515` | `6.783x` |

Best result:

- candidate: `vllm_default_b1_c30`
- chunks: `8`
- audio: `227.341s`
- throughput: `7.060x` realtime

### 60-Second Chunk Sweep

Command:

```bash
python test_scripts/benchmark_qwen3_asr_aligner_throughput.py \
  --input audios/enhanced_wav_gpu1_batch1 \
  --backends vllm,transformers \
  --vllm-profiles default \
  --batch-sizes 1,4 \
  --chunk-seconds 60 \
  --max-chunks 4 \
  --warmup-batches 1 \
  --timeout-sec 600 \
  --keep-going-after-fail
```

Results:

| Candidate | ASR sec | Align sec | Combined sec | Throughput |
| --- | ---: | ---: | ---: | ---: |
| `vllm_default_b1_c60` | `4.688` | `29.661` | `34.349` | `6.618x` |
| `vllm_default_b4_c60` | `1.363` | `36.651` | `38.014` | `5.981x` |
| `transformers_b1_c60` | `12.710` | `29.302` | `42.012` | `5.411x` |
| `transformers_b4_c60` | `4.706` | `36.412` | `41.118` | `5.529x` |

Best result:

- candidate: `vllm_default_b1_c60`
- throughput: `6.618x` realtime

### Practical Conclusion

Larger ASR batches significantly improve ASR-only speed, but the forced aligner becomes the bottleneck and slows down with larger batches in this local environment. End-to-end throughput is therefore not maximized by filling VRAM.

Recommended starting config:

```bash
ASR_BACKEND="vllm"
ASR_BACKEND_KWARGS='{"max_model_len":8192}'
ASR_MAX_BATCH_SIZE=1
PREPROCESS_CHUNK_SEC=30
```

Use the benchmark script before changing production defaults on a new machine. The best ASR-only setting may not be the best ASR + forced-aligner setting.

## RMS-Silence 5-15s Segment Experiment

Date: 2026-05-01

Experiment script:

```bash
python test_scripts/experiment_rms_silence_5_15.py
```

Policy tested:

- Start from the current cursor.
- Only search candidate cuts that make the segment length fall in `5-15s`.
- Compute short-frame RMS over the waveform.
- Mark low-RMS frames as silence with a per-file adaptive threshold.
- Choose the longest contiguous RMS-silence run inside the `5-15s` window.
- Cut at the center of that silence run.

### Default Sweep

Command:

```bash
python test_scripts/experiment_rms_silence_5_15.py \
  --input audios/enhanced_wav_gpu1_batch1 \
  --output-dir test_scripts/logs/rms_silence_5_15_default_all
```

Result:

- files: `40`
- segments: `1171`
- segment duration mean: `9.999s`
- segment duration median: `9.955s`
- segment duration range: `1.458-14.899s`
- silence hit rate: `96.58%`
- median chosen silence: `560.0ms`
- p90 chosen silence: `790.0ms`
- non-tail fallback segments: `0`

The minimum duration below `5s` came from final tail segments. All non-tail cuts stayed in the requested `5-15s` search window.

### Threshold Sensitivity

| Variant | Silence threshold | Min silence | Segments | Median duration | Silence hit | Median silence | P90 silence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `strict` | p20 * `1.5` | `100ms` | `1180` | `9.923s` | `96.61%` | `495.0ms` | `690.0ms` |
| `default` | p25 * `1.8` | `80ms` | `1171` | `9.955s` | `96.58%` | `560.0ms` | `790.0ms` |
| `loose` | p30 * `2.2` | `60ms` | `1171` | `9.908s` | `96.58%` | `695.0ms` | `1045.0ms` |

### Practical Conclusion

The waveform-only 5-15s policy is viable as a deterministic fallback or a fast rough-cut mode. It found usable RMS-silence cuts for about `96.6%` of all segments on the local enhanced WAV set, with median segment length near `10s`.

Recommended starting parameters:

```bash
MIN_SEG_SEC=5
MAX_SEG_SEC=15
RMS_SILENCE_PERCENTILE=25
RMS_SILENCE_THRESHOLD_MULTIPLIER=1.8
RMS_MIN_SILENCE_MS=80
```

This should not fully replace punctuation-aware cutting yet, because waveform silence does not know semantic sentence boundaries. It is a strong candidate for a fallback path when punctuation/LLM cutting fails, or as a pre-cut policy before transcript-aware refinement.

## Forced-Alignment Optimization Survey

Date: 2026-05-01

Question: can long-audio forced alignment be made faster by streaming forward through small windows?

Short answer: yes, this is consistent with how most modern forced aligners work. The main optimization family is not "make one huge alignment faster"; it is "avoid huge alignments by chunking, anchoring, batching similar lengths, and aligning locally."

### Relevant Existing Patterns

1. **VAD / silence pre-segmentation before alignment**

WhisperX explicitly uses VAD-based cut-and-merge for long-form transcription, then applies forced phoneme alignment for word-level timestamps. Its paper reports that pre-segmenting audio improves quality and enables batched inference speedups.

Implication for SegmentScribe: the 5-15s RMS-silence experiment fits this pattern. It gives the aligner smaller, cleaner units and avoids feeding long, padded sequences.

2. **CTC/Viterbi forced alignment uses a time-by-token search space**

Classic CTC forced alignment builds a trellis over audio frames and transcript tokens. NVIDIA's NeMo explanation describes the Viterbi matrix as `S x T`, where `S` is token/state length and `T` is audio time. TorchAudio's forced-alignment tutorial similarly constructs a trellis over frames and transcript labels.

Implication for SegmentScribe: reducing either audio duration or transcript length reduces the alignment search/work. Forward local windows are the natural way to do that.

3. **Long-audio aligners rely on chunking or segmentation**

NeMo Forced Aligner supports long audio, including 1+ hour files, but it is still a CTC-based timestamping pipeline and long-audio practicality depends on hardware/model choice. WhisperX handles long-form audio by segmenting first rather than aligning the entire recording as a monolith.

Implication for SegmentScribe: long recordings should be aligned as a sequence of anchored chunks, not as one giant forced-alignment request.

4. **Qwen3-ForcedAligner is non-autoregressive but still pads batch inputs**

The Qwen3 forced-aligner wrapper encodes each transcript, normalizes audio, creates padded processor inputs with `padding=True`, sends the full batch to the model, then reads timestamp-token positions from logits. The Hugging Face model card says Qwen3-ForcedAligner supports arbitrary timestamp units within up to 5 minutes of speech.

Implication for SegmentScribe: even though Qwen's aligner is not a classic CTC/Viterbi implementation at the public wrapper level, it still pays for the full padded audio/text batch. Small same-length windows should help; mixed-length large batches can be inefficient.

5. **Length bucketing and aligner-specific batch size matter**

Our local benchmark already showed ASR benefits from batching, but the forced aligner slows down when batch size rises. This matches the padding hypothesis: batch size can hurt if one long item forces the whole batch to a larger padded shape.

Implication for SegmentScribe: use independent controls:

```bash
ASR_MAX_BATCH_SIZE=4 or 8
ALIGNER_BATCH_SIZE=1 or 2
ALIGNER_WINDOW_SEC=5-15
```

### Candidate Optimization Designs

#### A. Pre-cut then align each segment

Pipeline:

1. Use VAD/RMS silence to cut audio into `5-15s` windows.
2. Run ASR on windows, batched through vLLM.
3. Run forced aligner on the same windows, with aligner batch size `1`.
4. Offset local timestamps back to source time.

Pros:

- Simple.
- Very likely faster than 30-60s alignment.
- Uses our successful RMS-silence experiment.

Cons:

- A bad pre-cut can split a word or semantic phrase.
- ASR context is shorter.

Best use: fast deterministic mode.

#### B. ASR large chunks, align small forward windows

Pipeline:

1. ASR `30s` chunks for better context and vLLM efficiency.
2. Split transcript into phrase windows using punctuation and/or estimated speech rate.
3. For each phrase, search audio only from previous aligned end to a bounded future window.
4. Align phrase with a small audio margin, e.g. `0.5-1.0s`.
5. Commit timestamps and advance.

Pros:

- Preserves better ASR context.
- Makes forced alignment local.
- Reduces drift by using previous end as an anchor.

Cons:

- More engineering complexity.
- Needs recovery logic when a phrase fails.

Best use: production-quality speedup path.

#### C. Hierarchical coarse-to-fine alignment

Pipeline:

1. Coarse anchors every sentence/phrase using ASR timestamps, punctuation, or RMS silence.
2. Align only within each anchor span.
3. If confidence is low or timestamps are non-monotonic, widen the local window and retry.

Pros:

- Good balance of speed and robustness.
- Avoids catastrophic drift.

Cons:

- Requires confidence checks and retry policy.

Best use: robust long-form audio.

#### D. Length-bucketed aligner batching

Pipeline:

1. Create many small alignment jobs.
2. Bucket by audio duration/text length.
3. Batch only similar-shape jobs.

Pros:

- Can recover GPU utilization without padding waste.

Cons:

- For Qwen3-ForcedAligner, batch size `1` may still be fastest; needs benchmark.

Best use: powerful GPUs processing many files.

### Recommended Next Experiment

Compare:

| Strategy | ASR unit | Aligner unit | Expected outcome |
| --- | --- | --- | --- |
| current baseline | `30s` | `30s` | known bottleneck |
| RMS pre-cut | `5-15s` | `5-15s` | likely faster aligner |
| large-ASR/local-align | `30s` | phrase + `0.5s` margin | likely best quality/speed tradeoff |
| local-align + retry | `30s` | phrase + adaptive margin | most robust |

Metrics:

- aligner seconds per audio second
- total pipeline throughput
- dropped/failed alignment windows
- timestamp monotonicity violations
- segment duration validity
- subjective cut quality on a small sample

### Sources

- Qwen3-ForcedAligner model card: `https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B`
- Qwen3 forced-aligner wrapper: `https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py`
- WhisperX paper: `https://arxiv.org/abs/2303.00747`
- WhisperX alignment implementation: `https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py`
- NVIDIA NeMo forced-alignment overview: `https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html`
- NVIDIA forced-alignment explanation: `https://research.nvidia.com/labs/conv-ai/blogs/2023/2023-08-forced-alignment/`
- TorchAudio forced-alignment tutorial: `https://docs.pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html`
