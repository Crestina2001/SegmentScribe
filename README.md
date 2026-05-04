# SegmentScribe

[English](README.md) | [简体中文](README.zh-CN.md)

SegmentScribe converts raw training audio into VoxCPM-compatible training
segments. It prepares source audio as numbered WAV files, optionally removes
background music, enhances speech with ModelScope ZipEnhancer or MossFormer2,
and slices transcript-aligned segments with manifests for downstream training.

## Setup

```cmd
conda create -n segmentScribe python=3.11 -y
conda activate segmentScribe
```

Install PyTorch with CUDA. Pick the CUDA build that matches your machine from
https://pytorch.org/get-started/locally/. Example:

```cmd
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Install the Python dependencies:

```cmd
pip install -r requirements.txt
```

## WebUI

Launch the local browser UI:

```cmd
python webui.py
```

The WebUI can run the same pipeline stages described below:

- transform source audio into numbered 16 kHz mono WAV files;
- remove background music with Demucs;
- denoise speech with MossFormer2 or ZipEnhancer;
- slice audio with `slide_rule`;
- review `summary.json`, `manifest.tsv`, transcripts, and segment audio;
- filter wrong-speaker outliers from VoxCPM JSONL manifests;
- normalize final segment loudness into a new safe dataset folder.

## Full Pipeline From Bash

You can run the whole pipeline without the WebUI by editing the config and
launching the runner:

```bash
bash scripts/run_full_pipeline.sh configs/full_pipeline.env
```

The config is a bash env file. It controls the source folder, workspace,
enabled stages, model paths, slicing mode, optional speaker filtering, and
optional volume normalization. Stage outputs are written under the workspace:

- `01_numbered_wavs`
- `02_vocals`
- `03_denoised`
- `03_presliced`
- `04_sliced`
- `05_normalized`

Use `SLICE_MODE="rule"` for `slide_rule`, or `SLICE_MODE="llm"` with
`LLM_MODEL` and provider settings for `slide_LLM`.

By default, slicing now pre-cuts each source into RMS-silence windows before
ASR/alignment:

```bash
PREPROCESS_CHUNK_MODE="rms_silence"
PREPROCESS_MIN_CHUNK_SEC=5
PREPROCESS_MAX_CHUNK_SEC=15
ASR_MAX_BATCH_SIZE=8
ALIGNER_MAX_BATCH_SIZE=1
ALIGNER_CONCURRENCY=1
```

This lets Qwen3-ASR and Qwen3-ForcedAligner see the same short audio window,
then stitches timestamps back into a source-level stream before punctuation
and rough cutting. Set `PREPROCESS_CHUNK_MODE="vad"` to use the older VAD-packed
`PREPROCESS_CHUNK_SEC` behavior.

Download the WebUI's default model checkpoints:

```cmd
python utils/download_models.py --models all --download_path checkpoints
```

This prepares the default WebUI paths under `checkpoints/` for Qwen3-ASR, the
Qwen3 forced aligner, MossFormer2, ZipEnhancer, and the SpeechBrain speaker
embedding model. It also prefetches Demucs and Silero VAD through their Python
package loaders. The same download command is available from the WebUI's
**Models** tab. `--download_path` is the local checkpoint destination for
hub snapshots. Use `--provider modelscope` or `--provider hf` to pick the
download source for the selected hub models. ZipEnhancer is currently
ModelScope-only, so omit `zipenhancer` when using `--provider hf`.

Examples:

```cmd
python utils/download_models.py --models all --provider modelscope --download_path checkpoints
python utils/download_models.py --models qwen3 mossformer speaker demucs silero --provider hf --download_path checkpoints
```

Install `ffmpeg` for audio conversion and non-WAV inputs:

```cmd
conda install -c conda-forge ffmpeg -y
```

For audio with background music, install Demucs as an optional vocal-separation
preprocessor:

```cmd
pip install -r music_removal/requirements.txt
```

For MossFormer2 enhancement, install the additional dependencies:

```cmd
pip install -r MossFormer/requirements.txt
```

## Preprocessing of dirty audio data(optional)

Convert a folder of audio files to numbered 16 kHz mono WAV files. This is
useful when source files have unusual formats or filenames:

```cmd
python utils/convert_to_numbered_wav.py \
  --input my_wav_folder \
  --output numbered_wavs \
  --recursive \
  --overwrite
```

## Remove background music(optional)

Usual denoising/audio enhancing models are trained on DNS2020-like dataset, which is designed to remove environmental noise, NOT background music. So we need additional steps if the source audios contain background music.

Command to use demucs

```cmd
python music_removal/extract_vocals_demucs.py \
  --input my_wav_folder \
  --output vocal_stems \
  --device cuda \
  --model htdemucs \
  --overwrite
```

Better quality but slower

```cmd
python music_removal/extract_vocals_demucs.py \
  --input my_wav_folder \
  --output vocal_stems \
  --device cuda \
  --model htdemucs_ft \
  --overwrite
```

## Denoising method 1: ZipEnhancer

CMD:

```cmd
python zip_enhancer/enhance_audio.py \
  --input my_wav_folder \
  --output enhanced_wavs \
  --device cuda:0 \
  --silence-aware-splits \
  --preprocess-workers 4 \
  --normalize match_original \
  --alignment-metric rms \
  --overwrite
```

About volume normalization: The modelscope code has internal volume normalization, which has been eliminated, but the model itself may still change the volume.

--normalize has these options: 1, None - no normalization of the model output; 2, match_original - in alignment of the original volume; 3, a value - a numeric dBFS value such as -20; 4, library_median - estimate one shared corpus RMS target using a duration-weighted speech median. Note that for safety issues, we will keep the value not too far away from the lowest volume one. So if there are certain audios with very low volumes, the final target value will be very low.

--alignment-metric has these options: 1, rms; 2, peak

About model downloading

The default enhancement model is **automatively** downloaded by ModelScope:

```text
iic/speech_zipenhancer_ans_multiloss_16k_base
```

Comment: ZipEnhancer is a newer model, but after some trials, I don't think it performs well. It is recommended to use the second method, which is more steady.

## Denoising method 2: MossFormer 2

MossFormer2 is bundled in `MossFormer/` and uses a downloaded checkpoint path
explicitly. First download the checkpoint:

```cmd
python MossFormer/download_checkpoints.py --model-path checkpoints/MossFormer2_SE_48K
```

Then enhance a folder:

```cmd
python MossFormer/enhance_folder.py \
  --input-dir my_wav_folder \
  --output-dir mossformer_enhanced \
  --model-path checkpoints/MossFormer2_SE_48K \
  --preprocess-workers 4 \
  --inference-batch-size 4 \
  --overwrite
```

`--preprocess-workers` loads and resamples upcoming files in background threads.
GPU inference still runs in the main process.
`--inference-batch-size` batches equal-length MossFormer windows inside each
long file to improve GPU occupancy; increase it until VRAM use is healthy.

## Optional: pre-split long denoised audio

For very long denoised files, split them before `slide_LLM` so each source
handed to ASR/LLM is closer to five minutes. The splitter searches a 60-second
window around each ideal cut and chooses the longest low-energy silence span.

```cmd
python utils/auto_slice_long_audio.py \
  --input mossformer_enhanced \
  --output mossformer_enhanced_presliced \
  --target-piece-seconds 300 \
  --search-window-seconds 60 \
  --overwrite
```

Then pass `mossformer_enhanced_presliced` as the slicing input.

## Download Qwen3 ASR models

Download the ASR and forced-aligner checkpoints before running `slide_rule`.
ModelScope is the default provider:

```cmd
python utils/download_qwen3_asr.py \
  --provider modelscope \
  --output-root checkpoints
```

Use Hugging Face instead:

```cmd
python utils/download_qwen3_asr.py \
  --provider hf \
  --output-root checkpoints
```

This creates:

- `checkpoints/Qwen3-ASR-1.7B`
- `checkpoints/Qwen3-ForcedAligner-0.6B`

## Rule-based slicing: slide_rule

`slide_rule` cuts prepared speech audio into transcript-aligned segments for
TTS data preparation. It runs ASR/alignment, rule-based punctuation handling,
rough segmentation, thin silence-aware trimming, and manifest writing.

Make sure `slide_LLM` and `slide_workflow` are available on `PYTHONPATH`,
because `slicing_utils` imports the ASR, prepass, rough-cut, and output-writing
helpers from those modules.

Default `transformers` backend:

```cmd
python -m slide_rule \
  --input mossformer_enhanced \
  --output-dir sliced_segments \
  --model-path checkpoints/Qwen3-ASR-1.7B \
  --aligner-path checkpoints/Qwen3-ForcedAligner-0.6B \
  --asr-backend transformers \
  --device cuda:0 \
  --dtype bfloat16 \
  --asr-max-batch-size 1 \
  --min-seg-sec 3 \
  --max-seg-sec 10 \
  --vad-backend auto \
  --overwrite
```

Optional `vllm` backend:

```cmd
python -m slide_rule \
  --input mossformer_enhanced \
  --output-dir sliced_segments \
  --model-path checkpoints/Qwen3-ASR-1.7B \
  --aligner-path checkpoints/Qwen3-ForcedAligner-0.6B \
  --asr-backend vllm \
  --device cuda:0 \
  --dtype bfloat16 \
  --asr-max-batch-size 1 \
  --min-seg-sec 3 \
  --max-seg-sec 6 \
  --overwrite
```

`vllm` is installed from `requirements.txt` only on Linux/WSL. Use the
`transformers` backend for native Windows environments.

Useful options:

- `--input`: a single audio file or a directory of audio files.
- `--output-dir`: output folder for audio clips, transcripts, traces, manifests,
  and the VoxCPM-style JSONL file.
- `--model-path`: ASR model path.
- `--aligner-path`: forced-aligner model path.
- `--language`: optional language hint passed to the ASR backend.
- `--dry-run`: run the pipeline without writing final audio clips.
- `--enable-punctuation-correction`: enable rule-based punctuation edits.
- `--rough-cut-strategy`: deterministic rough planner strategy. Default:
  `priority_silence_v3`, a combined strong+weak pause-percentile planner.

## LLM-assisted slicing: slide_LLM

`slide_LLM` uses the same ASR/alignment pre-pass, silence-aware thin cut,
output writer, manifest, and trace layout as `slide_rule`. By default it skips
punctuation correction and asks an LLM to classify punctuation pauses
(`good`/`ok`/`bad`) before the adapted `priority_silence_v2` rough planner cuts
segments. Configure your LLM provider in `.env` first; see `.env.example` for OpenAI-compatible,
Gemini-compatible, MiniMax, Anthropic, and DeepSeek keys/base URLs.

Example with Gemini through the bundled LLM gateway:

```cmd
python -m slide_LLM \
  --input mossformer_enhanced \
  --output-dir sliced_segments_llm \
  --model-path checkpoints/Qwen3-ASR-1.7B \
  --aligner-path checkpoints/Qwen3-ForcedAligner-0.6B \
  --asr-backend transformers \
  --device cuda:0 \
  --dtype bfloat16 \
  --asr-max-batch-size 1 \
  --llm-concurrency 8 \
  --min-seg-sec 3 \
  --max-seg-sec 10 \
  --vad-backend auto \
  --llm-model gemini-2.5-flash \
  --llm-provider gemini \
  --env-path .env \
  --overwrite
```

Useful LLM options:

- `--llm-model`: required default model for both LLM phases.
- `--enable-punctuation-correction`: enable LLM punctuation correction. Disabled
  by default.
- `--punct-llm-model`: optional override for punctuation correction when enabled.
- `--rough-llm-model`: optional override for rough cut planning.
- `--rough-cut-strategy`: rough planner strategy. Default:
  `llm_slice_v1`; use `llm_tool` for the older tool-calling
  rough planner. Deterministic strategies such as `priority_silence_v3` are
  also accepted.
  In `scripts/run_full_pipeline.sh`, `SLICE_MODE=llm` reads this from
  `LLM_ROUGH_CUT_STRATEGY`; `ROUGH_CUT_STRATEGY` is reserved for `SLICE_MODE=rule`.
- `--llm-provider`: optional provider name; if omitted, the gateway tries to
  infer it from the model name.
- `--env-path`: optional path to the `.env` file containing provider keys.
- `--llm-max-rounds`: max rough-cut repair rounds after a length error
  (default: `5`).
- `--asr-max-batch-size`: maximum number of pre-pass ASR chunks in one global
  ASR inference batch (default: `1`). When one source has fewer remaining
  chunks than the batch capacity, chunks from the next ready source can fill the
  same ASR batch.
- `--aligner-concurrency`: number of forced-aligner instances allowed to run at
  once (default: `1`). Values above `1` can improve throughput on short
  pre-sliced inputs, at the cost of loading additional aligner models.
- `--llm-concurrency`: maximum number of `slide_LLM` LLM calls in flight at
  once (default: `8`). The gateway's provider/model/global rate limits still
  apply underneath this cap.

When `--input` is a directory, `slide_LLM` schedules audio files concurrently
across a global ASR batcher and an LLM pool. Each source still runs in phase
order, but ASR batches can be filled with chunks from multiple audios, and one
file can be in ASR while other files are waiting on punctuation or rough-cut LLM
calls. Output directories are keyed by source filename stem, so duplicate stems
in different subdirectories are rejected before processing to avoid trace/audio
collisions.

## Final volume normalization(optional)

After `slide_rule` creates many short training segments, normalize volume at the
segment level and write a new safe dataset folder:

```cmd
python utils/normalize_corpus_volume.py \
  --input sliced_segments \
  --output normalized_segments \
  --target-lufs -20 \
  --max-volume-change-db 12 \
  --overwrite
```

The normalizer reads the `*_voxcpm.jsonl` file, measures only the segment audio
listed there, normalizes kept segments to the fixed LUFS target, and writes
normalized WAVs plus an updated JSONL under the new output folder. Segments are
pruned when they need more upward gain than `--max-volume-change-db` to reach
the target, would clip after gain, have too little active audio, or change
volume too much within the segment. Loud segments may be attenuated by any
amount. Pruned rows are recorded in
`pruned_volume_segments.jsonl`; reports are written to `volume_analysis.csv`
and `normalized_manifest.csv`.

## Filter speaker outliers(optional)

If the dataset should mostly contain one speaker, use SpeechBrain ECAPA speaker
embeddings to conservatively filter a VoxCPM JSONL manifest:

```cmd
python speaker_outlier_filter/filter_speaker_outliers.py \
  --input-jsonl audios/my_dataset_normalized/manifest.jsonl \
  --output-jsonl audios/my_dataset_normalized/manifest_speaker_filtered.jsonl \
  --dataset-root audios/my_dataset_normalized \
  --device cuda:0 \
  --overwrite
```

The tool keeps original audio files untouched and writes only new manifests and
audit reports. Kept rows are written to
`manifest_speaker_filtered.jsonl`, pruned rows are written to
`pruned_speaker_outliers.jsonl`, and the similarity report plus summary are
written to `speaker_similarity_report.csv` and `speaker_filter_summary.json`.
