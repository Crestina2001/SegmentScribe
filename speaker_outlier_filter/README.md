# Speaker Outlier Filter

Filter likely wrong-speaker segments from a VoxCPM-style JSONL manifest using
SpeechBrain ECAPA speaker embeddings. The tool only writes new manifests and
reports; it does not copy, delete, or rewrite audio files.

## Setup

Install the repo dependencies after installing the PyTorch build that matches
your machine:

```cmd
pip install -r requirements.txt
```

Audio is decoded and resampled with `librosa`, so this tool does not depend on
`torchaudio.load` or TorchCodec for manifest filtering. The default model is
downloaded on first use:

```text
speechbrain/spkrec-ecapa-voxceleb
```

It is cached under `checkpoints/spkrec-ecapa-voxceleb` by default.

## Usage

```cmd
python speaker_outlier_filter/filter_speaker_outliers.py ^
  --input-jsonl audios/zhouquanquan_final_normalized/zhouquanquan.jsonl ^
  --output-jsonl audios/zhouquanquan_final_normalized/zhouquanquan_speaker_filtered.jsonl ^
  --dataset-root audios/zhouquanquan_final_normalized ^
  --device cuda:0 ^
  --overwrite
```

For CPU:

```cmd
python speaker_outlier_filter/filter_speaker_outliers.py ^
  --input-jsonl audios/zhouquanquan_final_normalized/zhouquanquan.jsonl ^
  --output-jsonl audios/zhouquanquan_final_normalized/zhouquanquan_speaker_filtered.jsonl ^
  --device cpu
```

Inspect scores without writing filtered/pruned JSONL files:

```cmd
python speaker_outlier_filter/filter_speaker_outliers.py ^
  --input-jsonl audios/zhouquanquan_final_normalized/zhouquanquan.jsonl ^
  --output-jsonl audios/zhouquanquan_final_normalized/zhouquanquan_speaker_filtered.jsonl ^
  --dry-run ^
  --overwrite
```

## Outputs

Files are written next to `--output-jsonl`:

- `zhouquanquan_speaker_filtered.jsonl`: kept rows, preserving every original
  field and row order.
- `speaker_similarity_report.csv`: one row per input item, with cosine
  similarity, threshold, rank, and prune decision.
- `pruned_speaker_outliers.jsonl`: pruned rows with
  `speaker_outlier_pruned`, `speaker_similarity`, and
  `speaker_outlier_reason` fields. This file is only written when rows are
  pruned.
- `speaker_filter_summary.json`: row counts, threshold statistics, model
  source, model directory, device, and dry-run flag.

## Tuning

The default threshold is intentionally conservative:

```text
threshold = median(similarity) - 3.0 * 1.4826 * MAD
```

Useful options:

- `--mad-multiplier`: lower values prune more aggressively; higher values prune
  less.
- `--max-prune-ratio`: caps the number of removed rows. The default `0.10`
  means at most 10 percent of rows are pruned.
- `--min-rows`: below this row count, reports are written but no rows are
  pruned. The default is `8`.

This tool assumes the manifest mostly contains one dominant speaker. If a
dataset intentionally contains multiple speakers, treat the CSV report as a
review aid instead of an automatic cleanup decision.
