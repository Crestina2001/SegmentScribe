# Enhancement Benchmarks

Run the stable enhancement benchmark with CPU preprocessing workers:

```bash
bash test_scripts/benchmark_enhance_audio.sh
```

The script runs:

- `--preprocess-workers 4`

Output is written to:

- `audios/enhanced_wav_preprocess4`

Watch `nvidia-smi` during the run to confirm the single ZipEnhancer pipeline is
using the GPU.

Full logs and timing files are saved under:

```text
test_scripts/logs/YYYYMMDD_HHMMSS/
```

## Qwen3-ASR vLLM Batch Capacity

Probe the largest vLLM ASR batch size that completes on this machine:

```bash
python test_scripts/test_qwen3_asr_vllm_batch_capacity.py \
  --model-path checkpoints/Qwen3-ASR-1.7B \
  --aligner-path checkpoints/Qwen3-ForcedAligner-0.6B \
  --device cuda:0 \
  --max-batch-size 16 \
  --audio-seconds 30
```

By default this runs in `--mode single-process`: vLLM loads once, performs one
warmup call, then reports per-batch timings without repeated startup cost. Use
`--mode subprocess` when you want each candidate isolated in a fresh process
after possible OOM failures.

The tester defaults to `--max-model-len 8192`, which fits this repo's default
30-second ASR chunks better than Qwen3-ASR/vLLM's 65,536-token model default.
Override it with `--max-model-len 4096`, `--max-model-len 16384`, or set
`--max-model-len 0` to leave vLLM's default unchanged.

Use `--audio-file path/to/sample.wav` for a more realistic probe. Results and
per-batch logs are written under `test_scripts/logs/`.

## Qwen3-ASR + Forced Aligner Throughput

Sweep ASR backend, vLLM profile, batch size, and chunk size. Each candidate runs
in a fresh subprocess and reports ASR-only, forced-aligner-only, and combined
inference throughput:

```bash
python test_scripts/benchmark_qwen3_asr_aligner_throughput.py \
  --input audios/tian_raw_denoised \
  --model-path checkpoints/Qwen3-ASR-1.7B \
  --aligner-path checkpoints/Qwen3-ForcedAligner-0.6B \
  --device cuda:0 \
  --backends vllm,transformers \
  --batch-sizes 4,8,16,32 \
  --chunk-seconds 30,60 \
  --max-chunks 16 \
  --keep-going-after-fail
```

Built-in vLLM profiles:

- `default`: `{"max_model_len": 8192}`
- `fast`: moderate KV/cache pressure for quick startup
- `wide`: higher memory utilization and larger batching
- `eager`: skips CUDA graph capture for cases where compile/warmup dominates

Results are written as JSONL, CSV, and a best-candidate summary under
`test_scripts/logs/qwen3_asr_aligner_YYYYMMDD_HHMMSS/`.

Local dev smoke result on 2026-05-01 with `conda activate qwen3-asr`:

- best tested end-to-end candidate: `vllm_default_b1_c30`
- combined ASR + forced-align throughput: about `7.06x` realtime
- ASR alone improved strongly with larger batches, but forced alignment became
  the bottleneck and was fastest at batch size `1`

Recommended starting point for full-pipeline tuning on a strong GPU:

```bash
ASR_BACKEND="vllm"
ASR_BACKEND_KWARGS='{"max_model_len":8192}'
ASR_MAX_BATCH_SIZE=1
PREPROCESS_CHUNK_SEC=30
```

If you only care about ASR throughput before alignment, larger ASR batches can
be much faster; for this end-to-end pipeline, benchmark both phases together.
