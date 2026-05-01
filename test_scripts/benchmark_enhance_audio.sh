#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="audios/sweetie_numbered_wav"
OUTPUT_DIR="audios/enhanced_wav_preprocess4"
LOG_DIR="test_scripts/logs/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/preprocess4.log"
TIME_FILE="$LOG_DIR/preprocess4.time"

mkdir -p "$LOG_DIR"

echo "Benchmark: preprocess-workers=4"
echo "Log: ${LOG_FILE}"

{ { time python audio_enhancement/enhance_audio.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --device cuda:0 \
    --preprocess-workers 4 \
    --normalize match_original \
    --alignment-metric rms \
    --overwrite; } 2> >(tee "$TIME_FILE" >&2); } 2>&1 | tee "$LOG_FILE"

cat "$TIME_FILE"

echo
echo "Saved logs:"
find "$LOG_DIR" -maxdepth 1 -type f -print | sort
