#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_full_pipeline.sh configs/full_pipeline.env

The config file is a bash env file. Edit configs/full_pipeline.env before
running if you need different paths, stages, or model settings.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONFIG_PATH="${1:-configs/full_pipeline.env}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  echo >&2
  usage >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_PATH="${SOURCE_PATH:?Set SOURCE_PATH in the config}"
WORKSPACE="${WORKSPACE:?Set WORKSPACE in the config}"

is_true() {
  case "${1:-false}" in
    true|TRUE|1|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

append_flag_if_true() {
  local value="$1"
  local flag="$2"
  if is_true "${value}"; then
    COMMAND+=("${flag}")
  fi
}

run_command() {
  printf '\n==> '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

find_single_voxcpm_jsonl() {
  local dataset_dir="$1"
  shopt -s nullglob
  local matches=("${dataset_dir}"/*_voxcpm.jsonl)
  shopt -u nullglob

  if (( ${#matches[@]} == 0 )); then
    echo "No *_voxcpm.jsonl file found in ${dataset_dir}" >&2
    return 1
  fi
  if (( ${#matches[@]} > 1 )); then
    echo "Multiple *_voxcpm.jsonl files found in ${dataset_dir}:" >&2
    printf '  %s\n' "${matches[@]}" >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

mkdir -p "${WORKSPACE}"
CURRENT_INPUT="${SOURCE_PATH}"

if is_true "${DO_CONVERT:-true}"; then
  OUTPUT_PATH="${WORKSPACE}/01_numbered_wavs"
  COMMAND=(
    "${PYTHON_BIN}" utils/convert_to_numbered_wav.py
    --input "${CURRENT_INPUT}"
    --output "${OUTPUT_PATH}"
    --sample-rate "${SAMPLE_RATE:-16000}"
    --start-index "${START_INDEX:-1}"
    --digits "${DIGITS:-3}"
  )
  append_flag_if_true "${RECURSIVE:-true}" "--recursive"
  append_flag_if_true "${OVERWRITE:-false}" "--overwrite"
  run_command "${COMMAND[@]}"
  CURRENT_INPUT="${OUTPUT_PATH}"
fi

if is_true "${DO_MUSIC:-false}"; then
  OUTPUT_PATH="${WORKSPACE}/02_vocals"
  COMMAND=(
    "${PYTHON_BIN}" music_removal/extract_vocals_demucs.py
    --input "${CURRENT_INPUT}"
    --output "${OUTPUT_PATH}"
    --model "${DEMUCS_MODEL:-htdemucs}"
    --shifts "${DEMUCS_SHIFTS:-1}"
    --jobs "${DEMUCS_JOBS:-0}"
  )
  if [[ -n "${DEVICE:-}" ]]; then
    COMMAND+=(--device "${DEVICE}")
  fi
  if [[ -n "${DEMUCS_SEGMENT:-}" ]]; then
    COMMAND+=(--segment "${DEMUCS_SEGMENT}")
  fi
  if is_true "${RECURSIVE:-true}"; then
    COMMAND+=(--recursive)
  else
    COMMAND+=(--no-recursive)
  fi
  append_flag_if_true "${OVERWRITE:-false}" "--overwrite"
  run_command "${COMMAND[@]}"
  CURRENT_INPUT="${OUTPUT_PATH}"
fi

if is_true "${DO_DENOISE:-true}"; then
  OUTPUT_PATH="${WORKSPACE}/03_denoised"
  if [[ "${DENOISE_METHOD:-MossFormer2}" == "MossFormer2" ]]; then
    COMMAND=(
      "${PYTHON_BIN}" MossFormer/enhance_folder.py
      --input-dir "${CURRENT_INPUT}"
      --output-dir "${OUTPUT_PATH}"
      --model-path "${MOSSFORMER_MODEL_PATH:-checkpoints/MossFormer2_SE_48K}"
      --preprocess-workers "${PREPROCESS_WORKERS:-4}"
      --inference-batch-size "${INFERENCE_BATCH_SIZE:-4}"
    )
    append_flag_if_true "${MOSSFORMER_CPU:-false}" "--cpu"
  else
    COMMAND=(
      "${PYTHON_BIN}" zip_enhancer/enhance_audio.py
      --input "${CURRENT_INPUT}"
      --output "${OUTPUT_PATH}"
      --model "${ZIP_MODEL:-iic/speech_zipenhancer_ans_multiloss_16k_base}"
      --preprocess-workers "${PREPROCESS_WORKERS:-4}"
      --normalize "${ZIP_NORMALIZE:-match_original}"
      --alignment-metric "${ZIP_ALIGNMENT_METRIC:-rms}"
      --silence-aware-splits
    )
    if [[ -n "${DEVICE:-}" ]]; then
      COMMAND+=(--device "${DEVICE}")
    fi
  fi
  append_flag_if_true "${OVERWRITE:-false}" "--overwrite"
  run_command "${COMMAND[@]}"
  CURRENT_INPUT="${OUTPUT_PATH}"
fi

if is_true "${DO_PRESPLIT:-false}"; then
  OUTPUT_PATH="${WORKSPACE}/03_presliced"
  COMMAND=(
    "${PYTHON_BIN}" utils/auto_slice_long_audio.py
    --input "${CURRENT_INPUT}"
    --output "${OUTPUT_PATH}"
    --target-piece-seconds "${PRESPLIT_TARGET_PIECE_SECONDS:-300}"
    --search-window-seconds "${PRESPLIT_SEARCH_WINDOW_SECONDS:-60}"
    --frame-ms "${PRESPLIT_FRAME_MS:-30}"
    --hop-ms "${PRESPLIT_HOP_MS:-10}"
    --min-silence-ms "${PRESPLIT_MIN_SILENCE_MS:-500}"
  )
  if is_true "${RECURSIVE:-true}"; then
    COMMAND+=(--recursive)
  else
    COMMAND+=(--no-recursive)
  fi
  append_flag_if_true "${OVERWRITE:-false}" "--overwrite"
  run_command "${COMMAND[@]}"
  CURRENT_INPUT="${OUTPUT_PATH}"
fi

if is_true "${DO_SLICE:-true}"; then
  OUTPUT_PATH="${WORKSPACE}/04_sliced"
  SLICE_MODULE="slide_rule"
  if [[ "${SLICE_MODE:-rule}" == "llm" ]]; then
    SLICE_MODULE="slide_LLM"
  fi

  COMMAND=(
    "${PYTHON_BIN}" -m "${SLICE_MODULE}"
    --input "${CURRENT_INPUT}"
    --output-dir "${OUTPUT_PATH}"
    --model-path "${ASR_MODEL_PATH:-checkpoints/Qwen3-ASR-1.7B}"
    --aligner-path "${ALIGNER_PATH:-checkpoints/Qwen3-ForcedAligner-0.6B}"
    --asr-backend "${ASR_BACKEND:-transformers}"
    --device "${DEVICE:-cuda:0}"
    --dtype "${DTYPE:-bfloat16}"
    --asr-max-batch-size "${ASR_MAX_BATCH_SIZE:-1}"
    --min-seg-sec "${MIN_SEG_SEC:-3}"
    --max-seg-sec "${MAX_SEG_SEC:-10}"
    --vad-backend "${VAD_BACKEND:-auto}"
  )
  if [[ -n "${LANGUAGE:-}" ]]; then
    COMMAND+=(--language "${LANGUAGE}")
  fi
  if [[ "${SLICE_MODULE}" == "slide_rule" ]]; then
    append_flag_if_true "${ENABLE_PUNCTUATION_CORRECTION:-false}" "--enable-punctuation-correction"
  else
    if [[ -z "${LLM_MODEL:-}" ]]; then
      echo "SLICE_MODE=llm requires LLM_MODEL in the config." >&2
      exit 2
    fi
    COMMAND+=(--llm-model "${LLM_MODEL}")
    COMMAND+=(--llm-concurrency "${LLM_CONCURRENCY:-8}")
    COMMAND+=(--llm-max-rounds "${LLM_MAX_ROUNDS:-5}")
    if [[ -n "${PUNCT_LLM_MODEL:-}" ]]; then
      COMMAND+=(--punct-llm-model "${PUNCT_LLM_MODEL}")
    fi
    if [[ -n "${ROUGH_LLM_MODEL:-}" ]]; then
      COMMAND+=(--rough-llm-model "${ROUGH_LLM_MODEL}")
    fi
    if [[ -n "${LLM_PROVIDER:-}" ]]; then
      COMMAND+=(--llm-provider "${LLM_PROVIDER}")
    fi
    if [[ -n "${ENV_PATH:-}" ]]; then
      COMMAND+=(--env-path "${ENV_PATH}")
    fi
  fi
  append_flag_if_true "${OVERWRITE:-false}" "--overwrite"
  append_flag_if_true "${DRY_RUN:-false}" "--dry-run"
  run_command "${COMMAND[@]}"
fi

ACTIVE_JSONL=""
ACTIVE_DATASET_ROOT=""
SLICED_DIR="${WORKSPACE}/04_sliced"

if is_true "${DO_SPEAKER_FILTER:-false}"; then
  ACTIVE_JSONL="$(find_single_voxcpm_jsonl "${SLICED_DIR}")"
  ACTIVE_DATASET_ROOT="${SLICED_DIR}"
  FILTERED_JSONL="${ACTIVE_JSONL%.jsonl}_speaker_filtered.jsonl"
  COMMAND=(
    "${PYTHON_BIN}" speaker_outlier_filter/filter_speaker_outliers.py
    --input-jsonl "${ACTIVE_JSONL}"
    --output-jsonl "${FILTERED_JSONL}"
    --dataset-root "${ACTIVE_DATASET_ROOT}"
    --device "${SPEAKER_DEVICE:-${DEVICE:-cuda:0}}"
    --mad-multiplier "${SPEAKER_MAD_MULTIPLIER:-3.0}"
    --max-prune-ratio "${SPEAKER_MAX_PRUNE_RATIO:-0.10}"
    --min-rows "${SPEAKER_MIN_ROWS:-8}"
  )
  append_flag_if_true "${OVERWRITE:-false}" "--overwrite"
  run_command "${COMMAND[@]}"
  ACTIVE_JSONL="${FILTERED_JSONL}"
fi

if is_true "${DO_VOLUME_NORMALIZE:-false}"; then
  if [[ -z "${ACTIVE_JSONL}" ]]; then
    ACTIVE_JSONL="$(find_single_voxcpm_jsonl "${SLICED_DIR}")"
  fi
  if [[ -z "${ACTIVE_DATASET_ROOT}" ]]; then
    ACTIVE_DATASET_ROOT="${SLICED_DIR}"
  fi
  OUTPUT_PATH="${WORKSPACE}/05_normalized"
  COMMAND=(
    "${PYTHON_BIN}" utils/normalize_corpus_volume.py
    --input "${ACTIVE_DATASET_ROOT}"
    --output "${OUTPUT_PATH}"
    --jsonl "${ACTIVE_JSONL}"
    --target-lufs "${VOLUME_TARGET_LUFS:--20}"
    --max-volume-change-db "${VOLUME_MAX_VOLUME_CHANGE_DB:-12}"
    --max-dynamic-range-db "${VOLUME_MAX_DYNAMIC_RANGE_DB:-24}"
    --peak-margin-db "${VOLUME_PEAK_MARGIN_DB:-1}"
    --min-active-ratio "${VOLUME_MIN_ACTIVE_RATIO:-0.03}"
  )
  if [[ "${VOLUME_SAMPLE_RATE:-0}" != "0" ]]; then
    COMMAND+=(--sample-rate "${VOLUME_SAMPLE_RATE}")
  fi
  append_flag_if_true "${OVERWRITE:-false}" "--overwrite"
  run_command "${COMMAND[@]}"
fi

echo
echo "Pipeline complete. Workspace: ${WORKSPACE}"
