#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REASONER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REASONER_DIR}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

IMAGE_DIR="${IMAGE_DIR:-}"
FILE_LIST="${FILE_LIST:-}"
DETECTOR_JSON="${DETECTOR_JSON:-}"
HEATMAP_DIR="${HEATMAP_DIR:-}"
FALLBACK_HEATMAP_DIR="${FALLBACK_HEATMAP_DIR:-}"
OUTPUT_JSONL="${OUTPUT_JSONL:-output/pipeline_vllm.jsonl}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-}"
MERGED_MODEL="${MERGED_MODEL:-false}"

BATCH_SIZE="${BATCH_SIZE:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
RESIZE="${RESIZE:-1280}"
RESIZE_MODE="${RESIZE_MODE:-keep_ratio}"
TEST_NUM="${TEST_NUM:-0}"
RESUME="${RESUME:-true}"

DETECTOR_THRESHOLD="${DETECTOR_THRESHOLD:-0.5}"
LOC_THRESHOLD="${LOC_THRESHOLD:-0.5}"
LOC_MIN_THRESHOLD="${LOC_MIN_THRESHOLD:-0.0}"
THRESHOLD_STEP="${THRESHOLD_STEP:-0.05}"
MIN_AREA="${MIN_AREA:-10}"
MIN_COMPONENT_AREA="${MIN_COMPONENT_AREA:-16}"

RMC_MASK_DIR="${RMC_MASK_DIR:-}"
RMC_OUTPUT_JSONL="${RMC_OUTPUT_JSONL:-${OUTPUT_JSONL%.jsonl}_rmc.jsonl}"
RMC_STATS_JSON="${RMC_STATS_JSON:-${OUTPUT_JSONL%.jsonl}_rmc.stats.json}"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[reasoner-vllm] missing required variable: $name" >&2
    exit 1
  fi
}

require_var IMAGE_DIR
require_var DETECTOR_JSON
require_var HEATMAP_DIR
require_var MODEL_NAME_OR_PATH
if [[ "${MERGED_MODEL}" != "true" ]]; then
  require_var ADAPTER_CHECKPOINT
fi

args=(
  --image_dir "${IMAGE_DIR}"
  --detector_json "${DETECTOR_JSON}"
  --heatmap_dir "${HEATMAP_DIR}"
  --output_jsonl "${OUTPUT_JSONL}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --batch_size "${BATCH_SIZE}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --data_parallel_size "${DATA_PARALLEL_SIZE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --max_model_len "${MAX_MODEL_LEN}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --max_pixels "${MAX_PIXELS}"
  --resize "${RESIZE}"
  --resize_mode "${RESIZE_MODE}"
  --test_num "${TEST_NUM}"
  --detector_threshold "${DETECTOR_THRESHOLD}"
  --loc_threshold "${LOC_THRESHOLD}"
  --loc_min_threshold "${LOC_MIN_THRESHOLD}"
  --threshold_step "${THRESHOLD_STEP}"
  --min_area "${MIN_AREA}"
  --min_component_area "${MIN_COMPONENT_AREA}"
)

if [ -n "${FILE_LIST}" ]; then
  args+=(--file_list "${FILE_LIST}")
fi
if [ -n "${FALLBACK_HEATMAP_DIR}" ]; then
  args+=(--fallback_heatmap_dir "${FALLBACK_HEATMAP_DIR}")
fi
if [ "${RESUME}" = "true" ]; then
  args+=(--resume)
else
  args+=(--overwrite)
fi
if [ "${MERGED_MODEL}" = "true" ]; then
  args+=(--merged_model)
elif [ -n "${ADAPTER_CHECKPOINT}" ]; then
  args+=(--adapter_checkpoint "${ADAPTER_CHECKPOINT}")
fi

python pipeline_vllm.py "${args[@]}"

if [ -n "${RMC_MASK_DIR}" ]; then
  python scripts/postprocess/report_mask_consistency_postprocess.py \
    --mode jsonl \
    --source_jsonl "${OUTPUT_JSONL}" \
    --output_jsonl "${RMC_OUTPUT_JSONL}" \
    --mask_dir "${RMC_MASK_DIR}" \
    --image_root "${IMAGE_DIR}" \
    --bbox_coord pixel \
    --output_coord pixel \
    --stats_json "${RMC_STATS_JSON}"
fi
