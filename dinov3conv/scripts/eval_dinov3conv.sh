#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/eval_dinov3conv.env}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[dinov3conv-eval] config not found: $CONFIG_PATH" >&2
  echo "[dinov3conv-eval] copy configs/eval_dinov3conv.env.example to configs/eval_dinov3conv.env first" >&2
  exit 1
fi
shift || true

# shellcheck disable=SC1090
source "$CONFIG_PATH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT="${CHECKPOINT:-checkpoints/dinov3-conv}"
DINO_MODEL_PATH="${DINO_MODEL_PATH:-checkpoint/dinov3-vitb16-pretrain-lvd1689m}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLS_THRESHOLD="${CLS_THRESHOLD:-0.5}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}"
GATE_MASKS_BY_CLS="${GATE_MASKS_BY_CLS:-True}"
EVAL_DATASETS="${EVAL_DATASETS:-RealText:data/test/images:data/test/masks}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/eval/dinov3conv_metrics.json}"
PROB_OUTPUT_DIR="${PROB_OUTPUT_DIR:-outputs/eval/dinov3conv_probs}"
METRIC_SCRIPT="${METRIC_SCRIPT:-}"
METRIC_WORKERS="${METRIC_WORKERS:-8}"

IFS=';' read -r -a DATASET_ARGS <<< "$EVAL_DATASETS"

ARGS=(
  eval_dinov3_localizer_testsets.py
  --checkpoint "$CHECKPOINT"
  --dino_model_path "$DINO_MODEL_PATH"
  --datasets "${DATASET_ARGS[@]}"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --cls_threshold "$CLS_THRESHOLD"
  --mask_threshold "$MASK_THRESHOLD"
  --output_json "$OUTPUT_JSON"
  --prob_output_dir "$PROB_OUTPUT_DIR"
  --metric_workers "$METRIC_WORKERS"
)

if [[ "$GATE_MASKS_BY_CLS" =~ ^(False|false|0)$ ]]; then
  ARGS+=(--no-gate_masks_by_cls)
else
  ARGS+=(--gate_masks_by_cls)
fi
if [[ -n "$METRIC_SCRIPT" ]]; then
  ARGS+=(--metric_script "$METRIC_SCRIPT")
fi

echo "[dinov3conv-eval] checkpoint=$CHECKPOINT datasets=$EVAL_DATASETS"
"$PYTHON_BIN" "${ARGS[@]}" "$@"
