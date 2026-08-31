#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/predict_dinov3conv.env}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[dinov3conv-predict] config not found: $CONFIG_PATH" >&2
  echo "[dinov3conv-predict] copy configs/predict_dinov3conv.env.example to configs/predict_dinov3conv.env first" >&2
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
IMAGE_DIR="${IMAGE_DIR:-data/test/images}"
IMAGE_LIST="${IMAGE_LIST:-}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/predict/dinov3conv_predictions.json}"
LOC_PROB_DIR="${LOC_PROB_DIR:-outputs/predict/loc_probs}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLS_THRESHOLD="${CLS_THRESHOLD:-0.5}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}"

echo "[dinov3conv-predict] checkpoint=$CHECKPOINT image_dir=$IMAGE_DIR"
"$PYTHON_BIN" eval_dinov3_localizer_testsets.py \
  --checkpoint "$CHECKPOINT" \
  --dino_model_path "$DINO_MODEL_PATH" \
  --predict_image_dir "$IMAGE_DIR" \
  --predict_image_list "$IMAGE_LIST" \
  --prediction_output_json "$OUTPUT_JSON" \
  --prediction_loc_prob_dir "$LOC_PROB_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --cls_threshold "$CLS_THRESHOLD" \
  --mask_threshold "$MASK_THRESHOLD" \
  "$@"
