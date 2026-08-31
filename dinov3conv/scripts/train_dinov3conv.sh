#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/train_dinov3conv.env}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[dinov3conv-train] config not found: $CONFIG_PATH" >&2
  echo "[dinov3conv-train] copy configs/train_dinov3conv.env.example to configs/train_dinov3conv.env first" >&2
  exit 1
fi
shift || true

# shellcheck disable=SC1090
source "$CONFIG_PATH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29502}"

DINO_MODEL_PATH="${DINO_MODEL_PATH:-checkpoint/dinov3-vitb16-pretrain-lvd1689m}"
SFT_JSON_PATH="${SFT_JSON_PATH:-data/train_sft.json}"
VAL_SFT_JSON_PATH="${VAL_SFT_JSON_PATH:-data/val_sft.json}"
TRAIN_EXCLUDE_LIST="${TRAIN_EXCLUDE_LIST:-}"
MASK_ROOT="${MASK_ROOT:-data/train_masks}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/dinov3conv-localizer-1344x896}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

PRECISION="${PRECISION:-fp32}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-896}"
IMAGE_WIDTH="${IMAGE_WIDTH:-1344}"
DECODER_DIM="${DECODER_DIM:-256}"
MASK_HEAD_TYPE="${MASK_HEAD_TYPE:-conv}"
AUGMENT="${AUGMENT:-True}"
RANDOM_RESIZED_CROP_SCALE="${RANDOM_RESIZED_CROP_SCALE:-0.9}"
ROTATION_DEGREES="${ROTATION_DEGREES:-3.0}"
HFLIP_PROB="${HFLIP_PROB:-0.0}"
COLOR_JITTER="${COLOR_JITTER:-0.1}"
FREEZE_DINO="${FREEZE_DINO:-False}"
UNFREEZE_LAST_N_LAYERS="${UNFREEZE_LAST_N_LAYERS:--1}"
BATCH_SIZE="${BATCH_SIZE:-12}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
EPOCHS="${EPOCHS:-12}"
MAX_STEPS="${MAX_STEPS:--1}"
LR="${LR:-1e-5}"
HEAD_LR="${HEAD_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:--1}"
VAL_LIMIT_SAMPLES="${VAL_LIMIT_SAMPLES:--1}"
EVAL_STEPS="${EVAL_STEPS:-0}"
EVAL_EPOCHS="${EVAL_EPOCHS:-1}"
SAVE_EPOCHS="${SAVE_EPOCHS:-1}"
KEEP_LAST_EPOCH_CHECKPOINTS="${KEEP_LAST_EPOCH_CHECKPOINTS:-2}"
LOG_STEPS="${LOG_STEPS:-10}"
CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-1.0}"
MASK_BCE_WEIGHT="${MASK_BCE_WEIGHT:-1.0}"
MASK_DICE_WEIGHT="${MASK_DICE_WEIGHT:-1.0}"
MAX_MASK_POS_WEIGHT="${MAX_MASK_POS_WEIGHT:-100.0}"
CLS_THRESHOLD="${CLS_THRESHOLD:-0.5}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-False}"
SEED="${SEED:-42}"

if [[ "$MASK_HEAD_TYPE" != "conv" ]]; then
  echo "[dinov3conv-train] MASK_HEAD_TYPE must be conv for this package" >&2
  exit 1
fi

if [[ ! -f "$SFT_JSON_PATH" ]]; then
  echo "[dinov3conv-train] SFT json not found: $SFT_JSON_PATH" >&2
  exit 1
fi
if [[ -n "$VAL_SFT_JSON_PATH" && ! -f "$VAL_SFT_JSON_PATH" ]]; then
  echo "[dinov3conv-train] val SFT json not found: $VAL_SFT_JSON_PATH" >&2
  exit 1
fi

TRAIN_ARGS=(
  train_dinov3_localizer.py
  --dino_model_path "$DINO_MODEL_PATH"
  --sft_json_path "$SFT_JSON_PATH"
  --val_sft_json_path "$VAL_SFT_JSON_PATH"
  --exclude_list_path "$TRAIN_EXCLUDE_LIST"
  --mask_root "$MASK_ROOT"
  --output_dir "$OUTPUT_DIR"
  --init_checkpoint "$INIT_CHECKPOINT"
  --precision "$PRECISION"
  --image_size "$IMAGE_SIZE"
  --image_height "$IMAGE_HEIGHT"
  --image_width "$IMAGE_WIDTH"
  --decoder_dim "$DECODER_DIM"
  --mask_head_type conv
  --random_resized_crop_scale "$RANDOM_RESIZED_CROP_SCALE"
  --rotation_degrees "$ROTATION_DEGREES"
  --hflip_prob "$HFLIP_PROB"
  --color_jitter "$COLOR_JITTER"
  --unfreeze_last_n_layers "$UNFREEZE_LAST_N_LAYERS"
  --batch_size "$BATCH_SIZE"
  --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS"
  --epochs "$EPOCHS"
  --max_steps "$MAX_STEPS"
  --lr "$LR"
  --head_lr "$HEAD_LR"
  --weight_decay "$WEIGHT_DECAY"
  --num_workers "$NUM_WORKERS"
  --limit_samples "$LIMIT_SAMPLES"
  --val_limit_samples "$VAL_LIMIT_SAMPLES"
  --eval_steps "$EVAL_STEPS"
  --eval_epochs "$EVAL_EPOCHS"
  --save_epochs "$SAVE_EPOCHS"
  --keep_last_epoch_checkpoints "$KEEP_LAST_EPOCH_CHECKPOINTS"
  --log_steps "$LOG_STEPS"
  --cls_loss_weight "$CLS_LOSS_WEIGHT"
  --mask_bce_weight "$MASK_BCE_WEIGHT"
  --mask_dice_weight "$MASK_DICE_WEIGHT"
  --max_mask_pos_weight "$MAX_MASK_POS_WEIGHT"
  --cls_threshold "$CLS_THRESHOLD"
  --mask_threshold "$MASK_THRESHOLD"
  --seed "$SEED"
)

if [[ "$FREEZE_DINO" =~ ^(True|true|1)$ ]]; then
  TRAIN_ARGS+=(--freeze_dino)
else
  TRAIN_ARGS+=(--no-freeze_dino)
fi
if [[ "$AUGMENT" =~ ^(True|true|1)$ ]]; then
  TRAIN_ARGS+=(--augment)
else
  TRAIN_ARGS+=(--no-augment)
fi
if [[ "$DDP_FIND_UNUSED_PARAMETERS" =~ ^(True|true|1)$ ]]; then
  TRAIN_ARGS+=(--ddp_find_unused_parameters)
else
  TRAIN_ARGS+=(--no-ddp_find_unused_parameters)
fi

echo "[dinov3conv-train] output=$OUTPUT_DIR dino=$DINO_MODEL_PATH gpus=$NUM_GPUS"
if [[ "$NUM_GPUS" -gt 1 ]]; then
  "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    "${TRAIN_ARGS[@]}" "$@"
else
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}" "$@"
fi
