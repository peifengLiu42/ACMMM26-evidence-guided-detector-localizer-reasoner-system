#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var REALT_TRAIN_IMAGE_DIR
require_var REALT_TRAIN_MASK_DIR
require_var DOC_PRED_DIR
require_var RTM_PRED_DIR
require_var SPARSE_PRED_DIR

DIFFICULTY_DIR="${DIFFICULTY_DIR:-$REPO_DIR/outputs/difficulty_mining/realtext_train}"
WORKERS="${WORKERS:-32}"
LIMIT="${LIMIT:-0}"
TRAIN_EXCLUDE_LIST="${TRAIN_EXCLUDE_LIST:-${REALT_VAL_LIST:-}}"

python "$REPO_DIR/tools/realtext_difficulty_mining.py" \
  --image_dir "$REALT_TRAIN_IMAGE_DIR" \
  --gt_dir "$REALT_TRAIN_MASK_DIR" \
  --doc_pred_dir "$DOC_PRED_DIR" \
  --asc_pred_dir "$RTM_PRED_DIR" \
  --sparse_pred_dir "$SPARSE_PRED_DIR" \
  --output_dir "$DIFFICULTY_DIR" \
  ${TRAIN_EXCLUDE_LIST:+--exclude_list_path "$TRAIN_EXCLUDE_LIST"} \
  --workers "$WORKERS" \
  --limit "$LIMIT"
