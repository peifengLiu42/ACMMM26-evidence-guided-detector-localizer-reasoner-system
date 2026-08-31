#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CONDA_ENV="${CONDA_ENV:-llama-factory}"
export LLAMA_FACTORY_CLI="${LLAMA_FACTORY_CLI:-llamafactory-cli}"
export TRAIN_EXCLUDE_LIST="${TRAIN_EXCLUDE_LIST:-}"
REASONER_DATA_DIR="${REASONER_DATA_DIR:-data}"
REASONER_SFT_JSON="$REASONER_DATA_DIR/realtext_explain_train_sft.filtered.json"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
TRAIN_CONFIG="train_lora/qwen3vl4b_realtext_explain_lora_sft.yaml"
RUN_CONFIG="$TRAIN_CONFIG"
TMP_RUN_CONFIG=""
cleanup() {
  if [[ -n "$TMP_RUN_CONFIG" && -f "$TMP_RUN_CONFIG" ]]; then
    rm -f "$TMP_RUN_CONFIG"
  fi
}
trap cleanup EXIT

if [[ ! -f "$REASONER_SFT_JSON" ]]; then
  echo "[reasoner-train] filtered SFT json not found: $REASONER_SFT_JSON" >&2
  echo "[reasoner-train] run dataset/generate_explain_llm.py or dataset/filter_sft_by_exclude_list.py first" >&2
  exit 1
fi
if [[ -z "$MODEL_NAME_OR_PATH" ]]; then
  echo "[reasoner-train] set MODEL_NAME_OR_PATH to your local Qwen3-VL-4B-Instruct path or model id" >&2
  exit 1
fi

if [[ -n "$TRAIN_EXCLUDE_LIST" ]]; then
  python dataset/check_split_leakage.py \
    --exclude_list "$TRAIN_EXCLUDE_LIST" \
    --sft_json "$REASONER_SFT_JSON"
fi

TMP_RUN_CONFIG="$(mktemp /tmp/qwen3vl4b_realtext_explain_lora_sft.XXXXXX.yaml)"
RUN_CONFIG="$TMP_RUN_CONFIG"
python - "$TRAIN_CONFIG" "$RUN_CONFIG" "$REASONER_DATA_DIR" "$MODEL_NAME_OR_PATH" "$OUTPUT_DIR" "$RESUME_FROM_CHECKPOINT" <<'PY'
import sys
src, dst, data_dir, model_name_or_path, output_dir, resume_from_checkpoint = sys.argv[1:]
with open(src, "r", encoding="utf-8") as f:
    lines = f.readlines()
updates = {
    "dataset_dir": data_dir,
    "model_name_or_path": model_name_or_path,
}
if output_dir:
    updates["output_dir"] = output_dir
if resume_from_checkpoint:
    updates["resume_from_checkpoint"] = resume_from_checkpoint
seen = set()
for idx, line in enumerate(lines):
    key = line.split(":", 1)[0].strip()
    if key in updates:
        lines[idx] = f"{key}: {updates[key]}\n"
        seen.add(key)
for key, value in updates.items():
    if key not in seen:
        lines.append(f"{key}: {value}\n")
with open(dst, "w", encoding="utf-8") as f:
    f.writelines(lines)
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[reasoner-train] dry run passed"
  echo "[reasoner-train] ${LLAMA_FACTORY_CLI} train ${RUN_CONFIG}"
  exit 0
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

"${LLAMA_FACTORY_CLI}" train "$RUN_CONFIG"
