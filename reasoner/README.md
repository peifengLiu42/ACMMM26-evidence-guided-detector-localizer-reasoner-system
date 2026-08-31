# Qwen3-VL Reasoner

Qwen3-VL reasoner for generating structured forensic reports from image,
detector, and localizer evidence.

## Files

- `dataset/generate_explain_llm.py`: builds GT-box SFT data from masks and reports.
- `dataset/regen_realtext_mask.py`: creates zero masks for authentic images.
- `dataset/split_realtext_indomain_test.py`: creates an in-domain held-out list.
- `pipeline_vllm.py`: evidence-guided vLLM inference.
- `scripts/postprocess/report_mask_consistency_postprocess.py`: RMC.

## Build SFT Data

Create the held-out list if needed:

```bash
python dataset/split_realtext_indomain_test.py \
  --image_dir /path/to/RealTextV2/train/image \
  --output_list /path/to/RealTextV2/train/test.txt \
  --remaining_list /path/to/RealTextV2/train/train_remaining.txt \
  --ratio 0.10 \
  --seed 42
```

Create `regen_mask` if authentic masks are missing:

```bash
python dataset/regen_realtext_mask.py \
  --image_dir /path/to/RealTextV2/train/image \
  --mask_dir /path/to/RealTextV2/train/mask \
  --output_dir /path/to/RealTextV2/train/regen_mask
```

Generate filtered SFT data:

```bash
python dataset/generate_explain_llm.py \
  --image_dir /path/to/RealTextV2/train/image \
  --mask_dir /path/to/RealTextV2/train/regen_mask \
  --report_dir /path/to/RealTextV2/train/report \
  --output_file data/realtext_explain_train_sft.filtered.json \
  --exclude_list /path/to/RealTextV2/train/test.txt
```

## Train

Install LLaMA-Factory, then run:

```bash
MODEL_NAME_OR_PATH=/path/to/Qwen3-VL-4B-Instruct \
bash scripts/train_qwen3vl4b_realtext_explain.sh
```

or:

```bash
MODEL_NAME_OR_PATH=/path/to/Qwen3-VL-8B-Instruct \
bash scripts/train_qwen3vl8b_realtext_explain.sh
```

Optional variables:

```bash
REASONER_DATA_DIR=/path/to/data_dir
TRAIN_EXCLUDE_LIST=/path/to/RealTextV2/train/test.txt
OUTPUT_DIR=/path/to/save_dir
RESUME_FROM_CHECKPOINT=/path/to/checkpoint
DRY_RUN=1
```

## Inference

```bash
IMAGE_DIR=/path/to/RealTextV2/test/image \
DETECTOR_JSON=/path/to/detector_predictions.jsonl \
HEATMAP_DIR=/path/to/localizer_masks_or_heatmaps \
MODEL_NAME_OR_PATH=/path/to/Qwen3-VL-4B-Instruct \
ADAPTER_CHECKPOINT=/path/to/reasoner_lora_checkpoint \
bash scripts/run_pipeline_vllm.sh
```

For a merged model, set `MERGED_MODEL=true` and omit `ADAPTER_CHECKPOINT`.

## RMC

```bash
python scripts/postprocess/report_mask_consistency_postprocess.py \
  --mode jsonl \
  --source_jsonl output/predictions.jsonl \
  --output_jsonl output/predictions_rmc.jsonl \
  --mask_dir /path/to/final_masks \
  --image_root /path/to/RealTextV2/test/image \
  --bbox_coord pixel \
  --output_coord pixel \
  --force_forged_with_mask \
  --force_authentic_with_empty_mask \
  --stats_json output/predictions_rmc.stats.json
```

RMC aligns `[GROUNDING]` boxes with final masks. It does not rewrite semantic
explanations.
