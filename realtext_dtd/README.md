# RealTextV2 Difficulty Mining

This folder contains only our difficulty mining utilities. It does not include
DTD/RTM/SparseViT training code, inference code, weights, checkpoints, or data.

Use the upstream projects to generate prediction maps:

- DocTamper: https://github.com/qcf-568/DocTamper
- RTM: https://github.com/DrLuo/RTM
- SparseViT: https://github.com/scu-zjz/SparseViT

## Files

- `tools/realtext_difficulty_mining.py`: mines difficulty labels and hard regions.
- `tools/realtext_difficulty_postprocess_v3.py`: refines labels and error types.
- `scripts/mine_difficulty.sh`: wrapper using `configs/realtext_paths.env`.
- `configs/realtext_paths.env.example`: path template.

## Setup

```bash
pip install -r requirements.txt
cp configs/realtext_paths.env.example configs/realtext_paths.env
```

Edit `configs/realtext_paths.env`:

```text
REALT_TRAIN_IMAGE_DIR=/path/to/RealTextV2/train/image
REALT_TRAIN_MASK_DIR=/path/to/RealTextV2/train/regen_mask
TRAIN_EXCLUDE_LIST=/path/to/RealTextV2/train/test.txt

DOC_PRED_DIR=/path/to/doctamper/maps
RTM_PRED_DIR=/path/to/rtm/maps
SPARSE_PRED_DIR=/path/to/sparsevit/maps
DIFFICULTY_DIR=outputs/difficulty_mining/realtext_train
```

## Run

```bash
bash scripts/mine_difficulty.sh
```

Outputs in `DIFFICULTY_DIR`:

- `difficulty_manifest.csv`
- `difficulty_manifest.jsonl`
- `hard_regions.json`
- `easy.txt`, `medium.txt`, `hard.txt`
- `summary.json`

Optional post-processing:

```bash
python tools/realtext_difficulty_postprocess_v3.py \
  --manifest /path/to/difficulty_manifest.csv \
  --hard_regions /path/to/hard_regions.json \
  --output_dir /path/to/postprocessed_difficulty
```

## Acknowledgements

We thank the authors of DocTamper, RTM, and SparseViT for releasing their code.
