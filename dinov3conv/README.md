# DINOv3Conv Detector/Localizer

DINOv3-based model for image-level forgery detection and mask prediction.

## Files

- `model/dinov3_localizer.py`: model definition.
- `train_dinov3_localizer.py`: training.
- `eval_dinov3_localizer_testsets.py`: evaluation and prediction.
- `scripts/*.sh`: config-driven entry points.
- `configs/*.env.example`: path templates.

## Weights

- `DINO_MODEL_PATH`: DINOv3 backbone directory or model id.
- `CHECKPOINT`: DINOv3Conv checkpoint directory containing `pytorch_model.bin`.

`pytorch_model.bin` is the full `DINOv3TamperLocalizer.state_dict()`, including
the DINO backbone, classifier head, feature projection, and mask decoder.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
cp configs/train_dinov3conv.env.example configs/train_dinov3conv.env
# edit DINO_MODEL_PATH, SFT_JSON_PATH, VAL_SFT_JSON_PATH, MASK_ROOT, TRAIN_EXCLUDE_LIST
bash scripts/train_dinov3conv.sh configs/train_dinov3conv.env
```

SFT items need an image path, an assistant label containing `FORGED` or
`AUTHENTIC`, and either an explicit mask path or `MASK_ROOT`.

For multi-GPU training, set `CUDA_VISIBLE_DEVICES` and `NUM_GPUS` in the env file.

## Evaluate

```bash
cp configs/eval_dinov3conv.env.example configs/eval_dinov3conv.env
# edit DINO_MODEL_PATH, CHECKPOINT, EVAL_DATASETS
bash scripts/eval_dinov3conv.sh configs/eval_dinov3conv.env
```

`EVAL_DATASETS` format:

```text
Name:/path/to/images:/path/to/masks;Other:/path/to/images:/path/to/masks
```

## Predict

```bash
cp configs/predict_dinov3conv.env.example configs/predict_dinov3conv.env
# edit DINO_MODEL_PATH, CHECKPOINT, IMAGE_DIR
bash scripts/predict_dinov3conv.sh configs/predict_dinov3conv.env
```

Outputs: image-level probabilities and optional localization probability maps.
