# Evidence-Guided Detector-Localizer-Reasoner System

## Modules

```text
dinov3conv/    DINOv3Conv detector/localizer
realtext_dtd/  RealTextV2 difficulty mining utilities only
reasoner/      Qwen3-VL reasoner and report-mask consistency post-processing
```

`realtext_dtd/` does not include DTD/RTM/SparseViT training or inference code.
Generate their prediction maps with the original repositories.

## Required Inputs

- RealTextV2 images, masks, split lists, and expert reports.
- DINOv3 backbone for `dinov3conv/`.
- DINOv3Conv checkpoint for detector/localizer inference.
- DocTamper/RTM/SparseViT prediction maps for difficulty mining.
- Qwen3-VL base model and reasoner LoRA or merged checkpoint.

## Workflow

1. Prepare `RealTextV2/train/test.txt` as the fixed held-out stem list.
2. Train or run DINOv3Conv in `dinov3conv/`.
3. Generate upstream localizer maps, then run `realtext_dtd/` difficulty mining.
4. Build reasoner SFT data with held-out stems removed.
5. Train or run the Qwen3-VL reasoner.
6. Optionally run RMC to align `[GROUNDING]` boxes with final masks.

## Entry Points

- `dinov3conv/README.md`
- `realtext_dtd/README.md`
- `reasoner/README.md`

## Acknowledgements

We thank the authors of DocTamper, RTM, and SparseViT:

- DocTamper: https://github.com/qcf-568/DocTamper
- RTM: https://github.com/DrLuo/RTM
- SparseViT: https://github.com/scu-zjz/SparseViT
- Qwen3-VL: we acknowledge the Qwen3-VL work and its contribution to multimodal reasoning, which provides the foundation for our reasoner.
