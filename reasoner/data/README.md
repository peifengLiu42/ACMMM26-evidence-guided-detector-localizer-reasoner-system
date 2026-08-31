# Reasoner Data

Expected file:

- `realtext_explain_train_sft.filtered.json`

Generate it from RealTextV2 images, `regen_mask`, reports, and the held-out list:

```bash
python dataset/generate_explain_llm.py \
  --image_dir /path/to/RealTextV2/train/image \
  --mask_dir /path/to/RealTextV2/train/regen_mask \
  --report_dir /path/to/RealTextV2/train/report \
  --output_file data/realtext_explain_train_sft.filtered.json \
  --exclude_list /path/to/RealTextV2/train/test.txt
```
