# Notes

This directory is reserved for path-neutral notes about the RealTextV2
difficulty mining stage.

This release intentionally contains only our difficulty mining utilities. It
does not include DTD/RTM/SparseViT training or inference code. Generate
prediction maps with the upstream projects first, then use
`tools/realtext_difficulty_mining.py` to produce:

- `difficulty_manifest.csv`
- `difficulty_manifest.jsonl`
- `hard_regions.json`
- `easy.txt`, `medium.txt`, `hard.txt`
- `summary.json`
