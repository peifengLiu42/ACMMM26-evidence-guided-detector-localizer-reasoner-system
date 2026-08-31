#!/usr/bin/env python3
"""Filter SFT JSON samples by an excluded image stem/path list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove held-out images from an SFT JSON file.")
    parser.add_argument("--input_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--exclude_list", type=Path, required=True)
    return parser.parse_args()


def load_exclude_items(path: Path) -> set[str]:
    items: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value:
                continue
            value = value.replace("\\", "/")
            items.add(value)
            items.add(str(Path(value).with_suffix("")))
            items.add(Path(value).stem)
    return items


def sample_images(sample: dict[str, Any]) -> list[str]:
    images = sample.get("images", sample.get("image", sample.get("image_path", [])))
    if isinstance(images, list):
        return [str(item) for item in images]
    if images:
        return [str(images)]
    return []


def is_excluded(sample: dict[str, Any], exclude_items: set[str]) -> bool:
    for image in sample_images(sample):
        path = Path(image)
        text = str(path).replace("\\", "/")
        no_ext = str(path.with_suffix("")).replace("\\", "/")
        if text in exclude_items or no_ext in exclude_items or path.stem in exclude_items:
            return True
    return False


def main() -> None:
    args = parse_args()
    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {args.input_json}")

    exclude_items = load_exclude_items(args.exclude_list)
    kept = [sample for sample in data if not is_excluded(sample, exclude_items)]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"input: {len(data)}")
    print(f"kept: {len(kept)}")
    print(f"removed: {len(data) - len(kept)}")
    print(f"output_json: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
