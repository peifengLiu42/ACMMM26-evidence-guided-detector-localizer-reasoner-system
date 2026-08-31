#!/usr/bin/env python3
"""Regenerate RealTextV2 training masks.

The original RealTextV2 training split stores masks only for forged images.
This script creates a complete `regen_mask` tree aligned with `train/image`:
existing forged masks are copied from `train/mask`, and authentic images receive
an all-zero mask with the same spatial size as the image.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate RealTextV2 train/regen_mask.")
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mask_suffix", default="_mask.png")
    return parser.parse_args()


def build_mask_lookup(mask_dir: Path, mask_suffix: str) -> tuple[dict[str, Path], int]:
    lookup: dict[str, Path] = {}
    duplicate_count = 0
    for path in sorted(mask_dir.rglob(f"*{mask_suffix}")):
        if not path.is_file():
            continue
        if path.name in lookup:
            duplicate_count += 1
            continue
        lookup[path.name] = path
    return lookup, duplicate_count


def iter_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def regenerate_masks(image_dir: Path, mask_dir: Path, output_dir: Path, mask_suffix: str) -> dict[str, int]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_lookup, duplicate_count = build_mask_lookup(mask_dir, mask_suffix)
    image_paths = iter_images(image_dir)

    stats = {
        "images": len(image_paths),
        "copied_masks": 0,
        "generated_empty_masks": 0,
        "read_errors": 0,
        "duplicate_mask_names": duplicate_count,
    }

    for image_path in tqdm(image_paths, desc="Regenerating masks", unit="image"):
        rel_path = image_path.relative_to(image_dir)
        out_subdir = output_dir / rel_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        mask_name = f"{image_path.stem}{mask_suffix}"
        output_path = out_subdir / mask_name
        source_mask = mask_lookup.get(mask_name)

        if source_mask is not None:
            shutil.copy2(source_mask, output_path)
            stats["copied_masks"] += 1
            continue

        try:
            with Image.open(image_path) as image:
                Image.new("L", image.size, 0).save(output_path)
            stats["generated_empty_masks"] += 1
        except Exception as exc:
            stats["read_errors"] += 1
            print(f"[warn] failed to create empty mask for {image_path}: {exc}")

    return stats


def main() -> None:
    args = parse_args()
    stats = regenerate_masks(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        mask_suffix=args.mask_suffix,
    )
    print("Done.")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"output_dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
