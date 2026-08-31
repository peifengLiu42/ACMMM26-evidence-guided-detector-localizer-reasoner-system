#!/usr/bin/env python3
"""Create a RealTextV2 in-domain validation/test stem list.

The project uses a 10% split from `RealTextV2/train/image` as an in-domain
validation set. This script creates a deterministic stem list such as
`RealTextV2/train/test.txt`.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split RealTextV2 train images into an in-domain test list.")
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_list", type=Path, required=True)
    parser.add_argument("--remaining_list", type=Path, default=None)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write_relative_paths",
        action="store_true",
        help="Write relative image paths instead of filename stems.",
    )
    return parser.parse_args()


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def format_item(path: Path, image_dir: Path, write_relative_paths: bool) -> str:
    if write_relative_paths:
        return str(path.relative_to(image_dir))
    return path.stem


def write_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not 0.0 < args.ratio < 1.0:
        raise ValueError("--ratio must be in (0, 1)")

    image_paths = collect_images(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found under {args.image_dir}")

    shuffled = list(image_paths)
    random.Random(args.seed).shuffle(shuffled)
    split_size = int(len(shuffled) * args.ratio)
    selected = shuffled[:split_size]
    remaining = shuffled[split_size:]

    selected_items = [format_item(path, args.image_dir, args.write_relative_paths) for path in selected]
    remaining_items = [format_item(path, args.image_dir, args.write_relative_paths) for path in remaining]

    write_list(args.output_list, selected_items)
    if args.remaining_list is not None:
        write_list(args.remaining_list, remaining_items)

    print(f"images: {len(image_paths)}")
    print(f"selected: {len(selected_items)}")
    print(f"remaining: {len(remaining_items)}")
    print(f"seed: {args.seed}")
    print(f"ratio: {args.ratio}")
    print(f"output_list: {args.output_list.resolve()}")
    if args.remaining_list is not None:
        print(f"remaining_list: {args.remaining_list.resolve()}")


if __name__ == "__main__":
    main()
