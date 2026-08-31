#!/usr/bin/env python3
"""Check whether training artifacts contain held-out RealTextV2 samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report overlaps between an exclude list and generated training artifacts."
    )
    parser.add_argument("--exclude_list", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, action="append", default=[])
    parser.add_argument("--sft_json", type=Path, action="append", default=[])
    parser.add_argument("--csv_manifest", type=Path, action="append", default=[])
    parser.add_argument("--csv_image_column", default="image")
    parser.add_argument("--max_show", type=int, default=20)
    parser.add_argument(
        "--allow_leak",
        action="store_true",
        help="Return exit code 0 even if overlaps are found.",
    )
    return parser.parse_args()


def normalize_items(value: str) -> set[str]:
    value = value.strip().replace("\\", "/")
    if not value:
        return set()
    path = Path(value)
    return {value, str(path.with_suffix("")).replace("\\", "/"), path.stem}


def load_exclude_items(path: Path) -> set[str]:
    items: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            items.update(normalize_items(line))
    return items


def is_excluded(value: str, exclude_items: set[str]) -> bool:
    return bool(normalize_items(value) & exclude_items)


def sample_images(sample: dict[str, Any]) -> list[str]:
    images = sample.get("images", sample.get("image", sample.get("image_path", [])))
    if isinstance(images, list):
        return [str(item) for item in images]
    if images:
        return [str(images)]
    return []


def scan_image_dir(path: Path, exclude_items: set[str]) -> list[str]:
    overlaps = []
    for image_path in sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        rel = str(image_path.relative_to(path)).replace("\\", "/")
        if is_excluded(rel, exclude_items) or is_excluded(str(image_path), exclude_items):
            overlaps.append(str(image_path))
    return overlaps


def scan_sft_json(path: Path, exclude_items: set[str]) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")

    overlaps = []
    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            continue
        for image in sample_images(sample):
            if is_excluded(image, exclude_items):
                overlaps.append(f"{idx}: {image}")
                break
    return overlaps


def scan_csv_manifest(path: Path, image_column: str, exclude_items: set[str]) -> list[str]:
    overlaps = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if image_column not in (reader.fieldnames or []):
            raise ValueError(f"Column '{image_column}' not found in {path}")
        for row_idx, row in enumerate(reader, start=2):
            image = str(row.get(image_column, ""))
            if is_excluded(image, exclude_items):
                overlaps.append(f"line {row_idx}: {image}")
    return overlaps


def print_report(kind: str, path: Path, overlaps: Iterable[str], max_show: int) -> int:
    overlaps = list(overlaps)
    print(f"[{kind}] {path}: overlap={len(overlaps)}")
    for item in overlaps[:max_show]:
        print(f"  {item}")
    if len(overlaps) > max_show:
        print(f"  ... {len(overlaps) - max_show} more")
    return len(overlaps)


def main() -> None:
    args = parse_args()
    exclude_items = load_exclude_items(args.exclude_list)
    print(f"[exclude] {args.exclude_list}: rules={len(exclude_items)}")

    total_overlap = 0
    for path in args.image_dir:
        total_overlap += print_report("image_dir", path, scan_image_dir(path, exclude_items), args.max_show)
    for path in args.sft_json:
        total_overlap += print_report("sft_json", path, scan_sft_json(path, exclude_items), args.max_show)
    for path in args.csv_manifest:
        overlaps = scan_csv_manifest(path, args.csv_image_column, exclude_items)
        total_overlap += print_report("csv_manifest", path, overlaps, args.max_show)

    if total_overlap:
        print(f"[result] leakage risk: {total_overlap} held-out sample reference(s) found")
        if not args.allow_leak:
            raise SystemExit(1)
    print("[result] no overlap found" if total_overlap == 0 else "[result] overlaps allowed by --allow_leak")


if __name__ == "__main__":
    main()
