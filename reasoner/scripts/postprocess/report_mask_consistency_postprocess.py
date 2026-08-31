#!/usr/bin/env python3
"""Report-Mask Consistency Post-processing.

Replace existing [GROUNDING] boxes with the closest mask connected-component
boxes by IoU, then append mask boxes that were not matched to any report box.

The script supports two input formats:
  1. JSONL predictions with image_name/report fields.
  2. A directory tree of markdown reports.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GROUNDING_RE = re.compile(
    r"((?:\*\*\s*)?\[\s*GROUNDING\s*\](?:\s*\*\*)?\s*:\s*(?:\*\*\s*)?)"
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    r"(\s*\*\*)?",
    re.IGNORECASE,
)
CONCLUSION_RE = re.compile(
    r"((?:\*\*\s*)?\[?\s*Conclusion\s*\]?(?:\s*\*\*)?\s*:\s*(?:\*\*\s*)?)"
    r"(FORGED|AUTHENTIC|TAMPERED|FRAUDULENT|FORGERY)",
    re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>\s*</think>\s*", re.IGNORECASE | re.DOTALL)


def base_key(value: str | Path) -> str:
    stem = Path(str(value)).stem
    if stem.endswith("_mask"):
        stem = stem[:-5]
    stem = re.sub(r"[_-]?[Rr]eport$", "", stem)
    return stem


def collect_images(image_root: Path | None) -> dict[str, Path]:
    if image_root is None or not image_root.is_dir():
        return {}
    return {
        path.stem: path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def build_file_map(root: Path) -> dict[str, Path]:
    return {
        base_key(path): path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def image_size_for_key(key: str, item: dict | None, image_map: dict[str, Path]) -> tuple[int, int] | None:
    candidates: list[str] = [key]
    if item:
        for field in ("image_name", "image", "image_path", "relative_path", "file_name", "filename"):
            value = item.get(field)
            if value:
                candidates.append(base_key(value))
    for candidate in candidates:
        path = image_map.get(candidate)
        if path is not None:
            with Image.open(path) as img:
                return img.size
    return None


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            items.append(item)
    return items


def item_key(item: dict) -> str:
    for field in ("image_name", "image", "image_path", "relative_path", "file_name", "filename"):
        value = item.get(field)
        if value:
            return base_key(value)
    return ""


def mask_to_bboxes(
    mask_path: Path | None,
    min_component_area: int,
    min_area: int,
    mask_threshold: float,
    max_mask_bboxes: int,
) -> list[list[int]] | None:
    if mask_path is None or not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    threshold_value = int(round(mask_threshold * 255))
    mask_bin = (mask > threshold_value).astype(np.uint8)
    if not np.any(mask_bin):
        return []

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    boxes = []
    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_component_area or w * h < min_area:
            continue
        boxes.append([x, y, x + w - 1, y + h - 1, area])
    if max_mask_bboxes > 0:
        boxes.sort(key=lambda box: (-box[4], box[1], box[0]))
        boxes = boxes[:max_mask_bboxes]
    boxes.sort(key=lambda box: (box[1], box[0]))
    return [box[:4] for box in boxes]


def calc_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1 + 1.0)
    inter_h = max(0.0, y2 - y1 + 1.0)
    inter_area = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0] + 1.0) * max(0.0, box1[3] - box1[1] + 1.0)
    area2 = max(0.0, box2[2] - box2[0] + 1.0) * max(0.0, box2[3] - box2[1] + 1.0)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def to_pixel_box(box: list[float], coord_mode: str, image_size: tuple[int, int] | None) -> list[float]:
    if coord_mode == "pixel":
        return box
    if image_size is None:
        raise ValueError("image_root/image size is required when --bbox_coord is normalized")
    width, height = image_size
    return [
        box[0] / 999.0 * width,
        box[1] / 999.0 * height,
        box[2] / 999.0 * width,
        box[3] / 999.0 * height,
    ]


def format_box(box: list[int], output_coord: str, image_size: tuple[int, int] | None) -> list[int]:
    if output_coord == "pixel":
        return [int(v) for v in box]
    if image_size is None:
        raise ValueError("image_root/image size is required when --output_coord is normalized")
    width, height = image_size
    return [
        int(round(box[0] / max(1, width) * 999)),
        int(round(box[1] / max(1, height) * 999)),
        int(round(box[2] / max(1, width) * 999)),
        int(round(box[3] / max(1, height) * 999)),
    ]


def force_forged_conclusion(report: str) -> str:
    if CONCLUSION_RE.search(report):
        return CONCLUSION_RE.sub(lambda m: f"{m.group(1)}FORGED", report, count=1)
    return report


def force_authentic_conclusion(report: str) -> str:
    if CONCLUSION_RE.search(report):
        return CONCLUSION_RE.sub(lambda m: f"{m.group(1)}AUTHENTIC", report, count=1)
    return report


def append_extra_boxes(report: str, boxes: Iterable[list[int]]) -> str:
    boxes = list(boxes)
    if not boxes:
        return report

    insertion = []
    for idx, box in enumerate(boxes, 1):
        insertion.extend(
            [
                "",
                f"### ANOMALY_RMC_EXTRA_{idx:03d}: Report-Mask Consistency Region",
                f"[GROUNDING]: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]",
                "[REASON]: This additional suspicious region is provided by the mask consistency post-processing.",
            ]
        )

    summary_match = re.search(r"\n\s*##\s*SUMMARY", report, flags=re.IGNORECASE)
    if summary_match:
        before = report[: summary_match.start()].rstrip()
        after = report[summary_match.start() :].lstrip()
        return before + "\n\n" + "\n".join(insertion).lstrip() + "\n\n" + after
    return report.rstrip() + "\n\n" + "\n".join(insertion).lstrip()


def match_boxes_by_iou(
    report_boxes_pixel: list[list[float]],
    mask_boxes: list[list[int]],
    min_match_iou: float,
) -> tuple[dict[int, int], set[int]]:
    pairs = []
    for report_idx, report_box in enumerate(report_boxes_pixel):
        for mask_idx, mask_box in enumerate(mask_boxes):
            pairs.append((calc_iou(report_box, mask_box), report_idx, mask_idx))
    pairs.sort(key=lambda item: item[0], reverse=True)

    matched_report: set[int] = set()
    matched_mask: set[int] = set()
    matches: dict[int, int] = {}
    for iou, report_idx, mask_idx in pairs:
        if iou < min_match_iou:
            break
        if report_idx in matched_report or mask_idx in matched_mask:
            continue
        matches[report_idx] = mask_idx
        matched_report.add(report_idx)
        matched_mask.add(mask_idx)

    return matches, matched_mask


def postprocess_report(
    report: str,
    mask_boxes: list[list[int]],
    image_size: tuple[int, int] | None,
    bbox_coord: str,
    output_coord: str,
    min_match_iou: float,
    append_extra: bool,
    force_forged_with_mask: bool,
    force_authentic_with_empty_mask: bool,
) -> tuple[str, dict[str, int]]:
    stats = {
        "groundings": 0,
        "matched": 0,
        "unmatched_groundings": 0,
        "extra_appended": 0,
    }
    report = THINK_RE.sub("", report or "").replace("<think>\n\n</think>\n\n", "").lstrip()

    if mask_boxes and force_forged_with_mask:
        report = force_forged_conclusion(report)
    elif not mask_boxes and force_authentic_with_empty_mask:
        report = force_authentic_conclusion(report)

    matches = list(GROUNDING_RE.finditer(report))
    stats["groundings"] = len(matches)
    if not matches:
        if append_extra and mask_boxes:
            formatted = [format_box(box, output_coord, image_size) for box in mask_boxes]
            stats["extra_appended"] = len(formatted)
            return append_extra_boxes(report, formatted), stats
        return report, stats

    report_boxes = [[float(match.group(i)) for i in range(2, 6)] for match in matches]
    report_boxes_pixel = [to_pixel_box(box, bbox_coord, image_size) for box in report_boxes]
    match_map, matched_mask = match_boxes_by_iou(report_boxes_pixel, mask_boxes, min_match_iou)

    pieces = []
    cursor = 0
    for report_idx, match in enumerate(matches):
        pieces.append(report[cursor : match.start()])
        mask_idx = match_map.get(report_idx)
        if mask_idx is None:
            pieces.append(match.group(0))
            stats["unmatched_groundings"] += 1
        else:
            box = format_box(mask_boxes[mask_idx], output_coord, image_size)
            pieces.append(f"{match.group(1)}[{box[0]}, {box[1]}, {box[2]}, {box[3]}]{match.group(6) or ''}")
            stats["matched"] += 1
        cursor = match.end()
    pieces.append(report[cursor:])
    updated = "".join(pieces)

    if append_extra:
        extra_boxes = [box for idx, box in enumerate(mask_boxes) if idx not in matched_mask]
        if extra_boxes:
            formatted = [format_box(box, output_coord, image_size) for box in extra_boxes]
            stats["extra_appended"] = len(formatted)
            updated = append_extra_boxes(updated, formatted)

    return updated, stats


def process_jsonl(args: argparse.Namespace) -> dict[str, int]:
    if args.source_jsonl is None or args.output_jsonl is None:
        raise ValueError("--source_jsonl and --output_jsonl are required in jsonl mode")
    items = load_jsonl(args.source_jsonl)
    mask_map = build_file_map(args.mask_dir)
    image_map = collect_images(args.image_root)

    totals = {
        "items": 0,
        "missing_mask": 0,
        "empty_mask": 0,
        "groundings": 0,
        "matched": 0,
        "unmatched_groundings": 0,
        "extra_appended": 0,
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for item in tqdm(items, desc="RMC jsonl", unit="item"):
            key = item_key(item)
            mask_boxes = mask_to_bboxes(
                mask_map.get(key),
                args.min_component_area,
                args.min_area,
                args.mask_threshold,
                args.max_mask_bboxes,
            )
            if mask_boxes is None:
                totals["missing_mask"] += 1
                mask_boxes = []
            elif not mask_boxes:
                totals["empty_mask"] += 1

            report = item.get("report") or item.get("response") or ""
            new_report, local = postprocess_report(
                report=report,
                mask_boxes=mask_boxes,
                image_size=image_size_for_key(key, item, image_map),
                bbox_coord=args.bbox_coord,
                output_coord=args.output_coord,
                min_match_iou=args.min_match_iou,
                append_extra=not args.no_append_extra,
                force_forged_with_mask=args.force_forged_with_mask,
                force_authentic_with_empty_mask=args.force_authentic_with_empty_mask,
            )
            out_item = dict(item)
            if "report" in out_item or "response" not in out_item:
                out_item["report"] = new_report
            else:
                out_item["response"] = new_report
            f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

            totals["items"] += 1
            for name in ("groundings", "matched", "unmatched_groundings", "extra_appended"):
                totals[name] += local[name]
    return totals


def candidate_mask_keys(md_file: Path, report_dir: Path) -> list[str]:
    rel = md_file.relative_to(report_dir)
    keys = [base_key(rel), base_key(rel.name)]
    if rel.stem:
        keys.append(base_key(rel.stem))
    return list(dict.fromkeys(keys))


def process_markdown(args: argparse.Namespace) -> dict[str, int]:
    if args.report_dir is None or args.output_report_dir is None:
        raise ValueError("--report_dir and --output_report_dir are required in md mode")
    md_files = sorted(args.report_dir.rglob("*.md"))
    mask_map = build_file_map(args.mask_dir)
    image_map = collect_images(args.image_root)

    totals = {
        "items": 0,
        "missing_mask": 0,
        "empty_mask": 0,
        "groundings": 0,
        "matched": 0,
        "unmatched_groundings": 0,
        "extra_appended": 0,
    }
    for md_file in tqdm(md_files, desc="RMC markdown", unit="file"):
        rel = md_file.relative_to(args.report_dir)
        out_file = args.output_report_dir / rel
        key = ""
        mask_path = None
        for candidate in candidate_mask_keys(md_file, args.report_dir):
            if candidate in mask_map:
                key = candidate
                mask_path = mask_map[candidate]
                break
        if not key:
            key = base_key(md_file)

        mask_boxes = mask_to_bboxes(
            mask_path,
            args.min_component_area,
            args.min_area,
            args.mask_threshold,
            args.max_mask_bboxes,
        )
        if mask_boxes is None:
            totals["missing_mask"] += 1
            mask_boxes = []
        elif not mask_boxes:
            totals["empty_mask"] += 1

        report = md_file.read_text(encoding="utf-8")
        new_report, local = postprocess_report(
            report=report,
            mask_boxes=mask_boxes,
            image_size=image_size_for_key(key, None, image_map),
            bbox_coord=args.bbox_coord,
            output_coord=args.output_coord,
            min_match_iou=args.min_match_iou,
            append_extra=not args.no_append_extra,
            force_forged_with_mask=args.force_forged_with_mask,
            force_authentic_with_empty_mask=args.force_authentic_with_empty_mask,
        )

        out_file.parent.mkdir(parents=True, exist_ok=True)
        if new_report == report:
            shutil.copy2(md_file, out_file)
        else:
            out_file.write_text(new_report, encoding="utf-8")

        totals["items"] += 1
        for name in ("groundings", "matched", "unmatched_groundings", "extra_appended"):
            totals[name] += local[name]
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report-Mask Consistency Post-processing.")
    parser.add_argument("--mode", choices=["jsonl", "md"], required=True)
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, default=None)

    parser.add_argument("--source_jsonl", type=Path, default=None)
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--report_dir", type=Path, default=None)
    parser.add_argument("--output_report_dir", type=Path, default=None)

    parser.add_argument("--bbox_coord", choices=["pixel", "normalized"], default="pixel")
    parser.add_argument("--output_coord", choices=["pixel", "normalized"], default="pixel")
    parser.add_argument("--min_match_iou", type=float, default=0.0)
    parser.add_argument("--min_component_area", type=int, default=16)
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--max_mask_bboxes", type=int, default=0)
    parser.add_argument("--no_append_extra", action="store_true")
    parser.add_argument("--force_forged_with_mask", action="store_true")
    parser.add_argument("--force_authentic_with_empty_mask", action="store_true")
    parser.add_argument("--stats_json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mask_dir.is_dir():
        raise FileNotFoundError(f"Mask dir not found: {args.mask_dir}")
    if args.mode == "jsonl":
        stats = process_jsonl(args)
    else:
        stats = process_markdown(args)

    if args.stats_json is not None:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
