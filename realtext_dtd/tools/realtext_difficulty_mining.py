import argparse
import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser("Mine RealTextV2 difficulty levels from upstream prediction maps")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--doc_pred_dir", required=True)
    parser.add_argument("--asc_pred_dir", required=True)
    parser.add_argument("--sparse_pred_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exclude_list_path", default="")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--doc_threshold", type=float, default=0.5)
    parser.add_argument("--doc_threshold_mode", choices=("absolute", "relative", "otsu", "nonzero"), default="absolute")
    parser.add_argument("--asc_threshold", type=float, default=0.5)
    parser.add_argument("--asc_threshold_mode", choices=("absolute", "relative", "otsu", "nonzero"), default="relative")
    parser.add_argument("--sparse_threshold", type=float, default=0.5)
    parser.add_argument("--sparse_threshold_mode", choices=("absolute", "relative", "otsu", "nonzero"), default="relative")

    parser.add_argument("--easy_score", type=float, default=0.25)
    parser.add_argument("--easy_f1", type=float, default=0.75)
    parser.add_argument("--easy_recall", type=float, default=0.80)
    parser.add_argument("--easy_fn_ratio", type=float, default=0.15)
    parser.add_argument("--easy_fp_ratio", type=float, default=0.20)

    parser.add_argument("--hard_score", type=float, default=0.60)
    parser.add_argument("--hard_f1", type=float, default=0.40)
    parser.add_argument("--hard_fn_ratio", type=float, default=0.45)
    parser.add_argument("--hard_fp_ratio", type=float, default=0.80)
    parser.add_argument("--hard_missed_component_ratio", type=float, default=0.50)

    parser.add_argument("--component_miss_recall", type=float, default=0.10)
    parser.add_argument("--large_component_recall", type=float, default=0.30)
    parser.add_argument("--large_component_area_frac", type=float, default=0.05)
    parser.add_argument("--min_component_area", type=int, default=16)
    parser.add_argument("--region_min_area", type=int, default=16)
    parser.add_argument("--max_regions_per_image", type=int, default=30)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--save_error_masks", action="store_true")
    return parser.parse_args()


def clean_stem(path):
    stem = Path(path).stem
    for suffix in ("_mask", "_pred", "_heatmap", "_prob"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def collect_files(root, extensions):
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions])


def load_exclude_items(path):
    if not path:
        return set()
    items = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value:
                continue
            value = value.replace("\\", "/")
            items.add(value)
            items.add(str(Path(value).with_suffix("")))
            items.add(Path(value).stem)
    return items


def is_excluded(path, image_dir, exclude_items):
    if not exclude_items:
        return False
    rel = str(Path(path).relative_to(image_dir)).replace("\\", "/")
    rel_no_ext = str(Path(rel).with_suffix(""))
    stem = Path(path).stem
    return rel in exclude_items or rel_no_ext in exclude_items or stem in exclude_items


def build_stem_map(root, extensions):
    mapping = {}
    duplicates = {}
    for path in collect_files(root, extensions):
        stem = clean_stem(path)
        if stem in mapping:
            duplicates.setdefault(stem, [str(mapping[stem])]).append(str(path))
            continue
        mapping[stem] = path
    return mapping, duplicates


def read_gray(path):
    if path is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def resize_to(gray, shape, interpolation):
    if gray is None:
        return None
    h, w = shape
    if gray.shape[:2] == (h, w):
        return gray
    return cv2.resize(gray, (w, h), interpolation=interpolation)


def binarize(gray, threshold, mode):
    if gray is None:
        return None
    arr = gray.astype(np.float32)
    if mode == "nonzero":
        return arr > 0
    if mode == "otsu":
        if arr.max() <= arr.min():
            return np.zeros(arr.shape, dtype=bool)
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out > 0
    if mode == "relative":
        mn = float(arr.min())
        mx = float(arr.max())
        if mx <= mn:
            return np.zeros(arr.shape, dtype=bool)
        cutoff = mn + threshold * (mx - mn)
        return arr > cutoff

    cutoff = threshold
    if threshold <= 1.0 and arr.max() > 1.0:
        cutoff = threshold * 255.0
    return arr > cutoff


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def confusion(pred, gt):
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    return tp, fp, fn, tn


def f1_from_counts(tp, fp, fn):
    return safe_div(2 * tp, 2 * tp + fp + fn)


def iou_from_counts(tp, fp, fn):
    return safe_div(tp, tp + fp + fn)


def binary_iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return safe_div(inter, union)


def component_stats(gt, doc, asc, sparse, args):
    gt_u8 = gt.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gt_u8, connectivity=8)
    component_count = 0
    missed_count = 0
    large_missed = False
    gt_area = int(gt.sum())
    large_area = max(args.min_component_area, int(gt_area * args.large_component_area_frac))

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < args.min_component_area:
            continue
        comp = labels == idx
        component_count += 1
        doc_recall = safe_div(int(np.logical_and(comp, doc).sum()), area)
        if doc_recall < args.component_miss_recall:
            missed_count += 1
        if area >= large_area and doc_recall < args.large_component_recall:
            large_missed = True

    missed_ratio = safe_div(missed_count, component_count)
    return component_count, missed_count, missed_ratio, large_missed


def clamp_crop_box(x1, y1, x2, y2, width, height, crop_size):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = crop_size // 2
    left = max(0, min(cx - half, max(0, width - crop_size)))
    top = max(0, min(cy - half, max(0, height - crop_size)))
    right = min(width, left + crop_size)
    bottom = min(height, top + crop_size)
    return [int(left), int(top), int(right), int(bottom)]


def region_boxes(mask, width, height, args):
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    regions = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < args.region_min_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        bbox = [x, y, x + w, y + h]
        regions.append(
            {
                "bbox": bbox,
                "area": area,
                "crop_box": clamp_crop_box(x, y, x + w, y + h, width, height, args.crop_size),
            }
        )
    regions.sort(key=lambda item: item["area"], reverse=True)
    return regions[: args.max_regions_per_image]


def classify(record, args):
    hard = (
        record["difficulty_score"] >= args.hard_score
        or record["doc_f1"] < args.hard_f1
        or record["fn_ratio"] > args.hard_fn_ratio
        or record["fp_ratio"] > args.hard_fp_ratio
        or record["missed_component_ratio"] > args.hard_missed_component_ratio
        or record["large_missed_component"]
    )
    if hard:
        return "hard"

    easy = (
        record["difficulty_score"] < args.easy_score
        and record["doc_f1"] >= args.easy_f1
        and record["doc_recall"] >= args.easy_recall
        and record["fn_ratio"] <= args.easy_fn_ratio
        and record["fp_ratio"] <= args.easy_fp_ratio
        and not record["large_missed_component"]
    )
    return "easy" if easy else "medium"


def process_one(task):
    image_path, rel_path, maps, args_dict = task
    args = argparse.Namespace(**args_dict)
    stem = clean_stem(image_path)
    gt_path = maps["gt"].get(stem)
    doc_path = maps["doc"].get(stem)
    asc_path = maps["asc"].get(stem)
    sparse_path = maps["sparse"].get(stem)

    missing = []
    if gt_path is None:
        missing.append("gt")
    if doc_path is None:
        missing.append("doc")
    if asc_path is None:
        missing.append("asc")
    if sparse_path is None:
        missing.append("sparse")

    gt_gray = read_gray(gt_path)
    if gt_gray is None:
        return None, {"image": rel_path, "stem": stem, "missing": missing or ["gt_read_failed"]}

    h, w = gt_gray.shape[:2]
    gt = gt_gray > 0

    doc_gray = resize_to(read_gray(doc_path), (h, w), cv2.INTER_NEAREST)
    asc_gray = resize_to(read_gray(asc_path), (h, w), cv2.INTER_LINEAR)
    sparse_gray = resize_to(read_gray(sparse_path), (h, w), cv2.INTER_LINEAR)

    doc = binarize(doc_gray, args.doc_threshold, args.doc_threshold_mode)
    asc = binarize(asc_gray, args.asc_threshold, args.asc_threshold_mode)
    sparse = binarize(sparse_gray, args.sparse_threshold, args.sparse_threshold_mode)
    if doc is None:
        doc = np.zeros((h, w), dtype=bool)
    if asc is None:
        asc = np.zeros((h, w), dtype=bool)
    if sparse is None:
        sparse = np.zeros((h, w), dtype=bool)

    tp, fp, fn, tn = confusion(doc, gt)
    gt_area = int(gt.sum())
    pred_area = int(doc.sum())
    fn_mask = np.logical_and(gt, np.logical_not(doc))
    fp_mask = np.logical_and(doc, np.logical_not(gt))
    others = np.logical_or(asc, sparse)

    doc_f1 = f1_from_counts(tp, fp, fn)
    doc_iou = iou_from_counts(tp, fp, fn)
    doc_recall = safe_div(tp, gt_area)
    doc_precision = safe_div(tp, pred_area)
    fn_area = int(fn_mask.sum())
    fp_area = int(fp_mask.sum())
    fn_ratio = safe_div(fn_area, gt_area)
    fp_ratio = safe_div(fp_area, max(gt_area, 1))

    asc_tp = int(np.logical_and(asc, gt).sum())
    sparse_tp = int(np.logical_and(sparse, gt).sum())
    asc_recall = safe_div(asc_tp, gt_area)
    sparse_recall = safe_div(sparse_tp, gt_area)

    others_recover_doc_fn = safe_div(int(np.logical_and(fn_mask, others).sum()), fn_area)
    all_models_miss = safe_div(int(np.logical_and(fn_mask, np.logical_not(others)).sum()), gt_area)
    doc_unique_fp = safe_div(int(np.logical_and(fp_mask, np.logical_not(others)).sum()), fp_area)
    model_disagreement = 1.0 - binary_iou(doc, others)

    comp_count, missed_count, missed_ratio, large_missed = component_stats(gt, doc, asc, sparse, args)
    difficulty_score = (
        0.35 * fn_ratio
        + 0.20 * missed_ratio
        + 0.15 * float(large_missed)
        + 0.15 * min(fp_ratio, 1.0)
        + 0.15 * model_disagreement
    )

    hard_fn_by_others = np.logical_and(fn_mask, others)
    hard_fn_all_fail = np.logical_and(fn_mask, np.logical_not(others))
    hard_fp_unique = np.logical_and(fp_mask, np.logical_not(others))

    record = {
        "image": rel_path,
        "stem": stem,
        "gt_path": str(gt_path) if gt_path else "",
        "doc_pred_path": str(doc_path) if doc_path else "",
        "asc_pred_path": str(asc_path) if asc_path else "",
        "sparse_pred_path": str(sparse_path) if sparse_path else "",
        "width": int(w),
        "height": int(h),
        "gt_area": gt_area,
        "doc_pred_area": pred_area,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "doc_iou": doc_iou,
        "doc_f1": doc_f1,
        "doc_recall": doc_recall,
        "doc_precision": doc_precision,
        "fn_ratio": fn_ratio,
        "fp_ratio": fp_ratio,
        "asc_recall_on_gt": asc_recall,
        "sparse_recall_on_gt": sparse_recall,
        "others_recover_doc_fn": others_recover_doc_fn,
        "all_models_miss": all_models_miss,
        "doc_unique_fp": doc_unique_fp,
        "model_disagreement": model_disagreement,
        "component_count": comp_count,
        "missed_component_count": missed_count,
        "missed_component_ratio": missed_ratio,
        "large_missed_component": bool(large_missed),
        "difficulty_score": difficulty_score,
        "missing_predictors": ",".join(missing),
    }
    record["difficulty"] = classify(record, args)

    regions = {
        "image": rel_path,
        "stem": stem,
        "difficulty": record["difficulty"],
        "doc_f1": doc_f1,
        "fn_ratio": fn_ratio,
        "fp_ratio": fp_ratio,
        "others_recover_doc_fn": others_recover_doc_fn,
        "hard_fn_boxes": region_boxes(fn_mask, w, h, args),
        "hard_fn_by_others_boxes": region_boxes(hard_fn_by_others, w, h, args),
        "hard_fn_all_fail_boxes": region_boxes(hard_fn_all_fail, w, h, args),
        "hard_fp_boxes": region_boxes(fp_mask, w, h, args),
        "hard_fp_unique_boxes": region_boxes(hard_fp_unique, w, h, args),
    }

    if args.save_error_masks:
        err_dir = Path(args.output_dir) / "error_masks"
        for name, mask in (
            ("fn", fn_mask),
            ("fn_by_others", hard_fn_by_others),
            ("fp_unique", hard_fp_unique),
        ):
            out_dir = err_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{stem}.png"), mask.astype(np.uint8) * 255)

    return record, regions


def write_lines(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(str(line) + "\n")


def summarize(records, missing, duplicates):
    levels = {"easy": 0, "medium": 0, "hard": 0}
    for rec in records:
        levels[rec["difficulty"]] += 1

    def avg(key):
        return float(np.mean([rec[key] for rec in records])) if records else 0.0

    return {
        "num_records": len(records),
        "difficulty_counts": levels,
        "avg_doc_f1": avg("doc_f1"),
        "avg_doc_iou": avg("doc_iou"),
        "avg_fn_ratio": avg("fn_ratio"),
        "avg_fp_ratio": avg("fp_ratio"),
        "avg_others_recover_doc_fn": avg("others_recover_doc_fn"),
        "avg_all_models_miss": avg("all_models_miss"),
        "missing_count": len(missing),
        "duplicates": {k: len(v) for k, v in duplicates.items()},
    }


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_map, gt_dups = build_stem_map(args.gt_dir, MASK_EXTS)
    doc_map, doc_dups = build_stem_map(args.doc_pred_dir, MASK_EXTS)
    asc_map, asc_dups = build_stem_map(args.asc_pred_dir, MASK_EXTS)
    sparse_map, sparse_dups = build_stem_map(args.sparse_pred_dir, MASK_EXTS)
    duplicates = {"gt": gt_dups, "doc": doc_dups, "asc": asc_dups, "sparse": sparse_dups}

    image_paths = collect_files(image_dir, IMAGE_EXTS)
    exclude_items = load_exclude_items(args.exclude_list_path)
    if exclude_items:
        image_paths = [path for path in image_paths if not is_excluded(path, image_dir, exclude_items)]
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    maps = {"gt": gt_map, "doc": doc_map, "asc": asc_map, "sparse": sparse_map}
    args_dict = vars(args)
    tasks = [(str(path), str(path.relative_to(image_dir)), maps, args_dict) for path in image_paths]

    records = []
    regions = []
    missing = []
    with Pool(processes=args.workers) as pool:
        for record, region in tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks), desc="Mining difficulty"):
            if record is None:
                missing.append(region)
                continue
            records.append(record)
            regions.append(region)

    records.sort(key=lambda item: item["image"])
    regions.sort(key=lambda item: item["image"])

    fieldnames = list(records[0].keys()) if records else []
    with open(output_dir / "difficulty_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with open(output_dir / "difficulty_manifest.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(output_dir / "hard_regions.json", "w") as f:
        json.dump(regions, f, ensure_ascii=False, indent=2)

    by_level = {"easy": [], "medium": [], "hard": []}
    by_level_stem = {"easy": [], "medium": [], "hard": []}
    for rec in records:
        by_level[rec["difficulty"]].append(rec["image"])
        by_level_stem[rec["difficulty"]].append(rec["stem"])

    for level in ("easy", "medium", "hard"):
        write_lines(output_dir / f"{level}.txt", by_level[level])
        write_lines(output_dir / f"{level}_stems.txt", by_level_stem[level])

    with open(output_dir / "missing.json", "w") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)

    summary = summarize(records, missing, duplicates)
    summary["thresholds"] = {
        "doc": {"mode": args.doc_threshold_mode, "threshold": args.doc_threshold},
        "asc": {"mode": args.asc_threshold_mode, "threshold": args.asc_threshold},
        "sparse": {"mode": args.sparse_threshold_mode, "threshold": args.sparse_threshold},
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved difficulty mining outputs to: {output_dir}")


if __name__ == "__main__":
    main()
