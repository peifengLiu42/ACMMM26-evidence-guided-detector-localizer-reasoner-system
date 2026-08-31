import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


NUMERIC_FIELDS = {
    "width",
    "height",
    "gt_area",
    "doc_pred_area",
    "tp",
    "fp",
    "fn",
    "tn",
    "doc_iou",
    "doc_f1",
    "doc_recall",
    "doc_precision",
    "fn_ratio",
    "fp_ratio",
    "asc_recall_on_gt",
    "sparse_recall_on_gt",
    "others_recover_doc_fn",
    "all_models_miss",
    "doc_unique_fp",
    "model_disagreement",
    "component_count",
    "missed_component_count",
    "missed_component_ratio",
    "difficulty_score",
}


def parse_args():
    parser = argparse.ArgumentParser("Postprocess RealTextV2 difficulty manifest with empty-GT and error-type fixes")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hard_regions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--empty_fp_easy_area", type=int, default=32)
    parser.add_argument("--empty_fp_hard_area", type=int, default=256)
    parser.add_argument("--fn_hard_ratio", type=float, default=0.35)
    parser.add_argument("--fn_medium_ratio", type=float, default=0.15)
    parser.add_argument("--fp_hard_precision", type=float, default=0.60)
    parser.add_argument("--fp_medium_precision", type=float, default=0.80)
    parser.add_argument("--min_error_pixels", type=int, default=64)
    return parser.parse_args()


def to_float(row, key):
    try:
        return float(row.get(key, 0) or 0)
    except ValueError:
        return 0.0


def read_manifest(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in NUMERIC_FIELDS:
                if key in row:
                    row[key] = to_float(row, key)
            rows.append(row)
    return rows


def region_counts(region):
    out = {}
    for key in (
        "hard_fn_boxes",
        "hard_fn_by_others_boxes",
        "hard_fn_all_fail_boxes",
        "hard_fp_boxes",
        "hard_fp_unique_boxes",
    ):
        out[key + "_count"] = len(region.get(key, []))
    return out


def classify_positive(row, args):
    fn = to_float(row, "fn")
    fp = to_float(row, "fp")
    fn_ratio = to_float(row, "fn_ratio")
    precision = to_float(row, "doc_precision")
    recall = to_float(row, "doc_recall")
    f1 = to_float(row, "doc_f1")
    missed_ratio = to_float(row, "missed_component_ratio")
    large_missed = str(row.get("large_missed_component", "")).lower() == "true"
    others_recover = to_float(row, "others_recover_doc_fn")
    all_miss = to_float(row, "all_models_miss")
    doc_unique_fp = to_float(row, "doc_unique_fp")

    fn_heavy = fn >= max(args.min_error_pixels, 1.5 * fp) and fn_ratio >= args.fn_medium_ratio
    fp_heavy = fp >= max(args.min_error_pixels, 1.5 * fn) and precision < args.fp_medium_precision
    severe_fn = fn_ratio >= args.fn_hard_ratio or missed_ratio > 0.40 or large_missed
    severe_fp = fp_heavy and precision < args.fp_hard_precision

    if severe_fn and others_recover >= 0.20:
        return "hard", "fn_recoverable_by_others"
    if severe_fn and all_miss >= 0.20:
        return "hard", "fn_all_models_miss"
    if severe_fn or fn_heavy:
        level = "hard" if severe_fn else "medium"
        return level, "fn_dominant"
    if severe_fp and doc_unique_fp >= 0.30:
        return "hard", "fp_unique"
    if severe_fp or fp_heavy:
        level = "hard" if severe_fp else "medium"
        return level, "fp_dominant"
    if f1 >= 0.75 and recall >= 0.80 and precision >= 0.75 and fn_ratio <= 0.15:
        return "easy", "positive_good"
    if f1 >= 0.55:
        return "medium", "positive_minor_error"
    return "hard", "mixed_or_low_f1"


def classify(row, args):
    gt_area = to_float(row, "gt_area")
    pred_area = to_float(row, "doc_pred_area")
    fp = to_float(row, "fp")

    if gt_area <= 0:
        if pred_area <= args.empty_fp_easy_area:
            return "easy", "authentic_clean"
        if fp >= args.empty_fp_hard_area:
            return "hard", "authentic_fp_hard"
        return "medium", "authentic_fp_light"

    return classify_positive(row, args)


def write_lines(path, values):
    with open(path, "w") as f:
        for value in values:
            f.write(str(value) + "\n")


def summarize(rows):
    by_level = Counter(row["difficulty"] for row in rows)
    by_error = Counter(row["error_type"] for row in rows)
    cross = defaultdict(Counter)
    for row in rows:
        cross[row["difficulty"]][row["error_type"]] += 1

    def avg(key, subset):
        vals = [to_float(row, key) for row in subset]
        return float(np.mean(vals)) if vals else 0.0

    positives = [row for row in rows if to_float(row, "gt_area") > 0]
    authentic = [row for row in rows if to_float(row, "gt_area") <= 0]
    return {
        "num_records": len(rows),
        "difficulty_counts": dict(by_level),
        "error_type_counts": dict(by_error),
        "difficulty_error_type_counts": {k: dict(v) for k, v in cross.items()},
        "positive_count": len(positives),
        "authentic_or_empty_gt_count": len(authentic),
        "avg_doc_f1_all": avg("doc_f1", rows),
        "avg_doc_f1_positive": avg("doc_f1", positives),
        "avg_fn_ratio_positive": avg("fn_ratio", positives),
        "avg_fp_pixels_authentic": avg("fp", authentic),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    with open(args.hard_regions) as f:
        regions = json.load(f)
    region_map = {item["image"]: item for item in regions}

    for row in rows:
        row["difficulty_raw"] = row.get("difficulty", "")
        level, error_type = classify(row, args)
        row["difficulty"] = level
        row["error_type"] = error_type
        row.update(region_counts(region_map.get(row["image"], {})))

    rows.sort(key=lambda item: item["image"])
    fieldnames = list(rows[0].keys()) if rows else []

    with open(output_dir / "difficulty_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "difficulty_manifest.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_level = defaultdict(list)
    by_error = defaultdict(list)
    by_level_stem = defaultdict(list)
    by_error_stem = defaultdict(list)
    for row in rows:
        by_level[row["difficulty"]].append(row["image"])
        by_error[row["error_type"]].append(row["image"])
        by_level_stem[row["difficulty"]].append(row["stem"])
        by_error_stem[row["error_type"]].append(row["stem"])

    for level in ("easy", "medium", "hard"):
        write_lines(output_dir / f"{level}.txt", by_level[level])
        write_lines(output_dir / f"{level}_stems.txt", by_level_stem[level])

    error_dir = output_dir / "error_type_lists"
    error_dir.mkdir(exist_ok=True)
    for error_type, values in sorted(by_error.items()):
        write_lines(error_dir / f"{error_type}.txt", values)
        write_lines(error_dir / f"{error_type}_stems.txt", by_error_stem[error_type])

    summary = summarize(rows)
    summary["source_manifest"] = args.manifest
    summary["source_hard_regions"] = args.hard_regions
    summary["rules"] = {
        "empty_fp_easy_area": args.empty_fp_easy_area,
        "empty_fp_hard_area": args.empty_fp_hard_area,
        "fn_hard_ratio": args.fn_hard_ratio,
        "fn_medium_ratio": args.fn_medium_ratio,
        "fp_hard_precision": args.fp_hard_precision,
        "fp_medium_precision": args.fp_medium_precision,
        "min_error_pixels": args.min_error_pixels,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved v3 difficulty outputs to: {output_dir}")


if __name__ == "__main__":
    main()
