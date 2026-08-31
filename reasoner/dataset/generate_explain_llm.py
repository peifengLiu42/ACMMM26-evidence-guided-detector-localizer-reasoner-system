import os
import re
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

OUTPUT_FILE = "data/realtext_explain_train_sft.filtered.json"
MIN_BBOX_AREA = 0
BLACK_THRESHOLD = 20
MAX_WORKERS = min(os.cpu_count() or 1, 128)

SYSTEM_PROMPT = (
    "You are an expert AI image forensics and document analysis assistant. "
    "Your task is to rigorously evaluate image authenticity and generate a structured forensic report. "
    "Strictly follow this output format:\n\n"
    "I. Overall Assessment\n"
    "[Conclusion]: Clearly declare the status as FORGED or AUTHENTIC.\n"
    "[RISK_SCORE]: A numerical confidence score (0–100) representing the likelihood of manipulation.\n\n"
    "II. Detailed Anomaly Analysis\n"
    "For each detected anomaly, create a separate entry (e.g., ### ANOMALY_001).\n"
    "[GROUNDING]: Normalized bounding box coordinates in the format [xmin, ymin, xmax, ymax](0-999).\n"
    "[REASON]: A natural language explanation detailing visual artifacts (e.g., clumsy patching, noise inconsistency) "
    "and semantic contradictions (e.g., logical errors, identity fraud).\n\n"
    "III. Summary\n"
    "A final synthesis of the findings, summarizing how the identified anomalies collectively support the overall assessment."
)

GROUNDING_PATTERN = re.compile(r'\[GROUNDING\]\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GT-mask bbox prompt SFT data for the Qwen3-VL reasoner.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--exclude_list", default="")
    parser.add_argument("--min_bbox_area", type=int, default=MIN_BBOX_AREA)
    parser.add_argument("--black_threshold", type=int, default=BLACK_THRESHOLD)
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    return parser.parse_args()


def load_exclude_items(path):
    if not path or not os.path.exists(path):
        return set()
    items = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value:
                continue
            value = value.replace("\\", "/")
            items.add(value)
            items.add(str(Path(value).with_suffix("")))
            items.add(Path(value).stem)
    return items


def is_excluded(img_path, img_dir, exclude_items):
    if not exclude_items:
        return False
    rel = str(Path(img_path).relative_to(img_dir)).replace("\\", "/")
    rel_no_ext = str(Path(rel).with_suffix(""))
    stem = Path(img_path).stem
    return rel in exclude_items or rel_no_ext in exclude_items or stem in exclude_items


def normalize_to_999(val, max_dim):
    """坐标归一化至 0-999，防重复缩放截断"""
    if max_dim == 0:
        return 0
    if val <= 999 and max_dim <= 1000:
        return val
    mapped = int(round(val / max_dim * 999))
    return max(0, min(999, mapped))


def extract_mask_bboxes(mask_path, img_w, img_h, min_area, black_thresh):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return True, []
    is_black = np.all(mask < black_thresh)
    if is_black:
        return True, []
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            bboxes.append([
                normalize_to_999(x, img_w), normalize_to_999(y, img_h),
                normalize_to_999(x + w - 1, img_w), normalize_to_999(y + h - 1, img_h)
            ])
    bboxes.sort(key=lambda b: (b[1], b[0]))
    return False, bboxes


def extract_report_groundings(report_path, img_w, img_h):
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    norm_bboxes = []
    def replace_and_norm(match):
        x1, y1, x2, y2 = map(int, match.groups())
        nx1 = normalize_to_999(x1, img_w)
        ny1 = normalize_to_999(y1, img_h)
        nx2 = normalize_to_999(x2, img_w)
        ny2 = normalize_to_999(y2, img_h)
        norm_bboxes.append([nx1, ny1, nx2, ny2])
        return f"[GROUNDING]: [{nx1}, {ny1}, {nx2}, {ny2}]"
    modified_content = GROUNDING_PATTERN.sub(replace_and_norm, content)
    return norm_bboxes, modified_content.strip()


def worker_process(img_path, img_dir, mask_dir, report_dir, min_area, black_thresh):
    try:
        rel_path = Path(img_path).relative_to(img_dir)
        mask_path = mask_dir / rel_path.with_name(f"{Path(img_path).stem}_mask.png")
        report_path = report_dir / rel_path.with_name(f"{Path(img_path).stem}_report.md")

        if not mask_path.exists() or not report_path.exists():
            return None, "missing"

        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            return None, "read_error"

        h, w = img_cv.shape[:2]
        is_black, mask_bboxes = extract_mask_bboxes(mask_path, w, h, min_area, black_thresh)
        report_bboxes, report_content = extract_report_groundings(report_path, w, h)

        # Keep one target report entry per GT mask component.
        if len(mask_bboxes) != len(report_bboxes):
            return None, "mismatch"

        num_bboxes = len(mask_bboxes)
        if is_black:
            user_prompt = (
                f"<image>Expert forgery detector has analyzed this image and detected {num_bboxes} bbox(s), "
                f"indicating it is an authentic image. Please verify this assessment and provide a "
                f"detailed analysis report strictly following the required forensic format."
            )
        else:
            bbox_str = ", ".join([f"[{b[0]},{b[1]},{b[2]},{b[3]}]" for b in mask_bboxes])
            user_prompt = (
                f"<image>Expert forgery detector has identified {num_bboxes} potential tampered region(s) at: {bbox_str}. "
                f"Please analyze these specific areas in detail, explain the visual artifacts and logical contradictions, "
                f"and provide a comprehensive forgery analysis report strictly following the required forensic format."
            )

        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": report_content}
            ],
            "images": str(img_path)
        }
        return sample, "valid"
    except Exception:
        return None, "error"


def main():
    args = parse_args()
    img_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    report_dir = Path(args.report_dir)
    valid_exts = {'.jpg', '.jpeg', '.png'}

    image_files = [f for f in img_dir.rglob('*') if f.suffix.lower() in valid_exts and f.is_file()]
    exclude_items = load_exclude_items(args.exclude_list)
    if exclude_items:
        image_files = [f for f in image_files if not is_excluded(f, img_dir, exclude_items)]
    total = len(image_files)

    dataset = []
    stats = {"valid": 0, "missing": 0, "mismatch": 0, "read_error": 0, "error": 0}

    print(f"Start processing with workers={args.max_workers}; images={total}")
    if args.exclude_list:
        print(f"Exclude list: {args.exclude_list}; rules={len(exclude_items)}")
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(worker_process, str(img), str(img_dir), str(mask_dir), str(report_dir), args.min_bbox_area, args.black_threshold): img
            for img in image_files
        }

        for future in tqdm(as_completed(futures), total=total, desc="Processing"):
            result, status = future.result()
            stats[status] += 1
            if result is not None:
                dataset.append(result)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Stats: total={total}, valid={stats['valid']}")
    print(
        "Skipped: "
        f"missing={stats['missing']}, mismatch={stats['mismatch']}, "
        f"read_error={stats['read_error']}, error={stats['error']}"
    )
    print(f"Output: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    main()
