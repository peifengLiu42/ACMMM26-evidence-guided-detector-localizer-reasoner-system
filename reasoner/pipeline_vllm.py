#!/usr/bin/env python3
"""Detector-gated localizer-to-reasoner inference with vLLM.

This is the final reasoner-side inference path for the
detector-localizer-reasoner system:

1. read image-level detector predictions;
2. route detector-positive images to localizer heatmaps;
3. convert localizer responses to normalized grounding boxes;
4. feed image + detector/localizer evidence to the Qwen3-VL reasoner with vLLM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FORGED_CONCLUSIONS = {"FORGED", "TAMPERED", "FRAUDULENT", "FORGERY"}
REPORT_HEADER_RE = re.compile(
    r"(#\s*FORGERY\s+ANALYSIS\s+REPORT\s*\n+).*?(\*\*Overall Assessment:\*\*)",
    re.IGNORECASE | re.DOTALL,
)
THINK_RE = re.compile(r"^\s*<think>\s*</think>\s*", re.IGNORECASE | re.DOTALL)
GROUNDING_RE = re.compile(
    r"((?:\*\*\s*)?\[\s*GROUNDING\s*\](?:\s*\*\*)?\s*:\s*(?:\*\*\s*)?)"
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    r"(\s*\*\*)?",
    re.IGNORECASE,
)
CONCLUSION_RE = re.compile(
    r"(?:\*\*\s*)?\[?\s*Conclusion\s*\]?(?:\s*\*\*)?\s*:\s*(?:\*\*\s*)?"
    r"(FORGED|AUTHENTIC|TAMPERED|FRAUDULENT|FORGERY)",
    re.IGNORECASE,
)
RISK_SCORE_RE = re.compile(r"\[?\s*RISK_SCORE\s*\]?\s*:\s*(\d{1,3})", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are an expert AI image forensics and document analysis assistant. "
    "Your task is to rigorously evaluate image authenticity and generate a structured forensic report. "
    "Strictly follow this output format:\n\n"
    "I. Overall Assessment\n"
    "[Conclusion]: Clearly declare the status as FORGED or AUTHENTIC.\n"
    "[RISK_SCORE]: A numerical confidence score (0-100) representing the likelihood of manipulation.\n\n"
    "II. Detailed Anomaly Analysis\n"
    "For each detected anomaly, create a separate entry (e.g., ### ANOMALY_001).\n"
    "[GROUNDING]: Normalized bounding box coordinates in the format [xmin, ymin, xmax, ymax](0-999).\n"
    "[REASON]: A natural language explanation detailing visual artifacts (e.g., clumsy patching, noise inconsistency) "
    "and semantic contradictions (e.g., logical errors, identity fraud).\n\n"
    "III. Summary\n"
    "A final synthesis of the findings, summarizing how the identified anomalies collectively support the overall assessment."
)


def parse_resize_arg(value: str | None):
    if value is None or value.lower() in {"none", "null", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    for sep in ("x", "X", ",", " "):
        if sep in value:
            parts = value.split(sep)
            if len(parts) == 2:
                return int(parts[0].strip()), int(parts[1].strip())
    raise ValueError(f"Invalid resize value: {value}")


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def collect_images_from_file_list(image_dir: Path, file_list: Path) -> list[Path]:
    all_images = collect_images(image_dir)
    by_stem: dict[str, Path] = {}
    by_rel: dict[str, Path] = {}
    by_rel_no_suffix: dict[str, Path] = {}
    for path in all_images:
        rel = path.relative_to(image_dir).as_posix()
        by_stem[path.stem] = path
        by_rel[rel] = path
        by_rel_no_suffix[Path(rel).with_suffix("").as_posix()] = path

    image_paths: list[Path] = []
    missing: list[str] = []
    seen: set[Path] = set()
    with file_list.open("r", encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]

    for entry in entries:
        normalized = entry.replace("\\", "/")
        entry_path = Path(normalized)
        found = None
        if entry_path.is_absolute() and entry_path.exists():
            found = entry_path
        elif (image_dir / normalized).exists():
            found = image_dir / normalized
        elif normalized in by_rel:
            found = by_rel[normalized]
        elif normalized in by_rel_no_suffix:
            found = by_rel_no_suffix[normalized]
        elif entry_path.stem in by_stem:
            found = by_stem[entry_path.stem]

        if found is None:
            missing.append(entry)
            continue
        if found not in seen:
            image_paths.append(found)
            seen.add(found)

    if missing:
        print(f"Warning: missing {len(missing)} images from file_list; first 10: {missing[:10]}", file=sys.stderr)
    print(f"Loaded {len(image_paths)}/{len(entries)} images from file_list: {file_list}")
    return image_paths


def load_prediction_items(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"Expected object at {path}:{line_no}")
                items.append(item)
        return items

    data = json.load(path.open("r", encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        predictions = data.get("predictions")
        if isinstance(predictions, list):
            return [item for item in predictions if isinstance(item, dict)]
        for value in data.values():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
    raise ValueError(f"Unsupported detector JSON format: {path}")


def load_detector_predictions(detector_json: Path) -> dict[str, dict[str, Any]]:
    pred_map: dict[str, dict[str, Any]] = {}
    for item in load_prediction_items(detector_json):
        candidates: list[str] = []
        for key in ("image", "image_path", "relative_path", "file_name", "filename", "image_name"):
            value = item.get(key)
            if value:
                candidates.append(Path(str(value)).stem)
        for key in candidates:
            pred_map[key] = item
    return pred_map


def load_done_names(output_jsonl: Path) -> set[str]:
    if not output_jsonl.is_file():
        return set()
    done: set[str] = set()
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            report = item.get("report")
            image_name = item.get("image_name")
            if image_name and isinstance(report, str) and report.strip():
                done.add(str(image_name))
    return done


def detector_probability(item: dict[str, Any]) -> float | None:
    for key in ("prob_forged", "score", "forged_prob", "fake_prob", "probability"):
        if key in item:
            return float(item[key])
    probs = item.get("probs") or item.get("prob")
    if isinstance(probs, (list, tuple)) and len(probs) >= 2:
        return float(probs[1])
    return None


def detector_is_forged(item: dict[str, Any], threshold: float) -> bool:
    if "pred_label_id" in item:
        return int(item["pred_label_id"]) == 1
    if "pred_label" in item:
        return str(item["pred_label"]).lower() in {"forged", "fake", "tampered", "1"}
    probability = detector_probability(item)
    if probability is not None:
        return probability >= threshold

    report = item.get("report") or item.get("response")
    if isinstance(report, str):
        cleaned = report.replace("**", "")
        match = CONCLUSION_RE.search(cleaned)
        if match:
            return match.group(1).upper() in FORGED_CONCLUSIONS
        risk_match = RISK_SCORE_RE.search(cleaned)
        if risk_match:
            return int(risk_match.group(1)) > 0
        if "no anomalies detected" in cleaned.lower():
            return False
    raise ValueError(f"Cannot determine detector label from item: {item}")


def normalize_to_999(value: int, max_dim: int) -> int:
    if max_dim <= 0:
        return 0
    return int(max(0, min(999, round(value / max_dim * 999))))


def resolve_heatmap_path(heatmap_dir: Path, key: str) -> Path | None:
    direct = heatmap_dir / f"{key}.png"
    if direct.exists():
        return direct
    matches = sorted(heatmap_dir.rglob(f"{key}.png"))
    return matches[0] if matches else None


def heatmap_to_bboxes(
    heatmap_path: Path | None,
    start_threshold: float,
    threshold_step: float,
    min_area: int,
    min_component_area: int,
    min_threshold: float,
) -> tuple[list[list[int]], float]:
    if heatmap_path is None or not heatmap_path.exists():
        return [], -1.0
    heat = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if heat is None:
        return [], -1.0

    heat_h, heat_w = heat.shape[:2]
    threshold = start_threshold
    while threshold >= min_threshold:
        bin_mask = (heat > int(round(threshold * 255))).astype(np.uint8)
        if np.any(bin_mask):
            bboxes: list[list[int]] = []
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
            for label_id in range(1, num_labels):
                x = int(stats[label_id, cv2.CC_STAT_LEFT])
                y = int(stats[label_id, cv2.CC_STAT_TOP])
                w = int(stats[label_id, cv2.CC_STAT_WIDTH])
                h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
                component_area = int(stats[label_id, cv2.CC_STAT_AREA])
                if component_area < min_component_area or w * h < min_area:
                    continue
                bboxes.append(
                    [
                        normalize_to_999(x, heat_w),
                        normalize_to_999(y, heat_h),
                        normalize_to_999(x + w - 1, heat_w),
                        normalize_to_999(y + h - 1, heat_h),
                    ]
                )
            if bboxes:
                bboxes.sort(key=lambda box: (box[1], box[0]))
                return bboxes, threshold
        threshold = round(threshold - threshold_step, 10)
    return [], -1.0


def build_prompt(final_forged: bool, bboxes: list[list[int]]) -> str:
    num_bboxes = len(bboxes)
    if not final_forged:
        return (
            f"<image>Expert forgery detector has analyzed this image and detected {num_bboxes} bbox(s), "
            "indicating it is an authentic image. Please verify this assessment and provide a "
            "detailed analysis report strictly following the required forensic format."
        )

    bbox_str = ", ".join(f"[{x1},{y1},{x2},{y2}]" for x1, y1, x2, y2 in bboxes)
    return (
        f"<image>Expert forgery detector has identified {num_bboxes} potential tampered region(s) at: {bbox_str}. "
        "Please analyze these specific areas in detail, explain the visual artifacts and logical contradictions, "
        "and provide a comprehensive forgery analysis report strictly following the required forensic format."
    )


def clean_report(report: str) -> str:
    report = THINK_RE.sub("", report or "")
    report = report.replace("<think>\n\n</think>\n\n", "")
    report = report.replace("<think>\n</think>\n", "")
    return REPORT_HEADER_RE.sub(r"\1\2", report, count=1).lstrip()


def denormalize_grounding(report: str, image_path: Path, norm_base: int) -> str:
    with Image.open(image_path) as img:
        width, height = img.size

    def replace(match: re.Match) -> str:
        prefix = match.group(1)
        suffix = match.group(6) or ""
        nx1, ny1, nx2, ny2 = (float(match.group(i)) for i in range(2, 6))
        max_value = max(nx1, ny1, nx2, ny2)
        if max_value > norm_base:
            return match.group(0)
        scale = 1.0 if max_value <= 1.0 else float(norm_base)
        x1 = min(max(round(nx1 / scale * width), 0), width)
        y1 = min(max(round(ny1 / scale * height), 0), height)
        x2 = min(max(round(nx2 / scale * width), 0), width)
        y2 = min(max(round(ny2 / scale * height), 0), height)
        return f"{prefix}[{x1}, {y1}, {x2}, {y2}]{suffix}"

    return GROUNDING_RE.sub(replace, report)


def resize_image(image: Image.Image, resize, resize_mode: str) -> Image.Image:
    if resize is None:
        return image
    width, height = image.size
    if isinstance(resize, int):
        if resize_mode == "shrink_only" and max(width, height) <= resize:
            return image
        scale = resize / max(width, height)
        return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)

    target_w, target_h = resize
    if resize_mode == "keep_ratio":
        scale = min(target_w / width, target_h / height)
        new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        padded.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return padded
    return image.resize((target_w, target_h), Image.Resampling.LANCZOS)


def load_image_for_vllm(image_path: Path, resize, resize_mode: str, safety_max_pixels: int, large_image_resize: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    effective_resize = resize
    if resize is None and safety_max_pixels > 0 and width * height > safety_max_pixels:
        effective_resize = large_image_resize
        print(
            f"Large image fallback resize: {image_path} ({width}x{height}) -> long edge {large_image_resize}",
            file=sys.stderr,
        )
    return resize_image(image, effective_resize, resize_mode)


def build_messages(prompt: str, image: Image.Image, use_system_prompt: bool) -> list[dict[str, Any]]:
    user_text = prompt
    if user_text.startswith("<image>"):
        user_text = user_text[len("<image>") :].lstrip()
    messages: list[dict[str, Any]] = []
    if use_system_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": user_text},
            ],
        }
    )
    return messages


def prepare_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    detector_map = load_detector_predictions(args.detector_json)
    if args.file_list is not None:
        image_paths = collect_images_from_file_list(args.image_dir, args.file_list)
    else:
        image_paths = collect_images(args.image_dir)
    if args.test_num > 0:
        image_paths = image_paths[: args.test_num]
    if args.resume:
        done = load_done_names(args.output_jsonl)
        image_paths = [path for path in image_paths if path.name not in done]
        if done:
            print(f"Resume: skip {len(done)} existing report(s)")

    records: list[dict[str, Any]] = []
    for image_path in image_paths:
        key = image_path.stem
        detector_item = detector_map.get(key)
        if detector_item is None:
            if not args.missing_detector_as_authentic:
                print(f"Missing detector prediction for {image_path}", file=sys.stderr)
                continue
            detector_item = {}
            detector_forged = False
        else:
            detector_forged = detector_is_forged(detector_item, args.detector_threshold)

        boxes: list[list[int]] = []
        used_threshold = -1.0
        heatmap_path = None
        final_forged = detector_forged
        if detector_forged:
            heatmap_path = resolve_heatmap_path(args.heatmap_dir, key)
            boxes, used_threshold = heatmap_to_bboxes(
                heatmap_path,
                start_threshold=args.loc_threshold,
                threshold_step=args.threshold_step,
                min_area=args.min_area,
                min_component_area=args.min_component_area,
                min_threshold=args.loc_min_threshold,
            )
            if not boxes and args.fallback_heatmap_dir is not None:
                fallback_path = resolve_heatmap_path(args.fallback_heatmap_dir, key)
                boxes, used_threshold = heatmap_to_bboxes(
                    fallback_path,
                    start_threshold=args.fallback_loc_threshold,
                    threshold_step=args.threshold_step,
                    min_area=args.min_area,
                    min_component_area=args.min_component_area,
                    min_threshold=args.fallback_loc_min_threshold,
                )
                if fallback_path is not None:
                    heatmap_path = fallback_path
            if not boxes:
                final_forged = False

        prompt = build_prompt(final_forged=final_forged, bboxes=boxes)
        records.append(
            {
                "image_path": image_path,
                "image_name": image_path.name,
                "prompt": prompt,
                "detector_predicted_forged": detector_forged,
                "detector_prob_forged": detector_probability(detector_item),
                "localizer_boxes": boxes,
                "localizer_box_count": len(boxes),
                "localizer_heatmap_path": str(heatmap_path) if heatmap_path is not None else None,
                "localizer_used_threshold": used_threshold,
                "final_prompt_forged": final_forged,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--file_list", type=Path, default=None)
    parser.add_argument("--detector_json", type=Path, required=True)
    parser.add_argument("--heatmap_dir", type=Path, required=True)
    parser.add_argument("--fallback_heatmap_dir", type=Path, default=None)
    parser.add_argument("--output_jsonl", type=Path, required=True)

    parser.add_argument("--model_name_or_path", type=Path, required=True)
    parser.add_argument("--adapter_checkpoint", type=Path, default=None)
    parser.add_argument("--merged_model", action="store_true")
    parser.add_argument("--lora_name", default="realtext_reasoner")
    parser.add_argument("--max_lora_rank", type=int, default=32)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--resize", default="1280")
    parser.add_argument("--resize_mode", default="keep_ratio", choices=["keep_ratio", "force", "shrink_only"])
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce_eager", action="store_true")

    parser.add_argument("--test_num", type=int, default=0)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--overwrite", dest="resume", action="store_false")
    parser.add_argument("--detector_threshold", type=float, default=0.5)
    parser.add_argument("--loc_threshold", type=float, default=0.5)
    parser.add_argument("--loc_min_threshold", type=float, default=0.0)
    parser.add_argument("--fallback_loc_threshold", type=float, default=0.5)
    parser.add_argument("--fallback_loc_min_threshold", type=float, default=0.05)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--min_area", type=int, default=10)
    parser.add_argument("--min_component_area", type=int, default=16)
    parser.add_argument("--safety_max_pixels", type=int, default=80_000_000)
    parser.add_argument("--large_image_resize", type=int, default=1280)
    parser.add_argument("--grounding_norm_base", type=int, default=999)
    parser.add_argument("--no_denormalize_grounding", dest="denormalize_grounding", action="store_false")
    parser.add_argument("--no_system_prompt", dest="use_system_prompt", action="store_false")
    parser.add_argument("--missing_detector_as_authentic", action="store_true")
    parser.set_defaults(denormalize_grounding=True, use_system_prompt=True)
    args = parser.parse_args()
    args.resize_config = parse_resize_arg(args.resize)
    return args


def validate_args(args: argparse.Namespace) -> None:
    for path, label in (
        (args.image_dir, "image dir"),
        (args.detector_json, "detector JSON"),
        (args.heatmap_dir, "heatmap dir"),
        (args.model_name_or_path, "model"),
    ):
        if label.endswith("dir"):
            if not path.is_dir():
                raise FileNotFoundError(f"Missing {label}: {path}")
        elif not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.file_list is not None and not args.file_list.is_file():
        raise FileNotFoundError(f"Missing file list: {args.file_list}")
    if args.fallback_heatmap_dir is not None and not args.fallback_heatmap_dir.is_dir():
        raise FileNotFoundError(f"Missing fallback heatmap dir: {args.fallback_heatmap_dir}")
    if not args.merged_model and args.adapter_checkpoint is not None and not args.adapter_checkpoint.exists():
        raise FileNotFoundError(f"Missing adapter checkpoint: {args.adapter_checkpoint}")
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.max_model_len < 1:
        raise ValueError("Batch/length arguments must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu_memory_utilization must be in (0, 1)")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not args.resume:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text("", encoding="utf-8")

    records = prepare_records(args)
    if not records:
        print("No images to process.")
        return

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm_kwargs = dict(
        model=str(args.model_name_or_path),
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": args.max_pixels},
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    if args.adapter_checkpoint is not None and not args.merged_model:
        llm_kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank, max_loras=1)
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, skip_special_tokens=True)

    lora_request = None
    if args.adapter_checkpoint is not None and not args.merged_model:
        lora_request = LoRARequest(args.lora_name, 1, str(args.adapter_checkpoint))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("a", encoding="utf-8") as output_handle, tqdm(
        total=len(records), desc="vLLM reasoner", unit="img"
    ) as progress:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            images = [
                load_image_for_vllm(
                    record["image_path"],
                    args.resize_config,
                    args.resize_mode,
                    args.safety_max_pixels,
                    args.large_image_resize,
                )
                for record in batch
            ]
            messages = [build_messages(record["prompt"], image, args.use_system_prompt) for record, image in zip(batch, images)]
            chat_kwargs: dict[str, Any] = {"sampling_params": sampling_params, "use_tqdm": False}
            if lora_request is not None:
                chat_kwargs["lora_request"] = lora_request
            outputs = llm.chat(messages, **chat_kwargs)
            if len(outputs) != len(batch):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch)} inputs")

            for record, request_output in zip(batch, outputs):
                report = clean_report(request_output.outputs[0].text.strip())
                if args.denormalize_grounding:
                    report = denormalize_grounding(report, record["image_path"], args.grounding_norm_base)
                output_item = {
                    "image_name": record["image_name"],
                    "report": report,
                    "prompt": record["prompt"],
                    "detector_predicted_forged": record["detector_predicted_forged"],
                    "detector_prob_forged": record["detector_prob_forged"],
                    "localizer_boxes": record["localizer_boxes"],
                    "localizer_box_count": record["localizer_box_count"],
                    "localizer_heatmap_path": record["localizer_heatmap_path"],
                    "localizer_used_threshold": record["localizer_used_threshold"],
                    "final_prompt_forged": record["final_prompt_forged"],
                    "inference_backend": "vllm",
                    "vllm_model": str(args.model_name_or_path),
                    "vllm_adapter": None if args.adapter_checkpoint is None or args.merged_model else str(args.adapter_checkpoint),
                    "vllm_merged_model": bool(args.merged_model),
                }
                output_handle.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                output_handle.flush()
                progress.update(1)

            for image in images:
                image.close()

    print(f"Done. Wrote {len(records)} report(s) to {args.output_jsonl}")


if __name__ == "__main__":
    main()
