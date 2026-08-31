from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

_MODEL_SPEC = importlib.util.spec_from_file_location(
    "dinov3_localizer",
    Path(__file__).resolve().parent / "model" / "dinov3_localizer.py",
)
if _MODEL_SPEC is None or _MODEL_SPEC.loader is None:
    raise ImportError("Cannot load model/dinov3_localizer.py")
_MODEL_MODULE = importlib.util.module_from_spec(_MODEL_SPEC)
sys.modules[_MODEL_SPEC.name] = _MODEL_MODULE
_MODEL_SPEC.loader.exec_module(_MODEL_MODULE)
DINOv3LocalizerConfig = _MODEL_MODULE.DINOv3LocalizerConfig
DINOv3TamperLocalizer = _MODEL_MODULE.DINOv3TamperLocalizer


IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    image_path: str
    mask_path: str
    label: int


def get_resample(name: str) -> int:
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, name)


def find_mask(mask_dir: Path, image_path: Path) -> Path:
    candidates = [
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem}_mask.png",
        mask_dir / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = mask_dir / f"{image_path.stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No mask found for {image_path} in {mask_dir}")


def mask_has_positive(mask_path: Path) -> bool:
    mask = Image.open(mask_path).convert("L")
    return bool((np.asarray(mask) > 127).any())


class FolderMaskDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, image_height: int, image_width: int) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        image_paths = sorted(p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")
        self.samples: list[Sample] = []
        for image_path in image_paths:
            mask_path = find_mask(self.mask_dir, image_path)
            label = 1 if mask_has_positive(mask_path) else 0
            self.samples.append(Sample(str(image_path), str(mask_path), label))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_width, self.image_height), get_resample("BICUBIC"))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - IMAGE_MEAN) / IMAGE_STD

    def _load_mask(self, path: str) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.image_width, self.image_height), get_resample("NEAREST"))
        array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "pixel_values": self._load_image(sample.image_path),
            "masks": self._load_mask(sample.mask_path),
            "labels": torch.tensor(float(sample.label), dtype=torch.float32),
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
        }


class RecursiveImageDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        image_height: int,
        image_width: int,
        image_list_path: str = "",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        all_image_paths = sorted(p for p in self.image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if image_list_path:
            by_stem: dict[str, Path] = {}
            duplicate_stems: set[str] = set()
            for path in all_image_paths:
                if path.stem in by_stem:
                    duplicate_stems.add(path.stem)
                by_stem.setdefault(path.stem, path)
            if duplicate_stems:
                examples = ", ".join(sorted(duplicate_stems)[:10])
                raise RuntimeError(f"Duplicate image stems under {self.image_dir}: {examples}")

            requested_paths: list[Path] = []
            missing: list[str] = []
            with open(image_list_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    item = raw_line.strip()
                    if not item:
                        continue
                    item_path = Path(item)
                    if item_path.is_absolute():
                        candidate = item_path
                    elif item_path.suffix.lower() in IMAGE_EXTS:
                        candidate = self.image_dir / item_path
                    else:
                        candidate = by_stem.get(item)
                        if candidate is None:
                            missing.append(item)
                            continue
                    if not candidate.exists():
                        missing.append(item)
                        continue
                    requested_paths.append(candidate)
            if missing:
                examples = ", ".join(missing[:20])
                raise FileNotFoundError(f"Missing {len(missing)} images from {image_list_path}: {examples}")
            self.image_paths = requested_paths
        else:
            self.image_paths = all_image_paths
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_width, self.image_height), get_resample("BICUBIC"))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - IMAGE_MEAN) / IMAGE_STD

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        return {
            "pixel_values": self._load_image(str(image_path)),
            "image_path": str(image_path),
            "relative_path": str(image_path.relative_to(self.image_dir)),
        }


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in features], dim=0),
        "masks": torch.stack([item["masks"] for item in features], dim=0),
        "labels": torch.stack([item["labels"] for item in features], dim=0),
        "image_path": [item["image_path"] for item in features],
        "mask_path": [item["mask_path"] for item in features],
    }


def collate_predict_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in features], dim=0),
        "image_path": [item["image_path"] for item in features],
        "relative_path": [item["relative_path"] for item in features],
    }


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def image_metrics(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    pos_f1 = f1_from_counts(tp, fp, fn)
    neg_f1 = f1_from_counts(tn, fn, fp)
    pos_support = tp + fn
    neg_support = tn + fp
    total = max(pos_support + neg_support, 1.0)
    return {
        "image_acc": float((tp + tn) / max(tp + fp + fn + tn, 1.0)),
        "image_precision": float(tp / max(tp + fp, 1.0)),
        "image_recall": float(tp / max(tp + fn, 1.0)),
        "image_f1": pos_f1,
        "image_f1_negative": neg_f1,
        "weighted_image_f1": float((pos_f1 * pos_support + neg_f1 * neg_support) / total),
    }


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataset_name: str,
    dataset: FolderMaskDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cls_threshold: float,
    mask_threshold: float,
    gate_masks_by_cls: bool,
    probability_dir: Path | None = None,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    model.eval()
    counts = torch.zeros(8, device=device, dtype=torch.float64)
    classification_records: list[dict[str, Any]] = []
    loc_prob_dir = probability_dir / dataset_name / "loc_probs" if probability_dir else None
    if loc_prob_dir is not None:
        loc_prob_dir.mkdir(parents=True, exist_ok=True)
    iterator = tqdm(loader, desc=dataset_name, leave=False)
    for batch in iterator:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        outputs = model(pixel_values)

        probs = torch.sigmoid(outputs["cls_logits"])
        pred_labels = probs >= cls_threshold
        gt_labels = labels >= 0.5
        counts[0] += (pred_labels & gt_labels).sum()
        counts[1] += (pred_labels & ~gt_labels).sum()
        counts[2] += (~pred_labels & gt_labels).sum()
        counts[3] += (~pred_labels & ~gt_labels).sum()

        mask_probs = torch.sigmoid(outputs["mask_logits"])
        pred_masks = mask_probs >= mask_threshold
        if gate_masks_by_cls:
            pred_masks = pred_masks & pred_labels[:, None, None, None]
        gt_masks = masks >= 0.5
        counts[4] += (pred_masks & gt_masks).sum()
        counts[5] += (pred_masks & ~gt_masks).sum()
        counts[6] += (~pred_masks & gt_masks).sum()
        counts[7] += (~pred_masks & ~gt_masks).sum()

        if probability_dir is not None:
            mask_probs_cpu = mask_probs.detach().float().cpu().numpy()
            probs_cpu = probs.detach().float().cpu().tolist()
            labels_cpu = gt_labels.detach().cpu().tolist()
            pred_labels_cpu = pred_labels.detach().cpu().tolist()
            for idx, (image_path, mask_path) in enumerate(zip(batch["image_path"], batch["mask_path"])):
                image_stem = Path(image_path).stem
                loc_prob_path = loc_prob_dir / f"{image_stem}.png" if loc_prob_dir is not None else None
                if loc_prob_path is not None:
                    probability = np.clip(mask_probs_cpu[idx, 0], 0.0, 1.0)
                    probability_u8 = np.rint(probability * 255.0).astype(np.uint8)
                    Image.fromarray(probability_u8, mode="L").save(loc_prob_path)
                classification_records.append(
                    {
                        "dataset": dataset_name,
                        "image": image_path,
                        "mask": mask_path,
                        "gt_label": int(bool(labels_cpu[idx])),
                        "prob_forged": float(probs_cpu[idx]),
                        "pred_label": int(bool(pred_labels_cpu[idx])),
                        "loc_prob_path": str(loc_prob_path) if loc_prob_path is not None else "",
                    }
                )

    tp, fp, fn, tn, pix_tp, pix_fp, pix_fn, pix_tn = [float(x) for x in counts.detach().cpu().tolist()]
    pix_precision = pix_tp / max(pix_tp + pix_fp, 1.0)
    pix_recall = pix_tp / max(pix_tp + pix_fn, 1.0)
    metrics = {
        "dataset": dataset_name,
        "num_images": len(dataset),
        "positive_images": int(tp + fn),
        "negative_images": int(tn + fp),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        **image_metrics(tp, fp, fn, tn),
        "pixel_precision": float(pix_precision),
        "pixel_recall": float(pix_recall),
        "pixel_f1": float(2 * pix_precision * pix_recall / max(pix_precision + pix_recall, 1e-12)),
        "pixel_iou": float(pix_tp / max(pix_tp + pix_fp + pix_fn, 1.0)),
        "pixel_tp": int(pix_tp),
        "pixel_fp": int(pix_fp),
        "pixel_fn": int(pix_fn),
        "pixel_tn": int(pix_tn),
    }
    metrics["score_image_plus_pixel"] = float(metrics["image_f1"] + metrics["pixel_f1"])
    metrics["score_weighted_image_plus_pixel"] = float(metrics["weighted_image_f1"] + metrics["pixel_f1"])
    if probability_dir is not None:
        dataset_prob_dir = probability_dir / dataset_name
        dataset_prob_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = dataset_prob_dir / "classification_probs.jsonl"
        csv_path = dataset_prob_dir / "classification_probs.csv"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in classification_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["dataset", "image", "mask", "gt_label", "prob_forged", "pred_label", "loc_prob_path"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(classification_records)
        metrics["classification_probs_jsonl"] = str(jsonl_path)
        metrics["classification_probs_csv"] = str(csv_path)
        metrics["loc_prob_dir"] = str(loc_prob_dir)
    return metrics


def load_config(checkpoint: Path, fallback_dino_model_path: str) -> DINOv3LocalizerConfig:
    config_path = checkpoint / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["dino_model_path"] = fallback_dino_model_path or payload["dino_model_path"]
    return DINOv3LocalizerConfig(**payload)


def adapt_dino_state_keys(state: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, torch.Tensor]:
    model_keys = set(model.state_dict().keys())
    adapted = {}
    for key, value in state.items():
        if key in model_keys:
            adapted[key] = value
            continue
        candidates = []
        if key.startswith("dino.layer."):
            candidates.append(key.replace("dino.layer.", "dino.model.layer.", 1))
        elif key.startswith("dino.model.layer."):
            candidates.append(key.replace("dino.model.layer.", "dino.layer.", 1))
        matched = False
        for new_key in candidates:
            if new_key in model_keys:
                adapted[new_key] = value
                matched = True
                break
        if not matched:
            adapted[key] = value
    return adapted


def run_metric_script(
    metric_script: str,
    pred_dir: str,
    gt_dir: str,
    save_dir: str,
    threshold: float,
    workers: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        metric_script,
        "--pred_dir",
        pred_dir,
        "--gt_dir",
        gt_dir,
        "--save_dir",
        save_dir,
        "--threshold",
        str(threshold),
    ]
    if workers > 0:
        command.extend(["--workers", str(workers)])
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    metric_result = {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    result_files = sorted(Path(save_dir).glob("result_*.json"))
    if result_files:
        metric_result["result_json"] = str(result_files[-1])
    ans_files = sorted(Path(save_dir).glob("ans_*.txt"))
    if ans_files:
        metric_result["ans_txt"] = str(ans_files[-1])
    return metric_result


@torch.no_grad()
def predict_images(
    model: torch.nn.Module,
    image_dir: str,
    output_json: str,
    device: torch.device,
    image_height: int,
    image_width: int,
    batch_size: int,
    num_workers: int,
    cls_threshold: float,
    mask_threshold: float,
    loc_prob_dir: str = "",
    image_list_path: str = "",
) -> dict[str, Any]:
    dataset = RecursiveImageDataset(
        image_dir,
        image_height=image_height,
        image_width=image_width,
        image_list_path=image_list_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_predict_fn,
        drop_last=False,
    )
    loc_dir = Path(loc_prob_dir) if loc_prob_dir else None
    if loc_dir is not None:
        loc_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    records: list[dict[str, Any]] = []
    forged_count = 0
    loc_positive_count = 0
    iterator = tqdm(loader, desc="predict", leave=False)
    for batch in iterator:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        outputs = model(pixel_values)
        cls_probs = torch.sigmoid(outputs["cls_logits"]).detach().float().cpu().numpy()
        mask_probs = torch.sigmoid(outputs["mask_logits"]).detach().float().cpu().numpy()
        for idx, (image_path, relative_path) in enumerate(zip(batch["image_path"], batch["relative_path"])):
            cls_prob = float(cls_probs[idx])
            pred_label_id = int(cls_prob >= cls_threshold)
            forged_count += pred_label_id

            loc_prob = np.clip(mask_probs[idx, 0], 0.0, 1.0)
            loc_binary = loc_prob >= mask_threshold
            positive_pixels = int(loc_binary.sum())
            loc_positive = int(positive_pixels > 0)
            loc_positive_count += loc_positive

            loc_prob_path = ""
            if loc_dir is not None:
                rel = Path(relative_path)
                save_path = loc_dir / rel.with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.rint(loc_prob * 255.0).astype(np.uint8), mode="L").save(save_path)
                loc_prob_path = str(save_path)

            records.append(
                {
                    "image": image_path,
                    "relative_path": relative_path,
                    "prob_forged": cls_prob,
                    "pred_label": "Forged" if pred_label_id else "Authentic",
                    "pred_label_id": pred_label_id,
                    "loc_positive": bool(loc_positive),
                    "loc_positive_pixel_count": positive_pixels,
                    "loc_positive_fraction": float(positive_pixels / max(loc_prob.size, 1)),
                    "loc_prob_max": float(loc_prob.max()),
                    "loc_prob_mean": float(loc_prob.mean()),
                    "loc_prob_path": loc_prob_path,
                }
            )

    payload = {
        "image_dir": str(image_dir),
        "image_list": str(image_list_path),
        "num_images": len(records),
        "checkpoint": "",
        "cls_threshold": cls_threshold,
        "mask_threshold": mask_threshold,
        "predicted_forged_by_cls": forged_count,
        "predicted_authentic_by_cls": len(records) - forged_count,
        "predicted_positive_by_loc": loc_positive_count,
        "predicted_negative_by_loc": len(records) - loc_positive_count,
        "predictions": records,
    }
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dino_model_path", default="checkpoint/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--datasets", nargs="+", default=[], help="name:image_dir:mask_dir")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cls_threshold", type=float, default=0.5)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--gate_masks_by_cls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--prob_output_dir", default="")
    parser.add_argument("--metric_script", default="")
    parser.add_argument("--metric_workers", type=int, default=8)
    parser.add_argument("--predict_image_dir", default="")
    parser.add_argument("--predict_image_list", default="")
    parser.add_argument("--prediction_output_json", default="")
    parser.add_argument("--prediction_loc_prob_dir", default="")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    config = load_config(checkpoint, args.dino_model_path)
    model = DINOv3TamperLocalizer(config).to(device)
    state = adapt_dino_state_keys(torch.load(checkpoint / "pytorch_model.bin", map_location="cpu"), model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[eval] missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)

    if args.predict_image_dir:
        if not args.prediction_output_json:
            raise ValueError("--prediction_output_json is required with --predict_image_dir")
        payload = predict_images(
            model=model,
            image_dir=args.predict_image_dir,
            output_json=args.prediction_output_json,
            device=device,
            image_height=config.image_height,
            image_width=config.image_width,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cls_threshold=args.cls_threshold,
            mask_threshold=args.mask_threshold,
            loc_prob_dir=args.prediction_loc_prob_dir,
            image_list_path=args.predict_image_list,
        )
        payload["checkpoint"] = str(checkpoint)
        with open(args.prediction_output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps({k: v for k, v in payload.items() if k != "predictions"}, ensure_ascii=False))
        return

    if not args.datasets:
        raise ValueError("--datasets is required unless --predict_image_dir is used")

    results = []
    metric_results = []
    probability_dir = Path(args.prob_output_dir) if args.prob_output_dir else None
    combined_counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "pixel_tp": 0,
        "pixel_fp": 0,
        "pixel_fn": 0,
        "pixel_tn": 0,
        "num_images": 0,
    }
    for spec in args.datasets:
        name, image_dir, mask_dir = spec.split(":", 2)
        dataset = FolderMaskDataset(image_dir, mask_dir, config.image_height, config.image_width)
        result = evaluate_dataset(
            model=model,
            dataset_name=name,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cls_threshold=args.cls_threshold,
            mask_threshold=args.mask_threshold,
            gate_masks_by_cls=args.gate_masks_by_cls,
            probability_dir=probability_dir,
        )
        results.append(result)
        if args.metric_script and probability_dir is not None:
            metric_save_dir = probability_dir / name / "compute_matrix_global"
            metric_result = run_metric_script(
                metric_script=args.metric_script,
                pred_dir=str(probability_dir / name / "loc_probs"),
                gt_dir=mask_dir,
                save_dir=str(metric_save_dir),
                threshold=args.mask_threshold,
                workers=args.metric_workers,
            )
            metric_result["dataset"] = name
            metric_results.append(metric_result)
        for key in combined_counts:
            combined_counts[key] += int(result[key])

    tp = float(combined_counts["tp"])
    fp = float(combined_counts["fp"])
    fn = float(combined_counts["fn"])
    tn = float(combined_counts["tn"])
    pix_tp = float(combined_counts["pixel_tp"])
    pix_fp = float(combined_counts["pixel_fp"])
    pix_fn = float(combined_counts["pixel_fn"])
    pix_precision = pix_tp / max(pix_tp + pix_fp, 1.0)
    pix_recall = pix_tp / max(pix_tp + pix_fn, 1.0)
    combined = {
        "dataset": "combined",
        "num_images": int(combined_counts["num_images"]),
        "positive_images": int(tp + fn),
        "negative_images": int(tn + fp),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        **image_metrics(tp, fp, fn, tn),
        "pixel_precision": float(pix_precision),
        "pixel_recall": float(pix_recall),
        "pixel_f1": float(2 * pix_precision * pix_recall / max(pix_precision + pix_recall, 1e-12)),
        "pixel_iou": float(pix_tp / max(pix_tp + pix_fp + pix_fn, 1.0)),
        "pixel_tp": int(pix_tp),
        "pixel_fp": int(pix_fp),
        "pixel_fn": int(pix_fn),
        "pixel_tn": int(combined_counts["pixel_tn"]),
    }
    combined["score_image_plus_pixel"] = float(combined["image_f1"] + combined["pixel_f1"])
    combined["score_weighted_image_plus_pixel"] = float(combined["weighted_image_f1"] + combined["pixel_f1"])

    payload = {
        "checkpoint": str(checkpoint),
        "results": results,
        "combined": combined,
        "metric_script_results": metric_results,
    }
    text = json.dumps(payload, ensure_ascii=False)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
