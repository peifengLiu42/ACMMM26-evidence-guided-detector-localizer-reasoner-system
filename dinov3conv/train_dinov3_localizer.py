from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from tqdm import tqdm

_LOCALIZER_SPEC = importlib.util.spec_from_file_location(
    "dinov3_localizer",
    Path(__file__).resolve().parent / "model" / "dinov3_localizer.py",
)
if _LOCALIZER_SPEC is None or _LOCALIZER_SPEC.loader is None:
    raise ImportError("Cannot load model/dinov3_localizer.py")
_LOCALIZER_MODULE = importlib.util.module_from_spec(_LOCALIZER_SPEC)
sys.modules[_LOCALIZER_SPEC.name] = _LOCALIZER_MODULE
_LOCALIZER_SPEC.loader.exec_module(_LOCALIZER_MODULE)
DINOv3LocalizerConfig = _LOCALIZER_MODULE.DINOv3LocalizerConfig
DINOv3TamperLocalizer = _LOCALIZER_MODULE.DINOv3TamperLocalizer


IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class LocalizerSample:
    image_path: str
    mask_path: str
    label: int


def get_resample(name: str):
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, name)


def load_exclude_items(path: str | None) -> set[str]:
    if not path:
        return set()
    exclude_path = Path(path)
    if not exclude_path.exists():
        raise FileNotFoundError(f"exclude list not found: {exclude_path}")
    items: set[str] = set()
    with exclude_path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value:
                continue
            value = value.replace("\\", "/")
            items.add(value)
            items.add(str(Path(value).with_suffix("")))
            items.add(Path(value).stem)
    return items


def is_excluded_image(image_path: str, exclude_items: set[str]) -> bool:
    if not exclude_items:
        return False
    path = Path(image_path)
    text = str(path).replace("\\", "/")
    no_ext = str(path.with_suffix("")).replace("\\", "/")
    stem = path.stem
    return text in exclude_items or no_ext in exclude_items or stem in exclude_items


def parse_detection_label(text: str) -> int | None:
    upper = text.upper()
    conclusion_match = re.search(r"\[CONCLUSION\]\s*:?\s*\**\s*(FORGED|AUTHENTIC)", upper)
    if conclusion_match:
        return 1 if conclusion_match.group(1) == "FORGED" else 0
    if "FORGED" in upper:
        return 1
    if "AUTHENTIC" in upper:
        return 0
    return None


def derive_mask_path(image_path: str, mask_root: str | None = None) -> str:
    path = Path(image_path)
    if mask_root:
        part = path.parent.name
        return str(Path(mask_root) / part / f"{path.stem}_mask.png")
    text = str(path)
    if "/train/image/" in text:
        text = text.replace("/train/image/", "/train/regen_mask/")
    return str(Path(text).with_name(f"{path.stem}_mask.png"))


def get_assistant_text(item: dict[str, Any]) -> str:
    if "messages" in item:
        for message in item["messages"]:
            if message.get("role") == "assistant":
                return str(message.get("content", ""))
    if "conversations" in item and len(item["conversations"]) > 1:
        return str(item["conversations"][1].get("value", ""))
    return ""


def get_image_and_mask(item: dict[str, Any], mask_root: str | None) -> tuple[str | None, str | None]:
    images = item.get("images", item.get("image", item.get("image_path")))
    if isinstance(images, list):
        image_path = str(images[0]) if images else None
        mask_path = str(images[1]) if len(images) > 1 else None
    else:
        image_path = str(images) if images else None
        mask_path = None
    if image_path and not mask_path:
        mask_path = derive_mask_path(image_path, mask_root)
    return image_path, mask_path


class RealTextTamperLocalizationDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_size: int = 448,
        image_height: int | None = None,
        image_width: int | None = None,
        mask_root: str | None = None,
        skip_missing_forged_masks: bool = True,
        limit_samples: int = -1,
        is_train: bool = False,
        augment: bool = False,
        random_resized_crop_scale: float = 0.9,
        rotation_degrees: float = 3.0,
        hflip_prob: float = 0.0,
        color_jitter: float = 0.1,
        exclude_list_path: str | None = None,
    ) -> None:
        super().__init__()
        self.json_path = json_path
        self.image_size = int(image_size)
        self.image_height = int(image_height or image_size)
        self.image_width = int(image_width or image_size)
        self.mask_root = mask_root
        self.is_train = is_train
        self.augment = augment
        self.random_resized_crop_scale = float(random_resized_crop_scale)
        self.rotation_degrees = float(rotation_degrees)
        self.hflip_prob = float(hflip_prob)
        self.color_jitter = float(color_jitter)
        self.exclude_items = load_exclude_items(exclude_list_path)
        self.samples = self._load_samples(json_path, skip_missing_forged_masks)
        if limit_samples and limit_samples > 0:
            self.samples = self.samples[:limit_samples]

    def _load_samples(self, json_path: str, skip_missing_forged_masks: bool) -> list[LocalizerSample]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples: list[LocalizerSample] = []
        skipped = {
            "bad_label": 0,
            "bad_image": 0,
            "excluded": 0,
            "missing_forged_mask": 0,
        }
        for item in data:
            label = parse_detection_label(get_assistant_text(item))
            if label is None:
                skipped["bad_label"] += 1
                continue
            image_path, mask_path = get_image_and_mask(item, self.mask_root)
            if not image_path or not os.path.exists(image_path):
                skipped["bad_image"] += 1
                continue
            if is_excluded_image(image_path, self.exclude_items):
                skipped["excluded"] += 1
                continue
            mask_path = mask_path or ""
            if label == 1 and skip_missing_forged_masks and not os.path.exists(mask_path):
                skipped["missing_forged_mask"] += 1
                continue
            samples.append(LocalizerSample(image_path=image_path, mask_path=mask_path, label=label))
        if not samples:
            raise RuntimeError(f"No usable samples loaded from {json_path}. skipped={skipped}")
        self.skipped = skipped
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_pil(self, path: str) -> Image.Image:
        image = Image.open(path).convert("RGB")
        return image.resize((self.image_width, self.image_height), get_resample("BICUBIC"))

    def _load_mask_pil(self, path: str) -> Image.Image:
        if path and os.path.exists(path):
            mask = Image.open(path).convert("L")
            return mask.resize((self.image_width, self.image_height), get_resample("NEAREST"))
        return Image.fromarray(np.zeros((self.image_height, self.image_width), dtype=np.uint8), mode="L")

    def _random_resized_crop(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.random_resized_crop_scale >= 1.0:
            return image, mask
        min_scale = max(0.1, min(self.random_resized_crop_scale, 1.0))
        scale = random.uniform(min_scale, 1.0)
        crop_width = max(1, int(round(self.image_width * scale)))
        crop_height = max(1, int(round(self.image_height * scale)))
        if crop_width >= self.image_width and crop_height >= self.image_height:
            return image, mask
        left = random.randint(0, self.image_width - crop_width)
        top = random.randint(0, self.image_height - crop_height)
        box = (left, top, left + crop_width, top + crop_height)
        image = image.crop(box).resize((self.image_width, self.image_height), get_resample("BICUBIC"))
        mask = mask.crop(box).resize((self.image_width, self.image_height), get_resample("NEAREST"))
        return image, mask

    def _augment_pair(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if not (self.is_train and self.augment):
            return image, mask

        image, mask = self._random_resized_crop(image, mask)

        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = image.rotate(angle, resample=get_resample("BICUBIC"), fillcolor=(255, 255, 255))
            mask = mask.rotate(angle, resample=get_resample("NEAREST"), fillcolor=0)

        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if self.color_jitter > 0:
            low = max(0.0, 1.0 - self.color_jitter)
            high = 1.0 + self.color_jitter
            image = ImageEnhance.Brightness(image).enhance(random.uniform(low, high))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(low, high))
            image = ImageEnhance.Color(image).enhance(random.uniform(low, high))

        return image, mask

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - IMAGE_MEAN) / IMAGE_STD

    def _mask_to_tensor(self, mask: Image.Image) -> torch.Tensor:
        array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self._load_image_pil(sample.image_path)
        mask = self._load_mask_pil(sample.mask_path)
        image, mask = self._augment_pair(image, mask)
        return {
            "pixel_values": self._image_to_tensor(image),
            "masks": self._mask_to_tensor(mask),
            "labels": torch.tensor(float(sample.label), dtype=torch.float32),
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
        }


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in features], dim=0),
        "masks": torch.stack([item["masks"] for item in features], dim=0),
        "labels": torch.stack([item["labels"] for item in features], dim=0),
        "image_path": [item["image_path"] for item in features],
        "mask_path": [item["mask_path"] for item in features],
    }


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, local_rank, rank, world_size, torch.device("cuda", local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 0, 1, device


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def get_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def batch_mask_pos_weight(targets: torch.Tensor, max_pos_weight: float) -> torch.Tensor:
    pos = targets.sum()
    neg = targets.numel() - pos
    if pos.item() <= 0:
        value = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)
    else:
        value = (neg / pos).clamp(min=1.0, max=max_pos_weight)
    return value


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    labels = batch["labels"].to(outputs["cls_logits"].dtype)
    masks = batch["masks"].to(outputs["mask_logits"].dtype)
    cls_loss = F.binary_cross_entropy_with_logits(outputs["cls_logits"], labels)
    pos_weight = batch_mask_pos_weight(masks, args.max_mask_pos_weight)
    mask_bce = F.binary_cross_entropy_with_logits(outputs["mask_logits"], masks, pos_weight=pos_weight)
    mask_dice = dice_loss(outputs["mask_logits"], masks)
    loss = (
        args.cls_loss_weight * cls_loss
        + args.mask_bce_weight * mask_bce
        + args.mask_dice_weight * mask_dice
    )
    return {
        "loss": loss,
        "cls_loss": cls_loss.detach(),
        "mask_bce": mask_bce.detach(),
        "mask_dice": mask_dice.detach(),
    }


def parse_thresholds(value: str) -> list[float]:
    thresholds: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(float(item))
    return thresholds


def pixel_metrics_from_counts(tp: float, fp: float, fn: float) -> dict[str, float]:
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1.0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    step_name: str,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(14, device=device, dtype=torch.float64)
    sweep_thresholds = parse_thresholds(args.mask_threshold_sweep)
    sweep_totals = torch.zeros((len(sweep_thresholds), 3), device=device, dtype=torch.float64)
    predictions: list[dict[str, Any]] = []
    iterator = tqdm(loader, desc=f"eval {step_name}", disable=not is_main_process())
    for batch in iterator:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        outputs = model(pixel_values)
        losses = compute_loss(outputs, {"labels": labels, "masks": masks}, args)

        probs = torch.sigmoid(outputs["cls_logits"])
        pred_labels = probs >= args.cls_threshold
        gt_labels = labels >= 0.5
        totals[0] += losses["loss"].double() * labels.numel()
        totals[1] += losses["cls_loss"].double() * labels.numel()
        totals[2] += losses["mask_bce"].double() * labels.numel()
        totals[3] += losses["mask_dice"].double() * labels.numel()
        totals[4] += labels.numel()
        totals[5] += (pred_labels & gt_labels).sum()
        totals[6] += (pred_labels & ~gt_labels).sum()
        totals[7] += (~pred_labels & gt_labels).sum()
        totals[8] += (~pred_labels & ~gt_labels).sum()

        mask_probs = torch.sigmoid(outputs["mask_logits"])
        pred_masks = mask_probs >= args.mask_threshold
        if args.gate_masks_by_cls:
            pred_masks = pred_masks & pred_labels[:, None, None, None]
        gt_masks = masks >= 0.5
        totals[9] += (pred_masks & gt_masks).sum()
        totals[10] += (pred_masks & ~gt_masks).sum()
        totals[11] += (~pred_masks & gt_masks).sum()
        totals[12] += (~pred_masks & ~gt_masks).sum()
        totals[13] += gt_masks.numel()

        for t_idx, threshold in enumerate(sweep_thresholds):
            sweep_masks = mask_probs >= threshold
            if args.gate_masks_by_cls:
                sweep_masks = sweep_masks & pred_labels[:, None, None, None]
            sweep_totals[t_idx, 0] += (sweep_masks & gt_masks).sum()
            sweep_totals[t_idx, 1] += (sweep_masks & ~gt_masks).sum()
            sweep_totals[t_idx, 2] += (~sweep_masks & gt_masks).sum()

        if is_main_process() and len(predictions) < args.max_eval_predictions:
            for image_path, mask_path, prob, gt, pred in zip(
                batch["image_path"],
                batch["mask_path"],
                probs.detach().cpu().tolist(),
                gt_labels.detach().cpu().tolist(),
                pred_labels.detach().cpu().tolist(),
            ):
                predictions.append(
                    {
                        "image": image_path,
                        "mask": mask_path,
                        "prob_forged": float(prob),
                        "gt_label": "Forged" if gt else "Authentic",
                        "pred_label": "Forged" if pred else "Authentic",
                    }
                )

    totals = reduce_sum(totals)
    if sweep_totals.numel() > 0:
        sweep_totals = reduce_sum(sweep_totals)
    count = max(float(totals[4].item()), 1.0)
    tp, fp, fn, tn = [float(totals[i].item()) for i in range(5, 9)]
    pix_tp, pix_fp, pix_fn, pix_tn = [float(totals[i].item()) for i in range(9, 13)]

    image_precision = tp / max(tp + fp, 1.0)
    image_recall = tp / max(tp + fn, 1.0)
    pixel_precision = pix_tp / max(pix_tp + pix_fp, 1.0)
    pixel_recall = pix_tp / max(pix_tp + pix_fn, 1.0)
    metrics = {
        "loss": float(totals[0].item() / count),
        "cls_loss": float(totals[1].item() / count),
        "mask_bce": float(totals[2].item() / count),
        "mask_dice": float(totals[3].item() / count),
        "image_acc": float((tp + tn) / max(tp + fp + fn + tn, 1.0)),
        "image_precision": float(image_precision),
        "image_recall": float(image_recall),
        "image_f1": float(2 * image_precision * image_recall / max(image_precision + image_recall, 1e-12)),
        "pixel_precision": float(pixel_precision),
        "pixel_recall": float(pixel_recall),
        "pixel_f1": float(2 * pixel_precision * pixel_recall / max(pixel_precision + pixel_recall, 1e-12)),
        "pixel_iou": float(pix_tp / max(pix_tp + pix_fp + pix_fn, 1.0)),
        "num_images": int(totals[4].item()),
    }

    if sweep_thresholds:
        best_threshold = args.mask_threshold
        best_metrics = {
            "precision": metrics["pixel_precision"],
            "recall": metrics["pixel_recall"],
            "f1": metrics["pixel_f1"],
            "iou": metrics["pixel_iou"],
        }
        for threshold, counts in zip(sweep_thresholds, sweep_totals.detach().cpu().tolist()):
            sweep_metrics = pixel_metrics_from_counts(float(counts[0]), float(counts[1]), float(counts[2]))
            if sweep_metrics["f1"] > best_metrics["f1"]:
                best_threshold = float(threshold)
                best_metrics = sweep_metrics
        metrics.update(
            {
                "best_mask_threshold": float(best_threshold),
                "best_threshold_pixel_precision": best_metrics["precision"],
                "best_threshold_pixel_recall": best_metrics["recall"],
                "best_threshold_pixel_f1": best_metrics["f1"],
                "best_threshold_pixel_iou": best_metrics["iou"],
            }
        )

    if is_main_process() and args.save_eval_predictions:
        pred_dir = Path(args.output_dir) / "eval_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        with open(pred_dir / f"{step_name}.json", "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
    model.train()
    return metrics


def save_jsonl(path: str, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_checkpoint(
    model: torch.nn.Module,
    save_dir: str,
    args: argparse.Namespace,
    metrics: dict[str, float],
    step: int,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    if not is_main_process():
        return
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    module = unwrap_model(model)
    torch.save(module.state_dict(), path / "pytorch_model.bin")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), path / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), path / "scheduler.pt")
    with open(path / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(module.localizer_config.to_dict(), f, indent=2)
    trainer_state = {
        "step": step,
        "epoch": epoch,
        **metrics,
    }
    with open(path / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)
    with open(path / "training_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


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


def load_checkpoint_if_needed(model: torch.nn.Module, init_checkpoint: str) -> dict[str, Any]:
    if not init_checkpoint:
        return {}
    ckpt_path = Path(init_checkpoint)
    state_path = ckpt_path / "pytorch_model.bin"
    if not state_path.exists():
        raise FileNotFoundError(f"checkpoint weights not found: {state_path}")
    state = adapt_dino_state_keys(torch.load(state_path, map_location="cpu"), model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if is_main_process():
        print(f"[localizer] loaded checkpoint: {init_checkpoint}")
        print(f"[localizer] missing={len(missing)} unexpected={len(unexpected)}")
    trainer_state_path = ckpt_path / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_optimizer_scheduler_if_needed(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    init_checkpoint: str,
) -> bool:
    if not init_checkpoint:
        return False
    ckpt_path = Path(init_checkpoint)
    optimizer_path = ckpt_path / "optimizer.pt"
    scheduler_path = ckpt_path / "scheduler.pt"
    if optimizer_path.exists() and scheduler_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        scheduler_state = torch.load(scheduler_path, map_location="cpu")
        current_t_max = scheduler.state_dict().get("T_max")
        saved_t_max = scheduler_state.get("T_max")
        if current_t_max == saved_t_max:
            scheduler.load_state_dict(scheduler_state)
            scheduler_loaded = True
        else:
            scheduler_loaded = False
        if is_main_process():
            state_label = "optimizer/scheduler" if scheduler_loaded else "optimizer"
            print(f"[localizer] loaded {state_label} state: {init_checkpoint}")
            if not scheduler_loaded:
                print(
                    "[localizer] scheduler T_max changed; rebuilt scheduler for current run. "
                    f"saved_T_max={saved_t_max} current_T_max={current_t_max}"
                )
        return scheduler_loaded
    if is_main_process():
        missing = [
            str(path.name)
            for path in (optimizer_path, scheduler_path)
            if not path.exists()
        ]
        print(
            "[localizer] optimizer/scheduler state not found; "
            f"falling back to lightweight resume. missing={missing}"
        )
    return False


def prune_epoch_checkpoints(output_dir: str, keep_last: int) -> None:
    if keep_last <= 0 or not is_main_process():
        return
    root = Path(output_dir)
    if not root.exists():
        return
    epoch_dirs: list[tuple[int, Path]] = []
    for path in root.glob("epoch-*"):
        if not path.is_dir():
            continue
        try:
            epoch = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        epoch_dirs.append((epoch, path))
    epoch_dirs.sort(key=lambda item: item[0])
    stale_dirs = epoch_dirs[:-keep_last]
    for _, path in stale_dirs:
        shutil.rmtree(path, ignore_errors=True)
        print(f"[localizer] pruned old epoch checkpoint: {path}")


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    train_dataset = RealTextTamperLocalizationDataset(
        json_path=args.sft_json_path,
        image_size=args.image_size,
        image_height=args.image_height,
        image_width=args.image_width,
        mask_root=args.mask_root,
        skip_missing_forged_masks=args.skip_missing_forged_masks,
        limit_samples=args.limit_samples,
        is_train=True,
        augment=args.augment,
        random_resized_crop_scale=args.random_resized_crop_scale,
        rotation_degrees=args.rotation_degrees,
        hflip_prob=args.hflip_prob,
        color_jitter=args.color_jitter,
        exclude_list_path=args.exclude_list_path,
    )
    if args.val_sft_json_path:
        val_dataset = RealTextTamperLocalizationDataset(
            json_path=args.val_sft_json_path,
            image_size=args.image_size,
            image_height=args.image_height,
            image_width=args.image_width,
            mask_root=args.mask_root,
            skip_missing_forged_masks=args.skip_missing_forged_masks,
            limit_samples=args.val_limit_samples,
            is_train=False,
            augment=False,
        )
        return train_dataset, val_dataset

    indices = list(range(len(train_dataset)))
    random.Random(args.seed).shuffle(indices)
    val_size = max(1, int(len(indices) * args.val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(train_dataset, train_indices), Subset(train_dataset, val_indices)


def get_dataset_skipped(dataset: Dataset) -> dict[str, int]:
    source = dataset.dataset if isinstance(dataset, Subset) else dataset
    return getattr(source, "skipped", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino_model_path", default="checkpoint/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--sft_json_path", default="data/train_sft.json")
    parser.add_argument("--val_sft_json_path", default="data/val_sft.json")
    parser.add_argument("--exclude_list_path", default="")
    parser.add_argument("--mask_root", default="data/train_masks")
    parser.add_argument("--output_dir", default="runs/dinov3-tamper-localizer")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--resume_step_from_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--image_height", type=int, default=896)
    parser.add_argument("--image_width", type=int, default=1344)
    parser.add_argument("--decoder_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask_head_type", choices=["conv", "mask2former"], default="conv")
    parser.add_argument("--mask2former_num_queries", type=int, default=32)
    parser.add_argument("--mask2former_num_layers", type=int, default=3)
    parser.add_argument("--mask2former_num_heads", type=int, default=8)
    parser.add_argument("--mask2former_ffn_dim", type=int, default=1024)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random_resized_crop_scale", type=float, default=0.9)
    parser.add_argument("--rotation_degrees", type=float, default=3.0)
    parser.add_argument("--hflip_prob", type=float, default=0.0)
    parser.add_argument("--color_jitter", type=float, default=0.1)
    parser.add_argument("--freeze_dino", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--limit_samples", type=int, default=-1)
    parser.add_argument("--val_limit_samples", type=int, default=-1)
    parser.add_argument("--skip_missing_forged_masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cls_loss_weight", type=float, default=1.0)
    parser.add_argument("--mask_bce_weight", type=float, default=1.0)
    parser.add_argument("--mask_dice_weight", type=float, default=1.0)
    parser.add_argument("--max_mask_pos_weight", type=float, default=100.0)
    parser.add_argument("--cls_threshold", type=float, default=0.5)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--mask_threshold_sweep", default="0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--gate_masks_by_cls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_epochs", type=int, default=1)
    parser.add_argument("--save_epochs", type=int, default=1)
    parser.add_argument("--keep_last_epoch_checkpoints", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_eval_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_eval_predictions", type=int, default=512)
    parser.add_argument("--ddp_find_unused_parameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed, local_rank, rank, world_size, device = setup_distributed()
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    dtype = get_dtype(args.precision)
    config = DINOv3LocalizerConfig(
        dino_model_path=args.dino_model_path,
        image_size=args.image_size,
        image_height=args.image_height,
        image_width=args.image_width,
        decoder_dim=args.decoder_dim,
        dropout=args.dropout,
        mask_head_type=args.mask_head_type,
        mask2former_num_queries=args.mask2former_num_queries,
        mask2former_num_layers=args.mask2former_num_layers,
        mask2former_num_heads=args.mask2former_num_heads,
        mask2former_ffn_dim=args.mask2former_ffn_dim,
        freeze_dino=args.freeze_dino,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
    )
    model = DINOv3TamperLocalizer(config, torch_dtype=dtype).to(device)
    resume_state = load_checkpoint_if_needed(model, args.init_checkpoint) if args.init_checkpoint else {}

    train_dataset, val_dataset = build_datasets(args)
    if is_main_process():
        print(f"[localizer] train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
        print(f"[localizer] train_skipped={get_dataset_skipped(train_dataset)}")
        print(f"[localizer] trainable_params={unwrap_model(model).trainable_parameter_summary()}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.ddp_find_unused_parameters,
        )

    module = unwrap_model(model)
    dino_params = [p for n, p in module.named_parameters() if p.requires_grad and n.startswith("dino.")]
    head_params = [p for n, p in module.named_parameters() if p.requires_grad and not n.startswith("dino.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": dino_params, "lr": args.lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    start_global_step = int(resume_state.get("step", 0) or 0) if args.resume_step_from_checkpoint else 0
    planned_update_steps = math.ceil(len(train_loader) / args.grad_accumulation_steps) * args.epochs
    total_update_steps = start_global_step + planned_update_steps
    if args.max_steps > 0:
        total_update_steps = max(start_global_step, args.max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_update_steps, 1))
    loaded_optimizer_scheduler = False
    if args.init_checkpoint and args.resume_step_from_checkpoint:
        loaded_optimizer_scheduler = load_optimizer_scheduler_if_needed(
            optimizer=optimizer,
            scheduler=scheduler,
            init_checkpoint=args.init_checkpoint,
        )
    if start_global_step > 0 and not loaded_optimizer_scheduler:
        scheduler.step(start_global_step)

    best_image_f1 = float(resume_state.get("best_image_f1", 0.0) or 0.0)
    best_pixel_f1 = float(resume_state.get("best_pixel_f1", 0.0) or 0.0)
    best_val_loss = float(resume_state.get("best_val_loss", float("inf")) or float("inf"))
    global_step = start_global_step
    start_epoch = int(resume_state.get("epoch", 0) or 0) if args.resume_step_from_checkpoint else 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_log_path = os.path.join(args.output_dir, "train_log.jsonl")
    val_log_path = os.path.join(args.output_dir, "val_log.jsonl")

    stop_training = False
    for local_epoch in range(1, args.epochs + 1):
        epoch = start_epoch + local_epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        iterator = tqdm(train_loader, desc=f"epoch {epoch}", disable=not is_main_process())
        running_loss = 0.0
        running_count = 0
        for step_in_epoch, batch in enumerate(iterator, start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(pixel_values)
            losses = compute_loss(outputs, {"labels": labels, "masks": masks}, args)
            loss = losses["loss"] / args.grad_accumulation_steps
            loss.backward()
            running_loss += float(losses["loss"].detach().item())
            running_count += 1

            do_update = step_in_epoch % args.grad_accumulation_steps == 0 or step_in_epoch == len(train_loader)
            if not do_update:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if is_main_process() and (global_step % args.log_steps == 0 or global_step == start_global_step + 1):
                payload = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": running_loss / max(running_count, 1),
                    "lr": scheduler.get_last_lr()[0],
                    "head_lr": scheduler.get_last_lr()[-1],
                }
                save_jsonl(train_log_path, payload)
                iterator.set_postfix(loss=payload["loss"], step=global_step)
                running_loss = 0.0
                running_count = 0

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                metrics = evaluate(model, val_loader, device, args, step_name=f"step-{global_step}")
                if is_main_process():
                    payload = {"step": global_step, "epoch": epoch, **metrics}
                    save_jsonl(val_log_path, payload)
                    save_checkpoint(model, os.path.join(args.output_dir, "last"), args, metrics, global_step, epoch, optimizer, scheduler)
                    if metrics["image_f1"] >= best_image_f1:
                        best_image_f1 = metrics["image_f1"]
                        metrics["best_image_f1"] = best_image_f1
                        save_checkpoint(model, os.path.join(args.output_dir, "best-image-f1"), args, metrics, global_step, epoch, optimizer, scheduler)
                    if metrics["pixel_f1"] >= best_pixel_f1:
                        best_pixel_f1 = metrics["pixel_f1"]
                        metrics["best_pixel_f1"] = best_pixel_f1
                        save_checkpoint(model, os.path.join(args.output_dir, "best-pixel-f1"), args, metrics, global_step, epoch, optimizer, scheduler)
                    if metrics["loss"] <= best_val_loss:
                        best_val_loss = metrics["loss"]
                        metrics["best_val_loss"] = best_val_loss
                        save_checkpoint(model, os.path.join(args.output_dir, "best-val-loss"), args, metrics, global_step, epoch, optimizer, scheduler)
                if distributed:
                    dist.barrier()

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break

        if stop_training:
            break

        if args.eval_epochs > 0 and epoch % args.eval_epochs == 0:
            metrics = evaluate(model, val_loader, device, args, step_name=f"epoch-{epoch}")
            if is_main_process():
                payload = {"step": global_step, "epoch": epoch, **metrics}
                save_jsonl(val_log_path, payload)
                save_checkpoint(model, os.path.join(args.output_dir, "last"), args, metrics, global_step, epoch, optimizer, scheduler)
                if metrics["image_f1"] >= best_image_f1:
                    best_image_f1 = metrics["image_f1"]
                    metrics["best_image_f1"] = best_image_f1
                    save_checkpoint(model, os.path.join(args.output_dir, "best-image-f1"), args, metrics, global_step, epoch, optimizer, scheduler)
                if metrics["pixel_f1"] >= best_pixel_f1:
                    best_pixel_f1 = metrics["pixel_f1"]
                    metrics["best_pixel_f1"] = best_pixel_f1
                    save_checkpoint(model, os.path.join(args.output_dir, "best-pixel-f1"), args, metrics, global_step, epoch, optimizer, scheduler)
                if metrics["loss"] <= best_val_loss:
                    best_val_loss = metrics["loss"]
                    metrics["best_val_loss"] = best_val_loss
                    save_checkpoint(model, os.path.join(args.output_dir, "best-val-loss"), args, metrics, global_step, epoch, optimizer, scheduler)

        if args.save_epochs > 0 and epoch % args.save_epochs == 0:
            if is_main_process():
                save_checkpoint(
                    model,
                    os.path.join(args.output_dir, f"epoch-{epoch}"),
                    args,
                    {"best_image_f1": best_image_f1, "best_pixel_f1": best_pixel_f1, "best_val_loss": best_val_loss},
                    global_step,
                    epoch,
                    optimizer,
                    scheduler,
                )
                prune_epoch_checkpoints(args.output_dir, args.keep_last_epoch_checkpoints)
            if distributed:
                dist.barrier()

    if is_main_process():
        final_metrics = {"best_image_f1": best_image_f1, "best_pixel_f1": best_pixel_f1, "best_val_loss": best_val_loss}
        save_checkpoint(model, os.path.join(args.output_dir, "last"), args, final_metrics, global_step, epoch, optimizer, scheduler)
        print(f"[localizer] done step={global_step} metrics={final_metrics}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
