from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DINOv3LocalizerConfig:
    dino_model_path: str
    image_size: int = 448
    image_height: int = 448
    image_width: int = 448
    decoder_dim: int = 256
    dropout: float = 0.1
    mask_head_type: str = "conv"
    mask2former_num_queries: int = 32
    mask2former_num_layers: int = 3
    mask2former_num_heads: int = 8
    mask2former_ffn_dim: int = 1024
    freeze_dino: bool = False
    unfreeze_last_n_layers: int = -1
    trust_remote_code: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            out_dim = output_dim if layer_idx == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < num_layers - 1:
                layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Mask2FormerStyleDecoder(nn.Module):
    """Lightweight Mask2Former-style query decoder for one binary mask logit map."""

    def __init__(
        self,
        in_channels: int,
        decoder_dim: int,
        num_queries: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.decoder_dim = int(decoder_dim)
        self.pixel_proj = nn.Sequential(
            nn.Conv2d(in_channels, decoder_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(32, decoder_dim), num_channels=decoder_dim),
            nn.GELU(),
            ConvBlock(decoder_dim, decoder_dim),
        )
        self.query_feat = nn.Embedding(num_queries, decoder_dim)
        self.query_embed = nn.Embedding(num_queries, decoder_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_norm = nn.LayerNorm(decoder_dim)
        self.mask_embed = MLP(decoder_dim, decoder_dim, decoder_dim, num_layers=3)
        self.query_mask_score = nn.Linear(decoder_dim, 1)

    def _position_embedding(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dim_t = torch.arange(self.decoder_dim // 4, device=device, dtype=dtype)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / max(self.decoder_dim // 2, 1))
        pos_x = xx[..., None] / dim_t
        pos_y = yy[..., None] / dim_t
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1)
        if pos.shape[-1] < self.decoder_dim:
            pos = F.pad(pos, (0, self.decoder_dim - pos.shape[-1]))
        elif pos.shape[-1] > self.decoder_dim:
            pos = pos[..., : self.decoder_dim]
        return pos.reshape(1, height * width, self.decoder_dim)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        pixel_features = self.pixel_proj(feature_map)
        batch_size, channels, height, width = pixel_features.shape
        memory = pixel_features.flatten(2).transpose(1, 2)
        memory = memory + self._position_embedding(height, width, memory.device, memory.dtype)

        query_feat = self.query_feat.weight.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=memory.dtype)
        query_pos = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=memory.dtype)
        queries = self.transformer_decoder(query_feat + query_pos, memory)
        queries = self.query_norm(queries)

        mask_embed = self.mask_embed(queries)
        query_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, pixel_features)
        query_scores = self.query_mask_score(queries).squeeze(-1)
        query_weights = torch.softmax(query_scores, dim=1)
        return torch.einsum("bq,bqhw->bhw", query_weights, query_masks).unsqueeze(1)


class DINOv3TamperLocalizer(nn.Module):
    """DINOv3 based binary tamper classifier and dense tamper localizer."""

    def __init__(self, config: DINOv3LocalizerConfig, torch_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        from transformers import AutoModel

        self.localizer_config = config
        self.dino = AutoModel.from_pretrained(
            config.dino_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
        )

        hidden_size = getattr(self.dino.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.dino.config, "embed_dim", None)
        if hidden_size is None:
            raise ValueError("Cannot infer DINO hidden size from config.")

        self.hidden_size = int(hidden_size)
        self.patch_size = int(getattr(self.dino.config, "patch_size", 16))
        self.num_register_tokens = int(getattr(self.dino.config, "num_register_tokens", 0) or 0)
        self.mask_head_type = config.mask_head_type.lower()

        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, config.decoder_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_dim, 1),
        )
        if self.mask_head_type == "conv":
            self.feature_proj = nn.Sequential(
                nn.Conv2d(self.hidden_size, config.decoder_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(config.decoder_dim),
                nn.GELU(),
            )
            self.mask_decoder = nn.Sequential(
                ConvBlock(config.decoder_dim, config.decoder_dim),
                nn.Conv2d(config.decoder_dim, config.decoder_dim // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(config.decoder_dim // 2),
                nn.GELU(),
                nn.Conv2d(config.decoder_dim // 2, 1, kernel_size=1),
            )
        elif self.mask_head_type == "mask2former":
            self.feature_proj = nn.Identity()
            self.mask_decoder = Mask2FormerStyleDecoder(
                in_channels=self.hidden_size,
                decoder_dim=config.decoder_dim,
                num_queries=config.mask2former_num_queries,
                num_layers=config.mask2former_num_layers,
                num_heads=config.mask2former_num_heads,
                ffn_dim=config.mask2former_ffn_dim,
                dropout=config.dropout,
            )
        else:
            raise ValueError(f"Unsupported mask_head_type: {config.mask_head_type}")

        self.set_dino_trainability(config.freeze_dino, config.unfreeze_last_n_layers)

    def set_dino_trainability(self, freeze_dino: bool, unfreeze_last_n_layers: int = -1) -> None:
        for param in self.dino.parameters():
            param.requires_grad = not freeze_dino

        if freeze_dino and unfreeze_last_n_layers > 0:
            layers = self._get_dino_layers()
            for layer in layers[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # DINO's mask token is used for masked image modeling pretraining, not for
        # normal image classification/segmentation forward passes. Keeping it
        # trainable makes DDP fail because it never contributes to the loss.
        for name, param in self.dino.named_parameters():
            if "mask_token" in name:
                param.requires_grad = False

    def _get_dino_layers(self) -> list[nn.Module]:
        for attr in ("encoder", "layers", "blocks"):
            module = getattr(self.dino, attr, None)
            if module is None:
                continue
            if hasattr(module, "layer"):
                return list(module.layer)
            if hasattr(module, "layers"):
                return list(module.layers)
            if isinstance(module, (nn.ModuleList, list, tuple)):
                return list(module)
        if hasattr(self.dino, "layer"):
            return list(self.dino.layer)
        return []

    def _split_tokens(self, last_hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_token = last_hidden_state[:, 0]
        prefix_len = 1 + self.num_register_tokens
        patch_tokens = last_hidden_state[:, prefix_len:]
        if patch_tokens.shape[1] == 0:
            patch_tokens = last_hidden_state[:, 1:]
        return cls_token, patch_tokens

    def _tokens_to_feature_map(self, patch_tokens: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
        batch_size, num_patches, hidden_size = patch_tokens.shape
        height = max(image_hw[0] // self.patch_size, 1)
        width = max(image_hw[1] // self.patch_size, 1)
        expected = height * width
        if expected != num_patches:
            side = int(num_patches**0.5)
            if side * side != num_patches:
                raise RuntimeError(
                    f"Cannot reshape {num_patches} DINO patch tokens into a feature map "
                    f"for image size {image_hw} and patch size {self.patch_size}."
                )
            height = width = side
        return patch_tokens.transpose(1, 2).reshape(batch_size, hidden_size, height, width)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.dino(pixel_values=pixel_values, output_hidden_states=False)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            raise RuntimeError("DINO output does not contain hidden states.")

        cls_token, patch_tokens = self._split_tokens(hidden_states)
        pooled_patch = patch_tokens.mean(dim=1)
        head_dtype = next(self.cls_head.parameters()).dtype
        cls_features = torch.cat([cls_token, pooled_patch], dim=-1).to(dtype=head_dtype)
        cls_logits = self.cls_head(cls_features).squeeze(-1)

        feature_map = self._tokens_to_feature_map(
            patch_tokens,
            image_hw=(pixel_values.shape[-2], pixel_values.shape[-1]),
        )
        feature_map = feature_map.to(dtype=next(self.mask_decoder.parameters()).dtype)
        mask_logits = self.mask_decoder(self.feature_proj(feature_map))
        mask_logits = F.interpolate(
            mask_logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return {
            "cls_logits": cls_logits,
            "mask_logits": mask_logits,
        }

    def trainable_parameter_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {
            "dino": 0,
            "classification_head": 0,
            "mask_head": 0,
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            count = param.numel()
            if name.startswith("dino."):
                summary["dino"] += count
            elif name.startswith("cls_head."):
                summary["classification_head"] += count
            else:
                summary["mask_head"] += count
        summary["total"] = sum(summary.values())
        return summary
