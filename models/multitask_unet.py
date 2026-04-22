"""
models/multitask_unet.py — Step 4: The Diagnostician (Multi-Task 2D U-Net)
============================================================================

Shared encoder + two parallel decoder heads:

    Decoder A → lobe_mask   : 6 classes (BG + 5 anatomical lobes)
    Decoder B → lesion_mask : 2 classes (Healthy / Infected)

Architecture
------------
             ┌─────────────── Shared Encoder ───────────────┐
  Input ──► │ Enc1 → Enc2 → Enc3 → Enc4 → Bottleneck       │
             └───────────────┬──────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌── Decoder A (Lobes) ────┐   ┌── Decoder B (Lesions) ──┐
     │  4 UpBlocks → 1×1 Head │   │  4 UpBlocks → 1×1 Head  │
     └─────────────────────────┘   └──────────────────────────┘
       Output: [B, 6, H, W]         Output: [B, 2, H, W]

Design: GroupNorm (stable at small batches), bilinear upsampling (no checkerboard).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SegConfig


# ═══════════════════════════════════════════════════════════════════════════
#  Building Blocks
# ═══════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Two 3×3 convolutions, each followed by GroupNorm + LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = nn.functional.pad(
                x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]]
            )
        return self.conv(torch.cat([x, skip], dim=1))


# ═══════════════════════════════════════════════════════════════════════════
#  Shared Encoder
# ═══════════════════════════════════════════════════════════════════════════

class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: Tuple[int, ...]):
        super().__init__()
        c = channels
        self.enc1 = ConvBlock(in_channels, c[0])
        self.enc2 = DownBlock(c[0], c[1])
        self.enc3 = DownBlock(c[1], c[2])
        self.enc4 = DownBlock(c[2], c[3])
        self.bottleneck = DownBlock(c[3], c[4])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bn = self.bottleneck(e4)
        return [e1, e2, e3, e4], bn


# ═══════════════════════════════════════════════════════════════════════════
#  Decoder
# ═══════════════════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    def __init__(self, num_classes: int, channels: Tuple[int, ...]):
        super().__init__()
        c = channels
        self.up4 = UpBlock(c[4], c[3], c[3])
        self.up3 = UpBlock(c[3], c[2], c[2])
        self.up2 = UpBlock(c[2], c[1], c[1])
        self.up1 = UpBlock(c[1], c[0], c[0])
        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1)

    def forward(self, skips: List[torch.Tensor], bottleneck: torch.Tensor) -> torch.Tensor:
        x = self.up4(bottleneck, skips[3])
        x = self.up3(x, skips[2])
        x = self.up2(x, skips[1])
        x = self.up1(x, skips[0])
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-Task U-Net
# ═══════════════════════════════════════════════════════════════════════════

class MultiTaskUNet2D(nn.Module):
    """
    2D U-Net with shared encoder and two decoder heads:
        Head A → pathology segmentation (4 classes: BG/GGO/Consolidation/PE)
        Head B → infection detection    (2 classes: healthy / infected)

    The checkpoint was trained with these exact names:
        decoder_pathology  (4 classes)
        decoder_infection   (2 classes)

    Returns (pathology_logits, infection_logits).
    """

    def __init__(
        self,
        in_channels: int = 1,
        pathology_classes: int = 4,
        infection_classes: int = 2,
        encoder_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
    ):
        super().__init__()
        self.encoder = SharedEncoder(in_channels, encoder_channels)
        self.decoder_pathology = Decoder(pathology_classes, encoder_channels)
        self.decoder_infection = Decoder(infection_classes, encoder_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skips, bottleneck = self.encoder(x)
        pathology_logits = self.decoder_pathology(skips, bottleneck)
        infection_logits = self.decoder_infection(skips, bottleneck)
        return pathology_logits, infection_logits


# ═══════════════════════════════════════════════════════════════════════════
#  Factory & Loader
# ═══════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_seg_model(cfg: Optional[SegConfig] = None) -> MultiTaskUNet2D:
    if cfg is None:
        cfg = SegConfig()
    model = MultiTaskUNet2D(
        in_channels=cfg.in_channels,
        pathology_classes=cfg.pathology_classes,
        infection_classes=cfg.infection_classes,
        encoder_channels=cfg.encoder_channels,
    )
    print(f"[seg] MultiTaskUNet2D — {count_parameters(model):,} params  "
          f"(channels={cfg.encoder_channels})")
    return model


def load_seg_model(
    weights_path: str,
    cfg: Optional[SegConfig] = None,
    device: torch.device = torch.device("cpu"),
) -> MultiTaskUNet2D:
    model = build_seg_model(cfg).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[seg] Weights loaded from {weights_path}")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for low in [False, True]:
        cfg = SegConfig(low_vram=low)
        m = build_seg_model(cfg).to(device)
        x = torch.randn(1, 1, 512, 512, device=device)
        lobes, lesions = m(x)
        print(f"  low_vram={low}: input={x.shape} → lobes={lobes.shape}, lesions={lesions.shape}\n")
