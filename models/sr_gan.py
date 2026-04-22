"""
models/sr_gan.py — SRGAN Generator for 2D CT Slice Super-Resolution
=====================================================================

Architecture matched to the trained checkpoint (checkpoint_epoch0060.pth):
    Input  → Conv 9×9 → PReLU             (self.initial_conv)
    → 20 × Residual Block                 (self.res_blocks)
    → Conv 3×3 → BN + skip               (self.post_res_conv)
    → UpsampleBlock via PixelShuffle 2×   (self.upsample)
    → Conv 9×9 → output                   (self.final_conv)

Config: in_channels=1, out_channels=1, num_filters=64, num_residual_blocks=20, upscale_factor=2

Reference: Ledig et al., "Photo-Realistic Single Image Super-Resolution
           Using a Generative Adversarial Network" (CVPR 2017).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SRConfig


# ═══════════════════════════════════════════════════════════════════════════
#  Building Blocks
# ═══════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Standard residual block: Conv → BN → PReLU → Conv → BN + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """
    Sub-pixel convolution (PixelShuffle) upsampling block.
    Increases spatial resolution by 2×.
    """

    def __init__(self, in_channels: int, scale_factor: int = 2):
        super().__init__()
        out_channels = in_channels * (scale_factor ** 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ═══════════════════════════════════════════════════════════════════════════
#  SRGAN Generator (SRResNet)
# ═══════════════════════════════════════════════════════════════════════════

class SRGenerator(nn.Module):
    """
    SRResNet-style generator for single-channel medical image super-resolution.

    Variable names match the trained checkpoint exactly:
        self.initial_conv   → checkpoint keys: initial_conv.*
        self.res_blocks     → checkpoint keys: res_blocks.*
        self.post_res_conv  → checkpoint keys: post_res_conv.*
        self.upsample       → checkpoint keys: upsample.block.*
        self.final_conv     → checkpoint keys: final_conv.*

    Parameters
    ----------
    in_channels : int
        Input channels (1 for grayscale CT).
    out_channels : int
        Output channels (1 for grayscale CT).
    num_filters : int
        Base number of convolutional filters (default 64).
    num_residual_blocks : int
        Number of residual blocks (default 20 for this checkpoint).
    upscale_factor : int
        Super-resolution factor (default 2).

    Shape
    -----
    Input:  [B, 1, H, W]
    Output: [B, 1, H*upscale_factor, W*upscale_factor]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_filters: int = 64,
        num_residual_blocks: int = 20,
        upscale_factor: int = 2,
    ):
        super().__init__()
        assert upscale_factor in (2, 4, 8), "upscale_factor must be 2, 4, or 8"

        # ── Initial feature extraction ───────────────────────────────────
        # Checkpoint keys: initial_conv.0.weight, initial_conv.0.bias, initial_conv.1.weight
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # ── Residual trunk ───────────────────────────────────────────────
        # Checkpoint keys: res_blocks.{0..19}.block.*
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )
        # Checkpoint keys: post_res_conv.0.weight, post_res_conv.1.*
        self.post_res_conv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
        )

        # ── Upsampling (single block for 2×) ─────────────────────────────
        # Checkpoint keys: upsample.block.{0,2}.*
        # For 2× upscale: single UpsampleBlock (not wrapped in Sequential)
        # For 4×: would need two blocks
        if upscale_factor == 2:
            self.upsample = UpsampleBlock(num_filters, scale_factor=2)
        else:
            n_upsample = int(math.log2(upscale_factor))
            self.upsample = nn.Sequential(
                *[UpsampleBlock(num_filters, scale_factor=2) for _ in range(n_upsample)]
            )

        # ── Final reconstruction ─────────────────────────────────────────
        # Checkpoint keys: final_conv.0.weight, final_conv.0.bias
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_filters, out_channels, kernel_size=9, padding=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial_features = self.initial_conv(x)
        trunk = self.res_blocks(initial_features)
        trunk = self.post_res_conv(trunk) + initial_features   # global skip
        upsampled = self.upsample(trunk)
        return self.final_conv(upsampled)


# ═══════════════════════════════════════════════════════════════════════════
#  Factory & Loader
# ═══════════════════════════════════════════════════════════════════════════

def build_sr_model(cfg: Optional[SRConfig] = None) -> SRGenerator:
    """Build the SRGAN generator from config."""
    if cfg is None:
        cfg = SRConfig()
    model = SRGenerator(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        num_filters=cfg.num_filters,
        num_residual_blocks=cfg.num_residual_blocks,
        upscale_factor=cfg.upscale_factor,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sr_gan] SRGenerator built — {n_params:,} params  "
          f"(filters={cfg.num_filters}, blocks={cfg.num_residual_blocks}, "
          f"scale={cfg.upscale_factor}×)")
    return model


def load_sr_model(
    weights_path: str,
    cfg: Optional[SRConfig] = None,
    device: torch.device = torch.device("cpu"),
) -> SRGenerator:
    """Build + load trained weights."""
    model = build_sr_model(cfg).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=False)

    # Handle checkpoints that wrap state_dict inside a dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "generator_state_dict" in state:
        state = state["generator_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[sr_gan] Weights loaded from {weights_path}")
    return model


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_sr_model()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}  (2× super-resolved)")
