"""
models/dncnn.py — Step 1: The Physics Filter (DnCNN Denoiser)
==============================================================

A 17-layer Denoising Convolutional Neural Network that removes quantum
mottle (noise) from Ultra-Low-Dose CT scans via residual learning.

Architecture (per the blueprint):
    Layer 1:  Conv(1→64, 3×3) + ReLU
    Layers 2–16: Conv(64→64, 3×3) + BN + ReLU    (15 layers)
    Layer 17: Conv(64→1, 3×3)                     (no activation)

    Output = Input − Predicted_Noise   (residual learning)

The model predicts the NOISE pattern, then subtracts it from the
input to produce a clean image. This is the key insight of DnCNN.

Reference: Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning
           of Deep CNN for Image Denoising" (TIP 2017).

Shape: Input [B, 1, 512, 512] → Output [B, 1, 512, 512]
       Same spatial dimensions (no pooling, no upsampling).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DnCNNConfig


class DnCNN(nn.Module):
    """
    17-layer DnCNN for CT image denoising via residual learning.

    The network learns to predict the noise component. The clean output
    is obtained by subtracting the predicted noise from the input:

        clean = input - model(input)

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale CT).
    num_filters : int
        Width of intermediate convolutional layers (default 64).
    num_layers : int
        Total number of convolutional layers (default 17).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_filters: int = 64,
        num_layers: int = 17,
    ):
        super().__init__()
        assert num_layers >= 3, "DnCNN requires at least 3 layers"

        layers = []

        # Layer 1: Conv + ReLU (no batch norm)
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Layers 2 to (N-1): Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU(inplace=True))

        # Layer N: Conv (no BN, no activation) — predicts the noise residual
        layers.append(nn.Conv2d(num_filters, in_channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, 1, H, W]
            Noisy CT slice.

        Returns
        -------
        Tensor [B, 1, H, W]
            Denoised CT slice (input minus predicted noise).
        """
        noise = self.dncnn(x)
        return x - noise  # residual learning


# ═══════════════════════════════════════════════════════════════════════════
#  Factory & Loader
# ═══════════════════════════════════════════════════════════════════════════

def build_dncnn(cfg: Optional[DnCNNConfig] = None) -> DnCNN:
    """Build DnCNN from config."""
    if cfg is None:
        cfg = DnCNNConfig()
    model = DnCNN(
        in_channels=cfg.in_channels,
        num_filters=cfg.num_filters,
        num_layers=cfg.num_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dncnn] DnCNN built — {n_params:,} params  "
          f"(filters={cfg.num_filters}, layers={cfg.num_layers})")
    return model


def load_dncnn(
    weights_path: str,
    cfg: Optional[DnCNNConfig] = None,
    device: torch.device = torch.device("cpu"),
) -> DnCNN:
    """Build + load trained weights."""
    model = build_dncnn(cfg).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=False)

    # Handle various checkpoint dict formats
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[dncnn] Weights loaded from {weights_path}")
    return model


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_dncnn()
    x = torch.randn(1, 1, 512, 512)
    y = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}  (same size — denoising, no spatial change)")
    print(f"  Residual magnitude: {(x - y).abs().mean():.4f}")
