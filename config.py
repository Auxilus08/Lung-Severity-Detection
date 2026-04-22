"""
config.py — Centralised Configuration for the Lung Severity Detection Pipeline
================================================================================

Master Blueprint alignment:
    Step 0: Preprocessing    — HU window [-1250, +250], normalize, 512×512, → .npy
    Step 1: DnCNN            — 17-layer denoiser, residual learning
    Step 2: SRGAN            — PixelShuffle 2× super-resolution
    Step 3: Spatial Bridge   — .npy → .nii.gz with original affine/spacing
    Step 4: Segmentation     — 5-lobe anatomical mask + binary lesion mask
    Step 5: Severity Score   — 25-point per-lobe COVID-19 severity grading
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════
#  Path Configuration
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent

@dataclass
class PathConfig:
    """All filesystem paths used by the pipeline."""
    project_root: Path = PROJECT_ROOT

    # Data
    data_dir: Path       = PROJECT_ROOT / "data"
    train_images: Path   = PROJECT_ROOT / "data" / "tr_im.nii.gz"
    train_masks: Path    = PROJECT_ROOT / "data" / "tr_mask.nii.gz"
    test_images: Path    = PROJECT_ROOT / "data" / "val_im.nii.gz"

    # Intermediate (preprocessed .npy volumes)
    preprocessed_dir: Path = PROJECT_ROOT / "preprocessed"

    # Model weights
    dncnn_weights: Path    = PROJECT_ROOT / "weights" / "dncnn2d_epoch_0240.pth"
    sr_gan_weights: Path   = PROJECT_ROOT / "weights" / "best_sr_gan_model.pth"
    seg_weights: Path      = PROJECT_ROOT / "weights" / "best_segmentation_model.pth"

    # Outputs
    checkpoint_dir: Path   = PROJECT_ROOT / "checkpoints"
    prediction_dir: Path   = PROJECT_ROOT / "predictions"
    log_dir: Path          = PROJECT_ROOT / "logs"


# ═══════════════════════════════════════════════════════════════════════════
#  Step 0: Preprocessing — HU Windowing
# ═══════════════════════════════════════════════════════════════════════════
# Blueprint specifies -1250 to +250 HU (lung parenchyma window)
# This isolates lung tissue density and blacks out bones/air.

HU_MIN: float = -1250.0    # air / hyperinflated lung
HU_MAX: float = 250.0      # soft-tissue upper bound

SPATIAL_SIZE: int = 512     # slices zero-padded/resized to 512×512


# ═══════════════════════════════════════════════════════════════════════════
#  Step 4: Segmentation — Class Definitions
# ═══════════════════════════════════════════════════════════════════════════

# Task A — Anatomical Lobe Mapping (6 classes)
LOBE_CLASSES: int = 6
LOBE_NAMES = {
    0: "Background",
    1: "Right Upper Lobe (RUL)",
    2: "Right Middle Lobe (RML)",
    3: "Right Lower Lobe (RLL)",
    4: "Left Upper Lobe (LUL)",
    5: "Left Lower Lobe (LLL)",
}

# Task B — Binary Lesion Detection (2 classes)
LESION_CLASSES: int = 2
LESION_NAMES = {
    0: "Healthy",
    1: "Infected",
}

# Training masks from MedSeg have pathology-type labels:
#   0=BG, 1=GGO, 2=Consolidation, 3=Pleural Effusion
# For Task B, we collapse these: any label ≥ 1 → "Infected" (1)
MEDSEG_PATHOLOGY_NAMES = {
    0: "Background",
    1: "Ground-Glass Opacity (GGO)",
    2: "Consolidation",
    3: "Pleural Effusion",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Model Configurations
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DnCNNConfig:
    """Step 1: DnCNN denoiser configuration."""
    in_channels: int  = 1
    num_filters: int  = 64        # intermediate feature channels
    num_layers: int   = 17        # total convolutional layers (as per blueprint)


@dataclass
class SRConfig:
    """Step 2: SRGAN Generator configuration."""
    in_channels: int   = 1
    out_channels: int  = 1
    num_filters: int   = 64       # base filter width
    num_residual_blocks: int = 16 # residual blocks in the generator
    upscale_factor: int = 2       # 2× super-resolution (256→512)


@dataclass
class SegConfig:
    """Step 4: Multi-Task U-Net configuration."""
    in_channels: int        = 1
    pathology_classes: int  = 4   # BG, GGO, Consolidation, Pleural Effusion
    infection_classes: int  = 2   # Healthy, Infected

    # Encoder channel progression: enc1 → enc2 → enc3 → enc4 → bottleneck
    # Full:     (64, 128, 256, 512, 1024) — ~44M params
    # Low-VRAM: (32,  64, 128, 256,  512) — ~11M params
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024)
    low_vram: bool = False

    def __post_init__(self):
        if self.low_vram:
            self.encoder_channels = (32, 64, 128, 256, 512)


# ═══════════════════════════════════════════════════════════════════════════
#  Training Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """Training hyperparameters for the segmentation model."""
    # Image processing
    img_size: int          = 512     # blueprint: 512×512 standardised slices
    val_fraction: float    = 0.15

    # Optimisation
    epochs: int            = 150
    batch_size: int        = 4       # 512×512 needs smaller batch on 8 GB
    lr: float              = 1e-3
    weight_decay: float    = 1e-4
    grad_clip_max_norm: float = 1.0

    # Scheduler: CosineAnnealingWarmRestarts
    scheduler_T0: int      = 10
    scheduler_Tmult: int   = 2
    scheduler_eta_min: float = 1e-6

    # Multi-task loss weights
    task_a_weight: float   = 1.0     # lobe segmentation
    task_b_weight: float   = 2.0     # lesion detection (rarer, needs emphasis)

    # Per-class CE weights (inverse frequency)
    # Lobe classes are roughly balanced when present, BG dominates
    lobe_ce_weights: Tuple[float, ...] = (0.1, 1.0, 1.5, 1.0, 1.0, 1.0)
    # Lesion: BG dominates heavily → up-weight infected
    lesion_ce_weights: Tuple[float, ...] = (0.2, 5.0)

    # DataLoader
    num_workers: int       = 4
    pin_memory: bool       = True
    save_every_n_epochs: int = 10

    # Pipeline: prepend DnCNN + SRGAN during training?
    use_enhancement_pipeline: bool = False


# ═══════════════════════════════════════════════════════════════════════════
#  Inference Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InferConfig:
    """End-to-end inference settings."""
    img_size: int            = 512   # must match training resolution
    infer_batch_size: int    = 8     # slices per GPU batch
    use_dncnn: bool          = True  # Step 1: apply denoising
    use_sr: bool             = True  # Step 2: apply super-resolution
    use_amp: bool            = True  # mixed precision


# ═══════════════════════════════════════════════════════════════════════════
#  Step 5: Severity Scoring
# ═══════════════════════════════════════════════════════════════════════════

# 25-Point COVID-19 CT Severity Score (CO-RADS / CT-SS inspired)
# Each of the 5 lobes is scored 0–5 based on infection percentage:
SEVERITY_THRESHOLDS = {
    0: (0.0, 0.0),     # 0 pts: no involvement
    1: (0.001, 0.05),  # 1 pt:  < 5% involvement
    2: (0.05, 0.25),   # 2 pts: 5–25%
    3: (0.25, 0.50),   # 3 pts: 25–50%
    4: (0.50, 0.75),   # 4 pts: 50–75%
    5: (0.75, 1.01),   # 5 pts: > 75%
}

TRIAGE_GRADES = {
    "Mild":     (0, 7),    # 0–7 out of 25
    "Moderate": (8, 17),   # 8–17 out of 25
    "Severe":   (18, 25),  # 18–25 out of 25
}


# ═══════════════════════════════════════════════════════════════════════════
#  VRAM Quick Reference
# ═══════════════════════════════════════════════════════════════════════════
#
#  ┌────────────────┬──────────┬──────────┬────────────────────────┐
#  │ Model config   │ Img size │ Batch    │ Estimated VRAM         │
#  ├────────────────┼──────────┼──────────┼────────────────────────┤
#  │ low_vram=True  │ 512×512  │ 4        │ ~5 GB                  │
#  │ low_vram=False │ 512×512  │ 2        │ ~7 GB                  │
#  │ + DnCNN + SR   │ 512×512  │ same     │ +~1.5 GB over above    │
#  └────────────────┴──────────┴──────────┴────────────────────────┘
