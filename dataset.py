"""
dataset.py — 2D Slice Dataset for Multi-Task Segmentation Training
===================================================================

Loads preprocessed .npy slices (from Step 0) for training the
segmentation model (Step 4).

The MedSeg dataset masks have pathology-type labels:
    0=BG, 1=GGO, 2=Consolidation, 3=Pleural Effusion

The blueprint requires TWO output masks:
    Task A → lobe_mask   (6 classes: BG + 5 lobes)
    Task B → lesion_mask (2 classes: binary infected/healthy)

Since the MedSeg data does NOT contain lobe annotations, we handle
this with a flag:
    - If lobe masks are available → use them for Task A
    - If not → Task A is disabled (lobe_mask = zeros) and only
      Task B (lesion detection) is trained. The lobe decoder can
      later be fine-tuned on a lobe-annotated dataset.

For Task B, we collapse pathology labels: any ≥ 1 → Infected (1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    ToTensord,
)

from config import SPATIAL_SIZE, HU_MIN, HU_MAX


# ═══════════════════════════════════════════════════════════════════════════
#  HU windowing (for raw NIfTI loading path — .npy is already preprocessed)
# ═══════════════════════════════════════════════════════════════════════════

def hu_lung_window(arr: np.ndarray) -> np.ndarray:
    clipped = np.clip(arr, HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / (HU_MAX - HU_MIN)


# ═══════════════════════════════════════════════════════════════════════════
#  Augmentations
# ═══════════════════════════════════════════════════════════════════════════

_KEYS = ["image", "lobe_mask", "lesion_mask"]


def get_train_transforms() -> Compose:
    return Compose([
        RandFlipd(keys=_KEYS, prob=0.5, spatial_axis=0),
        RandFlipd(keys=_KEYS, prob=0.5, spatial_axis=1),
        RandRotate90d(keys=_KEYS, prob=0.5, max_k=3),
        RandAffined(
            keys=_KEYS, prob=0.3,
            rotate_range=(0.15,),
            scale_range=(0.1, 0.1),
            translate_range=(10, 10),
            mode=("bilinear", "nearest", "nearest"),
            padding_mode="zeros",
        ),
        ToTensord(keys=_KEYS),
    ])


def get_val_transforms() -> Compose:
    return Compose([ToTensord(keys=_KEYS)])


# ═══════════════════════════════════════════════════════════════════════════
#  Training Dataset
# ═══════════════════════════════════════════════════════════════════════════

class SegSliceDataset(Dataset):
    """
    Dataset for training the multi-task segmentation model.

    Supports two data loading paths:
        1. Preprocessed .npy (from Step 0) — faster, recommended
        2. Raw NIfTI files — direct loading with on-the-fly preprocessing

    Each __getitem__ returns:
        image      : FloatTensor [1, H, W]
        lobe_mask  : LongTensor  [H, W]   (0–5, or zeros if unavailable)
        lesion_mask: LongTensor  [H, W]   (0–1)
    """

    def __init__(
        self,
        image_data: np.ndarray,
        mask_data: np.ndarray,
        lobe_data: np.ndarray | None = None,
        is_train: bool = True,
        filter_empty: bool = True,
    ):
        """
        Parameters
        ----------
        image_data : ndarray [N, 1, H, W]  float32
            Preprocessed image slices.
        mask_data : ndarray [N, H, W]  int
            Pathology mask (0=BG, 1=GGO, 2=Consol, 3=PE).
        lobe_data : ndarray [N, H, W]  int | None
            Lobe mask (0=BG, 1–5=lobes). None if unavailable.
        is_train : bool
            Apply augmentations.
        filter_empty : bool
            Drop slices with no pathology (all-zero mask).
        """
        self.images = image_data
        self.masks = mask_data
        self.lobes = lobe_data
        self.has_lobes = lobe_data is not None

        # Build valid indices
        if filter_empty:
            self.indices = [
                i for i in range(len(self.images))
                if self.masks[i].max() > 0
            ]
            print(f"[dataset] Filtered: {len(self.indices)}/{len(self.images)} "
                  f"slices contain pathology")
        else:
            self.indices = list(range(len(self.images)))

        self.transforms = get_train_transforms() if is_train else get_val_transforms()

        if not self.has_lobes:
            print("[dataset] ⚠ No lobe annotations — Task A (lobe) will output zeros. "
                  "Only Task B (lesion) will be supervised.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]

        image = self.images[i].copy()                     # [1, H, W] float32
        pathology_mask = self.masks[i].copy()              # [H, W] int

        # Task A: Lobe mask
        if self.has_lobes:
            lobe_mask = self.lobes[i].copy().astype(np.int64)
        else:
            lobe_mask = np.zeros_like(pathology_mask, dtype=np.int64)

        # Task B: Binary lesion mask (any pathology ≥ 1 → infected)
        lesion_mask = (pathology_mask > 0).astype(np.int64)

        sample = {
            "image": image.astype(np.float32),
            "lobe_mask": lobe_mask,
            "lesion_mask": lesion_mask,
        }
        sample = self.transforms(sample)
        sample["lobe_mask"] = sample["lobe_mask"].long()
        sample["lesion_mask"] = sample["lesion_mask"].long()
        return sample


# ═══════════════════════════════════════════════════════════════════════════
#  Loaders from .npy and from NIfTI
# ═══════════════════════════════════════════════════════════════════════════

def load_from_npy(
    preprocessed_dir: str | Path,
    is_train: bool = True,
    filter_empty: bool = True,
) -> SegSliceDataset:
    """Load preprocessed .npy arrays from Step 0."""
    d = Path(preprocessed_dir)
    images = np.load(d / "tr_im.npy")
    masks = np.load(d / "tr_im_mask.npy") if (d / "tr_im_mask.npy").exists() \
        else np.load(d / "tr_mask.npy")

    # Check for lobe annotations
    lobe_path = d / "lobe_mask.npy"
    lobes = np.load(lobe_path) if lobe_path.exists() else None

    print(f"[dataset] Loaded .npy: images={images.shape}, masks={masks.shape}")
    return SegSliceDataset(images, masks, lobes, is_train, filter_empty)


def load_from_nifti(
    image_path: str | Path,
    mask_path: str | Path,
    lobe_path: str | Path | None = None,
    is_train: bool = True,
    filter_empty: bool = True,
) -> SegSliceDataset:
    """Load directly from NIfTI files with on-the-fly preprocessing."""
    import nibabel as nib

    print(f"[dataset] Loading NIfTI: {image_path}")
    img_vol = nib.load(str(image_path)).get_fdata(dtype=np.float32)
    mask_vol = nib.load(str(mask_path)).get_fdata(dtype=np.float32)

    n_slices = img_vol.shape[2]

    # Preprocess each slice
    from preprocess import spatial_standardize
    images = np.zeros((n_slices, 1, SPATIAL_SIZE, SPATIAL_SIZE), dtype=np.float32)
    masks = np.zeros((n_slices, SPATIAL_SIZE, SPATIAL_SIZE), dtype=np.int16)

    for s in range(n_slices):
        images[s, 0] = spatial_standardize(hu_lung_window(img_vol[:, :, s]), SPATIAL_SIZE)
        masks[s] = spatial_standardize(mask_vol[:, :, s], SPATIAL_SIZE, is_mask=True).astype(np.int16)

    lobes = None
    if lobe_path and Path(lobe_path).exists():
        lobe_vol = nib.load(str(lobe_path)).get_fdata(dtype=np.float32)
        lobes = np.zeros((n_slices, SPATIAL_SIZE, SPATIAL_SIZE), dtype=np.int16)
        for s in range(n_slices):
            lobes[s] = spatial_standardize(lobe_vol[:, :, s], SPATIAL_SIZE, is_mask=True).astype(np.int16)

    print(f"[dataset] Preprocessed: {n_slices} slices @ {SPATIAL_SIZE}×{SPATIAL_SIZE}")
    return SegSliceDataset(images, masks, lobes, is_train, filter_empty)


def build_train_val_datasets(
    data_dir: str | Path = "data",
    val_fraction: float = 0.15,
    seed: int = 42,
    use_npy: bool = False,
    preprocessed_dir: str | Path = "preprocessed",
):
    """
    Build train/val split from the MedSeg dataset.
    """
    if use_npy:
        full_ds = load_from_npy(preprocessed_dir, is_train=True, filter_empty=False)
    else:
        d = Path(data_dir)
        full_ds = load_from_nifti(
            d / "tr_im.nii.gz", d / "tr_mask.nii.gz",
            lobe_path=d / "lobe_mask.nii.gz",  # will gracefully handle if missing
            is_train=True, filter_empty=False,
        )

    n = len(full_ds)
    n_val = max(1, int(val_fraction * n))
    n_train = n - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"[dataset] Split: {n_train} train / {n_val} val (seed={seed})")
    return train_ds, val_ds
