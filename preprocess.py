"""
preprocess.py — Step 0: Data Standardization
==============================================

Translates raw hospital DICOM/NIfTI volumes into strict mathematical tensors.

Pipeline:
    1. HU Windowing: clip to [-1250, +250] → isolate lung tissue density
    2. Normalization: rescale to [0.0, 1.0]
    3. Spatial Standardization: zero-pad/resize to 512×512 (no distortion)
    4. Save as .npy arrays for fast downstream loading

Usage
-----
    # Preprocess a single NIfTI volume
    python preprocess.py --input data/tr_im.nii.gz --output_dir preprocessed/

    # Preprocess a directory of volumes
    python preprocess.py --input_dir data/ --output_dir preprocessed/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib

from config import HU_MIN, HU_MAX, SPATIAL_SIZE


# ═══════════════════════════════════════════════════════════════════════════
#  Core Preprocessing Functions
# ═══════════════════════════════════════════════════════════════════════════

def hu_lung_window(arr: np.ndarray) -> np.ndarray:
    """
    Step 0a: HU Windowing.
    Clips to [-1250, +250] and normalises to [0, 1].
    Isolates lung parenchyma density; blacks out bones and air.
    """
    clipped = np.clip(arr, HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / (HU_MAX - HU_MIN)


def spatial_standardize(
    slice_2d: np.ndarray,
    target_size: int = SPATIAL_SIZE,
    is_mask: bool = False,
) -> np.ndarray:
    """
    Step 0b: Spatial Standardization.
    Zero-pads to square, then resizes to target_size × target_size.
    Uses nearest-neighbour for masks, bilinear for images.

    The anatomy is never stretched or distorted — only padded and scaled.
    """
    import torch
    import torch.nn.functional as F

    h, w = slice_2d.shape
    # Zero-pad to square
    if h != w:
        diff = abs(h - w)
        pad_before = diff // 2
        pad_after = diff - pad_before
        if h > w:
            slice_2d = np.pad(slice_2d, ((0, 0), (pad_before, pad_after)),
                              mode='constant', constant_values=0)
        else:
            slice_2d = np.pad(slice_2d, ((pad_before, pad_after), (0, 0)),
                              mode='constant', constant_values=0)

    # Resize
    if slice_2d.shape[0] != target_size or slice_2d.shape[1] != target_size:
        t = torch.from_numpy(slice_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mode = "nearest" if is_mask else "bilinear"
        align = None if is_mask else False
        t = F.interpolate(t, size=(target_size, target_size), mode=mode, align_corners=align)
        slice_2d = t.squeeze().numpy()

    return slice_2d


def preprocess_volume(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
    target_size: int = SPATIAL_SIZE,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Preprocess an entire 3D volume slice-by-slice.

    Parameters
    ----------
    volume : ndarray [H, W, D]
        Raw HU values.
    mask : ndarray [H, W, D] | None
        Integer label mask (if available).
    target_size : int
        Output spatial size per slice.

    Returns
    -------
    processed_volume : ndarray [D, 1, target_size, target_size]  float32
    processed_mask   : ndarray [D, target_size, target_size]  int16 | None
    """
    n_slices = volume.shape[2]
    proc_vol = np.zeros((n_slices, 1, target_size, target_size), dtype=np.float32)
    proc_mask = None
    if mask is not None:
        proc_mask = np.zeros((n_slices, target_size, target_size), dtype=np.int16)

    for s in range(n_slices):
        # HU windowing + normalisation
        img = hu_lung_window(volume[:, :, s].astype(np.float32))
        # Spatial standardisation
        img = spatial_standardize(img, target_size, is_mask=False)
        proc_vol[s, 0] = img

        if mask is not None:
            m = mask[:, :, s].astype(np.float32)
            m = spatial_standardize(m, target_size, is_mask=True)
            proc_mask[s] = m.astype(np.int16)

    return proc_vol, proc_mask


# ═══════════════════════════════════════════════════════════════════════════
#  Metadata Extraction (for Step 3: Spatial Bridge)
# ═══════════════════════════════════════════════════════════════════════════

def extract_nifti_metadata(nifti_path: str | Path) -> dict:
    """
    Extract the hospital metadata needed for the Spatial Bridge (Step 3).

    Returns a dict with:
        affine:    4×4 transformation matrix (voxel → world coordinates)
        zooms:     voxel spacing (dx, dy, dz)
        shape:     original volume shape
        header:    full NIfTI header (pickled)
    """
    nii = nib.load(str(nifti_path))
    return {
        "affine": nii.affine.copy(),
        "zooms": tuple(nii.header.get_zooms()),
        "shape": nii.shape,
        "header": nii.header.copy(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    if args.input:
        files = [Path(args.input)]
    else:
        input_dir = Path(args.input_dir)
        files = sorted(input_dir.glob("*.nii.gz")) + sorted(input_dir.glob("*.nii"))

    for fpath in files:
        print(f"\n[preprocess] Processing: {fpath.name}")

        nii = nib.load(str(fpath))
        volume = nii.get_fdata(dtype=np.float32)
        print(f"  Original shape: {volume.shape}  HU=[{volume.min():.0f}, {volume.max():.0f}]")

        # Check for a matching mask file (same name in a masks/ dir)
        mask = None
        mask_path = fpath.parent / fpath.name.replace("_im", "_mask").replace("im.", "mask.")
        if mask_path.exists() and mask_path != fpath:
            mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
            print(f"  Found mask: {mask_path.name}  labels={np.unique(mask).astype(int).tolist()}")

        # Preprocess
        proc_vol, proc_mask = preprocess_volume(volume, mask, SPATIAL_SIZE)
        print(f"  Preprocessed: {proc_vol.shape}  range=[{proc_vol.min():.3f}, {proc_vol.max():.3f}]")

        # Save as .npy
        stem = fpath.name.replace(".nii.gz", "").replace(".nii", "")
        np.save(output_dir / f"{stem}.npy", proc_vol)
        print(f"  ✓ Saved: {stem}.npy")

        if proc_mask is not None:
            np.save(output_dir / f"{stem}_mask.npy", proc_mask)
            print(f"  ✓ Saved: {stem}_mask.npy")

        # Save metadata for Spatial Bridge (Step 3)
        meta = extract_nifti_metadata(fpath)
        np.save(output_dir / f"{stem}_meta.npy", meta, allow_pickle=True)
        print(f"  ✓ Saved: {stem}_meta.npy  (affine + spacing for Step 3)")


def parse_args():
    p = argparse.ArgumentParser("Step 0: Preprocessing — HU windowing + normalisation")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="Single NIfTI file")
    g.add_argument("--input_dir", help="Directory of NIfTI files")
    p.add_argument("--output_dir", default="./preprocessed")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
