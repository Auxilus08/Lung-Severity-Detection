"""
enhance_training_data.py — Pre-generate DnCNN + SRGAN enhanced training data
===============================================================================

Runs the DnCNN denoiser + SRGAN super-resolver on the training NIfTI data
and saves the enhanced slices as .npy for fast segmentation training.

This avoids loading all 3 models simultaneously during training (OOM on 4GB GPUs).

Usage
-----
    python enhance_training_data.py --data_dir ./data --output_dir ./enhanced_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.amp import autocast

from config import (
    HU_MIN, HU_MAX, SPATIAL_SIZE,
    DnCNNConfig, SRConfig,
)
from models.dncnn import load_dncnn
from models.sr_gan import load_sr_model


def hu_lung_window(arr: np.ndarray) -> np.ndarray:
    clipped = np.clip(arr, HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / (HU_MAX - HU_MIN)


def preprocess_slice(raw_slice: np.ndarray, target: int = SPATIAL_SIZE) -> np.ndarray:
    s = hu_lung_window(raw_slice.astype(np.float32))
    h, w = s.shape
    if h != w:
        d = abs(h - w)
        p1, p2 = d // 2, d - d // 2
        s = np.pad(s, ((0, 0), (p1, p2)) if h > w else ((p1, p2), (0, 0)),
                   mode='constant')
    if s.shape[0] != target:
        t = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(target, target), mode="bilinear", align_corners=False)
        s = t.squeeze().numpy()
    return s


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[enhance] Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models to CPU first
    cpu = torch.device("cpu")

    dncnn = None
    if not args.no_dncnn:
        print("[enhance] Loading DnCNN...")
        dncnn = load_dncnn(args.dncnn_weights, DnCNNConfig(), cpu)

    srgan = None
    if not args.no_sr:
        print("[enhance] Loading SRGAN...")
        srgan = load_sr_model(args.sr_weights, SRConfig(), cpu)

    sr_input_size = args.sr_input_size

    # Process each NIfTI file
    data_dir = Path(args.data_dir)
    for nii_name in ["tr_im.nii.gz", "val_im.nii.gz"]:
        nii_path = data_dir / nii_name
        if not nii_path.exists():
            print(f"[enhance] Skipping {nii_name} (not found)")
            continue

        print(f"\n[enhance] Processing: {nii_name}")
        nii = nib.load(str(nii_path))
        vol = nii.get_fdata(dtype=np.float32)
        n_slices = vol.shape[2]
        ts = SPATIAL_SIZE

        # Preprocess all slices
        enhanced = torch.zeros(n_slices, 1, ts, ts)
        for s in range(n_slices):
            enhanced[s, 0] = torch.from_numpy(preprocess_slice(vol[:, :, s], ts))

        # Run DnCNN (one slice at a time, model on GPU)
        if dncnn is not None:
            print(f"  Running DnCNN on {n_slices} slices...")
            dncnn.to(device)
            with torch.no_grad():
                for s in range(n_slices):
                    x = enhanced[s:s+1].to(device)
                    with autocast('cuda', enabled=(device.type == "cuda")):
                        x = dncnn(x).clamp(0, 1)
                    enhanced[s] = x.cpu()
            dncnn.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print("  DnCNN done ✓")

        # Run SRGAN (one slice at a time, model on GPU)
        if srgan is not None:
            print(f"  Running SRGAN on {n_slices} slices...")
            srgan.to(device)
            with torch.no_grad():
                for s in range(n_slices):
                    x = enhanced[s:s+1].to(device)
                    with autocast('cuda', enabled=(device.type == "cuda")):
                        x_low = F.interpolate(
                            x, size=(sr_input_size, sr_input_size),
                            mode="bilinear", align_corners=False
                        )
                        x = srgan(x_low).clamp(0, 1)
                        if x.shape[-1] != ts or x.shape[-2] != ts:
                            x = F.interpolate(x, size=(ts, ts),
                                              mode="bilinear", align_corners=False)
                    enhanced[s] = x.cpu()
            srgan.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print("  SRGAN done ✓")

        # Save as .npy [N, 1, H, W] — matches what SegSliceDataset expects
        out_name = nii_name.replace(".nii.gz", ".npy")
        result = enhanced.numpy()
        np.save(out_dir / out_name, result)
        print(f"  ✓ Saved: {out_dir / out_name}  shape={result.shape}  "
              f"range=[{result.min():.3f}, {result.max():.3f}]")

    # Copy mask files as-is (masks don't go through enhancement)
    for mask_name in ["tr_mask.nii.gz"]:
        mask_path = data_dir / mask_name
        if mask_path.exists():
            print(f"\n[enhance] Processing mask: {mask_name}")
            nii = nib.load(str(mask_path))
            mask_vol = nii.get_fdata(dtype=np.float32)
            n_slices = mask_vol.shape[2]

            from preprocess import spatial_standardize
            masks = np.zeros((n_slices, ts, ts), dtype=np.int16)
            for s in range(n_slices):
                masks[s] = spatial_standardize(
                    mask_vol[:, :, s], ts, is_mask=True
                ).astype(np.int16)

            out_name = mask_name.replace(".nii.gz", "_mask.npy").replace("tr_mask", "tr_im")
            np.save(out_dir / out_name, masks)
            print(f"  ✓ Saved: {out_dir / out_name}  shape={masks.shape}  "
                  f"labels={np.unique(masks).tolist()}")

    print(f"\n[enhance] Done! Enhanced data saved to {out_dir}/")
    print(f"[enhance] Train with:  python train.py --use_npy --preprocessed_dir {out_dir}")


def parse_args():
    p = argparse.ArgumentParser("Pre-generate enhanced training data")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="./enhanced_data")
    p.add_argument("--dncnn_weights", default="weights/dncnn2d_epoch_0240.pth")
    p.add_argument("--sr_weights", default="weights/best_sr_gan_model.pth")
    p.add_argument("--no_dncnn", action="store_true")
    p.add_argument("--no_sr", action="store_true")
    p.add_argument("--sr_input_size", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
