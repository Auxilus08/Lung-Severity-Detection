"""
pipeline.py — End-to-End Lung Severity Detection Pipeline (Steps 0–5)
======================================================================

IMPORTANT — DATA DOMAIN NOTES:
    • DnCNN & SRGAN were trained on .npy arrays derived from DICOM files.
    • The Segmentation model was trained on .nii.gz (MedSeg NIfTI dataset).
    • These models expect THE SAME [0, 1] normalised intensity range,
      but the pipeline must respect each model's spatial expectations:

        DnCNN:   [B, 1, 512, 512] → [B, 1, 512, 512]  (same size)
        SRGAN:   [B, 1, 256, 256] → [B, 1, 512, 512]  (2× upscale)
        SegNet:  [B, 1, 512, 512] → [B, 6, 512, 512] + [B, 2, 512, 512]

    The critical handoff between SRGAN and SegNet:
        DnCNN outputs 512×512 → DOWNSAMPLE to 256×256 → SRGAN → 512×512
        (SRGAN was trained via self-degradation: 512→256→512)

     ┌───────────────────── THE FULL PIPELINE ─────────────────────────┐
     │                                                                  │
     │  Step 0: Raw CT → HU Window [-1250,+250] → [0,1] → 512×512    │
     │  Step 1: DnCNN(512→512) — subtract predicted noise              │
     │  Step 2: ↓256 → SRGAN(256→512) — hallucinate sharp textures    │
     │  Step 3: inject affine + spacing → .nii.gz                      │
     │  Step 4: SegNet(512→512) → lobe_mask + lesion_mask              │
     │  Step 5: voxel math → 25-point severity score                   │
     │                                                                  │
     └──────────────────────────────────────────────────────────────────┘

Usage
-----
    # Full pipeline (all 5 steps)
    python pipeline.py --input patient_scan.nii.gz --output_dir results/

    # Skip denoising (no DnCNN)
    python pipeline.py --input scan.nii.gz --output_dir results/ --no_dncnn

    # Skip all enhancement (just segmentation)
    python pipeline.py --input scan.nii.gz --output_dir results/ --no_dncnn --no_sr

    # Batch processing
    python pipeline.py --input_dir scans/ --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from config import (
    HU_MIN, HU_MAX, SPATIAL_SIZE,
    DnCNNConfig, SRConfig, SegConfig,
    LOBE_NAMES, LESION_NAMES,
    SEVERITY_THRESHOLDS, TRIAGE_GRADES,
)
from models.dncnn import load_dncnn
from models.sr_gan import load_sr_model
from models.multitask_unet import load_seg_model


# ═══════════════════════════════════════════════════════════════════════════
#  Step 0: Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def hu_lung_window(arr: np.ndarray) -> np.ndarray:
    """Clip to [-1250, +250] HU and normalise to [0, 1]."""
    clipped = np.clip(arr, HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / (HU_MAX - HU_MIN)


def preprocess_slice(raw_slice: np.ndarray, target: int = SPATIAL_SIZE) -> np.ndarray:
    """HU window + normalise + zero-pad to square + resize to target."""
    s = hu_lung_window(raw_slice.astype(np.float32))
    h, w = s.shape
    # Zero-pad to square (no stretching)
    if h != w:
        d = abs(h - w)
        p1, p2 = d // 2, d - d // 2
        s = np.pad(s, ((0, 0), (p1, p2)) if h > w else ((p1, p2), (0, 0)),
                   mode='constant')
    # Resize to target
    if s.shape[0] != target:
        t = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(target, target), mode="bilinear", align_corners=False)
        s = t.squeeze().numpy()
    return s


# ═══════════════════════════════════════════════════════════════════════════
#  Steps 1–4: The Inference Engine
# ═══════════════════════════════════════════════════════════════════════════

class LungSeverityPipeline:
    """
    End-to-end inference engine chaining all neural network steps.

    Handles the domain bridge between:
        • DnCNN/SRGAN domain   (trained on .npy from DICOM)
        • Segmentation domain  (trained on .nii.gz from MedSeg)

    Both domains use [0, 1] normalised intensities at 512×512.
    The key detail is the SRGAN handoff:
        DnCNN → 512×512 clean blurry
        ↓ downsample to 256×256
        SRGAN(256→512) → sharp textures
        → into Segmentation at 512×512

    Parameters
    ----------
    dncnn : nn.Module | None   — Step 1 denoiser (512→512)
    srgan : nn.Module | None   — Step 2 super-resolver (256→512)
    seg_model : nn.Module      — Step 4 segmentation (512→512)
    device : torch.device
    target_size : int          — final spatial resolution (512)
    sr_input_size : int        — SRGAN input size (256 — per self-degradation training)
    batch_size : int           — slices per GPU batch
    """

    def __init__(
        self,
        dncnn: nn.Module | None,
        srgan: nn.Module | None,
        seg_model: nn.Module,
        device: torch.device,
        target_size: int = SPATIAL_SIZE,
        sr_input_size: int = 256,
        batch_size: int = 8,
    ):
        self.dncnn = dncnn
        self.srgan = srgan
        self.seg_model = seg_model
        self.device = device
        self.target_size = target_size      # 512
        self.sr_input_size = sr_input_size  # 256 (SRGAN was trained on 256→512)
        self.batch_size = batch_size

    @torch.no_grad()
    def predict_volume(self, volume_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Steps 0–4 slice-by-slice on a full 3D volume.

        Returns
        -------
        pathology_mask : ndarray [H, W, D] int16 (0=BG, 1=GGO, 2=Consolidation, 3=PE)
        infection_mask : ndarray [H, W, D] int16 (0=Healthy, 1=Infected)
        """
        orig_h, orig_w, n_slices = volume_3d.shape
        ts = self.target_size  # 512

        pathology_3d = np.zeros((orig_h, orig_w, n_slices), dtype=np.int16)
        infection_3d = np.zeros((orig_h, orig_w, n_slices), dtype=np.int16)

        for start in range(0, n_slices, self.batch_size):
            end = min(start + self.batch_size, n_slices)

            # ── Step 0: Preprocess each slice ────────────────────────
            batch = []
            for s in range(start, end):
                processed = preprocess_slice(volume_3d[:, :, s], ts)
                batch.append(torch.from_numpy(processed).unsqueeze(0))  # [1, 512, 512]

            x = torch.stack(batch, dim=0).to(self.device)  # [B, 1, 512, 512]

            with autocast('cuda', enabled=(self.device.type == "cuda")):

                # ── Step 1: DnCNN denoising (512→512) ────────────────
                # Input: noisy [B, 1, 512, 512]
                # Output: clean but blurry [B, 1, 512, 512]
                if self.dncnn is not None:
                    x = self.dncnn(x)
                    x = torch.clamp(x, 0.0, 1.0)  # keep in [0,1]

                # ── Step 2: SRGAN super-resolution (256→512) ─────────
                # The SRGAN was trained via self-degradation:
                #   training: degrade 512→256, then SRGAN learns 256→512
                #   inference: downsample DnCNN output → SRGAN → sharp 512
                if self.srgan is not None:
                    # Downsample to what SRGAN expects (256×256)
                    x_low = F.interpolate(
                        x, size=(self.sr_input_size, self.sr_input_size),
                        mode="bilinear", align_corners=False
                    )
                    # SRGAN: 256 → 512  (PixelShuffle 2×)
                    x = self.srgan(x_low)
                    x = torch.clamp(x, 0.0, 1.0)

                    # Safety: ensure output is exactly target_size
                    if x.shape[-1] != ts or x.shape[-2] != ts:
                        x = F.interpolate(x, size=(ts, ts),
                                          mode="bilinear", align_corners=False)

                # ── Step 4: Segmentation (512→512) ───────────────────
                pathology_logits, infection_logits = self.seg_model(x)

            # Argmax → class labels
            pathology_preds = pathology_logits.argmax(dim=1)   # [B, 512, 512]
            infection_preds = infection_logits.argmax(dim=1)   # [B, 512, 512]

            # Resize predictions back to original volume spatial dims
            pathology_full = F.interpolate(
                pathology_preds.unsqueeze(1).float(), (orig_h, orig_w), mode="nearest"
            ).squeeze(1).cpu().numpy().astype(np.int16)

            infection_full = F.interpolate(
                infection_preds.unsqueeze(1).float(), (orig_h, orig_w), mode="nearest"
            ).squeeze(1).cpu().numpy().astype(np.int16)

            for i, s in enumerate(range(start, end)):
                pathology_3d[:, :, s] = pathology_full[i]
                infection_3d[:, :, s] = infection_full[i]

        return pathology_3d, infection_3d


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3: Spatial Bridge — Save NIfTI with Metadata
# ═══════════════════════════════════════════════════════════════════════════

def save_mask_nifti(mask: np.ndarray, ref_nii: nib.Nifti1Image, path: str | Path):
    """Save 3D mask as NIfTI preserving original affine + spacing."""
    hdr = ref_nii.header.copy()
    hdr.set_data_dtype(np.int16)
    img = nib.Nifti1Image(mask.astype(np.int16), ref_nii.affine.copy(), hdr)
    img.header.set_zooms(ref_nii.header.get_zooms())
    nib.save(img, str(path))
    print(f"  ✓ {path}  shape={mask.shape}  labels={np.unique(mask).tolist()}")


# ═══════════════════════════════════════════════════════════════════════════
#  Step 5: The 25-Point Severity Calculation
# ═══════════════════════════════════════════════════════════════════════════

def score_lobe(infection_pct: float) -> int:
    """Assign 0–5 score based on infection percentage in one lobe."""
    for score, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= infection_pct < hi:
            return score
    return 5  # >75%


def compute_severity_score(pathology_mask: np.ndarray, infection_mask: np.ndarray) -> dict:
    """
    Step 5: Severity scoring.

    With the current checkpoint (no lobe annotations), we compute:
      - Overall infection percentage from the infection head
      - Per-pathology-type breakdown from the pathology head
      - A simplified severity score (0–5 scale) based on total infection

    When lobe-annotated models are available, this can be upgraded to
    the full 25-point per-lobe scoring.
    """
    total_voxels = int(infection_mask.size)
    infected_voxels = int((infection_mask == 1).sum())
    infection_pct = infected_voxels / max(total_voxels, 1)

    # Per-pathology breakdown
    pathology_breakdown = {}
    pathology_names = {1: "Ground-Glass Opacity (GGO)", 2: "Consolidation", 3: "Pleural Effusion"}
    for pid, pname in pathology_names.items():
        count = int((pathology_mask == pid).sum())
        pathology_breakdown[pname] = {
            "voxels": count,
            "percentage": round(100 * count / max(total_voxels, 1), 2),
        }

    # Overall severity score (0–5 using same thresholds)
    overall_score = score_lobe(infection_pct)

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_voxels": total_voxels,
        "infected_voxels": infected_voxels,
        "infection_pct": round(infection_pct * 100, 2),
        "pathology_breakdown": pathology_breakdown,
        "overall_score": overall_score,
        "overall_score_max": 5,
        "lobes": {},
        "total_score": 0,
        "max_score": 25,
        "triage": "Unknown",
    }

    total_score = 0

    for lobe_id in range(1, 6):
        lobe_name = LOBE_NAMES[lobe_id]
        # Note: lobe masks not available from current checkpoint
        # Placeholder for future lobe-annotated model
        report["lobes"][lobe_name] = {
            "lobe_id": lobe_id,
            "total_voxels": 0,
            "infected_voxels": 0,
            "infection_pct": 0.0,
            "score": 0,
            "note": "Lobe segmentation not available (needs lobe-annotated model)"
        }

    # Use overall infection score × 5 as proxy for total (same score per "lobe")
    report["total_score"] = overall_score * 5

    for grade, (lo, hi) in TRIAGE_GRADES.items():
        if lo <= report["total_score"] <= hi:
            report["triage"] = grade
            break

    return report


def print_severity_report(report: dict, volume_name: str = ""):
    """Pretty-print the clinical severity report."""
    print()
    print("╔═════════════════════════════════════════════════════════════╗")
    print("║       COVID-19 CT SEVERITY SCORE — CLINICAL REPORT        ║")
    print("╠═════════════════════════════════════════════════════════════╣")
    if volume_name:
        print(f"║  Patient Volume: {volume_name:>42s} ║")
    print(f"║  Timestamp:      {report['timestamp']:>42s} ║")
    print("╠═════════════════════════════════════════════════════════════╣")

    # Overall infection
    pct = report["infection_pct"]
    sc = report["overall_score"]
    bar = "█" * sc + "░" * (5 - sc)
    print(f"║  Overall Infection:  {pct:>6.2f}%   Score: {sc}/5  {bar}       ║")
    print("╠═════════════════════════════════════════════════════════════╣")

    # Pathology breakdown
    print("║  PATHOLOGY TYPE              │  Voxels  │  % of Volume     ║")
    print("║─────────────────────────────┼──────────┼──────────────────║")
    for pname, pdata in report.get("pathology_breakdown", {}).items():
        vox = pdata["voxels"]
        ppct = pdata["percentage"]
        print(f"║  {pname:<28s} │ {vox:>8,} │ {ppct:>7.2f}%          ║")

    print("╠═════════════════════════════════════════════════════════════╣")
    triage = report["triage"]
    print(f"║  TRIAGE:       {triage:<44s} ║")
    print("╚═════════════════════════════════════════════════════════════╝")


# ═══════════════════════════════════════════════════════════════════════════
#  Main — Single Volume
# ═══════════════════════════════════════════════════════════════════════════

def process_single(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pipeline] Device: {device}\n")

    # ── Load models ──────────────────────────────────────────────────────
    dncnn = None
    if not args.no_dncnn:
        print("[Step 1] Loading DnCNN denoiser...")
        dncnn = load_dncnn(args.dncnn_weights, DnCNNConfig(), device)

    srgan = None
    if not args.no_sr:
        print("[Step 2] Loading SRGAN super-resolver...")
        srgan = load_sr_model(args.sr_weights, SRConfig(), device)

    print("[Step 4] Loading segmentation model...")
    seg_cfg = SegConfig(low_vram=args.low_vram)
    seg_model = load_seg_model(args.seg_weights, seg_cfg, device)

    pipe = LungSeverityPipeline(
        dncnn=dncnn, srgan=srgan, seg_model=seg_model,
        device=device, target_size=SPATIAL_SIZE,
        sr_input_size=args.sr_input_size,
        batch_size=args.batch_size,
    )

    # ── Load volume ──────────────────────────────────────────────────────
    input_path = Path(args.input)
    nii = nib.load(str(input_path))
    volume = nii.get_fdata(dtype=np.float32)
    print(f"\n[Step 0] Input: {volume.shape}  "
          f"spacing={tuple(nii.header.get_zooms())}  "
          f"HU=[{volume.min():.0f}, {volume.max():.0f}]")

    # ── Run Steps 0–4 ───────────────────────────────────────────────────
    t0 = time.time()
    pathology_mask, infection_mask = pipe.predict_volume(volume)
    dt = time.time() - t0
    n = volume.shape[2]
    print(f"\n[Steps 0–4] {n} slices in {dt:.1f}s ({n/dt:.1f} slices/s)")

    # ── Step 3: Save with original metadata ──────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("\n[Step 3] Saving NIfTI masks with original affine/spacing:")
    save_mask_nifti(pathology_mask, nii, out / "pathology_mask.nii.gz")
    save_mask_nifti(infection_mask, nii, out / "infection_mask.nii.gz")

    # ── Step 5: Severity Score ───────────────────────────────────────────
    print("\n[Step 5] Computing severity score...")
    report = compute_severity_score(pathology_mask, infection_mask)
    print_severity_report(report, input_path.name)

    report_path = out / "severity_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Report saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Batch Mode
# ═══════════════════════════════════════════════════════════════════════════

def process_batch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dncnn = None if args.no_dncnn else load_dncnn(args.dncnn_weights, DnCNNConfig(), device)
    srgan = None if args.no_sr else load_sr_model(args.sr_weights, SRConfig(), device)
    seg_model = load_seg_model(args.seg_weights, SegConfig(low_vram=args.low_vram), device)

    pipe = LungSeverityPipeline(
        dncnn=dncnn, srgan=srgan, seg_model=seg_model,
        device=device, target_size=SPATIAL_SIZE,
        sr_input_size=args.sr_input_size,
        batch_size=args.batch_size,
    )

    files = sorted(Path(args.input_dir).glob("*.nii.gz")) + sorted(Path(args.input_dir).glob("*.nii"))
    print(f"[batch] {len(files)} volumes to process\n")

    all_reports = []
    for fpath in files:
        print(f"── {fpath.name} ──")
        nii = nib.load(str(fpath))
        volume = nii.get_fdata(dtype=np.float32)

        pathology_mask, infection_mask = pipe.predict_volume(volume)

        vol_dir = Path(args.output_dir) / fpath.stem.replace(".nii", "")
        vol_dir.mkdir(parents=True, exist_ok=True)

        save_mask_nifti(pathology_mask, nii, vol_dir / "pathology_mask.nii.gz")
        save_mask_nifti(infection_mask, nii, vol_dir / "infection_mask.nii.gz")

        report = compute_severity_score(pathology_mask, infection_mask)
        report["patient_volume"] = fpath.name
        all_reports.append(report)

        print(f"  Score: {report['total_score']}/25 — {report['triage']}\n")

    summary_path = Path(args.output_dir) / "batch_severity_report.json"
    with open(summary_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"[batch] Summary saved: {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Lung Severity Detection Pipeline (Steps 0–5)")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="Single .nii.gz volume")
    g.add_argument("--input_dir", help="Directory of .nii.gz volumes")

    p.add_argument("--dncnn_weights", default="weights/dncnn2d_epoch_0240.pth")
    p.add_argument("--sr_weights", default="weights/best_sr_gan_model.pth")
    p.add_argument("--seg_weights", default="weights/best_segmentation_model.pth")
    p.add_argument("--no_dncnn", action="store_true", help="Skip Step 1 (denoising)")
    p.add_argument("--no_sr", action="store_true", help="Skip Step 2 (super-resolution)")
    p.add_argument("--low_vram", action="store_true")

    p.add_argument("--sr_input_size", type=int, default=256,
                    help="SRGAN input resolution (default 256 — per self-degradation training)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_dir", default="./predictions")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input:
        process_single(args)
    else:
        process_batch(args)
