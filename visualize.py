"""
visualize.py — View NIfTI segmentation results as PNG images
==============================================================

Generates overlay images showing the CT scan with lobe and lesion
masks overlaid in color. No GUI required — saves directly to PNGs.

Usage
-----
    # Visualize pipeline results (auto-detects masks in the folder)
    python visualize.py --input data/val_im.nii.gz --pred_dir predictions/ --output_dir visualizations/

    # Visualize specific slices
    python visualize.py --input data/val_im.nii.gz --pred_dir predictions/ --slices 0 5 9

    # Just dump all slices as individual PNGs
    python visualize.py --input data/val_im.nii.gz --pred_dir predictions/ --all_slices
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from config import HU_MIN, HU_MAX, LOBE_NAMES


# ═══════════════════════════════════════════════════════════════════════════
#  Color Maps
# ═══════════════════════════════════════════════════════════════════════════

# Lobe colors (transparent BG + 5 distinct lobe colors)
LOBE_COLORS = [
    (0, 0, 0, 0),        # 0: BG — transparent
    (0.2, 0.6, 1.0, 0.4),  # 1: RUL — blue
    (0.0, 0.9, 0.4, 0.4),  # 2: RML — green
    (1.0, 0.8, 0.0, 0.4),  # 3: RLL — yellow
    (0.9, 0.3, 0.9, 0.4),  # 4: LUL — magenta
    (1.0, 0.4, 0.2, 0.4),  # 5: LLL — orange
]
LOBE_CMAP = ListedColormap(LOBE_COLORS)

# Lesion colors
LESION_COLORS = [
    (0, 0, 0, 0),           # 0: Healthy — transparent
    (1.0, 0.0, 0.0, 0.5),   # 1: Infected — red
]
LESION_CMAP = ListedColormap(LESION_COLORS)


def hu_normalize(arr):
    """Normalise raw HU to [0, 1] for display."""
    clipped = np.clip(arr, HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / (HU_MAX - HU_MIN)


# ═══════════════════════════════════════════════════════════════════════════
#  Single Slice Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def plot_slice(
    ct_slice: np.ndarray,
    lobe_slice: np.ndarray | None,
    lesion_slice: np.ndarray | None,
    slice_idx: int,
    save_path: Path,
):
    """Plot a single axial slice with overlays and save as PNG."""

    n_panels = 1 + (lobe_slice is not None) + (lesion_slice is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    col = 0

    # Panel 1: CT image
    axes[col].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[col].set_title(f"CT — Slice {slice_idx}", fontsize=13, fontweight="bold")
    axes[col].axis("off")
    col += 1

    # Panel 2: Lobe overlay
    if lobe_slice is not None:
        axes[col].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
        axes[col].imshow(lobe_slice, cmap=LOBE_CMAP, vmin=0, vmax=5, interpolation="nearest")
        axes[col].set_title("Lobe Segmentation", fontsize=13, fontweight="bold")
        axes[col].axis("off")

        # Legend
        patches = [
            mpatches.Patch(color=LOBE_COLORS[i], label=LOBE_NAMES[i])
            for i in range(1, 6)
        ]
        axes[col].legend(handles=patches, loc="lower right", fontsize=8,
                         framealpha=0.8, facecolor="black", labelcolor="white")
        col += 1

    # Panel 3: Lesion overlay
    if lesion_slice is not None:
        axes[col].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
        axes[col].imshow(lesion_slice, cmap=LESION_CMAP, vmin=0, vmax=1, interpolation="nearest")

        n_infected = int((lesion_slice == 1).sum())
        n_total = lesion_slice.size
        pct = 100 * n_infected / n_total if n_total > 0 else 0
        axes[col].set_title(f"Lesion Mask ({pct:.1f}% infected)", fontsize=13, fontweight="bold")
        axes[col].axis("off")

        patches = [
            mpatches.Patch(color=(1, 0, 0, 0.5), label="Infected"),
            mpatches.Patch(color=(0, 0.7, 0, 0.5), label="Healthy"),
        ]
        axes[col].legend(handles=patches, loc="lower right", fontsize=8,
                         framealpha=0.8, facecolor="black", labelcolor="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Summary Grid
# ═══════════════════════════════════════════════════════════════════════════

def plot_summary_grid(
    ct_vol: np.ndarray,
    lobe_vol: np.ndarray | None,
    lesion_vol: np.ndarray | None,
    save_path: Path,
    max_slices: int = 12,
):
    """Plot a grid of evenly-spaced slices for a quick overview."""
    n_slices = ct_vol.shape[2]
    indices = np.linspace(0, n_slices - 1, min(max_slices, n_slices), dtype=int)

    n_rows = len(indices)
    n_cols = 1 + (lobe_vol is not None) + (lesion_vol is not None)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    headers = ["CT Scan"]
    if lobe_vol is not None:
        headers.append("Lobe Overlay")
    if lesion_vol is not None:
        headers.append("Lesion Overlay")

    for ci, h in enumerate(headers):
        axes[0, ci].set_title(h, fontsize=14, fontweight="bold", color="white")

    for ri, si in enumerate(indices):
        ct_s = hu_normalize(ct_vol[:, :, si])

        col = 0
        axes[ri, col].imshow(ct_s, cmap="gray", vmin=0, vmax=1)
        axes[ri, col].set_ylabel(f"Slice {si}", fontsize=10, color="white")
        axes[ri, col].set_yticks([])
        axes[ri, col].set_xticks([])
        col += 1

        if lobe_vol is not None:
            axes[ri, col].imshow(ct_s, cmap="gray", vmin=0, vmax=1)
            axes[ri, col].imshow(lobe_vol[:, :, si], cmap=LOBE_CMAP, vmin=0, vmax=5)
            axes[ri, col].axis("off")
            col += 1

        if lesion_vol is not None:
            axes[ri, col].imshow(ct_s, cmap="gray", vmin=0, vmax=1)
            axes[ri, col].imshow(lesion_vol[:, :, si], cmap=LESION_CMAP, vmin=0, vmax=1)
            axes[ri, col].axis("off")

    plt.tight_layout()
    fig.patch.set_facecolor("black")
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"  ✓ Summary grid: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load CT
    print(f"[viz] Loading CT: {args.input}")
    ct_nii = nib.load(args.input)
    ct_vol = ct_nii.get_fdata(dtype=np.float32)
    n_slices = ct_vol.shape[2]
    print(f"  Shape: {ct_vol.shape}  ({n_slices} slices)")

    # Load masks (auto-detect from pred_dir)
    pred = Path(args.pred_dir)
    lobe_vol, lesion_vol = None, None

    # Try new names first (pathology/infection), then old names (lobe/lesion)
    pathology_path = pred / "pathology_mask.nii.gz"
    lobe_path = pred / "lobe_mask.nii.gz"
    if pathology_path.exists():
        lobe_vol = nib.load(str(pathology_path)).get_fdata().astype(np.int16)
        print(f"  Pathology: {pathology_path.name}  labels={np.unique(lobe_vol).tolist()}")
    elif lobe_path.exists():
        lobe_vol = nib.load(str(lobe_path)).get_fdata().astype(np.int16)
        print(f"  Lobes:  {lobe_path.name}  labels={np.unique(lobe_vol).tolist()}")

    infection_path = pred / "infection_mask.nii.gz"
    lesion_path = pred / "lesion_mask.nii.gz"
    if infection_path.exists():
        lesion_vol = nib.load(str(infection_path)).get_fdata().astype(np.int16)
        print(f"  Infection: {infection_path.name}  labels={np.unique(lesion_vol).tolist()}")
    elif lesion_path.exists():
        lesion_vol = nib.load(str(lesion_path)).get_fdata().astype(np.int16)
        print(f"  Lesion: {lesion_path.name}  labels={np.unique(lesion_vol).tolist()}")

    if lobe_vol is None and lesion_vol is None:
        print("  ⚠ No mask files found in pred_dir. Showing CT only.")

    # Determine which slices to render
    if args.all_slices:
        slice_indices = list(range(n_slices))
    elif args.slices:
        slice_indices = [s for s in args.slices if 0 <= s < n_slices]
    else:
        # Default: pick ~6 evenly spaced slices
        slice_indices = np.linspace(0, n_slices - 1, min(6, n_slices), dtype=int).tolist()

    # Render individual slices
    print(f"\n[viz] Rendering {len(slice_indices)} slices...")
    for si in slice_indices:
        ct_s = hu_normalize(ct_vol[:, :, si])
        lobe_s = lobe_vol[:, :, si] if lobe_vol is not None else None
        les_s = lesion_vol[:, :, si] if lesion_vol is not None else None

        save_path = out / f"slice_{si:03d}.png"
        plot_slice(ct_s, lobe_s, les_s, si, save_path)
        print(f"  ✓ {save_path.name}")

    # Render summary grid
    print("\n[viz] Rendering summary grid...")
    plot_summary_grid(ct_vol, lobe_vol, lesion_vol, out / "summary_grid.png")

    print(f"\n[viz] Done — {len(slice_indices) + 1} images saved to {out}/")
    print(f"  Open {out}/summary_grid.png for a quick overview")


def parse_args():
    p = argparse.ArgumentParser("Visualize NIfTI segmentation results as PNG images")
    p.add_argument("--input", required=True, help="Original CT scan (.nii.gz)")
    p.add_argument("--pred_dir", default="./predictions", help="Directory with mask NIfTI files")
    p.add_argument("--output_dir", default="./visualizations", help="Where to save PNG images")
    p.add_argument("--slices", type=int, nargs="+", help="Specific slice indices to render")
    p.add_argument("--all_slices", action="store_true", help="Render every slice")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
