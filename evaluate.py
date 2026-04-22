"""
evaluate.py — Evaluate segmentation model on training data with ground truth
=============================================================================
Computes Dice score, precision, recall per class on data where we have masks.
"""
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.amp import autocast

from pipeline import preprocess_slice
from models.multitask_unet import load_seg_model
from config import SegConfig, SPATIAL_SIZE
from preprocess import spatial_standardize


def dice_score(pred, gt, class_id):
    """Compute Dice for a single class."""
    p = (pred == class_id).astype(float)
    g = (gt == class_id).astype(float)
    inter = (p * g).sum()
    union = p.sum() + g.sum()
    if union == 0:
        return float('nan')  # class not present
    return (2.0 * inter) / (union + 1e-7)


def precision_recall(pred, gt, class_id):
    """Compute precision and recall for a single class."""
    p = (pred == class_id)
    g = (gt == class_id)
    tp = (p & g).sum()
    fp = (p & ~g).sum()
    fn = (~p & g).sum()
    precision = tp / (tp + fp + 1e-7) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn + 1e-7) if (tp + fn) > 0 else float('nan')
    return float(precision), float(recall)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}\n")

    # Load model
    seg_model = load_seg_model("weights/best_segmentation_model.pth", SegConfig(), device)

    # Load training data (we have ground truth for this)
    tr_nii = nib.load("data/tr_im.nii.gz")
    tr_vol = tr_nii.get_fdata(dtype=np.float32)
    tr_mask_nii = nib.load("data/tr_mask.nii.gz")
    tr_mask = tr_mask_nii.get_fdata(dtype=np.float32).astype(np.int16)

    n_slices = tr_vol.shape[2]
    ts = SPATIAL_SIZE

    # Also load enhanced data if available (to match inference conditions)
    try:
        enhanced = np.load("enhanced_data/tr_im.npy")  # [N, 1, H, W]
        use_enhanced = True
        print(f"[eval] Using enhanced data (DnCNN+SRGAN) — matches inference\n")
    except FileNotFoundError:
        use_enhanced = False
        print(f"[eval] Using raw data (no enhancement)\n")

    # Class names
    class_names = {0: "Background", 1: "GGO", 2: "Consolidation", 3: "Pleural Effusion"}
    infection_names = {0: "Healthy", 1: "Infected"}

    # Collect predictions
    all_path_dice = {c: [] for c in range(4)}
    all_inf_dice = {c: [] for c in range(2)}
    all_path_prec = {c: [] for c in range(4)}
    all_path_rec = {c: [] for c in range(4)}

    seg_model.to(device)
    seg_model.eval()

    print(f"[eval] Evaluating {n_slices} training slices...\n")

    with torch.no_grad():
        for s in range(n_slices):
            # Prepare input
            if use_enhanced:
                x = torch.from_numpy(enhanced[s:s+1]).to(device)
            else:
                processed = preprocess_slice(tr_vol[:, :, s], ts)
                x = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).to(device)

            # Prepare ground truth (resize mask to model size)
            gt_mask = spatial_standardize(tr_mask[:, :, s], ts, is_mask=True).astype(np.int16)
            gt_infection = (gt_mask > 0).astype(np.int16)

            # Predict
            with autocast('cuda', enabled=(device.type == 'cuda')):
                path_logits, inf_logits = seg_model(x)

            path_pred = path_logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.int16)
            inf_pred = inf_logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.int16)

            # Compute metrics per class
            for c in range(4):
                d = dice_score(path_pred, gt_mask, c)
                if not np.isnan(d):
                    all_path_dice[c].append(d)
                p, r = precision_recall(path_pred, gt_mask, c)
                if not np.isnan(p):
                    all_path_prec[c].append(p)
                    all_path_rec[c].append(r)

            for c in range(2):
                d = dice_score(inf_pred, gt_infection, c)
                if not np.isnan(d):
                    all_inf_dice[c].append(d)

    seg_model.cpu()

    # Print results
    print("=" * 70)
    print("  PATHOLOGY SEGMENTATION (4-class)")
    print("=" * 70)
    print(f"  {'Class':<25} {'Dice':>8} {'Precision':>10} {'Recall':>8} {'N':>5}")
    print("-" * 70)
    for c in range(4):
        d = np.mean(all_path_dice[c]) if all_path_dice[c] else float('nan')
        p = np.mean(all_path_prec[c]) if all_path_prec[c] else float('nan')
        r = np.mean(all_path_rec[c]) if all_path_rec[c] else float('nan')
        n = len(all_path_dice[c])
        print(f"  {class_names[c]:<25} {d:>8.4f} {p:>10.4f} {r:>8.4f} {n:>5}")

    # Overall pathology Dice (excluding background)
    fg_dices = []
    for c in range(1, 4):
        fg_dices.extend(all_path_dice[c])
    mean_fg = np.mean(fg_dices) if fg_dices else 0
    print(f"\n  Mean Foreground Dice:  {mean_fg:.4f}")

    print(f"\n{'=' * 70}")
    print("  INFECTION DETECTION (binary)")
    print("=" * 70)
    print(f"  {'Class':<25} {'Dice':>8} {'N':>5}")
    print("-" * 70)
    for c in range(2):
        d = np.mean(all_inf_dice[c]) if all_inf_dice[c] else float('nan')
        n = len(all_inf_dice[c])
        print(f"  {infection_names[c]:<25} {d:>8.4f} {n:>5}")

    inf_fg = np.mean(all_inf_dice[1]) if all_inf_dice[1] else 0
    print(f"\n  Infection Dice:  {inf_fg:.4f}")

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print("=" * 70)
    print(f"  Pathology Mean Dice (fg): {mean_fg:.4f}")
    print(f"  Infection Dice:           {inf_fg:.4f}")
    print(f"  Combined (0.6P + 0.4I):   {0.6*mean_fg + 0.4*inf_fg:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
