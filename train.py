"""
train.py — Training Loop for Step 4: The Diagnostician
========================================================

Trains the Multi-Task U-Net segmentation model:
    Head A → 5-lobe anatomical mapping    (6 classes)
    Head B → binary lesion detection      (2 classes)

Optionally chains frozen DnCNN + SRGAN before the U-Net during training
(--use_enhancement) to match inference-time conditions.

If lobe annotations are NOT available (MedSeg dataset has pathology labels
only), Task A loss is zeroed out and only Task B trains. The lobe decoder
can later be fine-tuned on a lobe-annotated dataset.

Usage
-----
    # Train with NIfTI data directly
    python train.py --data_dir ./data --epochs 150 --batch_size 4

    # Train with preprocessed .npy (faster)
    python train.py --use_npy --preprocessed_dir ./preprocessed --epochs 150

    # Train with enhancement pipeline
    python train.py --data_dir ./data --epochs 150 --use_enhancement
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from monai.losses import DiceLoss

from config import TrainConfig, SegConfig, SRConfig, DnCNNConfig
from dataset import build_train_val_datasets
from models.multitask_unet import build_seg_model
from models.dncnn import load_dncnn
from models.sr_gan import load_sr_model


# ═══════════════════════════════════════════════════════════════════════════
#  Combined Loss
# ═══════════════════════════════════════════════════════════════════════════

class CombinedSegLoss(nn.Module):
    """Dice + CrossEntropy for one segmentation head."""

    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.dice = DiceLoss(
            to_onehot_y=True, softmax=True,
            include_background=False, reduction="mean",
        )
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, target):
        return self.dice(logits, target.unsqueeze(1)) + self.ce(logits, target)


class MultiTaskLoss(nn.Module):
    """
    L = w_A × (Dice+CE)_lobe  +  w_B × (Dice+CE)_lesion

    If has_lobes is False, lobe loss is zeroed → only lesion trains.
    """

    def __init__(self, cfg: TrainConfig, device: torch.device, has_lobes: bool = False):
        super().__init__()
        self.w_a = cfg.task_a_weight if has_lobes else 0.0
        self.w_b = cfg.task_b_weight
        self.has_lobes = has_lobes

        self.loss_lobe = CombinedSegLoss(
            6, torch.tensor(cfg.lobe_ce_weights, device=device)
        )
        self.loss_lesion = CombinedSegLoss(
            2, torch.tensor(cfg.lesion_ce_weights, device=device)
        )

        if not has_lobes:
            print("[loss] ⚠ No lobe annotations — Task A weight set to 0.0")

    def forward(self, lobe_logits, lesion_logits, lobe_target, lesion_target):
        l_a = self.loss_lobe(lobe_logits, lobe_target) if self.has_lobes else torch.tensor(0.0)
        l_b = self.loss_lesion(lesion_logits, lesion_target)
        total = self.w_a * l_a + self.w_b * l_b
        return total, l_a, l_b


# ═══════════════════════════════════════════════════════════════════════════
#  Dice Metric
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def mean_dice(logits, target, num_classes):
    preds = logits.argmax(dim=1)
    scores = []
    for c in range(1, num_classes):
        pc, tc = (preds == c).float(), (target == c).float()
        inter = (pc * tc).sum()
        union = pc.sum() + tc.sum()
        if union > 0:
            scores.append((2.0 * inter / (union + 1e-7)).item())
    return sum(scores) / max(len(scores), 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Epoch Runners
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, enhancement, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    run = {"total": 0., "lobe": 0., "lesion": 0.}
    n = 0
    for i, batch in enumerate(loader):
        imgs = batch["image"].to(device, non_blocking=True)
        lobe_m = batch["lobe_mask"].to(device, non_blocking=True)
        les_m = batch["lesion_mask"].to(device, non_blocking=True)

        # Enhancement pipeline (frozen DnCNN → downsample → SRGAN)
        if enhancement is not None:
            with torch.no_grad():
                orig_size = imgs.shape[-2:]
                for stage in enhancement:
                    # If this is the SRGAN stage, downsample first (256→512 training)
                    if hasattr(stage, 'upsample'):  # SRGenerator has .upsample
                        imgs = nn.functional.interpolate(
                            imgs, size=(256, 256), mode="bilinear", align_corners=False)
                    imgs = stage(imgs)
                    imgs = torch.clamp(imgs, 0.0, 1.0)
                # Ensure output matches mask spatial dims
                if imgs.shape[-2:] != orig_size:
                    imgs = nn.functional.interpolate(
                        imgs, size=orig_size, mode="bilinear", align_corners=False)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            lobe_log, les_log = model(imgs)
            total, l_a, l_b = criterion(lobe_log, les_log, lobe_m, les_m)

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        run["total"] += total.item()
        run["lobe"] += l_a.item() if isinstance(l_a, torch.Tensor) else l_a
        run["lesion"] += l_b.item()
        n += 1

        if (i + 1) % 10 == 0:
            print(f"  [E{epoch}] batch {i+1}/{len(loader)}  loss={run['total']/n:.4f}")

    return {k: v / max(n, 1) for k, v in run.items()}


@torch.no_grad()
def validate(model, enhancement, loader, criterion, device, has_lobes):
    model.eval()
    run = {"total": 0., "lobe": 0., "lesion": 0., "dice_lobe": 0., "dice_lesion": 0.}
    n = 0
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        lobe_m = batch["lobe_mask"].to(device, non_blocking=True)
        les_m = batch["lesion_mask"].to(device, non_blocking=True)

        if enhancement is not None:
            orig_size = imgs.shape[-2:]
            for stage in enhancement:
                if hasattr(stage, 'upsample'):
                    imgs = nn.functional.interpolate(
                        imgs, size=(256, 256), mode="bilinear", align_corners=False)
                imgs = stage(imgs)
                imgs = torch.clamp(imgs, 0.0, 1.0)
            if imgs.shape[-2:] != orig_size:
                imgs = nn.functional.interpolate(
                    imgs, size=orig_size, mode="bilinear", align_corners=False)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            lobe_log, les_log = model(imgs)
            total, l_a, l_b = criterion(lobe_log, les_log, lobe_m, les_m)

        run["total"] += total.item()
        run["lobe"] += l_a.item() if isinstance(l_a, torch.Tensor) else l_a
        run["lesion"] += l_b.item()
        if has_lobes:
            run["dice_lobe"] += mean_dice(lobe_log, lobe_m, 6)
        run["dice_lesion"] += mean_dice(les_log, les_m, 2)
        n += 1

    return {k: v / max(n, 1) for k, v in run.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}\n")

    tcfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, num_workers=args.num_workers,
        img_size=args.img_size, val_fraction=args.val_fraction,
        task_a_weight=args.task_a_weight, task_b_weight=args.task_b_weight,
    )

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds, val_ds = build_train_val_datasets(
        data_dir=args.data_dir,
        val_fraction=tcfg.val_fraction,
        use_npy=args.use_npy,
        preprocessed_dir=args.preprocessed_dir,
    )

    train_loader = DataLoader(
        train_ds, batch_size=tcfg.batch_size, shuffle=True,
        num_workers=tcfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg.batch_size, shuffle=False,
        num_workers=tcfg.num_workers, pin_memory=True,
    )

    # Detect lobe availability from the underlying dataset
    base_ds = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
    has_lobes = getattr(base_ds, 'has_lobes', False)

    # ── Model ────────────────────────────────────────────────────────────
    seg_cfg = SegConfig(low_vram=args.low_vram)
    model = build_seg_model(seg_cfg).to(device)

    # ── Enhancement pipeline (optional) ──────────────────────────────────
    enhancement = None
    if args.use_enhancement:
        stages = []
        if args.dncnn_weights:
            dncnn = load_dncnn(args.dncnn_weights, DnCNNConfig(), device)
            for p in dncnn.parameters(): p.requires_grad_(False)
            stages.append(dncnn)
        if args.sr_weights:
            srgan = load_sr_model(args.sr_weights, SRConfig(), device)
            for p in srgan.parameters(): p.requires_grad_(False)
            stages.append(srgan)
        if stages:
            enhancement = stages
            print(f"[train] Enhancement pipeline: {len(stages)} frozen stages")

    # ── Loss / Opt / Scheduler ───────────────────────────────────────────
    criterion = MultiTaskLoss(tcfg, device, has_lobes=has_lobes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=tcfg.scheduler_T0, T_mult=tcfg.scheduler_Tmult,
        eta_min=tcfg.scheduler_eta_min,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Checkpointing ────────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, tcfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, enhancement, train_loader, criterion,
                             optimizer, scaler, device, epoch)
        vl = validate(model, enhancement, val_loader, criterion, device, has_lobes)
        scheduler.step()
        dt = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[E{epoch:03d}/{tcfg.epochs}] "
            f"trn={tr['total']:.4f} | "
            f"val={vl['total']:.4f} DLobe={vl['dice_lobe']:.4f} "
            f"DLesion={vl['dice_lesion']:.4f} | "
            f"lr={lr:.2e} | {dt:.1f}s"
        )

        # Best model (weight lobe 60%, lesion 40% — or just lesion if no lobes)
        if has_lobes:
            combined = 0.6 * vl["dice_lobe"] + 0.4 * vl["dice_lesion"]
        else:
            combined = vl["dice_lesion"]

        if combined > best_dice:
            best_dice = combined
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "has_lobes": has_lobes,
                "config": {"seg": vars(seg_cfg), "train": vars(tcfg)},
            }, ckpt_dir / "best_model.pth")
            print(f"  ✓ Best model saved (Dice = {best_dice:.4f})")

        if epoch % tcfg.save_every_n_epochs == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "best_dice": best_dice,
            }, ckpt_dir / f"ckpt_e{epoch:03d}.pth")

    torch.save({
        "epoch": tcfg.epochs, "model_state_dict": model.state_dict(),
        "best_dice": best_dice,
        "config": {"seg": vars(seg_cfg), "train": vars(tcfg)},
    }, ckpt_dir / "final_model.pth")
    print(f"\n[train] Done — best Dice = {best_dice:.4f}")


def parse_args():
    p = argparse.ArgumentParser("Train Multi-Task U-Net (Step 4)")

    p.add_argument("--data_dir", default="./data")
    p.add_argument("--use_npy", action="store_true")
    p.add_argument("--preprocessed_dir", default="./preprocessed")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--val_fraction", type=float, default=0.15)

    p.add_argument("--low_vram", action="store_true")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--task_a_weight", type=float, default=1.0)
    p.add_argument("--task_b_weight", type=float, default=2.0)

    p.add_argument("--use_enhancement", action="store_true")
    p.add_argument("--dncnn_weights", default="weights/dncnn2d_epoch_0240.pth")
    p.add_argument("--sr_weights", default="weights/best_sr_gan_model.pth")

    p.add_argument("--ckpt_dir", default="./checkpoints")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
