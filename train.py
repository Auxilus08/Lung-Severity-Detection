"""
train.py — Training Loop for Multi-Task U-Net Segmentation
============================================================

Train the Multi-Task U-Net with TWO decoder heads:
    Head A → pathology detection  (4 classes: BG, GGO, Consolidation, PE)
    Head B → binary infection     (2 classes: Healthy, Infected)

The MedSeg dataset provides pathology masks with labels:
    0=BG, 1=GGO, 2=Consolidation, 3=Pleural Effusion

Task A trains directly on these labels.
Task B collapses them to binary: any > 0 → Infected.

Usage
-----
    # Train from NIfTI directly (100 slices, ~4GB GPU)
    python train.py --data_dir ./data --epochs 150 --batch_size 2

    # Resume from checkpoint
    python train.py --data_dir ./data --epochs 150 --resume checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from monai.losses import DiceLoss

from config import TrainConfig, SegConfig
from dataset import build_train_val_datasets
from models.multitask_unet import build_seg_model


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
    L = w_A × (Dice+CE)_pathology  +  w_B × (Dice+CE)_infection

    Both heads are ALWAYS trained (unlike the old version that disabled Task A).
    """

    def __init__(self, cfg: TrainConfig, device: torch.device):
        super().__init__()
        self.w_a = cfg.task_a_weight  # always > 0
        self.w_b = cfg.task_b_weight

        # Task A: 4-class pathology (BG, GGO, Consolidation, PE)
        pathology_weights = torch.tensor([0.5, 2.0, 3.0, 3.0], device=device)
        self.loss_pathology = CombinedSegLoss(4, pathology_weights)

        # Task B: 2-class infection (Healthy, Infected)
        infection_weights = torch.tensor(cfg.lesion_ce_weights, device=device)
        self.loss_infection = CombinedSegLoss(2, infection_weights)

        print(f"[loss] Task A (pathology, 4-class) weight={self.w_a:.1f}")
        print(f"[loss] Task B (infection, binary)   weight={self.w_b:.1f}")

    def forward(self, path_logits, inf_logits, path_target, inf_target):
        l_a = self.loss_pathology(path_logits, path_target)
        l_b = self.loss_infection(inf_logits, inf_target)
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

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    accum_steps=4):
    """Train one epoch with gradient accumulation for low-VRAM GPUs."""
    model.train()
    run = {"total": 0., "pathology": 0., "infection": 0.}
    n = 0
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        imgs = batch["image"].to(device, non_blocking=True)
        path_m = batch["lobe_mask"].to(device, non_blocking=True)    # pathology mask (0,1,2,3)
        inf_m = batch["lesion_mask"].to(device, non_blocking=True)   # binary infection

        with autocast('cuda', enabled=(device.type == "cuda")):
            path_log, inf_log = model(imgs)
            total, l_a, l_b = criterion(path_log, inf_log, path_m, inf_m)
            total = total / accum_steps  # scale loss for accumulation

        scaler.scale(total).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        run["total"] += total.item() * accum_steps  # unscale for logging
        run["pathology"] += l_a.item()
        run["infection"] += l_b.item()
        n += 1

        if (i + 1) % 20 == 0:
            print(f"  [E{epoch}] batch {i+1}/{len(loader)}  "
                  f"loss={run['total']/n:.4f} path={run['pathology']/n:.4f} "
                  f"inf={run['infection']/n:.4f}")

    return {k: v / max(n, 1) for k, v in run.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    run = {"total": 0., "pathology": 0., "infection": 0.,
           "dice_pathology": 0., "dice_infection": 0.}
    n = 0
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        path_m = batch["lobe_mask"].to(device, non_blocking=True)
        inf_m = batch["lesion_mask"].to(device, non_blocking=True)

        with autocast('cuda', enabled=(device.type == "cuda")):
            path_log, inf_log = model(imgs)
            total, l_a, l_b = criterion(path_log, inf_log, path_m, inf_m)

        run["total"] += total.item()
        run["pathology"] += l_a.item()
        run["infection"] += l_b.item()
        run["dice_pathology"] += mean_dice(path_log, path_m, 4)
        run["dice_infection"] += mean_dice(inf_log, inf_m, 2)
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

    # ── Model ────────────────────────────────────────────────────────────
    seg_cfg = SegConfig(low_vram=args.low_vram)
    model = build_seg_model(seg_cfg).to(device)

    # ── Resume from checkpoint ───────────────────────────────────────────
    start_epoch = 1
    best_dice = 0.0
    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"[train] Resumed at epoch {start_epoch}, best_dice={best_dice:.4f}")

    # ── Loss / Opt / Scheduler ───────────────────────────────────────────
    criterion = MultiTaskLoss(tcfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=tcfg.scheduler_T0, T_mult=tcfg.scheduler_Tmult,
        eta_min=tcfg.scheduler_eta_min,
    )
    scaler = GradScaler('cuda', enabled=(device.type == "cuda"))

    # ── Checkpointing ────────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[train] Starting training: {tcfg.epochs} epochs, "
          f"batch_size={tcfg.batch_size}, lr={tcfg.lr}")
    print(f"[train] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples\n")

    for epoch in range(start_epoch, tcfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion,
                             optimizer, scaler, device, epoch)
        vl = validate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[E{epoch:03d}/{tcfg.epochs}] "
            f"trn={tr['total']:.4f} | "
            f"val={vl['total']:.4f} DPath={vl['dice_pathology']:.4f} "
            f"DInf={vl['dice_infection']:.4f} | "
            f"lr={lr:.2e} | {dt:.1f}s"
        )

        # Best model (weight pathology 60%, infection 40%)
        combined = 0.6 * vl["dice_pathology"] + 0.4 * vl["dice_infection"]

        if combined > best_dice:
            best_dice = combined
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
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
    p = argparse.ArgumentParser("Train Multi-Task U-Net Segmentation")

    p.add_argument("--data_dir", default="./data")
    p.add_argument("--use_npy", action="store_true")
    p.add_argument("--preprocessed_dir", default="./preprocessed")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--val_fraction", type=float, default=0.15)

    p.add_argument("--low_vram", action="store_true")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--task_a_weight", type=float, default=1.0,
                    help="Weight for pathology loss (Task A)")
    p.add_argument("--task_b_weight", type=float, default=2.0,
                    help="Weight for infection loss (Task B)")

    p.add_argument("--resume", default=None, help="Resume from checkpoint path")
    p.add_argument("--ckpt_dir", default="./checkpoints")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
