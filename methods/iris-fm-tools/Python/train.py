import os
import argparse
import time
import json
import torch
import madgrad
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from modules.dataset.iris_dataset import IrisDataset, prepare_datasets
from modules.dataset.task_configs import get_task
from modules.models.regression_head import RegressionHead


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

class WeightedSmoothL1Loss(nn.Module):

    def __init__(self, weights: list[float]):
        super().__init__()
        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32),
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_output = F.smooth_l1_loss(pred, target, reduction="none")
        return (per_output * self.weights.to(pred.device)).mean()


def build_criterion(task: dict) -> nn.Module:
    """Return WeightedSmoothL1Loss if the task defines loss_weights, else SmoothL1Loss."""
    weights = task.get("loss_weights", None)
    if weights is not None:
        print(f"  Loss: WeightedSmoothL1Loss  weights={weights}", flush=True)
        return WeightedSmoothL1Loss(weights)
    print(f"  Loss: SmoothL1Loss (uniform weights)", flush=True)
    return nn.SmoothL1Loss()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def build_optimizer(model, args):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "madgrad":
        return madgrad.MADGRAD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_samples = 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bs = features.size(0)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds = model(features)
                loss = criterion(preds, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            total_loss += loss.item() * bs
            total_samples += bs

    return total_loss / total_samples


def load_dino(args, device):
    if not args.dino_weights or not os.path.isfile(args.dino_weights):
        raise FileNotFoundError(f"DINO weights not found: {args.dino_weights}")
    print("Loading DINOv3 model...", flush=True)
    dino = torch.hub.load(
        args.dino_repo_dir, model="dinov3_vitl16", source="local",
        weights=args.dino_weights, trust_repo=True,
    )
    dino.to(device).eval()
    print("DINOv3 model loaded.", flush=True)
    return dino


def build_inference_checkpoint(model, train_ds, task, task_name, args, extra=None):
    """
    Lean inference-compatible checkpoint with model weights 
    and metadata needed for normalization and output decoding.
    """
    data = {
        "model_state": model.state_dict(),
        "task": task_name,
        "num_outputs": task["num_outputs"],
        "normalization": task["normalization"],
        "use_sigmoid": task["use_sigmoid"],
        "image_size": [args.image_size[0], args.image_size[1]],
        "args": vars(args),
    }
    if task["normalization"] == "zscore":
        data["label_mean"] = train_ds.get_label_mean()
        data["label_std"] = train_ds.get_label_std()
    else:
        data["norm_scale"] = train_ds.get_norm_scale()
    if extra:
        data.update(extra)
    return data


def make_datasets(args, task, task_name, dino_model, device, train_df, val_df=None):
    """
    Create train (and optionally val) IrisDataset instances.

    Translation augmentation is automatically enabled for the training set
    if the task defines a translate_fn.  The val set always uses augment=False.
    """
    tw, th = args.image_size
    # Augment train if the task supports it (has a translate_fn)
    train_augment = task.get("translate_fn") is not None

    train_ds = IrisDataset(
        df=train_df, image_dir=args.image_dir, task_name=task_name,
        feature_extractor=dino_model, device=args.device,
        target_w=tw, target_h=th, cache_dir=args.feature_cache,
        augment=train_augment,
        aug_translate_prob=0.5,
        aug_translate_max=0.20,
    )

    val_ds = None
    if val_df is not None and len(val_df) > 0:
        val_ds = IrisDataset(
            df=val_df, image_dir=args.image_dir, task_name=task_name,
            feature_extractor=dino_model, device=args.device,
            target_w=tw, target_h=th, cache_dir=args.feature_cache,
            label_mean=train_ds.label_mean,
            label_std=train_ds.label_std,
            norm_scale=train_ds.get_norm_scale(),
            augment=False,  
        )

    return train_ds, val_ds


def print_norm_stats(train_ds, task):
    if task["normalization"] == "zscore":
        print(f"\n  Z-score normalization:", flush=True)
        print(f"    mean = {train_ds.label_mean}", flush=True)
        print(f"    std  = {train_ds.label_std}", flush=True)
    elif task["normalization"] == "wh":
        print(f"\n  WH normalization (per-output): {train_ds.norm_scale.tolist()}", flush=True)
    else:
        print(f"\n  Image normalization: ÷ {train_ds.norm_scale}", flush=True)


# ==========================================================================
# MODE: split
# ==========================================================================

def mode_split(args, device, task, task_name):
    print(f"\n{'='*70}")
    print(f"  MODE: split  |  TASK: {task_name}  |  {task['num_outputs']} outputs")
    print(f"  Image size: {args.image_size[0]}×{args.image_size[1]}")
    print(f"{'='*70}", flush=True)

    dino = load_dino(args, device)
    train_df, val_df = prepare_datasets(args.data_csv, task_name, train_size=0.8)
    train_ds, val_ds = make_datasets(args, task, task_name, dino, device, train_df, val_df)
    print_norm_stats(train_ds, task)

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = RegressionHead(out_channels=task["num_outputs"], dropout=args.dropout,
                           use_sigmoid=task["use_sigmoid"]).to(device)
    
    criterion = build_criterion(task)
    optimizer = build_optimizer(model, args)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=args.lr * 0.01)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val, epochs_no_improve, start_epoch = float("inf"), 0, 0
    ckpt_name = f"{task_name}_best.pth"
    resume_name = f"{task_name}_resume.pth"

    # Resume
    rp = os.path.join(args.ckpt, resume_name)
    if args.resume and os.path.isfile(rp):
        ck = torch.load(rp, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            optimizer.load_state_dict(ck["optimizer_state"])
        if "scheduler_state" in ck:
            scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck.get("epoch", 0)
        best_val = ck.get("loss", float("inf"))
        print(f"Resumed from epoch {start_epoch} (val={best_val:.6f})", flush=True)
    elif args.resume:
        inf_path = os.path.join(args.ckpt, ckpt_name)
        if os.path.isfile(inf_path):
            ck = torch.load(inf_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            start_epoch = ck.get("epoch", 0)
            best_val = ck.get("loss", float("inf"))
            print(f"Resumed weights from {ckpt_name} (epoch {start_epoch}). "
                  f"Optimizer/scheduler reset.", flush=True)

    os.makedirs(args.ckpt, exist_ok=True)
    print(f"\nTraining for up to {args.epochs} epochs ...\n", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        tl = run_epoch(model, train_loader, criterion, device,
                       optimizer=optimizer, scaler=scaler, use_amp=use_amp)
        vl = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step(vl)
        el = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:>3}/{args.epochs}  |  Train {tl:.6f}  |  Val {vl:.6f}  |  "
              f"LR {lr:.2e}  |  {el:.1f}s", flush=True)

        if vl < best_val:
            best_val = vl
            epochs_no_improve = 0
            torch.save(
                build_inference_checkpoint(model, train_ds, task, task_name, args,
                                           {"epoch": epoch + 1, "loss": best_val}),
                os.path.join(args.ckpt, ckpt_name),
            )
            torch.save(
                {"model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict(),
                 "epoch": epoch + 1, "loss": best_val},
                os.path.join(args.ckpt, resume_name),
            )
            print(f"  |-> Saved best model (val {best_val:.6f})", flush=True)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}.", flush=True)
            break

    print(f"\nSplit complete. Best val: {best_val:.6f}", flush=True)


# ==========================================================================
# MODE: loso
# ==========================================================================

def mode_loso(args, device, task, task_name):
    if not task["has_subject_id"]:
        print(f"ERROR: LOSO not available for {task_name} (no subject_id).")
        return

    full_df = pd.read_csv(args.data_csv)
    subj_col = task["subject_id_col"]
    subjects = sorted(full_df[subj_col].unique())
    n = len(subjects)

    print(f"\n{'='*70}")
    print(f"  MODE: loso  |  TASK: {task_name}  |  {n} folds")
    print(f"  Image size: {args.image_size[0]}×{args.image_size[1]}")
    print(f"{'='*70}", flush=True)

    dino = load_dino(args, device)
    os.makedirs(args.ckpt, exist_ok=True)
    all_results = []

    for fold_idx, val_subj in enumerate(subjects):
        val_df = full_df[full_df[subj_col] == val_subj].reset_index(drop=True)
        train_df = full_df[full_df[subj_col] != val_subj].reset_index(drop=True)

        fc = task["val_filter_col"]
        if fc and fc in val_df.columns:
            val_df = val_df[val_df[fc] == "orig"].reset_index(drop=True)

        print(f"\n{'─'*60}")
        print(f"  FOLD {fold_idx+1}/{n}: hold out subject {val_subj}  "
              f"(train={len(train_df)}, val={len(val_df)})")
        print(f"{'─'*60}", flush=True)

        train_ds, val_ds = make_datasets(args, task, task_name, dino, device, train_df, val_df)

        kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=(device.type == "cuda"))
        tl = DataLoader(train_ds, shuffle=True, **kw)
        vl_loader = DataLoader(val_ds, shuffle=False, **kw)

        model = RegressionHead(out_channels=task["num_outputs"], dropout=args.dropout,
                               use_sigmoid=task["use_sigmoid"]).to(device)
        
        criterion = build_criterion(task)
        optimizer = build_optimizer(model, args)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=args.lr * 0.01)

        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        best_val, best_ep, no_imp = float("inf"), 0, 0
        fold_name = f"fold_{fold_idx:02d}_subj_{val_subj}"

        for epoch in range(args.epochs):
            t0 = time.perf_counter()
            train_loss = run_epoch(model, tl, criterion, device,
                                   optimizer=optimizer, scaler=scaler, use_amp=use_amp)
            val_loss = run_epoch(model, vl_loader, criterion, device, use_amp=use_amp)
            scheduler.step(val_loss)
            el = time.perf_counter() - t0
            lr = optimizer.param_groups[0]["lr"]

            if (epoch + 1) % 10 == 0 or val_loss < best_val:
                marker = "  ↳ best" if val_loss < best_val else ""
                print(f"  Ep {epoch+1:>3}  |  T {train_loss:.6f}  |  V {val_loss:.6f}  |  "
                      f"LR {lr:.2e}  |  {el:.1f}s{marker}", flush=True)

            if val_loss < best_val:
                best_val, best_ep, no_imp = val_loss, epoch + 1, 0
                torch.save(
                    build_inference_checkpoint(
                        model, train_ds, task, task_name, args,
                        {"epoch": best_ep, "loss": best_val,
                         "val_subject": str(val_subj), "fold_idx": fold_idx},
                    ),
                    os.path.join(args.ckpt, f"{fold_name}_best.pth"),
                )
            else:
                no_imp += 1

            if no_imp >= args.patience:
                print(f"  Early stop at epoch {epoch+1}", flush=True)
                break

        print(f"  -> Fold {fold_idx+1} best: {best_val:.6f} at epoch {best_ep}", flush=True)
        all_results.append({
            "fold": fold_idx + 1, "val_subject": str(val_subj),
            "val_images": len(val_df), "best_val_loss": float(best_val),
            "best_epoch": best_ep,
        })

    losses = [r["best_val_loss"] for r in all_results]
    epochs_list = [r["best_epoch"] for r in all_results]

    print(f"\n{'='*70}")
    print(f"  LOSO SUMMARY ({task_name})")
    print(f"{'='*70}")
    print(f"\n  {'Fold':>4s}  {'Subject':>8s}  {'Val Imgs':>8s}  {'Val Loss':>10s}  {'Epoch':>6s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*6}")

    for r in all_results:
        print(f"  {r['fold']:>4d}  {r['val_subject']:>8s}  {r['val_images']:>8d}  "
              f"{r['best_val_loss']:>10.6f}  {r['best_epoch']:>6d}")

    print(f"\n  Mean:   {np.mean(losses):.6f}")
    print(f"  Median: {np.median(losses):.6f}")
    print(f"  Std:    {np.std(losses):.6f}")
    print(f"  Median epoch: {int(np.median(epochs_list))}  (use for --mode final)")

    thr = np.mean(losses) + np.std(losses)
    hard = [r for r in all_results if r["best_val_loss"] > thr]
    if hard:
        print(f"\n  Hard subjects (> {thr:.4f}):")
        for r in hard:
            print(f"    Subject {r['val_subject']}: {r['best_val_loss']:.6f}")

    res = {
        "task": task_name, "num_outputs": task["num_outputs"],
        "num_folds": n, "image_size": list(args.image_size),
        "mean_val_loss": float(np.mean(losses)),
        "median_val_loss": float(np.median(losses)),
        "std_val_loss": float(np.std(losses)),
        "median_best_epoch": int(np.median(epochs_list)),
        "folds": all_results,
    }
    rp = os.path.join(args.ckpt, "loso_results.json")
    with open(rp, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  Results: {rp}")
    pd.DataFrame(all_results).to_csv(os.path.join(args.ckpt, "loso_summary.csv"), index=False)
    print(f"\nLOSO complete.", flush=True)


# ==========================================================================
# MODE: final
# ==========================================================================

def mode_final(args, device, task, task_name):
    loso_loss = None
    if args.loso_results and os.path.isfile(args.loso_results):
        with open(args.loso_results) as f:
            loso = json.load(f)
        target_epochs = loso.get("median_best_epoch", args.epochs)
        loso_loss = loso["mean_val_loss"]
        print(f"\n  LOSO: mean_val={loso_loss:.6f}, median_epoch={target_epochs}")
    else:
        target_epochs = args.epochs
        if args.loso_results:
            print(f"\n  WARNING: LOSO results not found at {args.loso_results}")
        print(f"  Using --epochs={target_epochs}")

    full_df = pd.read_csv(args.data_csv)

    print(f"\n{'='*70}")
    print(f"  MODE: final  |  TASK: {task_name}  |  {target_epochs} epochs")
    print(f"  Image size: {args.image_size[0]}×{args.image_size[1]}")
    print(f"  Training on ALL {len(full_df)} images")
    print(f"{'='*70}", flush=True)

    dino = load_dino(args, device)
    train_ds, _ = make_datasets(args, task, task_name, dino, device, full_df)
    print_norm_stats(train_ds, task)

    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = RegressionHead(out_channels=task["num_outputs"], dropout=args.dropout,
                           use_sigmoid=task["use_sigmoid"]).to(device)
    
    criterion = build_criterion(task)
    optimizer = build_optimizer(model, args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=target_epochs, eta_min=args.lr * 0.01)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    os.makedirs(args.ckpt, exist_ok=True)
    print(f"\nTraining final model ...\n", flush=True)

    for epoch in range(target_epochs):
        t0 = time.perf_counter()
        tl = run_epoch(model, train_loader, criterion, device,
                       optimizer=optimizer, scaler=scaler, use_amp=use_amp)
        scheduler.step()
        el = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:>3}/{target_epochs}  |  Train {tl:.6f}  |  "
              f"LR {lr:.2e}  |  {el:.1f}s", flush=True)

    ckpt_name = f"{task_name}_final.pth"
    ckpt_path = os.path.join(args.ckpt, ckpt_name)
    torch.save(
        build_inference_checkpoint(
            model, train_ds, task, task_name, args,
            {"epoch": target_epochs, "train_loss": float(tl),
             "train_all": True, "num_images": len(full_df)},
        ),
        ckpt_path,
    )

    print(f"\n{'='*70}")
    print(f"  Final model: {ckpt_path}")
    print(f"  Task:        {task_name} ({task['num_outputs']} outputs)")
    print(f"  Images:      {len(full_df)}")
    print(f"  Epochs:      {target_epochs}")
    print(f"  Final loss:  {tl:.6f}")
    if loso_loss:
        print(f"  LOSO est:    {loso_loss:.6f}")
    print(f"{'='*70}", flush=True)


# ==========================================================================
# Entry
# ==========================================================================

def main(args):
    device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)

    task = get_task(args.task)

    if args.mode == "loso" and not task["has_subject_id"]:
        print(f"ERROR: LOSO not available for {args.task} (no subject_id).")
        return

    if args.mode == "split":
        mode_split(args, device, task, args.task)
    elif args.mode == "loso":
        mode_loso(args, device, task, args.task)
    elif args.mode == "final":
        mode_final(args, device, task, args.task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified DINOv3 Iris Regression Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=["h8net", "cornernet", "eyelid_parabola", "eyelid_cubic", "circlenet"])
    parser.add_argument("--mode", required=True, choices=["split", "loso", "final"])
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--image_size", type=int, nargs=2, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--dino_repo_dir", default="./modules/dinov3")
    parser.add_argument("--dino_weights", default="./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "madgrad"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ckpt", default="./checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--loso_results", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
