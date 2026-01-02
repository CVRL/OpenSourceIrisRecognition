import os
import argparse
import pandas as pd
import numpy as np
import torch
import madgrad
from torch.utils.data import DataLoader
from modules.dataset.corner_dataset import IrisCornersDataset
from modules.backbone.corner_net import CornerNet

def main(args):
    device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)

    # ---------------------------
    # Load CSV
    # ---------------------------
    df = pd.read_csv(args.data_csv)
    subjects = df["subject_id"].unique()
    np.random.shuffle(subjects)
    print(f"Total subjects: {len(subjects)}", flush=True)

    # ---------------------------
    # Check DINO weights
    # ---------------------------
    if not args.dino_weights or not os.path.isfile(args.dino_weights):
        raise ValueError("DINO weights must be provided and exist!")

    print("Loading DINOv3 model...", flush=True)
    dino_model = torch.hub.load(
        args.dino_repo_dir,
        model="dinov3_vitl16",
        source="local",
        weights=args.dino_weights
    )
    dino_model.to(device)
    dino_model.eval()
    print("DINOv3 model loaded!", flush=True)

    # ---------------------------
    # Global best tracking
    # ---------------------------
    global_best = {
        'loss': float('inf'),
        'model_state': None,
        'fold_name': None,
        'optimizer': None
    }

    # ---------------------------
    # Subject-wise cross-validation
    # ---------------------------
    for fold_idx, val_subject in enumerate(subjects):
        print(f"\n=== Fold {fold_idx + 1}/{len(subjects)}: val subject {val_subject} ===", flush=True)

        train_df = df[df.subject_id != val_subject].reset_index(drop=True)
        val_df = df[df.subject_id == val_subject].reset_index(drop=True)

        # ---------------------------
        # Dataset and DataLoader
        # ---------------------------
        train_dataset = IrisCornersDataset(
            df=train_df,
            image_dir=args.image_dir,
            feature_extractor=dino_model,
            augment=True,
            aug_prob=0.7,
            device=device
        )
        val_dataset = IrisCornersDataset(
            df=val_df,
            image_dir=args.image_dir,
            feature_extractor=dino_model,
            device=device
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}", flush=True)

        # ---------------------------
        # Corner detection head
        # ---------------------------
        corner_model = CornerNet(in_channels=1024, out_channels=4)
        corner_model.to(device)

        # ---------------------------
        # Loss & optimizer
        # ---------------------------
        criterion = torch.nn.MSELoss()
        if args.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(corner_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = madgrad.MADGRAD(corner_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # ---------------------------
        # Mixed precision
        # ---------------------------
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler() if use_amp else None

        best_val_loss_fold = float('inf')
        epochs_no_improve = 0
        val_loss_history = []
        moving_avg_window = args.moving_avg_window

        # ---------------------------
        # Training loop
        # ---------------------------
        for epoch in range(args.epochs):
            corner_model.train()
            running_loss = 0.0
            for images, corners in train_loader:
                images, corners = images.to(device), corners.to(device)
                optimizer.zero_grad()

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = corner_model(images)
                    loss = criterion(outputs, corners)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * images.size(0)

            train_loss = running_loss / len(train_dataset)

            # ---------------------------
            # Validation
            # ---------------------------
            corner_model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for images, corners in val_loader:
                    images, corners = images.to(device), corners.to(device)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        outputs = corner_model(images)
                        val_loss_total += criterion(outputs, corners).item() * images.size(0)

            val_loss = val_loss_total / len(val_dataset)
            val_loss_history.append(val_loss)

            # Moving average
            if len(val_loss_history) >= moving_avg_window:
                avg_val_loss = np.mean(val_loss_history[-moving_avg_window:])
            else:
                avg_val_loss = np.mean(val_loss_history)

            print(f"[Fold {fold_idx + 1}] Epoch [{epoch + 1}/{args.epochs}] "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}", flush=True)

            # ---------------------------
            # Update best for fold
            # ---------------------------
            if avg_val_loss < best_val_loss_fold:
                best_val_loss_fold = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # ---------------------------
            # Update global best 
            # ---------------------------
            if avg_val_loss < global_best['loss']:
                global_best['loss'] = avg_val_loss
                global_best['model_state'] = corner_model.state_dict()
                global_best['fold_name'] = f"fold{fold_idx + 1}_subject{val_subject}"
                global_best['optimizer'] = args.optimizer

                os.makedirs(args.ckpt, exist_ok=True)
                ckpt_path = os.path.join(args.ckpt, f"CornerNet_{args.optimizer.upper()}_W{moving_avg_window}.pth")
                torch.save(global_best, ckpt_path)
                print(f"Global best model saved at {ckpt_path}", flush=True)

            # ---------------------------
            # Early stopping
            # ---------------------------
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Iris Canthi Detection Model with DINOv3")
    parser.add_argument('--data_csv', type=str, default="./metadata/corner_labels.csv")
    parser.add_argument('--image_dir', type=str, default="./bxgrid-canthi-dataset/images")
    parser.add_argument('--dino_repo_dir', type=str, default="./modules/dinov3")
    parser.add_argument('--dino_weights', type=str, default="./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument('--optimizer', type=str, default="adamw")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--moving_avg_window', type=int, default=1, help="Window size for moving average of validation loss")
    parser.add_argument('--ckpt', type=str, default="./checkpoint")
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
