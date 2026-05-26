import os
import cv2
import numpy as np
import torch
import argparse
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from modules.models.regression_head import RegressionHead
from modules.dataset.task_configs import get_task, TASKS, align_horizontal


DINO_NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_dino(dino_repo_dir, dino_weights, device):
    dino = torch.hub.load(
        dino_repo_dir, model="dinov3_vitl16", source="local",
        weights=dino_weights, trust_repo=True,
    )
    dino.to(device).eval()
    return dino


def load_model(ckpt_path, device):
    """Load model and all metadata from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    task_name = ckpt.get("task", None)
    num_outputs = ckpt.get("num_outputs", None)
    normalization = ckpt.get("normalization", "zscore")
    use_sigmoid = ckpt.get("use_sigmoid", False)
    image_size = ckpt.get("image_size", [640, 480])
    train_w, train_h = image_size

    if task_name is None:
        raise ValueError("Checkpoint missing 'task' key. Cannot determine task type.")

    task = get_task(task_name)

    model = RegressionHead(
        in_channels=1024, out_channels=num_outputs,
        use_sigmoid=use_sigmoid,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # Normalization parameters
    label_mean = ckpt.get("label_mean", None)
    label_std = ckpt.get("label_std", None)
    norm_scale = ckpt.get("norm_scale", None)

    if label_mean is not None:
        label_mean = label_mean.to(device)
    if label_std is not None:
        label_std = label_std.to(device)
    if norm_scale is not None:
        if isinstance(norm_scale, torch.Tensor):
            norm_scale = norm_scale.to(device)

    info = {
        "task_name": task_name,
        "task": task,
        "num_outputs": num_outputs,
        "normalization": normalization,
        "train_w": train_w,
        "train_h": train_h,
        "label_mean": label_mean,
        "label_std": label_std,
        "norm_scale": norm_scale,
    }

    return model, info


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def predict_single(img_pil, model, dino_model, info, device):
    """
    Predict from a PIL image. Returns denormalized params in training resolution.
    """
    train_w, train_h = info["train_w"], info["train_h"]

    # Resize to training resolution
    img_resized = img_pil.resize((train_w, train_h), Image.LANCZOS)
    img_tensor = TF.to_tensor(img_resized)
    img_tensor = DINO_NORMALIZE(img_tensor).unsqueeze(0).to(device)

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ):
        feats = dino_model.get_intermediate_layers(
            img_tensor, n=1, reshape=True, norm=True
        )
        pred = model(feats[-1])

    # Denormalize
    norm = info["normalization"]
    if norm == "zscore":
        real = pred * info["label_std"] + info["label_mean"]
    else:
        real = pred * info["norm_scale"]

    return real.squeeze().cpu().numpy()


def scale_to_original(params, info, orig_w, orig_h):
    """
    Scale predictions from training resolution to original image size.
    Uses the task-specific rescale function.
    """
    train_w, train_h = info["train_w"], info["train_h"]

    if orig_w == train_w and orig_h == train_h:
        return params

    sx = orig_w / train_w
    sy = orig_h / train_h

    rescale_fn = info["task"]["rescale_fn"]
    return rescale_fn(params, sx, sy)


# --------------------------------------------------------------------------
# Main inference loop
# --------------------------------------------------------------------------

def run_inference(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    dino_model = load_dino(args.dino_repo_dir, args.dino_weights, device)
    model, info = load_model(args.model_path, device)

    task_name = info["task_name"]
    task = info["task"]
    train_w, train_h = info["train_w"], info["train_h"]

    print(f"\nTask:        {task_name}")
    print(f"Outputs:     {info['num_outputs']}")
    print(f"Norm:        {info['normalization']}")
    print(f"Train size:  {train_w}×{train_h}")
    if info["normalization"] == "zscore":
        print(f"Label mean:  {info['label_mean'].cpu().numpy()}")
        print(f"Label std:   {info['label_std'].cpu().numpy()}")
    elif isinstance(info["norm_scale"], torch.Tensor):
        print(f"Norm scale:  {info['norm_scale'].cpu().numpy()}  (per-output)")
    else:
        print(f"Norm scale:  {info['norm_scale']}")

    os.makedirs(args.output_root, exist_ok=True)
    overlay_dir = os.path.join(args.output_root, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    # CornerNet: also save aligned images
    aligned_dir = None
    if task_name == "cornernet":
        aligned_dir = os.path.join(args.output_root, "aligned")
        os.makedirs(aligned_dir, exist_ok=True)

    viz_fn = task["viz_fn"]
    results = []

    for fname in sorted(os.listdir(args.image_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            continue

        img_path = os.path.join(args.image_dir, fname)
        img_orig = cv2.imread(img_path)
        if img_orig is None:
            print(f"  Skip: {fname} (read error)")
            continue

        orig_h, orig_w = img_orig.shape[:2]

        # Predict in training resolution
        img_pil = Image.open(img_path).convert("RGB")
        params_train = predict_single(img_pil, model, dino_model, info, device)

        # Scale to original image size
        params_orig = scale_to_original(params_train, info, orig_w, orig_h)

        # Format output
        label_cols = task["label_columns"]
        param_str = "  ".join(f"{c}={v:.2f}" for c, v in zip(label_cols, params_orig))

        # CornerNet: compute alignment angle
        if task_name == "cornernet":
            x1, y1, x2, y2 = params_orig[:4]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            print(f"{fname} ({orig_w}×{orig_h})  {param_str}  angle={angle:.2f}°")
        else:
            print(f"{fname} ({orig_w}×{orig_h})  {param_str}")

        # Draw overlay on original image
        overlay = viz_fn(img_orig, params_orig)
        cv2.imwrite(os.path.join(overlay_dir, fname), overlay)

        # CornerNet: save horizontally aligned image
        if aligned_dir is not None:
            aligned = align_horizontal(img_orig, params_orig)
            cv2.imwrite(os.path.join(aligned_dir, fname), aligned)

        entry = {"filename": fname, "image_width": orig_w, "image_height": orig_h}
        for col, val in zip(label_cols, params_orig):
            entry[col] = float(val)
        if task_name == "cornernet":
            entry["alignment_angle_deg"] = float(angle)
        results.append(entry)

    # Save predictions CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_root, "predictions.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nInference complete ({task_name}). {len(results)} images.")
    print(f"CSV:      {csv_path}")
    print(f"Overlays: {overlay_dir}")
    if aligned_dir is not None:
        print(f"Aligned:  {aligned_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified DINOv3 Iris Regression Inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="./inference_output")
    parser.add_argument("--dino_repo_dir", type=str, default="./modules/dinov3")
    parser.add_argument("--dino_weights", type=str, default="./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_inference(args)
