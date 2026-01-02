import os
import cv2
import math
import torch
import argparse
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from modules.backbone.corner_net import CornerNet


# -----------------------------------------------------
# Resize for DINO
# -----------------------------------------------------
def resize_transform(image: Image.Image, image_width: int, patch_size: int) -> torch.Tensor:
    w, h = image.size
    h_patches = image_width // patch_size
    w_patches = (w * image_width) // (h * patch_size)
    resized = TF.resize(image, (h_patches * patch_size, w_patches * patch_size))
    return TF.to_tensor(resized)


# -----------------------------------------------------
# Load DINOv3 model
# -----------------------------------------------------
def load_dino(dino_repo_dir, dino_weights, device):
    dino = torch.hub.load(
        dino_repo_dir,
        model="dinov3_vitl16",
        source="local",
        weights=dino_weights
    )
    dino.to(device)
    dino.eval()

    return dino


# -----------------------------------------------------
# Load CornerNet model
# -----------------------------------------------------
def load_corner_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CornerNet(in_channels=1024, out_channels=4)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    return model


# -----------------------------------------------------
# Predict single image
# -----------------------------------------------------
def predict_single(img: Image.Image, corner_model, dino_model, device):
    # Store original image size to map predictions later
    orig_w, orig_h = img.size

    # Resize image for DINO
    img_tensor = resize_transform(img, image_width=480, patch_size=16)

    # Normalize image
    img_tensor = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )(img_tensor)
    
    img_tensor = img_tensor.unsqueeze(0).to(device) 

    # Use mixed precision if CUDA
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    # Extract DINO features
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=dtype):
            feats = dino_model.get_intermediate_layers(
                img_tensor,
                n=range(24),
                reshape=True,
                norm=True
            )
            feats = feats[-1].to(device) 
    model_input = feats

    # Predict corners
    with torch.no_grad():
        pred = corner_model(model_input).squeeze().cpu()
    pred = torch.clamp(pred, 0, 1)

    # Map predictions to original image dimensions
    x1 = int(pred[0] * orig_w)
    y1 = int(pred[1] * orig_h)
    x2 = int(pred[2] * orig_w)
    y2 = int(pred[3] * orig_h)

    return x1, y1, x2, y2


# -----------------------------------------------------
# Draw corners
# -----------------------------------------------------
def draw_corners(img, x1, y1, x2, y2):
    # Draw left corner
    cv2.circle(img, (x1, y1), 8, (0, 255, 0), -1)
    cv2.putText(img, "L", (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw right corner
    cv2.circle(img, (x2, y2), 8, (0, 0, 255), -1)
    cv2.putText(img, "R", (x2 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return img


# -----------------------------------------------------
# Rotate image around center of two points
# -----------------------------------------------------
def rotate_image(img, x1, y1, x2, y2):
    # Compute midpoint as rotation center
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Compute angle to make the line horizontal
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    rotated = cv2.warpAffine(
        img, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(192, 192, 192)
    )

    return rotated


# -----------------------------------------------------
# Batch inference
# -----------------------------------------------------
def run_inference(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    dino_model = load_dino(args.dino_repo_dir, args.dino_weights, device)
    corner_model = load_corner_model(args.model_path, device)

    os.makedirs(args.output_root, exist_ok=True)
    pred_viz_dir = os.path.join(args.output_root, "pred_viz")
    pred_align_dir = os.path.join(args.output_root, "pred_align")
    os.makedirs(pred_viz_dir, exist_ok=True)
    os.makedirs(pred_align_dir, exist_ok=True)

    results = []

    for fname in os.listdir(args.image_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            continue

        img_path = os.path.join(args.image_dir, fname)
        img = Image.open(img_path).convert("RGB")

        x1, y1, x2, y2 = predict_single(img, corner_model, dino_model, device)
        print(f"{fname}: left=({x1},{y1}), right=({x2},{y2})")

        # Prediction visualization
        orig_img = cv2.imread(img_path)
        orig_img_viz = draw_corners(orig_img.copy(), x1, y1, x2, y2)
        cv2.imwrite(os.path.join(pred_viz_dir, fname), orig_img_viz)

        # Rotated visualization
        rotated_img = rotate_image(orig_img, x1, y1, x2, y2)
        cv2.imwrite(os.path.join(pred_align_dir, fname), rotated_img)

        results.append({
            "filename": fname,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        })

    df = pd.DataFrame(results) 
    csv_path = os.path.join(args.output_root, f"predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved at {csv_path}")


# -----------------------------------------------------
# Config
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/CornerNet_V1_5661_59_0.000131.pth")
    parser.add_argument("--image_dir", type=str, default="./example_input")
    parser.add_argument("--output_root", type=str, default="./inference_output")
    parser.add_argument("--dino_repo_dir", type=str, default="./modules/dinov3")
    parser.add_argument("--dino_weights", type=str, default="./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_inference(args)
