import os
import random
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# DINO normalization
dino_transform = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

def resize_transform(image: Image.Image, image_width: int, patch_size: int) -> torch.Tensor:
    # Resize image so height and width are multiples of PATCH_SIZE
    w, h = image.size
    h_patches = image_width // patch_size
    w_patches = (w * image_width) // (h * patch_size)
    resized = TF.resize(image, (h_patches * patch_size, w_patches * patch_size))
    return TF.to_tensor(resized)

def transform_points(points, M):
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.transform(pts, M).reshape(-1, 2)
    return [(int(round(x)), int(round(y))) for x, y in transformed]

def zoom_in_limits(x1, y1, x2, y2, img_w, img_h, max_scale=1.25):
    cx, cy = img_w / 2, img_h / 2
    dx1, dy1 = x1 - cx, y1 - cy
    dx2, dy2 = x2 - cx, y2 - cy

    scale_x1 = cx / (cx - dx1) if dx1 < 0 else (img_w - cx) / dx1 if dx1 > 0 else max_scale
    scale_y1 = cy / (cy - dy1) if dy1 < 0 else (img_h - cy) / dy1 if dy1 > 0 else max_scale
    scale_x2 = cx / (cx - dx2) if dx2 < 0 else (img_w - cx) / dx2 if dx2 > 0 else max_scale
    scale_y2 = cy / (cy - dy2) if dy2 < 0 else (img_h - cy) / dy2 if dy2 > 0 else max_scale

    safe_scale = min(scale_x1, scale_y1, scale_x2, scale_y2, max_scale)
    safe_scale = max(1.05, safe_scale)  
    return safe_scale


def random_iris_augmentation(image, x1, y1, x2, y2):
    # Random translation, rotation, zoom in/out
    h, w = image.shape[:2]
    corners = [(x1, y1), (x2, y2)]
    aug_type = random.choice(["translate", "rotate", "zoom_in", "zoom_out"])
    M = np.float32([[1, 0, 0], [0, 1, 0]])

    if aug_type == "translate":
        # Random shift within ±10% of image size
        tx = random.randint(-int(w * 0.15), int(w * 0.15))
        ty = random.randint(-int(h * 0.15), int(h * 0.15))
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        aug = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


    elif aug_type == "rotate":
        angle = random.uniform(-30, 30)
        cx, cy = w/2, h/2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0).astype(np.float32)
        aug = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    elif aug_type == "zoom_in":
        scale = random.uniform(1.05, zoom_in_limits(x1, y1, x2, y2, w, h))
        cx, cy = w/2, h/2
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale).astype(np.float32)
        aug = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    else:  # zoom_out
        scale = random.uniform(0.65, 0.95)
        new_w, new_h = max(1,int(round(w*scale))), max(1,int(round(h*scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros_like(image)
        tx, ty = (w-new_w)//2, (h-new_h)//2
        canvas[ty:ty+new_h, tx:tx+new_w] = resized
        aug = canvas
        M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)

    if aug.shape[0]!=h or aug.shape[1]!=w:
        aug = cv2.resize(aug, (w, h), interpolation=cv2.INTER_LINEAR)

    new_corners = transform_points(corners, M)
    (nx1, ny1), (nx2, ny2) = new_corners
    nx1, ny1 = int(np.clip(nx1,0,w-1)), int(np.clip(ny1,0,h-1))
    nx2, ny2 = int(np.clip(nx2,0,w-1)), int(np.clip(ny2,0,h-1))

    return aug, nx1, ny1, nx2, ny2

class IrisCornersDataset(Dataset):
    def __init__(self, df, image_dir, feature_extractor=None, augment=False, aug_prob=0.5, device="cpu"):
        if feature_extractor is None:
            raise ValueError("You must provide a DINO model as feature_extractor!")

        self.df = df
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.aug_prob = aug_prob
        self.device = torch.device(device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])

        # Read image (handle gray/RGB/RGBA)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read {img_path}")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        orig_h, orig_w = img.shape[:2]
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']

        # Apply augmentation with probability aug_prob
        if self.augment and random.random() < self.aug_prob:
            img, x1, y1, x2, y2 = random_iris_augmentation(img, x1, y1, x2, y2)

        # DINO feature extraction
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = resize_transform(img_pil, image_width=480, patch_size=16)
        img_tensor = dino_transform(img_tensor).unsqueeze(0).to(self.device)

        dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=dtype):
                feats = self.feature_extractor.get_intermediate_layers(
                    img_tensor, n=range(24), reshape=True, norm=True
                )
                img_out = feats[-1].squeeze(0).detach().cpu()

        # Normalize corner coordinates
        coords = torch.tensor([
            x1 / orig_w, 
            y1 / orig_h,
            x2 / orig_w, 
            y2 / orig_h
        ], dtype=torch.float32)

        return img_out, coords