import os
import random
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split as _tts

from modules.dataset.task_configs import get_task


DINO_NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


class IrisDataset(Dataset):
    """
    Unified dataset for all iris regression tasks.

    Args:
        df:                    DataFrame with filepath and label columns.
        image_dir:             Root directory containing the images.
        task_name:             Task key from TASKS registry.
        feature_extractor:     DINOv3 model (eval mode, frozen).
        device:                Torch device for the backbone.
        target_w:              Target image width (must be divisible by 16).
        target_h:              Target image height (must be divisible by 16).
        cache_dir:             If set, original-image features cached to disk.
        label_mean:            Override normalization mean (for val set).
        label_std:             Override normalization std (for val set).
        norm_scale:            Override normalization scale (for val set).
        augment:               Enable translation augmentation.
        aug_translate_prob:    Probability of applying translation per sample.
        aug_translate_max:     Max translation as a fraction of image dimension.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        task_name: str,
        feature_extractor=None,
        device: str = "cpu",
        target_w: int = 640,
        target_h: int = 480,
        cache_dir: str | None = None,
        label_mean: np.ndarray | None = None,
        label_std: np.ndarray | None = None,
        norm_scale=None,
        augment: bool = False,
        aug_translate_prob: float = 0.5,
        aug_translate_max: float = 0.20,
    ):
        if feature_extractor is None:
            raise ValueError("A DINOv3 model must be provided.")

        assert target_w % 16 == 0 and target_h % 16 == 0, \
            f"Target size ({target_w}×{target_h}) must be divisible by patch_size=16"

        self.task = get_task(task_name)
        self.task_name = task_name
        self.image_dir = image_dir
        self.target_w = target_w
        self.target_h = target_h
        self.cache_dir = cache_dir
        self.label_columns = self.task["label_columns"]
        self.filename_col = self.task["filename_col"]
        self.num_outputs = self.task["num_outputs"]

        # Augmentation config — only active if task defines a translate_fn
        self.translate_fn = self.task.get("translate_fn", None)
        self.augment = augment and (self.translate_fn is not None)
        self.aug_translate_prob = aug_translate_prob
        self.aug_translate_max = aug_translate_max

        if augment and self.translate_fn is None:
            print(f"  [aug] augment=True but task '{task_name}' has no translate_fn — "
                  f"augmentation disabled.", flush=True)
        elif self.augment:
            print(f"  [aug] Translation augmentation enabled: "
                  f"p={aug_translate_prob}, max_shift={aug_translate_max*100:.0f}%",
                  flush=True)
            print(f"  [aug] Photometric jitter enabled (brightness/contrast/saturation)",
                  flush=True)

        # Photometric jitter — only for tasks that opt in via aug_color_jitter=True
        self.color_jitter = (
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.04)
            if self.augment and self.task.get("aug_color_jitter", False) else None
        )

        # ------------------------------------------------------------------
        # Per-image label rescaling
        # ------------------------------------------------------------------
        df = df.copy()
        for col in self.label_columns:
            df[col] = df[col].astype(np.float32)

        rescale_fn = self.task["rescale_fn"]
        raw_all = df[self.label_columns].values.astype(np.float32)
        n_rescaled = 0

        for i in range(len(df)):
            img_path = os.path.join(image_dir, str(df.iloc[i][self.filename_col]))
            with Image.open(img_path) as img_hdr:
                orig_w_i, orig_h_i = img_hdr.size  # header-only read
            sx_i = target_w / orig_w_i
            sy_i = target_h / orig_h_i
            if sx_i != 1.0 or sy_i != 1.0:
                raw_all[i] = rescale_fn(raw_all[i], sx_i, sy_i)
                n_rescaled += 1

        if n_rescaled > 0:
            df[self.label_columns] = raw_all
            print(f"  Labels rescaled per-image: {n_rescaled}/{len(df)} images "
                  f"had original dimensions different from {target_w}×{target_h}",
                  flush=True)
        else:
            print(f"  Labels: no rescaling needed (all images are {target_w}×{target_h})",
                  flush=True)

        self.df = df.reset_index(drop=True)

        # ------------------------------------------------------------------
        # Normalization
        # ------------------------------------------------------------------
        norm_type = self.task["normalization"]

        if norm_type == "zscore":
            if label_mean is None or label_std is None:
                raw = self.df[self.label_columns].values.astype("float32")
                self.label_mean = raw.mean(axis=0)
                self.label_std = raw.std(axis=0)
                self.label_std[self.label_std < 1e-8] = 1.0
            else:
                self.label_mean = label_mean.astype("float32")
                self.label_std = label_std.astype("float32")
            self.norm_scale = None

        elif norm_type == "wh":
            if norm_scale is not None:
                if isinstance(norm_scale, torch.Tensor):
                    self.norm_scale = norm_scale.numpy().astype(np.float32)
                else:
                    self.norm_scale = np.asarray(norm_scale, dtype=np.float32)
            else:
                axes = self.task["norm_axes"]
                assert axes is not None, \
                    f"Task '{task_name}' has normalization='wh' but no norm_axes defined."
                self.norm_scale = np.array(
                    [float(target_w) if ax in ("x", "r") else float(target_h)
                     for ax in axes],
                    dtype=np.float32,
                )
            self.label_mean = None
            self.label_std = None
            col_scale = dict(zip(self.label_columns, self.norm_scale.tolist()))
            print(f"  WH normalization (per-output divisors): {col_scale}", flush=True)

        else:  # "image"
            self.norm_scale = norm_scale if norm_scale is not None else float(target_w)
            self.label_mean = None
            self.label_std = None
            print(f"  Image normalization: divide by {self.norm_scale}", flush=True)

        # ------------------------------------------------------------------
        # Backbone
        # ------------------------------------------------------------------
        self.device = torch.device(device)
        self.feature_extractor = feature_extractor.to(self.device)
        self.feature_extractor.eval()

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_label_mean(self) -> torch.Tensor | None:
        return torch.from_numpy(self.label_mean) if self.label_mean is not None else None

    def get_label_std(self) -> torch.Tensor | None:
        return torch.from_numpy(self.label_std) if self.label_std is not None else None

    def get_norm_scale(self) -> torch.Tensor | float | None:
        if self.norm_scale is None:
            return None
        if isinstance(self.norm_scale, np.ndarray):
            return torch.from_numpy(self.norm_scale.copy())
        return self.norm_scale

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_and_resize(self, path: str) -> Image.Image:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read: {path}")
        h, w = img.shape[:2]
        if w != self.target_w or h != self.target_h:
            interp = cv2.INTER_AREA if (w > self.target_w) else cv2.INTER_LANCZOS4
            img = cv2.resize(img, (self.target_w, self.target_h), interpolation=interp)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _apply_translate(
        self, img_pil: Image.Image, raw_labels: np.ndarray
    ) -> tuple[Image.Image, np.ndarray]:

        max_dx = self.aug_translate_max * self.target_w
        max_dy = self.aug_translate_max * self.target_h
        dx = int(random.uniform(-max_dx, max_dx))
        dy = int(random.uniform(-max_dy, max_dy))

        img_arr = np.array(img_pil)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(
            img_arr, M, (self.target_w, self.target_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(192, 192, 192),
        )

        updated_labels = self.translate_fn(raw_labels, dx, dy, self.target_w)
        return Image.fromarray(shifted), updated_labels

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, img_pil: Image.Image) -> torch.Tensor:
        img_tensor = TF.to_tensor(img_pil)
        img_tensor = DINO_NORMALIZE(img_tensor).unsqueeze(0).to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ):
            feats = self.feature_extractor.get_intermediate_layers(
                img_tensor, n=1, reshape=True, norm=True
            )
            return feats[-1].squeeze(0).detach().cpu()

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cache_path(self, filename: str) -> str | None:
        if self.cache_dir is None:
            return None
        stem = os.path.splitext(filename)[0]
        return os.path.join(self.cache_dir, f"{stem}_{self.target_w}x{self.target_h}.pt")

    def _load_cached(self, path: str) -> torch.Tensor | None:
        if path and os.path.isfile(path):
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def _save_cached(self, path: str, tensor: torch.Tensor):
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            torch.save(tensor, tmp)
            os.replace(tmp, path)

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row[self.filename_col]
        raw = row[self.label_columns].values.astype("float32")

        augmented = False
        img_pil = None

        if self.augment and random.random() < self.aug_translate_prob:
            img_pil = self._load_and_resize(os.path.join(self.image_dir, filename))

            # Translation
            img_pil, raw = self._apply_translate(img_pil, raw)

            # Photometric jitter (only for tasks with aug_color_jitter=True)
            if self.color_jitter is not None:
                img_pil = self.color_jitter(img_pil)

            augmented = True

        # ------------------------------------------------------------------
        # Feature extraction
        # ------------------------------------------------------------------
        # Augmented samples always need fresh features (cache stores the
        # original un-shifted image).  Non-augmented samples use the cache.
        if augmented:
            features = self._extract_features(img_pil)
        else:
            cache_path = self._cache_path(filename)
            features = self._load_cached(cache_path)
            if features is None:
                if img_pil is None:
                    img_pil = self._load_and_resize(
                        os.path.join(self.image_dir, filename)
                    )
                features = self._extract_features(img_pil)
                self._save_cached(cache_path, features)

        # ------------------------------------------------------------------
        # Normalization (applied after any label update from augmentation)
        # ------------------------------------------------------------------
        if self.task["normalization"] == "zscore":
            normalized = (raw - self.label_mean) / self.label_std
        else:
            normalized = raw / self.norm_scale

        return features, torch.from_numpy(normalized)


# ======================================================================
# Train / Val splitting
# ======================================================================

def prepare_datasets(
    csv_path: str,
    task_name: str,
    train_size: float = 0.8,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task = get_task(task_name)
    df = pd.read_csv(csv_path)

    if task["has_subject_id"]:
        subj_col = task["subject_id_col"]
        subjects = df[subj_col].unique()
        train_subs, val_subs = _tts(subjects, train_size=train_size, random_state=seed)
        train_df = df[df[subj_col].isin(train_subs)].reset_index(drop=True)
        val_df = df[df[subj_col].isin(val_subs)].reset_index(drop=True)

        print(f"--- Subject-Disjoint Split ({task_name}) ---", flush=True)
        print(f"  Total subjects : {len(subjects)}", flush=True)
        print(f"  Train          : {len(train_df):>6} images  ({len(train_subs)} subjects)", flush=True)
        print(f"  Val            : {len(val_df):>6} images  ({len(val_subs)} subjects)", flush=True)
    else:
        train_df, val_df = _tts(df, train_size=train_size, random_state=seed, shuffle=True)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        print(f"--- Random Split ({task_name}) ---", flush=True)
        print(f"  Total  : {len(df):>6} images", flush=True)
        print(f"  Train  : {len(train_df):>6} images", flush=True)
        print(f"  Val    : {len(val_df):>6} images", flush=True)

    filter_col = task["val_filter_col"]
    if filter_col and filter_col in val_df.columns:
        before = len(val_df)
        val_df = val_df[val_df[filter_col] == "orig"].reset_index(drop=True)
        print(f"  Val filtered   : {before} → {len(val_df)} (orig only)", flush=True)

    print(f"  Outputs        : {task['num_outputs']} ({task['label_columns']})", flush=True)
    print(f"  Normalization  : {task['normalization']}", flush=True)

    return train_df, val_df
