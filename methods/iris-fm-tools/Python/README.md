## Installation

**Requirements:** Python 3.11, CUDA-capable GPU recommended.

```bash
# 1. Clone the repository 
git clone --recurse-submodules https://github.com/CVRL/OpenSourceIrisRecognition/methods/iris-fm-tools.git
cd iris-fm-tools

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate dinov3
```

## Pretrained Weights

All pretrained task checkpoints and the DINOv3 backbone weights can be downloaded from [this Google folder](https://drive.google.com/drive/folders/1XwFMKeNyBMkHOcbJ-8P6iwcDNxZQ9MzD?usp=sharing).

The folder contains:

| File | Description |
|---|---|
| `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | DINOv3 ViT-L/16 backbone (frozen, shared by all tasks) |
| `circlenet.pth` | CircleNet: circular approximations of the inner and outer iris boundaries |
| `cornernet_live.pth` | CornerNet: lateral and medial eye canthi detection, live irises |
| `cornernet_pmi.pth` | CornerNet: lateral and medial eye canthi detection, post-mortem irises |
| `eyelidnet_parabola.pth` | EyelidNet: parabolic approximation of eyelid curves |
| `eyelidnet_cubic.pth` | EyelidNet: cubic approximation of eyelid curves |
| `h8net.pth` | H8Net: estimation of projective transformation matrix for off-axis gaze correction |

**Placement:** After downloading, place the entire `models/` folder in the repository root so the layout matches exactly:

```
iris-fm-tools/
└── models/
    ├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
    ├── circlenet.pth
    ├── cornernet_live.pth
    ├── cornernet_pmi.pth
    ├── eyelidnet_parabola.pth
    ├── eyelidnet_cubic.pth
    └── h8net.pth
```

All scripts and CLI examples reference weights from `./models/` relative to the repository root. No path changes are needed if the folder is placed correctly.

## Inference

The task, normalization parameters, and training resolution are all stored inside the `.pth` checkpoints and restored automatically — no additional flags are required beyond the model path and image directory.

```bash
bash scripts/inference.sh
```

Or call directly for any individual model, for instance:

```bash
# CircleNet
python inference.py \
    --model_path  ./models/circlenet.pth \
    --image_dir   ./test_images \
    --output_root ./inference_output/circlenet \
    --dino_repo_dir ./modules/dinov3 \
    --dino_weights  ./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --device cuda

# CornerNet (live)
python inference.py \
    --model_path  ./models/cornernet_live.pth \
    --image_dir   ./test_images \
    --output_root ./inference_output/cornernet_live \
    --dino_repo_dir ./modules/dinov3 \
    --dino_weights  ./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --device cuda
```

**Output layout:**

```
inference_output/<model>/
├── overlays/        # input images with predicted annotations drawn
├── aligned/         # (CornerNet only) horizontally aligned images
└── predictions.csv  # predicted parameters for every image
```

## Training

Three training modes are available for all tasks:

| Mode | Description |
|------|-------------|
| `split` | Subject-disjoint (or random) 80/20 train/val split with early stopping |
| `loso` | Leave-One-Subject-Out cross-validation; saves per-fold checkpoints and a `loso_results.json` summary |
| `final` | Full-dataset training; pass `--loso_results` to inherit the median best epoch automatically |

**Via script:**

```bash
bash scripts/h8net_loso.sh
```

**Direct CLI:**

```bash
python train.py \
    --task          h8net \
    --mode          loso \
    --data_csv      ./data/homography_labels.csv \
    --image_dir     ./data/images \
    --image_size    384 288 \
    --dino_repo_dir ./modules/dinov3 \
    --dino_weights  ./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --feature_cache ./feature_cache \
    --optimizer     adamw \
    --epochs        200 \
    --batch_size    64 \
    --lr            1e-4 \
    --weight_decay  1e-4 \
    --dropout       0.3 \
    --patience      20 \
    --num_workers   0 \
    --ckpt          ./checkpoint_loso/h8net \
    --device        cuda
```

Replace `--task` and `--data_csv` with the appropriate values for other models. Equivalent scripts for every task × mode combination are provided in the `scripts/` folder.

### Checkpoint format

Every saved `.pth` is inference-compatible and self-contained:

```
model_state, task, num_outputs, normalization, use_sigmoid,
image_size, args,
label_mean / label_std    (zscore tasks)
norm_scale                (wh / image tasks)
```

Optimizer and scheduler states are saved separately as `{task}_resume.pth` and are used exclusively by `--resume`. 

**Resume training**

```bash
python train.py --task h8net --mode split ... --resume
# Restores from h8net_resume.pth (full optimizer state).
# Falls back to h8net_best.pth (weights only) if resume file is absent.
```

## Project Structure

```
iris-fm-tools/
├── assets/                              # Teaser images for each model
│   ├── teaser_circlenet.png
│   ├── teaser_cornernet.png
│   ├── teaser_eyelidnet_cubic.png
│   ├── teaser_eyelidnet_parabola.png
│   └── teaser_h8net.png
│
├── data/                                # Label CSVs (one per task / imaging condition)
│   ├── circle_labels.csv
│   ├── corner_labels_live.csv
│   ├── corner_labels_pmi.csv
│   ├── eyelid_labels_cubic.csv
│   ├── eyelid_labels_parabola.csv
│   └── homography_labels.csv
│
├── models/                              # Pretrained checkpoints + DINOv3 backbone weights
│   ├── circlenet.pth
│   ├── cornernet_live.pth
│   ├── cornernet_pmi.pth
│   ├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
│   ├── eyelidnet_cubic.pth
│   ├── eyelidnet_parabola.pth
│   └── h8net.pth
│
├── modules/
│   ├── dataset/
│   │   ├── iris_dataset.py              # Unified dataset: loading, per-image rescaling,
│   │   │                                #   DINOv3 feature caching, and augmentation
│   │   └── task_configs.py              # Per-task label columns, normalization strategy,
│   │                                    #   rescale / translate / visualization functions
│   ├── dinov3/                          # DINOv3 repository (local torch.hub source)
│   └── models/
│       └── regression_head.py           # Shared prediction head
│
├── scripts/                             # SGE job scripts
│   ├── inference.sh
│   ├── circlenet_{split,final}.sh
│   ├── cornernet_{split,loso,final}.sh
│   ├── eyelid_cubic_{split,loso,final}.sh
│   ├── eyelid_parabola_{split,loso,final}.sh
│   └── h8net_{split,loso,final}.sh
│
├── environment.yml
├── inference.py                         # Unified inference entry-point
└── train.py                             # Unified training entry-point
```
