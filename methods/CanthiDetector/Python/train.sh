#!/bin/bash
#$ -N canthi
#$ -q gpu
#$ -l gpu=1
#$ -l h="qa-a10*"
#$ -j y
#$ -o logs/train.log

# -------------------------------
# Exit immediately if any command fails
# -------------------------------
set -e
set -o pipefail

# Periodically flush stdout buffer every 60 seconds
fsync -d 60 "$SGE_STDOUT_PATH" &

# -------------------------------
# Navigate to project directory
# -------------------------------
cd ~/canthi-detector

# -------------------------------
# Activate virtual environment
# -------------------------------
conda activate canthi

# -------------------------------
# Run the training script
# -------------------------------
stdbuf -oL python train.py \
    --data_csv "./metadata/corner_labels.csv" \
    --image_dir "./bxgrid-canthi-dataset/images" \
    --dino_repo_dir "./modules/dinov3" \
    --dino_weights "./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    --optimizer "adamw" \
    --epochs 1000 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --patience 20 \
    --moving_avg_window 1 \
    --ckpt "./checkpoint" \
    --device "cuda"
