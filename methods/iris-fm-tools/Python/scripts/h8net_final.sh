#!/bin/bash
#$ -N H8Net-Final
#$ -q gpu
#$ -l gpu=1
#$ -l h="qa-a10*"
#$ -j y
#$ -o logs/h8net_final.log

set -e
set -o pipefail
fsync -d 60 "$SGE_STDOUT_PATH" &

cd ~/dinov3-iris-regression
conda activate dinov3

stdbuf -oL python train.py \
    --task "h8net" \
    --mode "final" \
    --data_csv "./data/homography_labels.csv" \
    --image_dir "./data/images" \
    --image_size 320 240 \
    --dino_repo_dir "./modules/dinov3" \
    --dino_weights "./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    --feature_cache "./feature_cache" \
    --epochs 54 \
    --optimizer "adamw" \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --dropout 0.3 \
    --num_workers 0 \
    --ckpt "./checkpoint/h8net" \
    --device "cuda"
