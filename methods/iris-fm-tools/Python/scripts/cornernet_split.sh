#!/bin/bash
#$ -N Corner-Split
#$ -q gpu
#$ -l gpu=1
#$ -l h="qa-a10*"
#$ -j y
#$ -o logs/cornernet_split.log

set -e
set -o pipefail
fsync -d 60 "$SGE_STDOUT_PATH" &

cd ~/dinov3-iris-regression
conda activate dinov3

stdbuf -oL python train.py \
    --task "cornernet" \
    --mode "split" \
    --data_csv "./data/corner_labels_live.csv" \
    --image_dir "./data/images" \
    --image_size 640 480 \
    --dino_repo_dir "./modules/dinov3" \
    --dino_weights "./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    --feature_cache "./feature_cache" \
    --optimizer "adamw" \
    --epochs 300 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --dropout 0.3 \
    --patience 20 \
    --num_workers 0 \
    --ckpt "./checkpoint/cornernet" \
    --device "cuda"
