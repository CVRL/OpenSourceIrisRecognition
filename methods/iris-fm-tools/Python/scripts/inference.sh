#!/bin/bash
#$ -N Inference
#$ -q gpu
#$ -l gpu=1
#$ -l h="qa-a10*"
#$ -j y
#$ -o logs/inference.log

set -e
set -o pipefail

cd ~/dinov3-iris-regression
conda activate dinov3

# -----------------------------------------------
# Inference - auto-detects task from checkpoint
# Change --model_path and --image_dir as needed
# -----------------------------------------------

# Example: CircleNet
stdbuf -oL python inference.py \
    --model_path "./models/circlenet.pth" \
    --image_dir "./test_images" \
    --output_root "./inference_output/circlenet" \
    --dino_repo_dir "./modules/dinov3" \
    --dino_weights "./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    --device "cuda"

# Example: H8Net 
stdbuf -oL python inference.py \
    --model_path "./models/h8net.pth" \
    --image_dir "./test_images" \
    --output_root "./inference_output/h8net" \
    --dino_repo_dir "./modules/dinov3" \
    --dino_weights "./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    --device "cuda"

# Example: CornerNet live
stdbuf -oL python inference.py \
    --model_path "./models/cornernet_live.pth" \
    --image_dir "./test_images" \
    --output_root "./inference_output/cornernet_live" \
    --dino_repo_dir "./modules/dinov3" \
    --dino_weights "./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    --device "cuda"

# Example: CornerNet pmi
stdbuf -oL python inference.py \
    --model_path "./models/cornernet_pmi.pth" \
    --image_dir "./test_images" \
    --output_root "./inference_output/cornernet_pmi" \
    --dino_repo_dir "./modules/dinov3" \
    --dino_weights "./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    --device "cuda"

# Example: EyelidNet parabola 
stdbuf -oL python inference.py \
    --model_path "./models/eyelidnet_parabola.pth" \
    --image_dir "./test_images" \
    --output_root "./inference_output/eyelid_parabola" \
    --dino_repo_dir "./modules/dinov3" \
    --dino_weights "./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    --device "cuda"

# Example: EyelidNet cubic 
stdbuf -oL python inference.py \
    --model_path "./models/eyelidnet_cubic.pth" \
    --image_dir "./test_images" \
    --output_root "./inference_output/eyelid_cubic" \
    --dino_repo_dir "./modules/dinov3" \
    --dino_weights "./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    --device "cuda"
