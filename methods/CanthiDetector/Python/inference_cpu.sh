# Exit immediately if any command fails
set -e
set -o pipefail

python inference.py \
    --model_path "./models/CornerNet_V2_9100_2574_0.000411.pth" \
    --image_dir "./example_input" \
    --output_root "./inference_output" \
    --dino_repo_dir "./modules/dinov3" \
    --dino_weights "./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    --device "cpu"
