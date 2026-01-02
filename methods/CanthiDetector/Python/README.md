## Installation

1. Clone the repository

```bash
git clone https://github.com/CVRL/OpenSourceIrisRecognition/tree/main/methods/canthi-detector
cd canthi-detector
```

2. Create and activate conda environment

```bash
conda env create -f environment.yml
conda activate canthi
```

3. Download DINOv3 Model

Download the pretrained **DINOv3 ViT-L/16** model (`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`) from the [model repository](https://notredame.box.com/s/9km8z2nvrh649pkavo9hvp7l8ijqk34x) and place it in the `./models/` folder.


## Inference

You can run inference either by using the bash script:

```bash
bash inference_cuda.sh
```
or 

```bash
bash inference_cpu.sh
```

or by calling the Python script directly:

```bash
python inference.py --model_path "./models/CornerNet_5661_59_0.000131.pth" --image_dir "./example_input" --output_root "./inference_output" --dino_repo_dir "./modules/dinov3" --dino_weights "./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" --device "cuda"
```

Predictions are saved in `./inference_output/`:

```
inference_output/
├── pred_viz/     # Images with predicted corners
├── pred_align/   # Rotated/aligned images
└── predictions.csv
```

**Note:** Input images **must be 640×480 px**. The models were trained with NIR (ISO/IEC 19794-6-compliant) iris images. The models were neither trained nor tested for non-ISO/IEC-compliant iris images.

## Currently Available Pre-trained Models

1. V1: `CornerNet_V1_5661_59_0.000131.pth`
2. V2: `CornerNet_V2_9100_2574_0.000411.pth`
3. PMI_V1: `CornerNet_PMI_838_422_0.000974.pth`

### Model V1: CornerNet_V1_5661_59_0.000131.pth

Training Dataset:
- Subset of [public Notre Dame data](https://cvrl.nd.edu/projects/data/)

Dataset Statistics:
- Total number of samples: 898
- Number of unique subjects: 18

Training Strategy:
- Subject-disjoint cross-validation
- Leave-one-subject-out validation
- One subject used for validation, remaining subjects for training
- Best global model across folds was saved

Data Augmentation (training set only):
- Augmentation probability: 0.7
- For a given training sample, if the random probability meets the threshold, **one** augmentation is randomly selected and applied
- Only a single augmentation is applied per image (not all augmentations)
- Possible augmentation operations:
	* Random translation
	* Random rotation
	* Random zoom-in
	* Random zoom-out


### Model V2: CornerNet_V2_9100_2574_0.000411.pth

Training Dataset:
- Subset of [public Notre Dame data](https://cvrl.nd.edu/projects/data/)

Original Dataset Statistics:
- Total number of samples: 898
- Number of unique subjects: 18

Training Strategy:
- Subject-disjoint split
- 4 subjects randomly selected for validation
- 14 subjects used for training

Data Augmentation:
- Heavy augmentation applied to training data
- For each original training sample, 12 augmented samples generated
- Augmentation methods:
	* Random translation
	* Random rotation
	* Random zoom-in
	* Random zoom-out
- Each augmentation applied 3 times per sample

Final Training Set Size:
- 9,100 samples


### Model PMI_V1: CornerNet_PMI_838_422_0.000974.pth

Training Dataset:
- Subset of public post-mortem iris datasets: [[Warsaw]](http://zbum.ia.pw.edu.pl/EN/node/46) and [[NIJ]](https://nij.ojp.gov/library/publications/software-tool-and-methodology-enhancement-unidentified-decedent-systems)

Dataset Statistics:
- Total number of samples: 1,260
- Number of unique PMIs: 211

Training / Validation Split:
- Training samples: 838
- Validation samples: 422

Sample Selection Strategy (For Each PMI):
- Randomly selected 2–3 right-eye iris images
- Randomly selected 2–3 left-eye iris images

## Training

You can train your own canthi detector with your own annotated data. First, prepare your training data:

- **Images:** Must be of **640×480 px** resolution. Place them in the `./example_input/` folder. 
- **Annotations:** CSV file (e.g., `corner_labels.csv`) with columns: `filename, subject_id, x1, y1, x2, y2`.

Start training either by running the bash script:

```bash
bash train.sh
```

or by running the Python script directly:

```bash
python train.py --data_csv "./metadata/corner_labels.csv" --image_dir "./bxgrid-canthi-dataset/images" --dino_repo_dir "./modules/dinov3" --dino_weights "./models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" --device "cuda"
```

Checkpoints are saved in `./checkpoint/`. Optional arguments in `train.py` include: `--epochs`, `--batch_size`, `--lr`, `--optimizer`, `--patience`.

**Note:** Pre-trained DINOv3 model is required for training, since it deliveres embeddings for the regression model being trained (DINO model is frozen and not updated).
