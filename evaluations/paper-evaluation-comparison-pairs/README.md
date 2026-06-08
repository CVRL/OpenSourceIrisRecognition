# Iris Recognition Dataset Evaluation Pairs

This repository provides the pre-generated genuine and imposter evaluation pairs used in our [paper](https://arxiv.org/abs/2605.20735) to ensure reproducibility when comparing open-source algorithms. 

## Datasets Overview

We utilize eight distinct datasets offering a wide variety of image qualities, demographics, and sensor types. *(Note: A copy of the Notre Dame datasets used in this work can be requested at the [CVRL Datasets Page](https://cvrl.nd.edu/projects/data/)).*

* **Q-FIRE:** Investigates the impact of image quality on recognition. Features NIR videos with intentional variations in resolution, lighting, focus, and motion blur. Samples were curated via a three-step process to ensure strict ISO compliance.
* **Warsaw-Biobase Post-Mortem Iris v3.0 (WBPMI):** A challenging benchmark of post-mortem NIR images. Due to tissue decay, irises lose shape and texture, effectively testing algorithmic robustness against non-ISO-compliant artifacts.
* **CASIA-Iris-Thousand V4:** Images from 1,000 subjects. Primary variations include eyeglasses and specular reflections, making it ideal for evaluating scalability and feature uniqueness.
* **CASIA-Iris-Lamp V4:** Captured with fluctuating illumination to intentionally cause pupil expansion and contraction. Excellent for testing robustness against significant non-linear iris texture deformations.
* **IITD-Iris:** Features an Indian demographic captured with real-world acquisition artifacts and no artificial constraints. Highly useful for evaluating the generalizability of segmentation algorithms.
* **NDIris3D (ND3D):** We utilize the authentic (non-PAD), noiseless subset captured under ideal conditions with varying NIR illumination angles. 
* **IIITD-CLI:** A contact lens database. We strictly utilize the baseline (no-lens) subset to evaluate core, unobstructed algorithmic performance.
* **Notre Dame VII-Q-R2:** A specialized subset prepared for NIST. Features a wide spectrum of naturally occurring real-world variations, including heavily closed eyelids, off-axis captures, non-ideal sizing, and varying exposure levels.

---

## Pair Generation Protocol

To standardize our evaluations, we generated specific image pairs for comparison. The pair generation logic strategically manages combinatorial explosion during imposter matching.

### Matching Rules
* **Laterality Strictness:** Comparisons are strictly intra-ocular. Left eyes are only ever compared to left eyes (L vs L), and right eyes are only compared to right eyes (R vs R).

### Pair Types
* **Genuine Pairs:** Formed by computing all possible combinations of images belonging to the exact same identity and the same eye type. 
* **Imposter Pairs:** Formed by comparing images from different identities (matching the same eye type). To maintain a balanced and computationally manageable dataset size, a fixed maximum number of images (`n_sample_imposter`) is randomly sampled per identity before generating the cross-identity combinations. This sampling was repeated every time a pair of identities was selected for comparison.

## Usage
The generated pairs are provided as `genuine.csv` and `imposter.csv` files within their respective dataset directories. Each CSV contains two columns (`image1`, `image2`) listing the precise filenames/relative file path to be compared during evaluation. The file names are provided without an image extension.
