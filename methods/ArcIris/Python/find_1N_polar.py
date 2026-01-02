from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm

def main(cfg):

    irisRec = irisRecognition(cfg)

    if not os.path.exists("./1N_data/enroll_im_polar/"):
        os.mkdir("./1N_data/enroll_im_polar/")
        os.mkdir("./1N_data/enroll_m_polar/")

    with open("./1N_data/enroll_1N.txt") as enrollFile:
        for line in tqdm(enrollFile):
            im = Image.fromarray(np.array(Image.open("./1N_data/" + line.strip()).convert("RGB"))[:, :, 0], "L")
            im = irisRec.fix_image(im)
            mask, pupil_xyr, iris_xyr = irisRec.segment_and_circApprox(im)
            im_polar, mask_polar = irisRec.cartToPol_torch(im, mask, pupil_xyr, iris_xyr)
            im_name = line.strip().split("/")[1]
            Image.fromarray(im_polar, "L").save("./1N_data/enroll_im_polar/" + im_name.split(".")[0] + ".png")
            Image.fromarray(mask_polar, "L").save("./1N_data/enroll_m_polar/" + im_name.split(".")[0] + ".png")
    
    if not os.path.exists("./1N_data/search_im_polar/"):
        os.mkdir("./1N_data/search_im_polar/")
        os.mkdir("./1N_data/search_m_polar/")
    
    with open("./1N_data/search_1N.txt") as searchFile:
        for line in tqdm(searchFile):
            im = Image.fromarray(np.array(Image.open("./1N_data/" + line.strip()).convert("RGB"))[:, :, 0], "L")
            im = irisRec.fix_image(im)
            mask, pupil_xyr, iris_xyr = irisRec.segment_and_circApprox(im)
            im_polar, mask_polar = irisRec.cartToPol_torch(im, mask, pupil_xyr, iris_xyr)
            im_name = line.strip().split("/")[1]
            Image.fromarray(im_polar, "L").save("./1N_data/search_im_polar/" + im_name.split(".")[0] + ".png")
            Image.fromarray(mask_polar, "L").save("./1N_data/search_m_polar/" + im_name.split(".")[0] + ".png")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg_baseline.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))