from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os
import cv2
import numpy as np

def main(cfg):

    irisRec = irisRecognition(cfg)
    
    if not os.path.exists('./templates/'):
        os.mkdir('./templates/')

    # Get the list of images to process
    filename_list = []
    image_list = []
    extensions = ["bmp", "png", "gif", "jpg", "jpeg", "tiff", "tif"]
    for ext in extensions:
        for filename in glob.glob("./data/*." + ext):
            im = Image.fromarray(np.array(Image.open(filename).convert("RGB"))[:, :, 0], "L")
            image_list.append(im)
            filename_list.append(os.path.basename(filename))

    # Segmentation, normalization and encoding
    vectors_list = []
    for im,fn in zip(image_list,filename_list):
        
        print(fn)

        # convert to ISO-compliant aspect ratio (4:3) and resize to ISO-compliant resolution: 640x480
        im = irisRec.fix_image(im)

        # segmentation mask and circular approximation:
        pupil_xyr, iris_xyr = irisRec.circApprox(im)

        # cartesian to polar transformation:
        im_polar = irisRec.cartToPol_torch(im, pupil_xyr, iris_xyr)

        # human-driven BSIF encoding:
        vector = irisRec.extractVector(im_polar)
        #print(code.shape)
        vectors_list.append(vector)

        # DEBUG: save selected processing results
        np.savez_compressed("./templates/" + os.path.splitext(fn)[0] + "_tmpl.npz",vector)

    # Matching (all-vs-all, as an example)
    for vector1, fn1, i in zip(vectors_list, filename_list, range(len(vectors_list))):
        for vector2, fn2, j in zip(vectors_list, filename_list, range(len(vectors_list))):
            if i < j:
                score = irisRec.matchVectors(vector1, vector2)
                print("{} <-> {} : {:.3f}".format(fn1,fn2,score))
     
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg_baseline.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))