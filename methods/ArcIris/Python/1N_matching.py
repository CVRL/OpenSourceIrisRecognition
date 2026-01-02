from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle as pkl


def cmc(search_uids, search_vectors, enrolled_uids, enrolled_vectors, irisRec, topk=30):
    valid_queries = 0
    all_rank = []
    sum_rank = np.zeros(topk)
    for search_uid, search_vector in tqdm(zip(search_uids, search_vectors), total=len(search_uids)):
        # Calculate the distances for each query
        distmat = []
        for enrolled_uid, enrolled_vector in zip(enrolled_uids, enrolled_vectors):
            # Get the label from the image
            dist = irisRec.matchVectors(search_vector, enrolled_vector)
            distmat.append([dist, enrolled_uid])
        
        distmat.sort()

        # Find matches
        matches = np.zeros(len(distmat))
        # Zero if no match 1 if match
        for i in range(0, len(distmat)):
            if distmat[i][1] == search_uid:
                # Match found
                matches[i] = 1
        rank = np.zeros(topk)
        for i in range(0, topk):
            if matches[i] == 1:
                rank[i] = 1
                # If 1 is found then break as you dont need to look further path k
                break
        all_rank.append(rank)
        valid_queries +=1
    #print(all_rank)
    sum_all_ranks = np.zeros(len(all_rank[0]))
    for i in range(0,len(all_rank)):
        my_array = all_rank[i]
        for g in range(0, len(my_array)):
            sum_all_ranks[g] = sum_all_ranks[g] + my_array[g]
    sum_all_ranks = np.array(sum_all_ranks)
    cmc_results = np.cumsum(sum_all_ranks) / valid_queries
    return cmc_results, sum_all_ranks

def main(cfg):

    irisRec = irisRecognition(cfg)

    enrolled_uids = []
    enrolled_vectors = []
    with open("./1N_data/enroll_1N.txt", "r") as enrollFile:
        for line in tqdm(enrollFile):
            im_name = line.strip().split("/")[1].split(".")[0]
            enrolled_uids.append(im_name.split("+")[0])
            im_polar = np.array(Image.open("./1N_data/enroll_im_polar/" + im_name + ".png").convert("L"))
            enrolled_vectors.append(irisRec.extractVector(np.array(im_polar)))
    
    search_uids = []
    search_vectors = []
    with open("./1N_data/search_1N.txt", "r") as searchFile:
        for line in tqdm(searchFile):
            im_name = line.strip().split("/")[1].split(".")[0]
            search_uids.append(im_name.split("+")[0])
            im_polar = np.array(Image.open("./1N_data/search_im_polar/" + im_name + ".png").convert("L"))
            search_vectors.append(irisRec.extractVector(np.array(im_polar)))

    cmc_results, sum_all_ranks = cmc(search_uids, search_vectors, enrolled_uids, enrolled_vectors, irisRec, args.topk)
    #print(cmc_results)
    cmc_dict = {"cmc": cmc_results, "NPSAR": sum_all_ranks}
    print(cmc_results)
    with open("cmc_dict" + args.tag + ".pkl", "wb") as cmcdictfile:
        pkl.dump(cmc_dict, cmcdictfile)
     
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg_baseline.yaml",
                        help="path of the configuration file")
    parser.add_argument("--tag", default="_tripletnn")
    parser.add_argument("--topk", type=int, default=30)
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))