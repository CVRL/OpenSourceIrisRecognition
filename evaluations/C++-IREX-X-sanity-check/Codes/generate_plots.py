import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from sklearn.metrics import roc_curve, roc_auc_score
import os

def calculate_d_prime(genuine_scores, imposter_scores):
    genuine_scores = np.array(genuine_scores)
    imposter_scores = np.array(imposter_scores)

    g_mean = np.mean(genuine_scores)
    g_var = np.var(genuine_scores)
    i_mean = np.mean(imposter_scores)
    i_var = np.var(imposter_scores)
    d_prime = abs(g_mean - i_mean)/math.sqrt(0.5 * (g_var + i_var))

    return d_prime

def generate_ytrue_yscore(genuine_scores, imposter_scores):
    y_true = []
    y_score = []
    for score in genuine_scores:
        y_true.append(1)
        y_score.append(1-score)
    for score in imposter_scores:
        y_true.append(0)
        y_score.append(1-score)
    return y_true, y_score

def get_uid(filename):
    filenameparts = filename.split('-')
    if len(filenameparts) == 8:
        uid = filenameparts[1] + '_' + filenameparts[2] + '_' + filenameparts[4]
    else:
        filenameparts = filename.split('_')
        uid = filenameparts[1] + '_' + filenameparts[2] + '_' + filenameparts[3].lower()
    return uid

genuine_scores = []
imposter_scores = []

with open(sys.argv[1], 'r') as subsetFile:
    i = 0
    for line in subsetFile:
        lineparts = line.split(',')
        score = float(lineparts[2].strip())

        id1 = get_uid(lineparts[0].strip())
        id2 = get_uid(lineparts[1].strip())
        
        if id1 == id2:
            genuine_scores.append(float(score))
        else:
            imposter_scores.append(float(score))

d_prime = calculate_d_prime(genuine_scores, imposter_scores)
y_true, y_score = generate_ytrue_yscore(genuine_scores, imposter_scores)

fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

plt.plot(fpr, tpr, label='AUC = '+str(round(auc,4)) + '')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xscale("log")
# show the legend
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(sys.argv[2], os.path.split(sys.argv[1])[1].split('.')[0] + '_roc.png'))
plt.close()

plt.figure(figsize=(10, 7))

n_bins = 50

range_min = min(genuine_scores+imposter_scores)
range_max = max(genuine_scores+imposter_scores)

genuine_counts, genuine_bins = np.histogram(genuine_scores, bins=n_bins, range=[range_min,range_max])
genuine_probs = [genuine_count/genuine_counts.sum() for genuine_count in genuine_counts]
imposter_counts, imposter_bins = np.histogram(imposter_scores, bins=n_bins, range=[range_min,range_max])
imposter_probs = [imposter_count/imposter_counts.sum() for imposter_count in imposter_counts]

plt.grid(True)

genuine_bin_centers = 0.5*(genuine_bins[1:]+genuine_bins[:-1])
plt.step(genuine_bin_centers, genuine_probs, where='mid', color='blue', linestyle=('solid'), label='Genuine', marker='.')
imposter_bin_centers = 0.5*(imposter_bins[1:]+imposter_bins[:-1])
plt.step(imposter_bin_centers, imposter_probs, where='mid', color='red', linestyle=('solid'), label='Imposter', marker='x', markersize=5)

plt.plot([], [], ' ', label=" ")
plt.plot([], [], ' ', label="d\'= " + str(round(d_prime, 3)))
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                     mode="expand", borderaxespad=0, ncol=4, fontsize=10, 
                     edgecolor='black', handletextpad=0.3)

plt.xlabel('Score')
plt.ylabel('Probability')
plt.tight_layout()
plt.savefig(os.path.join(sys.argv[2], os.path.split(sys.argv[1])[1].split('.')[0] + '_histogram.png'))
