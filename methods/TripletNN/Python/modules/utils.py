import yaml
import numpy as np
from sklearn.metrics import roc_curve, auc

def get_cfg(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    return cfg

def load_CCNet():

    # CCNet model loading (iris segmentation)
    model = UNet(NUM_CLASSES, NUM_CHANNELS)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    if args.state:
        try:
            if args.cuda:
                model.load_state_dict(torch.load(args.state))
            else:
                model.load_state_dict(torch.load(args.state, map_location=torch.device('cpu')))
                # print("model state loaded")
        except AssertionError:
            print("assertion error")
            model.load_state_dict(torch.load(args.state,
                map_location = lambda storage, loc: storage))
    model.eval()
    softmax = nn.LogSoftmax(dim=1)

    return model, softmax

def get_performance_metrics(genuine_scores,impostor_scores):

    # AUROC
    scores_combined = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(impostor_scores))])
    fpr, tpr, thresholds = roc_curve(labels, scores_combined)
    roc_auc = auc(fpr, tpr)

    # d'
    dprime = np.abs(np.mean(genuine_scores) - np.mean(impostor_scores)) / (np.sqrt(0.5*(np.var(genuine_scores)+np.var(impostor_scores))))

    return roc_auc, dprime

