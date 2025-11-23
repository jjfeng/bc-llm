"""
Script to evaluate model for human-annotated concepts on ood images
"""

import os
import sys
import logging
import argparse
import torch
import pandas as pd
import numpy as np
import pickle, joblib

from sklearn.linear_model import LogisticRegressionCV

sys.path.append(os.getcwd())
from scripts.evaluate_pocky import get_ci, load_images
from scripts.evaluate_bayesian import eval_ensemble

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--indices-csv", type=str)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled training data")
    parser.add_argument("--in-mdl", type=str, help="the learned concept model")
    parser.add_argument("--max-obs", type=int, default=-1)
    parser.add_argument("--trained-classes", nargs="*", type=int, default=None)
    parser.add_argument("--dataset-folder", type=str, help="folder of CUB data")
    parser.add_argument("--log-file", type=str, default="_output/log_plot.txt")
    parser.add_argument("--csv-file", type=str, default="_output/ood.csv")
    parser.add_argument("--results-csv", type=str)
    args = parser.parse_args()
    args.partition = "test"
    args.max_obs = -1
    return args

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.in_mdl, "rb") as f:
        model = pickle.load(f)
    X_test, target_domain = load_images(args)
    X_test = X_test[model.feature_names_in_]

    pred_probs = model.predict_proba(X_test)
    entropies = -np.sum(np.log(pred_probs) * pred_probs, axis=1)

    logging.info("entropies first two %s", entropies[:2])
    
    logging.info("entropy CI: %s", get_ci(entropies))
    entropy_res = get_ci(entropies)
    pd.DataFrame({
        "method": ["Human+CBM"],
        "trained_classes": [" ".join(map(str, args.trained_classes))],
        "target_domain": [target_domain],
        "entropy": [entropy_res[0]],
        "entropy_lower_ci": [entropy_res[1]],
        "entropy_upper_ci": [entropy_res[2]],
    }).to_csv(args.csv_file)

if __name__ == "__main__":
    main(sys.argv[1:])
