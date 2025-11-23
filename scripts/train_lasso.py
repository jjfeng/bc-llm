"""
Script to train on human-annotated concepts
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
from scripts.train_bayesian import load_data_partition

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--indices-csv", type=str)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled training data")
    parser.add_argument("--out-mdl", type=str, help="the learned concept model")
    parser.add_argument("--log-file", type=str, default="_output/log_train_concept.txt")
    args = parser.parse_args()
    args.partition = "train"
    args.max_obs = -1
    return args

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_df = load_data_partition(args, init_concepts_file=None, text_summary_column=None)
    X = data_df[data_df.columns[4:-1]]
    y = data_df.y

    lr = LogisticRegressionCV(penalty="l1", solver="liblinear", max_iter=args.epochs, cv=3, verbose=False, n_jobs=-1)
    lr.fit(X, y)

    with open(args.out_mdl, "wb") as f:
        pickle.dump(lr, f)

if __name__ == "__main__":
    main(sys.argv[1:])
