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
from scripts.evaluate_bayesian import eval_ensemble

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--indices-csv", type=str)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled training data")
    parser.add_argument("--in-mdl", type=str, help="the learned concept model")
    parser.add_argument("--results-csv", type=str)
    args = parser.parse_args()
    args.partition = "test"
    args.max_obs = -1
    return args

def main(args):
    args = parse_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_df = load_data_partition(args, init_concepts_file=None, text_summary_column=None)
    X_test = data_df[data_df.columns[4:-1]]
    y_test = data_df.y

    with open(args.in_mdl, "rb") as f:
        model = pickle.load(f)

    pred_prob = model.predict_proba(X_test)
    is_multiclass = len(np.unique(y_test)) > 2
    _, res_df = eval_ensemble(y_test, pred_prob, is_multiclass=is_multiclass)
    res_df['method'] = 'Human+CBM'
    res_df['seed'] = args.seed
    res_df.to_csv(args.results_csv, index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
