"""
Script to train a lasso model on LLM probability outputs
"""

import os
import sys
import logging
import argparse
import torch
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegressionCV

sys.path.append(os.getcwd())
from scripts.train_concept import make_dataset, setup

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-obs", type=int, default=50000)
    parser.add_argument("--learner-type", type=str, choices=["lasso", "lasso_std"], help="whether to standardize the data to have variance 1")
    parser.add_argument("--embeddings-file", type=str, help="npz of the embedding data")
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled training data")
    parser.add_argument("--out-mdl", type=str, help="the learned concept model")
    parser.add_argument("--out-coef-csv", type=str, help="the learned concept model, coef csv")
    parser.add_argument("--log-file", type=str, default="_output/log_train_concept.txt")
    parser.add_argument(
            "--llm-model-type",
            type=str,
            default="meta-llama/Meta-Llama-3-8B-Instruct",
            choices=["meta-llama/Meta-Llama-3-8B-Instruct"]
            )
    args = parser.parse_args()
    return args

def prep_lasso_data(learner_type, X):
    if "std" in learner_type:
        X = X/np.sqrt(np.var(X, axis=0, keepdims=True))
    return X

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, english_filter = setup(args)

    dataset, _ = make_dataset(args, english_filter, device)

    lr = LogisticRegressionCV(penalty="l2", solver="liblinear", max_iter=args.epochs, cv=3, verbose=False, n_jobs=-1)
    X = prep_lasso_data(args.learner_type, dataset.tensors[0][:,0,:].cpu().numpy())
    lr.fit(
            X[:args.max_obs],
            dataset.tensors[1].cpu().numpy().flatten()[:args.max_obs])

    logging.info("num nonzero coef %d", np.sum(np.abs(lr.coef_[0]) > 1e-3))
    subset_feat_idxs = np.where(np.abs(lr.coef_[0]) > 1e-3)[0]
    lr_df = pd.DataFrame({
        "token_id": subset_feat_idxs,
        "token": english_filter.batch_decode(subset_feat_idxs),
        "coef": lr.coef_[0,subset_feat_idxs],
        "abs_coef": np.abs(lr.coef_[0,subset_feat_idxs])
        })
    lr_df = lr_df.sort_values("abs_coef", ascending=False)
    logging.info(lr_df)
    print("LASSO", lr_df)
    lr_df.to_csv(args.out_coef_csv)

    with open(args.out_mdl, "wb") as f:
        pickle.dump(lr, f)

if __name__ == "__main__":
    main(sys.argv[1:])
