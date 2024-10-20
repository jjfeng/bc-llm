"""
Script to train a LR oracle model
"""

import os
import sys
import logging
import argparse
import torch
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learner-type", type=str, default="oracle")
    parser.add_argument("--oracle-columns", type=str)
    parser.add_argument("--max-obs", type=int, default=None)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled training data")
    parser.add_argument("--out-mdl", type=str, help="the learned concept model")
    parser.add_argument("--out-coef-csv", type=str, help="the learned concept model, coef csv")
    parser.add_argument("--log-file", type=str, default="_output/log_train_concept.txt")
    args = parser.parse_args()
    args.oracle_columns = list(map(int, args.oracle_columns.split(",")))
    return args

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    np.random.seed(args.seed)
    
    labelled_oracle_df = pd.read_csv(args.in_dataset_file, index_col=0)
    print(labelled_oracle_df)

    lr = LogisticRegression()
    lr.fit(
        labelled_oracle_df.iloc[:,args.oracle_columns],
        labelled_oracle_df.iloc[:,-1])
    lr_result = pd.DataFrame({
        "x": np.concatenate([labelled_oracle_df.columns[args.oracle_columns], ["intercept"]]),
        "param": np.concatenate([lr.coef_.flatten(), lr.intercept_])
    })
    lr.oracle_columns = args.oracle_columns
    print(lr_result)
    logging.info(lr_result)
    lr_result.to_csv(args.out_coef_csv)

    with open(args.out_mdl, "wb") as f:
        pickle.dump(lr, f)

if __name__ == "__main__":
    main(sys.argv[1:])
