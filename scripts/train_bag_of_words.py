"""
Script to train a simple bag of words model 
"""
import os
import sys
import logging
import argparse
import torch
import numpy as np
import pickle

from sklearn.metrics import roc_auc_score

sys.path.append(os.getcwd())
from scripts.train_bayesian import load_data_partition, get_word_count_data
import src.common as common

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
            "--learner-type", 
            type=str, 
            default="count_l2", 
            choices=['count_elasticnet','tfidf_l2', 'tfidf_l1', 'count_l2', 'count_l1'], 
            help="model types"
            )
    parser.add_argument("--max-obs", type=int, default=-1)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled training data")
    parser.add_argument("--indices-csv", type=str, help="csv of training indices")
    parser.add_argument("--out-mdl", type=str, help="the learned bag of words model")
    parser.add_argument("--out-vectorizer-file", type=str, help="the learned vectorizer")
    parser.add_argument("--log-file", type=str, default="_output/log_train_concept.txt")
    parser.add_argument("--text-column", type=str, default="sentence")
    args = parser.parse_args()
    args.partition = "train"
    args.count_vectorizer, args.penalty = args.learner_type.split("_")
    return args

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    np.random.seed(args.seed)
    
    data_df = load_data_partition(args, init_concepts_file=None, text_summary_column=None)

    X_train, word_names = get_word_count_data(
            data_df, 
            args.count_vectorizer, 
            args.text_column, 
            vectorizer_out_file=args.out_vectorizer_file
            )
    y_train = data_df['y'].to_numpy().flatten()

    results = common.train_LR(X_train, y_train, penalty=args.penalty)

    with open(args.out_mdl, "wb") as f:
        pickle.dump(results["model"], f)

    print(f"AUC {results['auc']}")
    logging.info("AUC %f", results["auc"]) 

if __name__ == "__main__":
    main(sys.argv[1:])
