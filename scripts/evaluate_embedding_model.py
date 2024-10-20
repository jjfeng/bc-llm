"""
Script to evaluate a LR embedding model
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
from sklearn.metrics import roc_auc_score

from sentence_transformers import SentenceTransformer

sys.path.append(os.getcwd())
from scripts.train_bayesian import load_data_partition, get_word_count_data, load_llms

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-obs", type=int, default=-1)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled data")
    parser.add_argument("--indices-csv", type=str, help="csv of test indices")
    parser.add_argument("--in-mdl", type=str, help="the learned concept model")
    parser.add_argument(
            "--embedding-model-name",
            type=str,
            default='all-MiniLM-L6-v2',
            )
    parser.add_argument("--log-file", type=str, default="_output/log_test_concept.txt")
    args = parser.parse_args()
    args.partition = "test"
    return args

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    np.random.seed(args.seed)
    
    data_df = load_data_partition(args, init_concepts_file=None, text_summary_column=None)

    model = SentenceTransformer(args.embedding_model_name)
    embeddings = model.encode(data_df.sentence.tolist())
    print("embeddings", embeddings.shape)
    
    with open(args.in_mdl, "rb") as f:
        lr = pickle.load(f)
    
    pred_prob = lr.predict_proba(embeddings)[:,1]
    print("LR ROC", roc_auc_score(y_score=pred_prob, y_true=data_df.y))

if __name__ == "__main__":
    main(sys.argv[1:])
