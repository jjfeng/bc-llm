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
from scripts.evaluate_bayesian import eval_ensemble

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
    parser.add_argument("--in-mdl", type=str, help="the learned bag of words model")
    parser.add_argument("--in-vectorizer-file", type=str, help="the learned vectorizer")
    parser.add_argument("--text-column", type=str, default="sentence")
    parser.add_argument("--results-csv", type=str, default="_output/log_results.csv")
    args = parser.parse_args()
    args.partition = "test"
    args.count_vectorizer, args.penalty = args.learner_type.split("_")
    return args

def main(args):
    args = parse_args(args)
    np.random.seed(args.seed)
    
    data_df = load_data_partition(args, init_concepts_file=None, text_summary_column=None)

    with open(args.in_vectorizer_file, 'rb') as file:
        vectorizer = pickle.load(file)

    with open(args.in_mdl, 'rb') as file:
        model = pickle.load(file)

    X_test = vectorizer.transform(data_df[args.text_column]).toarray()
    y_test = data_df['y'].to_numpy().flatten()

    pred_prob = model.predict_proba(X_test)
    is_multiclass = len(np.unique(y_test)) > 2

    _, res_df = eval_ensemble(y_test, pred_prob, is_multiclass=is_multiclass)
    print(res_df)
    res_df['method'] = 'Bag of Words'
    res_df['seed'] = args.seed
    res_df.to_csv(args.results_csv, index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
