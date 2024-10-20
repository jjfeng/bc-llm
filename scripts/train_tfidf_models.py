"""
Script to train models that use simple features, like count or tf-idf
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import pickle

import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

sys.path.append(os.getcwd())
from scripts.train_concept import make_dataset, setup

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-obs", type=int, default=50000)
    parser.add_argument("--learner-type", type=str, choices=['tfidf_l2', 'tfidf_l1', 'count_l2', 'count_l1'], help="model types")
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled training data")
    parser.add_argument("--out-mdl", type=str, help="the learned concept model")
    parser.add_argument("--out-coef-csv", type=str, help="the learned concept model, coef csv")
    parser.add_argument("--log-file", type=str, default="_output/log_train_concept.txt")
    args = parser.parse_args()
    args.count_vectorizer, args.model = args.learner_type.split("_")
    return args

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    np.random.seed(args.seed)

    dset_train = pd.read_csv(args.in_dataset_file)
    print("DSET SIZE", dset_train.shape)
    if args.count_vectorizer == 'count':
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            ngram_range=(1,1), binary=True)
    elif args.count_vectorizer == 'tfidf':
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            ngram_range=(1,1))
    else:
        raise NotImplementedError("vectorizer not recognized")
    print("NUM NA", pd.isna(dset_train.llm_output).sum())

    dset_train = dset_train[~dset_train.llm_output.isna()]
    print("dset_train", dset_train.shape)

    X_train = vectorizer.fit_transform(dset_train.llm_output)
    y_train = dset_train['y']

    if args.model == 'l2':
        model = LogisticRegressionCV(cv=5, max_iter=1000)
    elif args.model == 'l1':
        model = LogisticRegressionCV(
            cv=5, max_iter=1000, penalty='l1', solver='liblinear')
    model.fit(X_train[:args.max_obs], y_train[:args.max_obs])

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:,1])
    logging.info("train auc %f", train_auc)

    # interpret the model
    coefs = model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()
    df = pd.DataFrame(
        {
            'feature_names': feature_names,
            'coef': coefs,
            'abs_coef': np.abs(coefs),
        }).sort_values("abs_coef", ascending=False)
    print(df)
    df.to_csv(args.out_coef_csv, index=False)

    with open(args.out_mdl, "wb") as f:
        pickle.dump({
            "model": model,
            "vectorizer": vectorizer,
            }, f)

if __name__ == "__main__":
    main(sys.argv[1:])
