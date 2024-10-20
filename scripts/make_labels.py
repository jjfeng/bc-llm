"""
Script for making labels
"""
import os
import sys
import logging
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

sys.path.append(os.getcwd())
from src.llm.llm_api import LLMApi
from src.llm.llm_local import LLMLocal
import src.common as common
from src.training_history import TrainingHistory


def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--y-params", type=str, help="coef and intercept for y model, comma separated")
    parser.add_argument("--in-dataset-file", type=str, help="csv of the data we want to learn concepts for")
    parser.add_argument("--columns", nargs="*", type=str, help="the columns in the data to use for making the labels")
    parser.add_argument("--in-training-history-file", type=str, default=None)
    parser.add_argument("--out-extractions", type=str, default="_output/note_extractions.pkl")
    parser.add_argument("--prompt-concepts-file", type=str, default="exp_multi_concept/prompts/concept_questions.txt")
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--log-file", type=str, help="log file")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--labelled-data-file", type=str, help="csv file with the necessary data elements for training model")
    parser.add_argument(
            "--llm-model-type",
            type=str,
            default="meta-llama/Meta-Llama-3.1-8B-Instruct",
            choices=[
                "gpt-4o-mini",
                "meta-llama/Meta-Llama-3.1-8B-Instruct", 
                "meta-llama/Meta-Llama-3.1-70B-Instruct", 
                ]
            )
    args = parser.parse_args()
    args.y_params = np.array(list(map(float, args.y_params.split(","))), dtype=float)
    return args

def train_oracle(x, y, train_frac=0.8):
    shuffle_idxs = np.random.choice(x.shape[0], size=x.shape[0], replace=False)
    x = x[shuffle_idxs]
    y = y[shuffle_idxs]
    num_train = int(x.shape[0] * train_frac)
    train_x = x[:num_train]
    train_y = y[:num_train]
    test_x = x[num_train:]
    test_y = y[num_train:]
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    pred_prob = lr.predict_proba(test_x)[:,1]
    test_auc = roc_auc_score(test_y, pred_prob)

    logging.info("train/test oracle ROC %f", test_auc)
    print("train/test oracle ROC", test_auc) 

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    np.random.seed(args.seed)
    logging.info(args)

    notes_df = pd.read_csv(args.in_dataset_file)

    coef_idxs = np.where(args.y_params[:-1] != 0)[0]
    coefs = args.y_params[coef_idxs].reshape((-1,1))
    if args.columns:
        x = notes_df[args.columns].to_numpy()
    elif args.in_training_history_file is not None:
        history = TrainingHistory().load(args.in_training_history_file)
        concept_dicts = history.get_last_concepts()
        if args.use_api:
            llm = LLMApi(args.seed, args.llm_model_type, logging)
        else:
            llm = LLMLocal(args.seed, args.llm_model_type, logging)
        all_extracted_features = {}
        if os.path.exists(args.out_extractions):
            with open(args.out_extractions, "rb") as f:
                all_extracted_features = pickle.load(f)
        all_extracted_features = common.extract_features_by_llm(
            llm, 
            notes_df,
            concept_dicts,
            all_extracted_features_dict=all_extracted_features,
            prompt_file=args.prompt_concepts_file,
            batch_size=args.batch_size,
            extraction_file=args.out_extractions
        )
        x = np.concatenate(
            [all_extracted_features[concept_dict["concept"]] for concept_dict in concept_dicts],
            axis=1
            )
        
        # HACK: PURELY FOR CHECKING agreement between MIMIC annotations and LLM annotations
        # column_mappings = [
        #     ["label_employment_False"],
        #     ['label_alcohol_Present', 'label_alcohol_Past'],
        #     ['label_tobacco_Present', 'label_tobacco_Past'],
        #     ['label_drugs_Present']
        # ]
        # for i, mapped_cols in enumerate(column_mappings):
        #     logging.info("COLUMN MAPPING %s", mapped_cols)
        #     x_col_given = 0
        #     for c in mapped_cols:
        #         x_col_given += notes_df[c].to_numpy()
        #     agreement_rate = np.mean(x[:,i] == x_col_given)
        #     logging.info(f"agreement rate col{i}: agree {agreement_rate}")
    else:
        x = notes_df.iloc[:,coef_idxs].to_numpy()
    logging.info("X MEAN %s", x.mean(axis=0))

    # generate Y via logistic regression
    notes_df["true_prob"] = 1/(1 + np.exp(-(x @ coefs + args.y_params[-1])))
    notes_df["y"] = np.random.binomial(n=1, p=notes_df.true_prob, size=notes_df.shape[0])
    print(notes_df)
    print("PREVALENCE", notes_df.y.mean())
    notes_df.to_csv(args.labelled_data_file)

    logging.info("oracle ROC %f", roc_auc_score(notes_df.y, notes_df.true_prob))
    print("oracle ROC", roc_auc_score(notes_df.y, notes_df.true_prob))

    train_oracle(x, notes_df.y.to_numpy().flatten())

if __name__ == "__main__":
    main(sys.argv[1:])
