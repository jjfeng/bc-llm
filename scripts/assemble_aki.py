"""
Creates notes for concept extraction from the AKI dataset
"""
import os
import sys
import logging
import argparse
import pandas as pd
import medspacy
from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sys.path.append(os.getcwd())
import src.common as common

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--note-dataset-file",
                        type=str,
                        help="location of your notes data"
                        )
    parser.add_argument("--tabular-dataset-file",
                        type=str,
                        help="location of your labelled data"
                        )
    parser.add_argument("--max-obs", type=int, default=None)
    parser.add_argument("--out-csv", type=str,
                        help="the name of the output csv")
    parser.add_argument("--log-file", type=str, default="_output/log_aki.txt",
                        help="log file")
    args = parser.parse_args()
    return args

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)

    tabular_df = pd.read_csv(args.tabular_dataset_file).reset_index(drop=True)
    notes_df = pd.read_csv(args.note_dataset_file).reset_index(drop=True)

    merged_df = tabular_df.merge(notes_df[~notes_df.preop_note_text.isna()], on=["deid_case_id","deid_mrn"]).rename(
        {
            "preop_note_text": "sentence",
            "aki_outcome": "y",
        }, axis=1)
    logging.info("prevalence ORIG %f", merged_df.y.mean())
    case_idxs = np.where(merged_df.y == 1)[0]
    num_control = case_idxs.size
    control_idxs = np.random.choice(np.where(merged_df.y == 0)[0], num_control, replace=False)
    print("case", case_idxs.size, "control", control_idxs.size)
    merged_df = merged_df.iloc[np.concatenate([control_idxs, case_idxs])]

    logging.info("prevalence in filtered %f", merged_df.y.mean())

    merged_df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
