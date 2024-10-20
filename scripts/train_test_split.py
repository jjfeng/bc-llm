"""
Create train validation tests split by shuffling the dataset
"""
import sys
import pickle
import logging
import argparse
import os

import numpy as np
import pandas as pd


def parse_args(args):
    """parse command line arguments"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-obs", type=int, default=None)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--data-csv", type=str)
    parser.add_argument(
        "--indices-csv", type=str, default="_output/train_test_indices.csv"
    )
    args = parser.parse_args()
    return args

# returns a list to assign patients to train, test, validation 
def _split(num_rows: int, test_frac: float) -> list[str]:
    num_test = int(test_frac * num_rows)
    num_train = num_rows - num_test
    print("NUM TEST", num_test)    
    partition_data = ["train"] * num_train + ["test"] * num_test
    return partition_data

def main(args):
    args = parse_args(args)
    np.random.seed(args.seed)

    data_df = pd.read_csv(args.data_csv)
    num_obs = data_df.shape[0] if args.max_obs is None else args.max_obs

    # split data into train/ test/ validation
    print("INDEX", data_df.index)
    df = pd.DataFrame({
        "idx": np.random.choice(data_df.index, num_obs, replace=False),
        "partition": _split(num_obs, args.test_frac)
    }).sort_values('partition', ascending=True)
    
    df.to_csv(args.indices_csv, index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
