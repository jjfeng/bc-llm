import os
import argparse
import logging

import scipy.stats as stats
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="aggregate result files"
    )
    parser.add_argument(
        "--result-files",
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--groupby-cols",
        nargs="*")
    parser.add_argument(
        "--add-col",
        type=str,
    )
    parser.add_argument(
        "--add-val",
        type=str,
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="_output/res.csv",
    )
    args = parser.parse_args()
    print(args.result_files)
    return args


def main():
    args = parse_args()
    # print('AGG args', vars(args))

    all_res = []
    for res_file in args.result_files:
        try:
            res = pd.read_csv(res_file, index_col=0)
            all_res.append(res)
        except FileNotFoundError as e:
            print(e)
            continue

    all_res = pd.concat(all_res).reset_index()
    if args.add_col is not None:
        all_res[args.add_col] = args.add_val
    all_res.to_csv(args.csv_file, index=False)
    print(all_res)

    # if args.groupby_cols is not None:
    #     print(all_res.groupby(args.groupby_cols).mean())


if __name__ == "__main__":
    main()
