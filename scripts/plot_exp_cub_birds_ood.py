import os
import argparse
import logging
import sys

import scipy.stats as stats
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

sys.path.append(os.getcwd())
from scripts.plot_exp_cub_birds import get_ci_str

def parse_args():
    parser = argparse.ArgumentParser(
        description="plot for OOD experiments for exp_cub_birds"
    )
    parser.add_argument(
        "--title",
        type=str,
    )
    parser.add_argument(
        "--in-result-csv",
        type=str,
    )
    parser.add_argument(
        "--log-file",
        type=str,
    )
    parser.add_argument(
        "--plot-entropies-file",
        type=str,
    )
    args = parser.parse_args()
    return args

def plot_rankings(res_df, file_name, max_better=True):
    res_df["method"] = res_df.method.replace({
        "bayesian": "BC-LLM",
        "baseline": "LLM+CBM",
        "boosting": "Boosting LLM+CBM",
    })

    plt.clf()
    plt.subplots(figsize=(5,8))
    cmap = sns.color_palette(n_colors=5)
    sns.boxplot(
        data=res_df,
        x="entropy",
        y="method",
        whis=[2.5, 97.5]
    )
    sns.swarmplot(
        data=res_df,
        x="entropy",
        y="method",
        # hue="bird"
    )
    plt.tight_layout()
    plt.savefig(file_name)
    print("plotted at", file_name)

def main():
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    
    birds_df = pd.read_csv(args.in_result_csv)
    plot_rankings(birds_df, args.plot_entropies_file)
    print(birds_df)
    method_cols = birds_df.method.unique()
    res_df = pd.DataFrame({
        "method": method_cols,
        "entropy": [get_ci_str(birds_df[birds_df.method == method_col].entropy) for method_col in method_cols],
    })
    print(res_df)
    

if __name__ == "__main__":
    main()