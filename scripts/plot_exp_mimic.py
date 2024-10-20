import os
import argparse
import logging

import scipy.stats as stats
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(
        description="plot for exp_mimic"
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--in-result-csv",
        type=str,
    )
    parser.add_argument(
        "--coverage-csv",
        type=str,
    )
    parser.add_argument(
        "--plot-file",
        type=str,
    )
    parser.add_argument(
        "--coverage-plot-file",
        type=str,
    )
    parser.add_argument(
        "--coverage-detailed-plot-file",
        type=str,
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    palette = sns.color_palette()
    
    sns.set_context('paper', font_scale=2.5)

    res = pd.read_csv(args.in_result_csv)[['auc', 'log_lik', 'seed', 'method', 'max_train_obs']]
    print(res.groupby(["method", "max_train_obs"]).mean())
    res = res.melt(id_vars=["method", "seed", "max_train_obs"], var_name="metric")
    coverage_df = pd.read_csv(args.coverage_csv)
    coverage_df = coverage_df[coverage_df.metric.isin(['recall', 'precision'])]
    coverage_df['value'] = coverage_df.abs_corr > args.coverage_threshold
    res = pd.concat([res, coverage_df])
    res['metric'] = res['metric'].map({
        'auc': 'AUC',
        'log_lik': 'Log Likelihood',
        'recall': 'Recall',
        'precision': 'Precision'
    })
    res['method'] = res['method'].map({
        'baseline': 'One-pass',
        'bayesian': 'BC-LLM',
        'boosting': 'Boosting',
        'oracle': 'Pre-specified',
        'Bag of Words': 'Bag-of-words',
    })
    res = res.rename({
        'method': 'Method',
        'max_train_obs': 'Num train obs',
        'value': 'Value',
        'metric': 'Metric',
    }, axis=1)

    g = sns.relplot(
        data=res,
        x="Num train obs",
        y="Value",
        hue="Method",
        col="Metric",
        kind="line",
        facet_kws={"sharey": False, "sharex": True},
        palette=palette,
        # col_wrap=2,
        hue_order=[
            'BC-LLM',
            'One-pass',
            'Boosting',
            'Pre-specified',
            'Bag-of-words'
        ]
    )
    g.set_titles("{col_name}")
    g.axes.flat[1].set_ylim((-1.15, -0.5))
    plt.savefig(args.plot_file)

    # plt.clf()
    # print(coverage_df)
    # sns.relplot(
    #     data=coverage_df,
    #     x="max_train_obs",
    #     y="is_match", # is_match
    #     col="metric",
    #     hue="method",
    #     kind="line",
    #     palette=palette[1:],
    # )
    # plt.savefig(args.coverage_plot_file)
    # plt.clf()
    # sns.relplot(
    #     data=coverage_df[coverage_df.metric == "recall"],
    #     x="max_train_obs",
    #     y="is_match", # is_match
    #     col="oracle_concept",
    #     hue="method",
    #     kind="line",
    #     palette=palette[1:],
    # )
    # plt.savefig(args.coverage_detailed_plot_file)

if __name__ == "__main__":
    main()