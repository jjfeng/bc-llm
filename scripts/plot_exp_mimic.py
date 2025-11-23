import os
import argparse
import logging

import scipy.stats as stats
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

DEFAULT_HUE_ORDER = [
            'BC-LLM',
            'LLM+CBM',
            'Boosting LLM+CBM',
            'Human(Oracle)+CBM',
            'Bag-of-words'
        ]
ABLATION_COHERE_HUE_ORDER = [
            'BC-LLM Ablation',
            'BC-LLM Cohere',
            'LLM+CBM Cohere',
            'Boosting LLM+CBM Cohere',
            'Human(Oracle)+CBM Cohere',
            'Bag-of-words Cohere'
        ]
DEFAULT_MAP = {
        'baseline': 'LLM+CBM',
        'bayesian': 'BC-LLM',
        'boosting': 'Boosting LLM+CBM',
        'oracle': 'Human(Oracle)+CBM',
        'Bag of Words': 'Bag-of-words',
    }
COHERE_MAP = {
        'baseline': 'LLM+CBM Cohere',
        'bayesian': 'BC-LLM Cohere',
        'boosting': 'Boosting LLM+CBM Cohere',
        'oracle': 'Human(Oracle)+CBM Cohere',
        'Bag of Words': 'Bag-of-words Cohere',
    }
ABLATION_MAP = {
        'bayesian': 'BC-LLM Ablation',
    }

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
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--coverage-csv",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        '--maps',
        nargs="+",
        type=str
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

    map_list = []
    for map_str in args.maps:
        print(map_str)
        if map_str == "ablation":
            map_list.append(ABLATION_MAP)
        elif map_str == "cohere":
            map_list.append(COHERE_MAP)
        else:
            map_list.append(DEFAULT_MAP)

    all_res = []
    for csv_f, map_dict in zip(args.in_result_csv, map_list):
        res = pd.read_csv(csv_f)[['auc', 'brier', 'seed', 'method', 'max_train_obs']]
        print(res.method.unique())
        res['method'] = res['method'].map(map_dict)
        all_res.append(res)
    all_res = pd.concat(all_res).reset_index(drop=True)
    all_res = all_res.melt(id_vars=["method", "seed", "max_train_obs"], var_name="metric")
    all_res = all_res[all_res.metric.isin(['auc', 'log_lik', 'brier'])]
    
    all_coverages = []
    for csv_f, map_dict in zip(args.coverage_csv, map_list):
        cov_df = pd.read_csv(csv_f)
        cov_df['method'] = cov_df['method'].map(map_dict)
        print(cov_df)
        all_coverages.append(cov_df)
    coverage_df = pd.concat(all_coverages).reset_index(drop=True)
    coverage_df = coverage_df[coverage_df.metric.isin(['recall', 'precision'])]
    coverage_df['value'] = coverage_df.abs_corr > args.coverage_threshold
    print(coverage_df)
    print(coverage_df[['metric', 'method', 'max_train_obs', 'value']].groupby(['method', 'metric', 'max_train_obs',]).mean())
    
    all_res = pd.concat([all_res, coverage_df])
    all_res['metric'] = all_res['metric'].map({
        'auc': 'AUC',
        'log_lik': 'Log Likelihood',
        'brier': 'Brier',
        'recall': 'Recall',
        'precision': 'Precision'
    })
    
    all_res = all_res.rename({
        'method': 'Method',
        'max_train_obs': 'Num train obs',
        'value': 'Value',
        'metric': 'Metric',
    }, axis=1)
    print(all_res)
    g = sns.relplot(
        data=all_res,
        x="Num train obs",
        y="Value",
        hue="Method",
        col="Metric",
        kind="line",
        facet_kws={"sharey": False, "sharex": True},
        palette=palette,
        col_wrap=4,
        hue_order=DEFAULT_HUE_ORDER, # ABLATION_COHERE_HUE_ORDER
    )
    g.set_titles("{col_name}")
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