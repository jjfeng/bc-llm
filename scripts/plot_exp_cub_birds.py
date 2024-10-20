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
        description="plot for exp_cub_birds"
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
    args = parser.parse_args()
    return args

def make_pivot_table(birds_df, metric):
    birds_res_df = birds_df.pivot(index=["bird", "max_train_frac"], columns="method", values=[metric]).reset_index(drop=False)
    birds_res_df['bird'] = birds_res_df['bird'].map({
        'class_1_2_3': 'Albatross',
        'class_5_6_7_8': 'Auklet',
        'class_9_10_11_12': 'Blackbird',
        'class_14_15_16': 'Bunting',
        'class_37_38_39_40_41_42_43' :'Flycatcher',
        'class_113_114_115_116_117_118_119_120_121_122_123_124_125_126_127_128_129_130_131_132_133': 'Sparrow',
        'class_141_142_143_144_145_146_147':'Tern'
    })
    birds_res_df.columns = ['bird', 'max_train_frac'] + birds_res_df.columns.droplevel().tolist()[2:]

    latex_str = birds_res_df[birds_res_df.max_train_frac == 1][['bird', 'bayesian', 'baseline', 'boosting', 'ResNet']].to_latex(float_format="%.3f", index=False)
    return latex_str

def main():
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    
    birds_df = pd.read_csv(args.in_result_csv)

    latex_str = make_pivot_table(birds_df, metric="auc_str")
    logging.info(latex_str)
    print(latex_str)
    latex_str = make_pivot_table(birds_df, metric="log_lik_str")
    logging.info(latex_str)
    print(latex_str)

    # print(res.groupby(["method", "max_train_frac"]).mean())
    # res = res.melt(id_vars=["method", "seed", "max_train_frac"], var_name="metric")
    # sns.relplot(
    #     data=res,
    #     x="max_train_frac",
    #     y="auc",
    #     hue="method",
    #     # col="metric",
    #     kind="line",
    #     # facet_kws={"sharey": False},
    # )
    # plt.title(args.title)
    # plt.tight_layout()
    # plt.savefig(args.plot_file)

if __name__ == "__main__":
    main()