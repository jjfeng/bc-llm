import os
import argparse
import logging

import scipy.stats as stats
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

METHOD_COLS = ["bayesian", "baseline", "boosting", "Human+CBM", "Labelfree", "ResNet"]

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
    parser.add_argument(
        "--brier-csv-file",
        type=str,
    )
    parser.add_argument(
        "--ece-csv-file",
        type=str,
    )
    parser.add_argument(
        "--acc-csv-file",
        type=str,
    )
    parser.add_argument(
        "--auc-csv-file",
        type=str,
    )
    args = parser.parse_args()
    return args

def make_pivot_table(birds_df, metric):
    birds_res_df = birds_df.pivot(index=["bird", "max_train_frac"], columns="method", values=[metric]).reset_index(drop=False)
    birds_res_df['bird'] = birds_res_df['bird'].map({
    'class_1_2_3':  "Albatross",
    'class_5_6_7_8':  "Auklet",
    'class_9_10_11_12':  "Blackbird",
    'class_14_15_16':  "Bunting",
    'class_18_19':  "Catbird",
    'class_23_24_25':  "Cormorant",
    'class_26_27':  "Cowbird",
    'class_29_30':  "Crow",
    'class_31_32_33':  "Cuckoo",
    'class_34_35':  "Finch",
    'class_37_38_39_40_41_42_43':  "Flycatcher",
    'class_47_48':  "Goldfinch",
    'class_50_51_52_53':  "Grebe",
    'class_54_55_56_57':  "Grosbeak",
    'class_59_60_61_62_63_64_65_66':  "Gull",
    'class_67_68_69':  "Hummingbird",
    'class_71_72':  "Jaeger",
    'class_73_74_75':  "Jay",
    'class_77_78':  "Kingbird",
    'class_79_80_81_82_83':  "Kingfisher",
    'class_95_96_97_98':  "Oriole",
    'class_107_108':  "Raven",
    'class_111_112':  "Shrike",
    'class_113_114_115_116_117_118_119_120_121_122_123_124_125_126_127_128_129_130_131_132_133':  "Sparrow",
    'class_135_136_137_138':  "Swallow",
    'class_139_140':  "Tanager",
    'class_141_142_143_144_145_146_147':  "Tern",
    'class_149_150':  "Thrasher",
    'class_158_159_160_161_162_163_164_165_166_167_168_169_170_171_172_173_174_175_176_177_178_179_180_181_182': "Warbler",
    'class_183_184':  "Waterthrush",
    'class_185_186':  "Waxwing",
    'class_151_152_153_154_155_156_157':  "Vireo",
    'class_187_188_189_190_191_192':  "Woodpecker",
    'class_193_194_195_196_197_198_199':  "Wren",
    })
    birds_res_df.columns = ['bird', 'max_train_frac'] + birds_res_df.columns.droplevel().tolist()[2:]

    tab = birds_res_df[birds_res_df.max_train_frac == 1][['bird'] + METHOD_COLS].sort_values('bird')
    print(tab)
    return tab

def plot_rankings(res_df, file_name, max_better=True):
    if max_better:
        logging.info("bayesian %f", np.mean(np.isclose(res_df['bayesian'], res_df.max(axis=1, numeric_only=True))))
        logging.info("baseline %f", np.mean(np.isclose(res_df['baseline'], res_df.max(axis=1, numeric_only=True))))
        logging.info("boosting %f", np.mean(np.isclose(res_df['boosting'], res_df.max(axis=1, numeric_only=True))))
        logging.info("ResNet %f", np.mean(np.isclose(res_df['ResNet'], res_df.max(axis=1, numeric_only=True))))
        logging.info("Human+CBM %f", np.mean(np.isclose(res_df['Human+CBM'], res_df.max(axis=1, numeric_only=True))))
        logging.info("Labelfree %f", np.mean(np.isclose(res_df['Labelfree'], res_df.max(axis=1, numeric_only=True))))
    else:
        logging.info("bayesian %f", np.mean(np.isclose(res_df['bayesian'], res_df.min(axis=1, numeric_only=True))))
        logging.info("baseline %f", np.mean(np.isclose(res_df['baseline'], res_df.min(axis=1, numeric_only=True))))
        logging.info("boosting %f", np.mean(np.isclose(res_df['boosting'], res_df.min(axis=1, numeric_only=True))))
        logging.info("ResNet %f", np.mean(np.isclose(res_df['ResNet'], res_df.min(axis=1, numeric_only=True))))
        logging.info("Human+CBM %f", np.mean(np.isclose(res_df['Human+CBM'], res_df.min(axis=1, numeric_only=True))))
        logging.info("Labelfree %f", np.mean(np.isclose(res_df['Labelfree'], res_df.min(axis=1, numeric_only=True))))

    res_df_pivot = res_df.melt(id_vars="bird", value_vars=METHOD_COLS, value_name="value", var_name="Method")
    res_df_pivot["Method"] = res_df_pivot.Method.replace({
        "bayesian": "BC-LLM",
        "baseline": "LLM+CBM",
        "boosting": "Boosting LLM+CBM",
    })

    plt.clf()
    plt.subplots(figsize=(5,8))
    cmap = sns.color_palette(n_colors=5)
    sns.boxplot(
        data=res_df_pivot,
        x="value",
        y="Method",
        whis=[2.5, 97.5]
    )
    sns.swarmplot(
        data=res_df_pivot,
        x="value",
        y="Method",
        # hue="bird"
    )

    plt.tight_layout()
    plt.savefig(file_name)
    print("plotted at", file_name)

def get_ci_str(vals):
    mean_ent = np.mean(vals)
    std_dev_ent = np.sqrt(np.var(vals)/vals.size)
    return f"{mean_ent:.3f} \err"+"{"+ f"{mean_ent - 1.96 * std_dev_ent:.3f}, {mean_ent + 1.96 * std_dev_ent:.3f}" + "}"

def main():
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    
    birds_df = pd.read_csv(args.in_result_csv)

    auc_tab = make_pivot_table(birds_df, metric="auc")
    brier_tab = make_pivot_table(birds_df, metric="brier")
    acc_tab = make_pivot_table(birds_df, metric="acc")
    ece_tab = make_pivot_table(birds_df, metric="ece")
    res_df = pd.DataFrame({
        "method": METHOD_COLS,
        "acc": [get_ci_str(acc_tab[method_col]) for method_col in METHOD_COLS],
        "auc": [get_ci_str(auc_tab[method_col]) for method_col in METHOD_COLS],
        "brier": [get_ci_str(brier_tab[method_col]) for method_col in METHOD_COLS],
        "ece": [get_ci_str(ece_tab[method_col]) for method_col in METHOD_COLS],
    })
    print(res_df)
    logging.info(res_df.to_latex(index=False))

    logging.info("----AUC-----")
    auc_tab = make_pivot_table(birds_df, metric="auc")
    auc_tab.to_csv(args.auc_csv_file)
    print("AUC", auc_tab.mean(numeric_only=True))
    logging.info("AUC %s", auc_tab.mean(numeric_only=True))
    # plot_rankings(auc_tab, args.auc_csv_file.replace("csv", "png"))

    logging.info("----ACC-----")
    acc_tab = make_pivot_table(birds_df, metric="acc")
    acc_tab.to_csv(args.acc_csv_file)
    print("ACC", acc_tab.mean(numeric_only=True))
    logging.info("ACC %s", acc_tab.mean(numeric_only=True))
    # plot_rankings(acc_tab, args.acc_csv_file.replace("csv", "png"))
    
    logging.info("----Brier-----")
    brier_tab = make_pivot_table(birds_df, metric="brier")
    brier_tab.to_csv(args.brier_csv_file)
    print("brier", brier_tab.mean(numeric_only=True))
    logging.info("brier %s", brier_tab.mean(numeric_only=True))
    # plot_rankings(brier_tab, args.brier_csv_file.replace("csv", "png"), max_better=False)

    logging.info("----ECE-----")
    ece_tab = make_pivot_table(birds_df, metric="ece")
    ece_tab.to_csv(args.ece_csv_file)
    print("ece", ece_tab.mean(numeric_only=True))
    logging.info("ece %s", ece_tab.mean(numeric_only=True))

    # pivot_tab = make_pivot_table(birds_df, metric="log_lik_str")
    # logging.info(pivot_tab.to_latex(float_format="%.3f", index=False))
    # print(pivot_tab.to_latex(float_format="%.3f", index=False))

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