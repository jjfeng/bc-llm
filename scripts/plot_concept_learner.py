"""
Script to plot concept learner
"""

import os
import sys
import logging
import argparse
import torch
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from scripts.train_concept import make_dataset, setup
from src.common import load_model

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-concepts", type=int, default=1)
    parser.add_argument("--num-top-words", type=int, default=20)
    parser.add_argument("--perplexity", type=int, default=10)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the training data")
    parser.add_argument("--embeddings-file", type=str, help="npz of the embeddings")
    parser.add_argument("--learner-type", type=str, choices=["kernel", "direct"])
    parser.add_argument("--in-concept-mdl", type=str, help="the learned concept model")
    parser.add_argument("--plot-attention-file", type=str, default="_output/attention_concepts.png")
    parser.add_argument("--plot-dim-red-file", type=str, default="_output/tsne_concepts.png")
    parser.add_argument("--words-to-highlight", type=str, help="comma separated list of words to highlight in plots")
    parser.add_argument("--log-file", type=str, default="_output/log_plot.txt")
    parser.add_argument("--bootstrap-params", type=str, help="file that stores the values for the bootstrapped parameters for the concept model")
    # currently the code works with llama models
    parser.add_argument(
            "--llm-model-type",
            type=str,
            default="meta-llama/Meta-Llama-3-8B-Instruct",
            choices=["meta-llama/Meta-Llama-3-8B-Instruct"]
            )
    args = parser.parse_args()
    args.words_to_highlight = args.words_to_highlight.split(",")
    return args

def get_top_attention_df(word_df, num_top_words):
    return pd.concat([
        word_df[(word_df.seed == seed) & (word_df.coef == coef)].sort_values(by='attention', ascending=False).iloc[:num_top_words]
        for seed in word_df.seed.unique() for coef in word_df[word_df.seed == seed].coef.unique()])

def plot_dim_red(all_gammas, all_top_dicts, filtered_embeddings, plot_file, num_top_words=10, num_other_words = 40, key="standardized", words_to_highlight=[]):
    all_gammas = np.concatenate(all_gammas, axis=0)
    print("ALL GAMMAS", all_gammas.shape)

    # plot only concepts with top attention or top variation in pr(c|x)
    word_df = all_top_dicts[key]
    # words with top variation in pr(c|x)
    top_prob_sd_df = word_df[(word_df.seed == 0) & (word_df.coef > 0)].sort_values(by='prob_sd', ascending=False).iloc[:num_other_words]
    other_word_df = top_prob_sd_df[['filter_token_id', 'word', 'coef', 'attention']]

    # words with top attention
    top_attent_df = get_top_attention_df(word_df, num_top_words)
    print("TOP ATTNET", top_attent_df)
    top_word_df = top_attent_df[['filter_token_id', 'word', 'coef', 'attention']].drop_duplicates()
    plot_word_df = pd.concat([top_word_df, other_word_df]).drop_duplicates()

    numerics_df = word_df[['filter_token_id', 'prob_sd']].groupby('filter_token_id').max().reset_index()
    print(word_df)
    print(numerics_df)
    plot_word_df = plot_word_df.merge(numerics_df, on=['filter_token_id'])
    print("PLOT WORD", plot_word_df)

    # Apply dimension reduction
    gamma_norms = np.linalg.norm(all_gammas, axis=1, keepdims=True)
    all_plot_embeddings = np.concatenate([
        all_gammas/gamma_norms,
        filtered_embeddings[plot_word_df.filter_token_id],
        ], axis=0)
    num_gammas = all_gammas.shape[0]

    tsne = PCA(n_components=2)
    dim_red_embeddings = tsne.fit_transform(all_plot_embeddings)
    gamma_embeddings = tsne.transform(all_gammas/gamma_norms)
    attentions = plot_word_df.attention.to_numpy()

    # Actually make the plot
    # TODO: remove hack

    plt.clf()
    plt.figure(figsize=(15,8))
    COLORS = ["blue", "red"]
    plt.scatter(
        dim_red_embeddings[:,0],
        dim_red_embeddings[:,1],
        c=["green"] * num_gammas + [COLORS[int(i)] for i in (plot_word_df.coef < 0).to_list()],
        s=np.concatenate([[10] * num_gammas, np.power(attentions/attentions.mean(), 2) * 16]))
    for i, txt in enumerate(plot_word_df.word):
        fontweight = 'bold' if txt.strip().strip("Ġ").lower() in words_to_highlight else 'light'
        plt.annotate(txt, dim_red_embeddings[num_gammas + i], weight=fontweight)
    plt.savefig(plot_file)

def plot_attentions(all_top_dicts, num_top_words, plot_file):
    # TODO: if a word doesnt appear in the top for some of the bootstrapped models, then
    # it doesnt have an attention value and it is not included in boxplots. This plotting needs to be fixed
    # so that we get attentions for all words, and then only plot the words with largest average
    # attention or something like that.
    plt.clf()
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False, sharex=False, figsize=(14,8))

    for key, ax in [("raw", ax1), ("standardized", ax2)]:
        sns.boxplot(ax=ax, data=get_top_attention_df(all_top_dicts[key], num_top_words), x="word", y="attention", hue="coef")
        ax.set_ylim(bottom=0)
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.savefig(plot_file)

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)

    device, english_filter = setup(args)

    dataset, llm_token_embed = make_dataset(args, english_filter, device)
    dataloader = DataLoader(dataset, batch_size=len(list(dataset)), shuffle=False)

    # Load the one model and the bootstrapped models
    cl_model = load_model(args.learner_type, english_filter, llm_token_embed, num_concepts=args.num_concepts, device=device, file=args.in_concept_mdl, logging=logging)
    all_top_dicts = cl_model.print_top_tokens(dataloader)
    for key, df in all_top_dicts.items():
        df['seed'] = 0
        print(df)

    all_gammas = [cl_model.get_gamma().detach().cpu().T]
    if args.bootstrap_params is not None:
        for i, state_dict in enumerate(torch.load(args.bootstrap_params)):
            cl_model.load_state_dict(state_dict)
            top_dicts = cl_model.print_top_tokens(dataloader)
            all_gammas.append(cl_model.get_gamma().detach().cpu().T)
            for key, df in top_dicts.items():
                df['seed'] = i + 1
                print("KEY", key, df.iloc[:5])
                all_top_dicts[key] = pd.concat([all_top_dicts[key], df])

    # Plot dimension reduction -- gammas and word embeddings
    plot_dim_red(all_gammas, all_top_dicts, cl_model.embed_weights_norm.detach().cpu().numpy(), args.plot_dim_red_file, num_top_words=args.num_top_words, num_other_words=args.num_top_words * 4, words_to_highlight=args.words_to_highlight)

    # Plot attention of top tokens
    plot_attentions(all_top_dicts, num_top_words=args.num_top_words * 2, plot_file=args.plot_attention_file)

if __name__ == "__main__":
    main(sys.argv[1:])
