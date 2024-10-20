import sys
import os
import argparse
import logging
import pickle

import scipy.stats as stats
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import scipy
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.cluster.hierarchy import dendrogram, linkage

sys.path.append(os.getcwd())
from src.training_history import TrainingHistory

COLORS = ['red', 'purple', 'green']

def parse_args():
    parser = argparse.ArgumentParser(
        description="plot embeddings for concepts"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--history-files",
        nargs="*",
        type=str,
    )
    parser.add_argument("--num-posterior-iters", type=int, default=1)
    parser.add_argument(
            "--extraction-files",
            type=str,
            nargs="*"
            )
    parser.add_argument(
        "--plot-hierarchical-file",
        type=str,
    )
    args = parser.parse_args()
    return args

def get_equiv_str(concept):
    if concept == "Does this image contain bright feathers?":
        return "Does this image depict a bird with bright feathers?"
    elif concept == "Does this image feature an orange chest?":
        return "Does this image feature a bird with an orange chest?"
    else:
        return concept

def make_clean_str(concept):
    image_verbs = ["depict", "represent", "illustrate", "show", "contain", "feature", "include", "portray", "have", "showcase"]
    for verb in image_verbs:
        concept = concept.replace(f"Does this image {verb} ", "")
        concept = concept.replace(f"Does the image {verb} ", "")
    concept = concept.replace(f"a bird found in an oceanic environment", "an ocean environment")
    return concept.replace("?", "").replace("a bird with ", "").replace("a bird in ", "").replace("birds with ", "").replace("a bird that is ", "")

def dist_func(x, y):
    corr = scipy.stats.pearsonr(x,y)[0]
    if np.isnan(corr):
        return 1
    else:
        return 1 - np.abs(corr)

def main():
    args = parse_args()
    np.random.seed(args.seed)

    all_extracted_features = {}
    for extraction_file in args.extraction_files:
        with open(extraction_file, "rb") as f:
            all_extracted_features = all_extracted_features | pickle.load(f)
    print(all_extracted_features.keys())

    all_concepts_df = []
    all_concept_sizes = []
    all_concepts_to_embed_dfs = []
    for idx, history_file in enumerate(args.history_files):
        print("history", idx)
        logging.info(f"========history file {history_file}")
        history = TrainingHistory().load(history_file)
        start_iter = max(0, history.num_iters - args.num_posterior_iters)
        concepts_to_embed = list(
            [
                get_equiv_str(concept_dict['concept'])
                for concept_dicts in history._concepts[start_iter:history.num_iters] for concept_dict in concept_dicts
            ])
        concepts_to_embed = pd.Series(concepts_to_embed)
        concepts_to_embed_df = concepts_to_embed.value_counts()/(history.num_iters - start_iter + 1)
        all_concepts_to_embed_dfs.append(concepts_to_embed_df)
        posterior_probs = concepts_to_embed_df.to_numpy()
        concepts_df = pd.DataFrame({
            concepts_to_embed_df.index[c_idx]: [posterior_probs[c_idx]]
            for c_idx in np.arange(concepts_to_embed_df.index.size)
        })
        all_concepts_df.append(concepts_df)
        all_concept_sizes.append((100 * posterior_probs)**2)
    
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    for plot_idx, idx in enumerate([0,1]):
        concepts_df = all_concepts_df[idx].T
        print(concepts_df)
        all_embeddings = np.array([all_extracted_features[c][:,0] for c in concepts_df.index])
        Z = scipy.cluster.hierarchy.linkage(all_embeddings, 'average', metric=dist_func)
        hierarchy.dendrogram(
            Z,
            labels=[make_clean_str(txt + f" ({weight:.2f})") for txt, weight in zip(concepts_df.index, concepts_df[0])],
            leaf_rotation=0,
            ax=axes[plot_idx] if len(all_concepts_df) > 1 else axes,
            color_threshold=0,
            orientation='right')
    plt.tight_layout()
    plt.savefig(args.plot_hierarchical_file)

if __name__ == "__main__":
    main()