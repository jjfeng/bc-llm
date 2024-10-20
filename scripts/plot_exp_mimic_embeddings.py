import sys
import os
import argparse
import logging
import pickle

import sklearn
import scipy
import scipy.spatial
import scipy.stats as stats
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn import manifold

sys.path.append(os.getcwd())
from src.training_history import TrainingHistory

COLORS = sns.color_palette()

def parse_args():
    parser = argparse.ArgumentParser(
        description="plot embeddings for concepts"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--history-files",
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--extraction-files",
        type=str,
        nargs="*"
    )
    parser.add_argument("--num-posterior-iters", type=int, default=1)
    parser.add_argument(
        "--plot-embeds-file",
        type=str,
    )
    parser.add_argument(
        "--plot-hierarchical-file",
        type=str,
        default="_output/hier.png"
    )
    parser.add_argument(
        "--concepts-csv-file",
        type=str,
    )
    args = parser.parse_args()
    return args

def clean_str(concept):
    image_verbs = ["depict", "represent", "illustrate", "indicate", "report", "show", "contain", "feature", "include", "portray", "have", "showcase", "refer", "suggest"]
    for verb in image_verbs:
        concept = concept.replace(f"Does this note {verb}", "Does the note mention")
        concept = concept.replace(f"Does the note {verb}", "Does the note mention")
    concept = concept.replace(f"Does the note mention that the patient", "Does the note mention the patient")
    return concept#.replace("Does the note mention the patient ", "").replace("Does the note mention ", "")

def make_clean_str(concept):
    concept = clean_str(concept)
    concept = concept.replace("Does the note mention ", "").replace("that the patient ", "").replace("the patient ", "")
    concept = concept.replace("engaging in ", "").replace("experiencing ", "").replace("having ", "").replace("undergoing ", "")
    concept = concept.replace("the patient's vital signs being abnormal", "abnormal vital signs")
    return concept.replace("?", "")

def dist_func(x, y):
    return 1 - np.abs(scipy.stats.pearsonr(x,y)[0])

def main():
    args = parse_args()
    np.random.seed(args.seed)

    all_extracted_features = {}
    for extraction_file in args.extraction_files:
        with open(extraction_file, "rb") as f:
            all_extracted_features = all_extracted_features | pickle.load(f)
    
    all_concepts_df = []
    all_concept_sizes = []
    all_concepts_to_embed_dfs = []
    for idx, history_file in enumerate(args.history_files):
        logging.info(f"========history file {history_file}")
        history = TrainingHistory().load(history_file)
        start_iter = max(0, history.num_iters - args.num_posterior_iters)
        concepts_to_embed = list(
            [
                concept_dict['concept']
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
    for plot_idx, idx in enumerate([0,3]):
        concepts_df = all_concepts_df[idx].T
        print(concepts_df)
        all_embeddings = np.array([all_extracted_features[c][:,0] for c in concepts_df.index])
        Z = scipy.cluster.hierarchy.linkage(all_embeddings, 'single', metric=dist_func)
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