import sys
import os
import argparse
import logging
import pickle

import scipy
import scipy.stats as stats
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

sys.path.append(os.getcwd())
from src.training_history import TrainingHistory

COLORS = sns.color_palette()

def parse_args():
    parser = argparse.ArgumentParser(
        description="plot embeddings for concepts"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-train-obs",
        type=int
    )
    parser.add_argument(
        "--method-name",
        type=str
    )
    parser.add_argument(
        "--oracle-history-file",
        type=str
    )
    parser.add_argument(
        "--model-history-file",
        type=str
    )
    parser.add_argument(
        "--model-extraction-file",
        type=str,
    )
    parser.add_argument(
        "--oracle-extraction-file",
        type=str,
    )
    parser.add_argument(
        "--coverage-csv",
        type=str,
    )
    parser.add_argument("--num-posterior-iters", type=int, default=1)
    args = parser.parse_args()
    return args

def clean_str(concept):
    image_verbs = ["depict", "represent", "illustrate", "indicate", "report", "show", "contain", "feature", "include", "portray", "have", "showcase", "refer", "suggest"]
    for verb in image_verbs:
        concept = concept.replace(f"Does this note {verb}", "Does the note mention")
        concept = concept.replace(f"Does the note {verb}", "Does the note mention")
    return concept

def make_clean_str(concept):
    concept = clean_str(concept)
    concept = concept.replace("Does the note mention or imply ", "").replace("that the patient ", "").replace("the patient ", "")
    return concept.replace("?", "")

def main():
    args = parse_args()
    np.random.seed(args.seed)

    with open(args.oracle_extraction_file, "rb") as f:
        oracle_extracted_feature_dict = pickle.load(f)
    with open(args.model_extraction_file, "rb") as f:
        model_extracted_feature_dict = pickle.load(f)

    oracle_history = TrainingHistory().load(args.oracle_history_file)
    history = TrainingHistory().load(args.model_history_file)
    
    oracle_concepts = oracle_history.get_last_concepts()
    start_iter = max(0, history.num_iters - args.num_posterior_iters)
    num_meta_concepts = len(oracle_concepts)
    correlation_dfs = []
    for oracle_concept_dict in oracle_concepts:
        oracle_concept_str = oracle_concept_dict['concept']
        oracle_concept_extract = oracle_extracted_feature_dict[oracle_concept_str].flatten()
        for posterior_iter, model_concept_dicts in enumerate(history._concepts[start_iter:history.num_iters]):
            for concept_idx, model_concept_dict in enumerate(model_concept_dicts):
                model_concept_str = model_concept_dict['concept']
                model_concept_extract = model_extracted_feature_dict[model_concept_str].flatten()
                corr = stats.pearsonr(oracle_concept_extract, model_concept_extract)[0]
                correlation_dfs.append(pd.DataFrame({
                    "oracle_concept": [make_clean_str(oracle_concept_str)],
                    "model_concept": [make_clean_str(model_concept_str)],
                    "abs_corr": [np.abs(corr)],
                    "iter": [posterior_iter],
                    "concept_idx": [concept_idx],
                    # "corr": [corr],
                }))
    correlation_df = pd.concat(correlation_dfs).drop_duplicates().reset_index(drop=True)

    top_corr_df = correlation_df.sort_values("abs_corr", ascending=False).groupby('oracle_concept').head(1)
    top_corr_df = top_corr_df.sort_values('oracle_concept')
    top_corr_df['metric'] = 'recall'
    print(top_corr_df)
    bottom_corr_df = correlation_df.sort_values("abs_corr", ascending=False).groupby(['iter', 'model_concept']).head(1)
    bottom_corr_df = bottom_corr_df.sort_values(['iter', 'concept_idx'])
    bottom_corr_df['metric'] = 'precision'
    print(bottom_corr_df)
    print(bottom_corr_df.iloc[-20:])
    
    corr_df = pd.concat([top_corr_df, bottom_corr_df])
    corr_df['seed'] = args.seed
    corr_df['method'] = args.method_name
    corr_df['max_train_obs'] = args.max_train_obs
    corr_df.to_csv(args.coverage_csv)

            
                

    
    
    
if __name__ == "__main__":
    main()