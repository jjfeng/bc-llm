"""
Script to do bayesian inference over concepts
"""

import time
import re
import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import pickle
import torch

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score

sys.path.append(os.getcwd())
from src.concept_learner_model import ConceptLearnerModel
from src.llm.llm import LLM
from src.llm.llm_api import LLMApi
from src.llm.llm_local import LLMLocal
from src.training_history import TrainingHistory
import src.common as common

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-meta-concepts", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-greedy-epochs", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--num-restricted-epochs", type=int, default=0)
    parser.add_argument("--max-obs", type=int, default=0)
    parser.add_argument("--init-concepts-file", type=str, help="init concepts extract")
    parser.add_argument("--min-prevalence", type=float, default=0)
    parser.add_argument("--text-summary-column", type=str, default="llm_output", choices=['llm_output', 'sentence', 'spacy_output'])
    parser.add_argument("--indices-csv", type=str, help="csv of training indices")
    parser.add_argument("--keep-x-cols", type=str, nargs="*")
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--learner-type", type=str, default="count_l2", choices=['count_elasticnet','tfidf_l2', 'tfidf_l1', 'count_l2', 'count_l1', 'count_rf'], help="model types")
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled training data")
    parser.add_argument("--log-file", type=str, default="_output/log_train_concept.txt")
    parser.add_argument("--out-mdl", type=str, default="_output/bayesian_posterior.csv")
    parser.add_argument("--out-extractions-file", type=str, default="_output/note_extractions.pkl")
    parser.add_argument("--out-posterior", type=str, default="_output/posterior.pkl") 
    parser.add_argument("--prompt-iter-file", type=str, default="exp_multi_concept/prompts/bayesian_iter.txt")
    parser.add_argument("--prompt-iter-type", type=str, choices=["marginal", "conditional", "error_obs", "error_obs_w_conditional"], default="conditional")
    parser.add_argument("--prompt-concepts-file", type=str)
    parser.add_argument("--prompt-prior-file", type=str)
    parser.add_argument("--inverse-penalty-param", type=int, default=20000)
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--aucs-plot-file", type=str, default="_output/aucs.png")
    parser.add_argument("--training-history-file", type=str, default=None)
    parser.add_argument("--init-history-file", type=str)
    parser.add_argument("--is-image", action="store_true", default=False)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--num-top-residual-words", type=int, default=40)
    parser.add_argument("--do-greedy", action="store_true", default=False)
    parser.add_argument("--max-section-length", type=int, default=None)
    parser.add_argument(
            "--llm-model-type",
            type=str,
            default=None,
            choices=[
                "gpt-4o-mini",
                "versa-gpt-4o-2024-05-13",
                "meta-llama/Meta-Llama-3.1-8B-Instruct", 
                "meta-llama/Meta-Llama-3.1-70B-Instruct", 
                "meta-llama/Llama-3.2-11B-Vision-Instruct" 
                ]
            )
    parser.add_argument(
            "--llm-iter-type",
            type=str,
            default=None,
            choices=[
                "gpt-4o-mini",
                "versa-gpt-4o-2024-05-13",
                "meta-llama/Meta-Llama-3.1-8B-Instruct", 
                "meta-llama/Meta-Llama-3.1-70B-Instruct", 
                "meta-llama/Llama-3.2-11B-Vision-Instruct" 
                ]
            )
    parser.add_argument(
            "--llm-extraction-type",
            type=str,
            default=None,
            choices=[
                "gpt-4o-mini",
                "versa-gpt-4o-2024-05-13",
                "meta-llama/Meta-Llama-3.1-8B-Instruct", 
                "meta-llama/Meta-Llama-3.1-70B-Instruct", 
                "meta-llama/Llama-3.2-11B-Vision-Instruct" 
                ]
            )
    args = parser.parse_args()
    args.partition = "train"
    args.count_vectorizer, args.residual_model_type = args.learner_type.split("_")
    args.keep_x_cols = pd.Series(args.keep_x_cols) if args.keep_x_cols is not None else None
    return args

def load_data_partition(args, init_concepts_file=None, text_summary_column: str = "llm_output"):
    # Read in the LLM-extracted concepts from the notes
    dset = pd.read_csv(args.in_dataset_file)
    
    partition_df = pd.read_csv(args.indices_csv)
    print("partition_df", partition_df)

    if init_concepts_file is not None:
        dset_init_concepts = pd.read_csv(init_concepts_file)
        assert np.all(dset_init_concepts.sentence == dset.sentence)
        
        dset['llm_output'] = dset_init_concepts["llm_output"]

    # filter
    dset_partition = dset.iloc[partition_df[partition_df.partition == args.partition].idx].reset_index(drop=True)

    logging.info("DSET SIZE %s y prevalence %f", dset.shape, dset.y.mean())
    
    if args.max_obs > 0:
        dset_partition = dset_partition.iloc[:args.max_obs]
    logging.info("DSET PARTITION size %s", dset_partition.shape)

    print(dset_partition)
    print("FINAL NON-NA DSET", dset_partition.shape)
    return dset_partition

def comma_tokenizer(text):
    # a token is comma-separated phrases
    comma_phrases = [z.strip().lower() for z in text.split(',')]
    return comma_phrases

def get_word_count_data(
        dset, 
        count_vectorizer, 
        text_summary_column: str = "llm_output", 
        min_prevalence: float = 0, 
        vectorizer_out_file=None
        ):
    # Get basic word count vectorized data
    if count_vectorizer == 'count':
        vectorizer = CountVectorizer(tokenizer=comma_tokenizer, binary=True, strip_accents="ascii", lowercase=True)
    elif count_vectorizer == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1,1))
    else:
        raise NotImplementedError("vectorizer not recognized")

    vectorized_sentences = vectorizer.fit_transform(dset[text_summary_column]).toarray()
    if vectorizer_out_file:
        with open(vectorizer_out_file, "wb") as f:
            pickle.dump(vectorizer, f)
    word_names = vectorizer.get_feature_names_out()
    print("WORD FREQ MAX", vectorized_sentences.mean(axis=0).max())
    print("WORD FREQ MIN", vectorized_sentences.mean(axis=0).min())
    print("WORD FREQ MEAN", vectorized_sentences.mean(axis=0).mean())
    print("WORD FREQ MEDIAN", np.median(vectorized_sentences.mean(axis=0)))

    word_prevalences = vectorized_sentences.mean(axis=0)
    mask = word_prevalences >= min_prevalence
    vectorized_sentences = vectorized_sentences[:, mask]
    word_names = word_names[mask]

    return vectorized_sentences, word_names

def load_llms(args) -> dict[str, LLM]:
    if args.llm_model_type:
        if args.use_api:
            llm = LLMApi(args.seed, args.llm_model_type, logging)
        else:
            llm = LLMLocal(args.seed, args.llm_model_type, logging)
        return {
            "iter": llm,
            "extraction": llm
        }
    else:
        if args.use_api:
            llm_iter = LLMApi(args.seed, args.llm_iter_type, logging)
            llm_extraction = LLMApi(args.seed, args.llm_extraction_type, logging)
        elif args.llm_iter_type == args.llm_extraction_type:
            llm_iter = LLMLocal(args.seed, args.llm_iter_type, logging)
            llm_extraction = llm_iter
        else:
            raise NotImplementedError("only get to load one llm in local setting")
        return {
            "iter": llm_iter,
            "extraction": llm_extraction
        }

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    history = TrainingHistory(args.keep_x_cols).load(args.init_history_file)
    
    data_df = load_data_partition(args, init_concepts_file=args.init_concepts_file, text_summary_column=args.text_summary_column)

    X_features, feat_names = get_word_count_data(data_df, args.count_vectorizer, args.text_summary_column, min_prevalence=args.min_prevalence)
    print("data_df for prior generation", data_df.shape)
    print("num classes", data_df.y.unique().size)
    print("x_features", X_features.shape)

    # load any past extractions
    all_extracted_features_dict = {}
    if os.path.exists(args.out_extractions_file):
        with open(args.out_extractions_file, "rb") as f:
            all_extracted_features_dict = pickle.load(f)
    
    llm_dict = load_llms(args)

    assert args.num_restricted_epochs == 0

    bayesian_cbm = ConceptLearnerModel(
        init_history=history,
        llm_iter=llm_dict['iter'],
        llm_extraction=llm_dict['extraction'],
        num_classes=data_df.y.unique().size,
        num_meta_concepts=args.num_meta_concepts,
        prompt_iter_type=args.prompt_iter_type,
        prompt_iter_file=args.prompt_iter_file,
        prompt_concepts_file=args.prompt_concepts_file,
        prompt_prior_file=args.prompt_prior_file,
        out_extractions_file=args.out_extractions_file,
        residual_model_type=args.residual_model_type,
        inverse_penalty_param=args.inverse_penalty_param,
        train_frac=args.train_frac,
        num_greedy_epochs=args.num_greedy_epochs,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        do_greedy=args.do_greedy,
        is_image=args.is_image,
        all_extracted_features_dict=all_extracted_features_dict,
        max_section_length=args.max_section_length,
        force_keep_columns=args.keep_x_cols,
        num_top=args.num_top_residual_words,
        max_new_tokens=args.max_new_tokens,
    )
    bayesian_cbm.fit(
        data_df,
        X_features=X_features,
        feat_names=feat_names,
        training_history_file=args.training_history_file,
        aucs_plot_file=args.aucs_plot_file)

if __name__ == "__main__":
    main(sys.argv[1:])
