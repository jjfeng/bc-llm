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
from dotenv import load_dotenv

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from src.concept_learner_model import ConceptLearnerModel

sys.path.append('llm-api-main/lab_llm')
from lab_llm.constants import LLMModel, OpenAi, convert_to_llm_type
from lab_llm.llm_api import LLMApi
from lab_llm.llm_cache import LLMCache
from lab_llm.duckdb_handler import DuckDBHandler
from lab_llm.error_callback_handler import ErrorCallbackHandler
from src.training_history import TrainingHistory
import src.common as common

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-meta-concepts", type=int, default=5)
    parser.add_argument("--cache-file", type=str, default="cache.db")
    parser.add_argument("--batch-size", type=int, default=4, help="The number of LLM queries to batch together at once")
    parser.add_argument("--batch-concept-size", type=int, default=20, help="The number of concepts to annotate simultaneously, useful when there are a lot of concepts to annotate")
    parser.add_argument("--batch-obs-size", type=int, default=1, help="The number of observations to annotate simultaneously, useful for speeding up concept annotation when prompts are long")
    parser.add_argument("--num-greedy-epochs", type=int, default=0)
    parser.add_argument("--num-greedy-holdout", type=int, default=1)
    parser.add_argument("--is-greedy-metric-acc", action="store_true", default=False)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--num-minibatch", type=int, default=None)
    parser.add_argument("--max-obs", type=int, default=0)
    parser.add_argument("--init-concepts-file", type=str, help="init concepts extract")
    parser.add_argument("--min-prevalence", type=float, default=0)
    parser.add_argument("--text-summary-column", type=str, default="llm_output", choices=['llm_output', 'sentence', 'spacy_output'])
    parser.add_argument("--indices-csv", type=str, help="csv of training indices")
    parser.add_argument("--keep-x-cols", type=str, nargs="*")
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--learner-type", type=str, default="count_l2", choices=['count_elasticnet','tfidf_l2', 'tfidf_l1', 'count_l2', 'count_l1', 'count_rf'], help="model types")
    parser.add_argument("--final-learner-type", type=str, default=None, choices=['l1', 'l2', None], help="model type for the final CBM -- usually should be None to indicate an unpenalized logistic regression")
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled training data")
    parser.add_argument("--log-file", type=str, default="_output/log_train_concept.txt")
    parser.add_argument("--out-mdl", type=str, default="_output/bayesian_posterior.csv")
    parser.add_argument("--out-extractions-file", type=str, default="_output/note_extractions.pkl")
    parser.add_argument("--out-posterior", type=str, default="_output/posterior.pkl") 
    parser.add_argument("--config-file", type=str)
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
            )
    parser.add_argument(
            "--llm-iter-type",
            type=str,
            )
    parser.add_argument(
            "--llm-extraction-type",
            type=str,
            )
    args = parser.parse_args()
    args.partition = "train"
    assert args.use_api
    args.count_vectorizer, args.residual_model_type = args.learner_type.split("_")
    args.keep_x_cols = pd.Series(args.keep_x_cols) if args.keep_x_cols is not None else None
    args.cache = DuckDBHandler(args.cache_file)
    return args

def load_data_partition(args, init_concepts_file=None, text_summary_column: str = "llm_output"):
    # Read in the LLM-extracted concepts from the notes
    dset = pd.read_csv(args.in_dataset_file)
    
    partition_df = pd.read_csv(args.indices_csv)
    print("partition_df", partition_df, (partition_df.partition == "train").sum())

    if init_concepts_file is not None:
        dset_init_concepts = pd.read_csv(init_concepts_file)
        if 'sentence' in dset_init_concepts.columns:
            assert np.all(dset_init_concepts.sentence == dset.sentence)
        elif 'image_path' in dset_init_concepts.columns:
            assert np.all(dset_init_concepts.image_path == dset.image_path)
        
        dset['llm_output'] = dset_init_concepts["llm_output"]

    # filter
    dset_partition = dset.iloc[partition_df[partition_df.partition == args.partition].idx].reset_index(drop=True)
    if text_summary_column in dset_partition.columns:
        dset_partition = dset_partition[~dset_partition[text_summary_column].isna()]
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

    vectorized_sentences = vectorizer.fit_transform(dset[text_summary_column]).toarray().astype(int)
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
    print("FINAL X SHAPE", vectorized_sentences.shape)

    return vectorized_sentences, word_names

def load_llms(args) -> dict[str, LLMApi]:
    assert args.cache_file is not None
    
    logger = logging.getLogger(__name__)
    load_dotenv()
    cache = LLMCache(DuckDBHandler(args.cache_file))
    if args.llm_model_type is not None:
        args.llm_model_type = convert_to_llm_type(args.llm_model_type)
    if args.llm_iter_type is not None:
        args.llm_iter_type = convert_to_llm_type(args.llm_iter_type)
    if args.llm_extraction_type is not None:
        args.llm_extraction_type = convert_to_llm_type(args.llm_extraction_type)

    if args.llm_model_type:
        llm = LLMApi(cache, seed=10, model_type=args.llm_model_type, error_handler=ErrorCallbackHandler(logger), logging=logging, timeout=120)
        return {
            "iter": llm,
            "extraction": llm
        }
    else:
        llm_iter = LLMApi(cache, seed=10, model_type=args.llm_iter_type, error_handler=ErrorCallbackHandler(logger), logging=logging)
        llm_extraction = LLMApi(cache, seed=10, model_type=args.llm_extraction_type, error_handler=ErrorCallbackHandler(logger), logging=logging)
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

    all_extracted_features_dict = {}
    llm_dict = load_llms(args)

    config_dict = {}
    if args.config_file:
        with open(args.config_file, "r") as f:
            config_dict = json.load(f)

    bayesian_cbm = ConceptLearnerModel(
        init_history=history,
        llm_iter=llm_dict['iter'],
        llm_extraction=llm_dict['extraction'],
        num_classes=data_df.y.unique().size,
        num_minibatch=args.num_minibatch,
        num_meta_concepts=args.num_meta_concepts,
        prompt_iter_type=args.prompt_iter_type,
        prompt_iter_file=args.prompt_iter_file,
        config=config_dict,
        prompt_concepts_file=args.prompt_concepts_file,
        prompt_prior_file=args.prompt_prior_file,
        out_extractions_file=args.out_extractions_file,
        residual_model_type=args.residual_model_type,
        final_learner_type=args.final_learner_type,
        inverse_penalty_param=args.inverse_penalty_param,
        train_frac=args.train_frac,
        num_greedy_epochs=args.num_greedy_epochs,
        num_greedy_holdout=args.num_greedy_holdout,
        is_greedy_metric_acc=args.is_greedy_metric_acc,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        batch_concept_size=args.batch_concept_size,
        batch_obs_size=args.batch_obs_size,
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
