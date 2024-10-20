"""
Script to train a model on LLM concepts
"""
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import pickle

sys.path.append(os.getcwd())
from scripts.train_bayesian import load_data_partition, get_word_count_data, load_llms
from src.concept_learner_model import ConceptLearnerModel
import src.common as common
from src.training_history import TrainingHistory

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-obs", type=int, default=50000)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled training data")
    parser.add_argument("--keep-x-cols", type=str, nargs="*", help="tabular columns to force keep")
    parser.add_argument("--init-concepts-file", type=str, help="init concepts extract")
    parser.add_argument("--min-prevalence", type=float, default=0)
    parser.add_argument("--indices-csv", type=str, help="csv of training indices")
    parser.add_argument("--num-meta-concepts", type=int, default=5)
    parser.add_argument("--log-file", type=str, default="_output/log_train_baseline_concept.txt")
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--text-summary-column", type=str, default="llm_output", choices=['llm_output', 'sentence', 'spacy_output'])
    parser.add_argument("--max-section-length", type=int, default=None)
    parser.add_argument("--learner-type", type=str, default="count_l2", choices=['count_elasticnet','tfidf_l2', 'tfidf_l1', 'count_l2', 'count_l1'], help="model types")
    parser.add_argument("--baseline-init-file", type=str, default="exp_multi_concept/prompts/baseline_init.txt")
    parser.add_argument("--prompt-concepts-file", type=str, default="exp_multi_concept/prompts/concept_questions.txt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--in-training-history-file", type=str, default=None)
    parser.add_argument("--out-training-history-file", type=str, default=None)
    parser.add_argument("--out-extractions", type=str, default="_output/note_extractions.pkl")
    parser.add_argument("--is-image", action="store_true", default=False)
    parser.add_argument("--combine-by-id", type=str, default=None)
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
                "meta-llama/Llama-3.2-11B-Vision-Instruct",
                "meta-llama/Meta-Llama-3.1-70B-Instruct"
                ]
            )
    args = parser.parse_args()
    args.partition = "train"
    args.count_vectorizer, args.model = args.learner_type.split("_")
    args.keep_x_cols = pd.Series(args.keep_x_cols) if args.keep_x_cols is not None else None
    return args

def generate_prior_prompt(data_df, X_train, y_train, word_names, args, num_top: int = 40):
    word_names = word_names.tolist()
    is_multiclass = np.unique(y_train).size > 2
    X_keep = None
    if args.keep_x_cols is not None:
        print("KEEP COLS", args.keep_x_cols)
        print(common.TABULAR_PREFIX + args.keep_x_cols)
        X_keep = data_df[args.keep_x_cols].to_numpy()
        
    top_df = ConceptLearnerModel.fit_residual(args.model, word_names, X_keep, X_train, y_train, penalty_downweight_factor=100, is_multiclass=is_multiclass, num_top=num_top)
    print(top_df)
    
    normalization_factor = np.max(np.abs(top_df.coef))
    top_df['coef'] = top_df.coef/normalization_factor if normalization_factor > 0 else top_df.coef

    print("---------------------------")
    with open(args.baseline_init_file, 'r') as file:
        prompt_template = file.read()
        prompt_template = prompt_template.replace(
                "{top_features_df}", 
                top_df[['feature_name', 'coef']].to_csv(index=False, float_format='%.3f')
                )
        prompt_template = prompt_template.replace("{num_meta_concepts}", str(args.num_meta_concepts))
    print(prompt_template)
    return prompt_template
        
def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)

    history = TrainingHistory(args.keep_x_cols)

    data_df = load_data_partition(args, init_concepts_file=args.init_concepts_file, text_summary_column=args.text_summary_column)

    X_words_train, word_names = get_word_count_data(data_df, args.count_vectorizer, args.text_summary_column, min_prevalence=args.min_prevalence)
    y_train = data_df['y'].to_numpy().flatten()
    logging.info("y train prevalence %f", y_train.mean())
    logging.info("data_df shape %s", data_df.shape)
    logging.info("X_words shape %s", X_words_train.shape)

    llm_dict = load_llms(args)

    all_extracted_features = {}
    if os.path.exists(args.out_extractions):
        with open(args.out_extractions, "rb") as f:
            all_extracted_features = pickle.load(f)
    
    if args.in_training_history_file is not None:
        history = history.load(args.in_training_history_file)
        print(history._concepts)
        concept_dicts = history.get_last_concepts()
    else:
        init_llm_prompt = generate_prior_prompt(
            data_df,
            X_words_train,
            y_train,
            word_names=word_names,
            args=args
            )
        llm_response = llm_dict['iter'].get_output(init_llm_prompt, max_new_tokens=2500, is_image=args.is_image)
        logging.info(f"LLM PROMPT {init_llm_prompt}")
        logging.info(f"LLM RESPONSE {llm_response}")
        concept_dicts = ConceptLearnerModel.get_candidate_concepts(llm_response, keep_num_candidates=args.num_meta_concepts)
        print("concept dicts", concept_dicts)
    history.add_concepts(concept_dicts)

    all_extracted_features = common.extract_features_by_llm(
        llm_dict['extraction'],
        data_df, 
        concept_dicts,
        all_extracted_features_dict=all_extracted_features,
        prompt_file=args.prompt_concepts_file,
        batch_size=args.batch_size,
        extraction_file=args.out_extractions,
        is_image=args.is_image,
        max_section_length=args.max_section_length
    )
    
    X_train = common.get_features(concept_dicts, all_extracted_features, data_df, force_keep_columns=args.keep_x_cols)
    for concept_dict in concept_dicts:
        logging.info("concept %s", concept_dict['concept'])

    model_results = common.train_LR(X_train, y_train, penalty=None)
    history.add_auc(model_results["auc"])
    print("baseline train auc", model_results["auc"])
    logging.info("baseline train auc %s", model_results["auc"])
    history.add_coef(model_results["coef"])
    logging.info("baseline COEF %s", model_results["coef"])
    history.add_intercept(model_results["intercept"])
    history.add_model(model_results["model"])

    history.save(args.out_training_history_file)

if __name__ == "__main__":
    main(sys.argv[1:])
